// hybrid_dynamic.cu
// Despacho híbrido DINÂMICO por bloco com controle de feedback:
//   - Semáforo (mutex) nas filas de I/O compartilhadas
//   - O dispatcher é o UNICO que decide o destino de cada bloco:
//       * mede o TEMPO DE RESPOSTA real da GPU (CUDA events) e da CPU
//         (contador atomico de blocos concluidos + relogio)
//       * estima o backlog restante de cada lado (pendentes * resposta/bloco)
//       * entrega o bloco ao lado com MENOR tempo de conclusao estimado
//   - Escala dinamicamente conforme a resposta medida (feedback)
//
// Compilar: nvcc -O3 -arch=sm_89 -o hybrid_dynamic hybrid_dynamic.cu -lpthread
// Rodar:    ./hybrid_dynamic [N] [json] [fma] [gpu_lag]

#include <cuda_runtime.h>
#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

constexpr size_t N_DEFAULT = 64ULL * 1024 * 1024;
constexpr size_t BLOCK_ELEMS = 1ULL << 20;   // 1M elem por bloco
constexpr int REPEATS = 3;

static int g_fma = 128;
static float g_lag = 1.0f;                   // fator de resposta da GPU (argv[4])
__constant__ int c_fma;

// ------------------------ Kernel ------------------------
__global__ void compute_offsets(const float* __restrict__ in, float* __restrict__ out,
                                float alpha, const size_t* __restrict__ gpu_base,
                                int batch, size_t be, float lag) {
    int b = blockIdx.x;
    size_t base = gpu_base[b];
    for (size_t i = base + threadIdx.x; i < base + be; i += blockDim.x) {
        float v = in[i];
        int n = (int)(c_fma * lag);
        for (int j = 0; j < n; ++j)
            v = fmaf(v, alpha, 0.001f * j);
        out[i] = v;
    }
}

static inline void cpu_compute(const float* __restrict__ in, float* __restrict__ out,
                               float alpha, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        float v = in[i];
        for (int j = 0; j < g_fma; ++j)
            v = fmaf(v, alpha, 0.001f * j);
        out[i] = v;
    }
}

// ------------------------ Semaforo de I/O (fila compartilhada) ------------------------
struct IoQueue {
    pthread_mutex_t mtx;
    std::vector<size_t> ids;
    size_t pos;
    IoQueue() { pthread_mutex_init(&mtx, nullptr); pos = 0; }
    ~IoQueue() { pthread_mutex_destroy(&mtx); }
    void init(const std::vector<size_t>& v) { ids = v; pos = 0; }
    long pop() {
        pthread_mutex_lock(&mtx);
        long id = -1;
        if (pos < ids.size()) id = (long)ids[pos++];
        pthread_mutex_unlock(&mtx);
        return id;
    }
    void push(size_t id) {
        pthread_mutex_lock(&mtx);
        ids.push_back(id);
        pthread_mutex_unlock(&mtx);
    }
    long remaining() {
        pthread_mutex_lock(&mtx);
        long r = (long)(ids.size() - pos);
        pthread_mutex_unlock(&mtx);
        return r;
    }
};

struct CpuArg {
    const float* in; float* out; float alpha;
    size_t be; int core;
    IoQueue* queue;                    // fila da CPU (controlada pelo dispatcher)
    std::atomic<long>* cpu_done;
};

static void* cpu_worker(void* arg_) {
    CpuArg* a = (CpuArg*)arg_;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(a->core, &set);
    sched_setaffinity(0, sizeof(set), &set);
    long id;
    while ((id = a->queue->pop()) >= 0) {
        cpu_compute(a->in, a->out, a->alpha, (size_t)id * a->be, (size_t)id * a->be + a->be);
        a->cpu_done->fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

// ------------------------ Dispatcher dinâmico (controlador) ------------------------
struct DispatchStats {
    long gpu_blocks = 0, cpu_blocks = 0;
    double gpu_ms_per_block = 0, cpu_ms_per_block = 0;
    std::vector<std::vector<double>> trace;   // {t_ms, est_gpu, est_cpu, bloco_destino}
};

struct PendingBatch { cudaEvent_t s, e; long n; };

struct DispatcherArg {
    const float* in; float* out; float alpha; size_t be;
    IoQueue* main_queue;   // blocos ainda nao despachados
    IoQueue* cpu_queue;    // blocos dados a CPU
    size_t* gpu_base_host; size_t* d_gpu_base;
    std::atomic<long>* cpu_done; std::atomic<long>* gpu_done;
    int block;
    double cpu_init_ms_per_block;
    DispatchStats* st;
};

static void* gpu_dispatcher(void* arg_) {
    DispatcherArg* dp = (DispatcherArg*)arg_;
    auto& in = dp->in; auto& out = dp->out; auto& alpha = dp->alpha; auto be = dp->be;
    auto& mainq = *dp->main_queue; auto& cpuq = *dp->cpu_queue;
    auto& cpu_done = *dp->cpu_done; auto& gpu_done = *dp->gpu_done;
    auto& st = *dp->st; int block = dp->block;

    double gpu_ms_per_block = 0.02;   // estimativa inicial
    double cpu_ms_per_block = dp->cpu_init_ms_per_block;
    long gpu_assigned = 0, cpu_assigned = 0;
    long batch_n = 0;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    std::vector<PendingBatch> pending;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_last = t_start;
    long cpu_last = 0;
    long probe = 1;   // sonda: 1 bloco GPU no inicio para medir a resposta real

    double loop_ms = 0, drain_ms = 0, evq_ms = 0;
    auto t_phase = t_start;
    while (true) {
        long rem = mainq.remaining();
        if (rem <= 0) break;

        // resposta real da CPU (blocos concluidos por tempo)
        auto tn = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(tn - t_last).count();
        long dcpu = cpu_done.load(std::memory_order_relaxed);
        if (dt > 4.0 && dcpu > cpu_last) {
            cpu_ms_per_block = 0.5 * cpu_ms_per_block +
                               0.5 * (dt / (double)(dcpu - cpu_last));
            cpu_last = dcpu; t_last = tn;
        }

        // resposta real da GPU (lotes concluidos via CUDA events)
        for (auto it = pending.begin(); it != pending.end(); ) {
            if (cudaEventQuery(it->e) == cudaSuccess) {
                float ms = 0;
                cudaEventElapsedTime(&ms, it->s, it->e);
                cudaEventDestroy(it->s); cudaEventDestroy(it->e);
                double per = ms / (double)it->n;
                gpu_ms_per_block = (gpu_ms_per_block == 0) ? per
                                  : 0.6 * gpu_ms_per_block + 0.4 * per;
                gpu_done.fetch_add(it->n, std::memory_order_relaxed);
                it = pending.erase(it);
            } else ++it;
        }

        // feedback: reparte os blocos restantes pela RAZÃO das taxas medidas
        // (terminar junto: n_gpu/n_cpu = r_gpu/r_cpu)
        double r_g = 1.0 / std::fmax(gpu_ms_per_block, 1e-6);
        double r_c = 1.0 / std::fmax(cpu_ms_per_block, 1e-6);
        double frac_cpu = r_c / (r_g + r_c);
        long cpu_target = (long)std::llround((double)rem * frac_cpu);
        long cpu_pending = cpu_assigned - cpu_done.load(std::memory_order_relaxed);

        // backlog estimado (para o traco)
        long pend_gpu = gpu_assigned - gpu_done.load(std::memory_order_relaxed);
        long pend_cpu = cpu_assigned - cpu_done.load(std::memory_order_relaxed);
        double est_gpu = std::fmax(0.0, (double)pend_gpu * gpu_ms_per_block);
        double est_cpu = std::fmax(0.0, (double)pend_cpu * cpu_ms_per_block);

        long id = mainq.pop();           // dispatcher é o único que decide
        if (id < 0) break;

        int destino;
        bool force_flush = false;
        if (probe > 0) {           // sonda inicial: 1 bloco GPU p/ medir resposta real
            probe--;
            destino = 1;
            force_flush = true;
        } else {
            destino = (cpu_pending < cpu_target) ? 0 : 1;   // 0=CPU, 1=GPU
        }
        if (destino == 1) {
            dp->gpu_base_host[st.gpu_blocks + batch_n] = (size_t)id * be;
            gpu_assigned++;
            batch_n++;
            if (force_flush || batch_n >= 32 || mainq.remaining() <= 0) {
                cudaEvent_t evS, evE;
                cudaEventCreate(&evS); cudaEventCreate(&evE);
                cudaEventRecord(evS, stream);
                compute_offsets<<<(int)batch_n, block, 0, stream>>>(
                    in, out, alpha, dp->d_gpu_base + st.gpu_blocks,
                    (int)batch_n, be, g_lag);
                cudaEventRecord(evE, stream);
                pending.push_back({evS, evE, batch_n});
                st.gpu_blocks += batch_n;
                batch_n = 0;
            }
        } else {
            cpuq.push((size_t)id);
            cpu_assigned++;
        }

        double now_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        if ((int)st.trace.size() < 2000)
            st.trace.push_back({now_ms, est_gpu, est_cpu, (double)destino});
        loop_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_phase).count();
        t_phase = std::chrono::high_resolution_clock::now();
    }
    auto t_drain0 = std::chrono::high_resolution_clock::now();
    // drena lotes GPU restantes e pendentes
    if (batch_n > 0) {
        cudaEvent_t evS, evE;
        cudaEventCreate(&evS); cudaEventCreate(&evE);
        cudaEventRecord(evS, stream);
        compute_offsets<<<(int)batch_n, block, 0, stream>>>(
            in, out, alpha, dp->d_gpu_base + st.gpu_blocks, (int)batch_n, be, g_lag);
        cudaEventRecord(evE, stream);
        pending.push_back({evS, evE, batch_n});
        st.gpu_blocks += batch_n;
    }
    for (auto& p : pending) {
        cudaEventSynchronize(p.e);
        float ms = 0; cudaEventElapsedTime(&ms, p.s, p.e);
        gpu_done.fetch_add(p.n, std::memory_order_relaxed);
        gpu_ms_per_block = (gpu_ms_per_block == 0) ? (double)ms / p.n
                          : 0.6 * gpu_ms_per_block + 0.4 * (double)ms / p.n;
        cudaEventDestroy(p.s); cudaEventDestroy(p.e);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    drain_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_drain0).count();

    if (getenv("HYBRID_DEBUG")) {
        printf("[dbg] gpu_blocks=%ld cpu_blocks=%ld gpu_ms_per_block=%.4f cpu_ms_per_block=%.3f "
               "loop_ms=%.1f drain_ms=%.1f evq_ms=%.1f\n",
               st.gpu_blocks, st.cpu_blocks, gpu_ms_per_block, cpu_ms_per_block,
               loop_ms, drain_ms, evq_ms);
    }

    st.gpu_ms_per_block = gpu_ms_per_block;
    st.cpu_ms_per_block = cpu_ms_per_block;
    return nullptr;
}

// ------------------------ Execução dinâmica ------------------------
static double run_dynamic(const float* d_in, float* d_out, float* h_in, float* h_out,
                          float alpha, size_t total, const std::vector<int>& phys,
                          int block, size_t* gpu_base_host, size_t* d_gpu_base,
                          double cpu_init_ms_per_block, DispatchStats* st) {
    size_t nb = total / BLOCK_ELEMS;
    std::vector<size_t> ids;
    for (size_t i = 0; i < nb; ++i) ids.push_back(i);
    IoQueue mainq, cpuq;
    mainq.init(ids);
    cpuq.init({});

    std::atomic<long> cpu_done(0), gpu_done(0);

    std::vector<CpuArg> cargs;
    std::vector<pthread_t> ths;
    for (size_t i = 0; i < phys.size(); ++i)
        cargs.push_back({h_in, h_out, alpha, BLOCK_ELEMS, phys[i], &cpuq, &cpu_done});

    auto t0 = std::chrono::high_resolution_clock::now();
    pthread_t disp;
    DispatcherArg darg{d_in, d_out, alpha, BLOCK_ELEMS, &mainq, &cpuq,
                       gpu_base_host, d_gpu_base, &cpu_done, &gpu_done,
                       block, cpu_init_ms_per_block, st};
    pthread_create(&disp, nullptr, gpu_dispatcher, &darg);
    auto t_create = std::chrono::high_resolution_clock::now();
    for (auto& a : cargs) { pthread_t t; pthread_create(&t, nullptr, cpu_worker, &a); ths.push_back(t); }
    auto t_workers = std::chrono::high_resolution_clock::now();
    pthread_join(disp, nullptr);
    auto t_join_d = std::chrono::high_resolution_clock::now();
    for (auto t : ths) pthread_join(t, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    st->cpu_blocks = cpu_done.load(std::memory_order_relaxed);
    st->gpu_blocks = gpu_done.load(std::memory_order_relaxed);
    if (getenv("HYBRID_DEBUG"))
        printf("[dbg] phases: create_disp=%.3f workers=%.3f join_disp=%.3f join_workers=%.3f\n",
               std::chrono::duration<double, std::milli>(t_create - t0).count(),
               std::chrono::duration<double, std::milli>(t_workers - t_create).count(),
               std::chrono::duration<double, std::milli>(t_join_d - t_workers).count(),
               std::chrono::duration<double, std::milli>(t1 - t_join_d).count());
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ------------------------ Medições do modelo (report) ------------------------
__global__ void empty_launch_kernel() {}
static double medir_latencia_gpu() {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 200; ++i) empty_launch_kernel<<<1, 1>>>();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 200.0;
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : N_DEFAULT;
    std::string out_json = (argc > 2) ? argv[2] : "resultado_dynamic.json";
    g_fma = (argc > 3) ? atoi(argv[3]) : 128;
    g_lag = (argc > 4) ? (float)atof(argv[4]) : 1.0f;
    if (g_fma < 1) g_fma = 1;
    if (g_lag < 0.1f) g_lag = 0.1f;
    N = (N / BLOCK_ELEMS) * BLOCK_ELEMS;
    size_t nb = N / BLOCK_ELEMS;

    std::vector<int> phys;
    int ncpu = std::thread::hardware_concurrency();
    for (int c = 0; c < ncpu && (int)phys.size() < 32; c += 2) phys.push_back(c);
    if (phys.empty()) phys.push_back(0);

    cudaSetDevice(0);
    cudaMemcpyToSymbol(c_fma, &g_fma, sizeof(int));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int block = 256;
    {
        int mg = 0;
        cudaOccupancyMaxPotentialBlockSize(&mg, &block, (const void*)compute_offsets, 0, 0);
        if (block <= 0) block = 256;
    }

    float* h_in; float* h_out;
    posix_memalign((void**)&h_in, 64, N * sizeof(float));
    posix_memalign((void**)&h_out, 64, N * sizeof(float));
    for (size_t i = 0; i < N; ++i) h_in[i] = (float)(i % 1024) / 1024.0f;
    float* d_in; float* d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    float alpha = 0.5f;

    size_t* gpu_base_host = (size_t*)malloc(nb * sizeof(size_t));
    size_t* d_gpu_base; cudaMalloc(&d_gpu_base, nb * sizeof(size_t));

    double lat_ms = medir_latencia_gpu();

    // estimativa inicial da resposta CPU (1 bloco em 1 nucleo -> escalado pelos cores)
    float* tmp_in = (float*)malloc(BLOCK_ELEMS * sizeof(float));
    for (size_t i = 0; i < BLOCK_ELEMS; ++i) tmp_in[i] = 0.5f;
    auto tc0 = std::chrono::high_resolution_clock::now();
    cpu_compute(tmp_in, h_out, alpha, 0, BLOCK_ELEMS);
    auto tc1 = std::chrono::high_resolution_clock::now();
    double cpu_init_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count()
                         / (double)phys.size();
    free(tmp_in);

    DispatchStats st;
    double best = 1e18;
    for (int k = 0; k < REPEATS; ++k) {
        DispatchStats s;
        double t = run_dynamic(d_in, d_out, h_in, h_out, alpha, N, phys, block,
                               gpu_base_host, d_gpu_base, cpu_init_ms, &s);
        if (getenv("HYBRID_DEBUG")) printf("[dbg] run %d: %.3f ms\n", k, t);
        if (t < best) { best = t; st = s; }
    }

    printf("\n=== CONTROLADOR DINAMICO (por resposta) ===\n");
    printf("Latencia GPU medida: %.4f ms/launch | CPU 1 bloco/16 cores: %.3f ms (est. inicial)\n",
           lat_ms, cpu_init_ms);
    printf("Blocos: %ld GPU vs %ld CPU (total %zu)\n", st.gpu_blocks, st.cpu_blocks, nb);
    printf("Resposta medida: GPU %.4f ms/bloco | CPU %.3f ms/bloco\n",
           st.gpu_ms_per_block, st.cpu_ms_per_block);
    printf("Tempo total dinamico: %.3f ms\n", best);
    printf("Traco do controlador: %zu amostras\n", st.trace.size());

    // JSON
    std::ostringstream js;
    js << "{\n";
    js << "  \"experimento\": \"Despacho dinamico por resposta (feedback)\",\n";
    js << "  \"gpu\": {\"nome\": \"" << prop.name << "\", \"cc\": " << prop.major << "."
       << prop.minor << ", \"sms\": " << prop.multiProcessorCount
       << ", \"bloco_otimo\": " << block << "},\n";
    js << "  \"cpu\": {\"workers_fisicos\": " << phys.size() << "},\n";
    js << "  \"n_elementos\": " << N << ",\n";
    js << "  \"fma_por_elemento\": " << g_fma << ",\n";
    js << "  \"gpu_lag\": " << g_lag << ",\n";
    js << "  \"tempo_total_ms\": " << best << ",\n";
    js << "  \"blocos\": {\"total\": " << nb << ", \"gpu\": " << st.gpu_blocks
       << ", \"cpu\": " << st.cpu_blocks << "},\n";
    js << "  \"resposta\": {\"gpu_ms_por_bloco\": " << st.gpu_ms_per_block
       << ", \"cpu_ms_por_bloco\": " << st.cpu_ms_per_block << "},\n";
    js << "  \"traco\": [\n";
    for (size_t i = 0; i < st.trace.size(); ++i)
        js << "    {\"t_ms\": " << st.trace[i][0] << ", \"est_gpu_ms\": " << st.trace[i][1]
           << ", \"est_cpu_ms\": " << st.trace[i][2] << ", \"destino\": " << (int)st.trace[i][3]
           << "}" << (i + 1 < st.trace.size() ? "," : "") << "\n";
    js << "  ]\n";
    js << "}\n";
    std::ofstream ofs(out_json);
    ofs << js.str();
    ofs.close();
    printf("JSON salvo em: %s\n", out_json.c_str());

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gpu_base);
    free(h_in); free(h_out); free(gpu_base_host);
    return 0;
}
