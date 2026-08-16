// hybrid_adaptive.cu
// Controlador adaptativo por TEMPO DE RESPOSTA medido (feedback).
// Fluxo de controle:
//   main_queue (todos os blocos)
//        |
//   [gpu_dispatcher]  <- UNICO que decide o destino de cada bloco
//        |-- GPU: acumula lote, lanca kernel, espera evento, mede resposta
//        |        (grid = blocos do lote, um bloco CUDA por bloco de dados)
//        \-- CPU: empurra na cpu_queue (semaforo de I/O)
//                       |
//              [16 workers CPU] consomem a fila e contam concluidos
//
//   frac_cpu = r_cpu/(r_gpu+r_cpu)  ->  cada lado termina junto
//   (r_gpu e r_cpu sao as respostas MEDIDAS, atualizadas a cada lote)
//
// Compilar: nvcc -O3 -arch=sm_89 -o hybrid_adaptive hybrid_adaptive.cu -lpthread
// Rodar:    ./hybrid_adaptive [N] [json] [fma] [gpu_lag]

#include <cuda_runtime.h>
#include <cuda.h>

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
constexpr int BATCH = 8;                     // blocos de dados por lote de decisao

static int g_fma = 128;
static float g_lag = 1.0f;
__constant__ int c_fma;

__global__ void compute_batch(const float* __restrict__ in, float* __restrict__ out,
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

// Semaforo de I/O: fila compartilhada
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
    IoQueue* queue;
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

struct Result {
    long gpu_blocks = 0, cpu_blocks = 0;
    double gpu_ms_per_block = 0, cpu_ms_per_block = 0;
    std::vector<std::vector<double>> trace;   // {t_ms, frac_cpu, destino}
};

struct DispArg {
    const float* in; float* out; float alpha; size_t be;
    IoQueue* mainq; IoQueue* cpuq;
    size_t* gpu_base_host; size_t* d_gpu_base;
    std::atomic<long>* cpu_done;
    int block;
    double cpu_init_ms_per_block;
    Result* r;
};

static void* gpu_dispatcher(void* arg_) {
    DispArg* dp = (DispArg*)arg_;
    auto& mainq = *dp->mainq; auto& cpuq = *dp->cpuq;
    auto& cpu_done = *dp->cpu_done;
    auto& R = *dp->r;

    double gpu_ms_per_block = dp->cpu_init_ms_per_block;   // otimista: mede logo no 1o lote
    double cpu_ms_per_block = dp->cpu_init_ms_per_block;
    long gpu_assigned = 0, cpu_assigned = 0;
    bool gpu_measured = false;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_last = t_start;
    long cpu_last = 0;

    while (true) {
        long rem = mainq.remaining();
        if (rem <= 0) break;

        // resposta CPU (workers concluidos por tempo)
        auto tn = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(tn - t_last).count();
        long dcpu = cpu_done.load(std::memory_order_relaxed);
        if (dt > 3.0 && dcpu > cpu_last) {
            double per = dt / (double)(dcpu - cpu_last);
            cpu_ms_per_block = (cpu_ms_per_block == 0) ? per
                              : 0.5 * cpu_ms_per_block + 0.5 * per;
            cpu_last = dcpu; t_last = tn;
        }

        // reparte o que resta pela razão das respostas (terminar junto)
        double r_g = 1.0 / std::fmax(gpu_ms_per_block, 1e-6);
        double r_c = 1.0 / std::fmax(cpu_ms_per_block, 1e-6);
        double frac_cpu = r_c / (r_g + r_c);
        long cpu_target = (long)std::llround((double)rem * frac_cpu);
        long cpu_pending = cpu_assigned - cpu_done.load(std::memory_order_relaxed);

        bool to_cpu = (cpu_pending < cpu_target);
        int take = (int)std::min<long>(BATCH, rem);

        if (to_cpu) {
            for (int i = 0; i < take; ++i) {
                long id = mainq.pop();
                if (id < 0) break;
                cpuq.push((size_t)id);
                cpu_assigned++;
            }
        } else {
            long n = 0;
            for (int i = 0; i < take; ++i) {
                long id = mainq.pop();
                if (id < 0) break;
                dp->gpu_base_host[R.gpu_blocks + n] = (size_t)id * be;
                gpu_assigned++;
                n++;
            }
            cudaEvent_t evS, evE;
            cudaEventCreate(&evS); cudaEventCreate(&evE);
            cudaEventRecord(evS, stream);
            compute_batch<<<(int)n, dp->block, 0, stream>>>(
                dp->in, dp->out, dp->alpha, dp->d_gpu_base + R.gpu_blocks,
                (int)n, dp->be, g_lag);
            cudaEventRecord(evE, stream);
            cudaEventSynchronize(evE);            // espera e MEDE a resposta real
            float ms = 0; cudaEventElapsedTime(&ms, evS, evE);
            cudaEventDestroy(evS); cudaEventDestroy(evE);
            R.gpu_blocks += n;
            if (!gpu_measured) {
                gpu_ms_per_block = ms / (double)n;
                gpu_measured = true;
            } else {
                gpu_ms_per_block = 0.5 * gpu_ms_per_block + 0.5 * (ms / (double)n);
            }
        }

        double now_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        if ((int)R.trace.size() < 4000)
            R.trace.push_back({now_ms, frac_cpu, to_cpu ? 0.0 : 1.0});
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    R.gpu_ms_per_block = gpu_ms_per_block;
    R.cpu_ms_per_block = cpu_ms_per_block;
    return nullptr;
}

static double run_adaptive(const float* d_in, float* d_out, float* h_in, float* h_out,
                           float alpha, size_t total, const std::vector<int>& phys,
                           int block, size_t* gpu_base_host, size_t* d_gpu_base,
                           double cpu_init_ms_per_block, Result* R) {
    size_t nb = total / BLOCK_ELEMS;
    std::vector<size_t> ids;
    for (size_t i = 0; i < nb; ++i) ids.push_back(i);
    IoQueue mainq, cpuq;
    mainq.init(ids);
    cpuq.init({});

    std::atomic<long> cpu_done(0);

    std::vector<CpuArg> cargs;
    std::vector<pthread_t> ths;
    for (size_t i = 0; i < phys.size(); ++i)
        cargs.push_back({h_in, h_out, alpha, BLOCK_ELEMS, phys[i], &cpuq, &cpu_done});

    auto t0 = std::chrono::high_resolution_clock::now();
    pthread_t disp;
    DispArg darg{d_in, d_out, alpha, BLOCK_ELEMS, &mainq, &cpuq,
                 gpu_base_host, d_gpu_base, &cpu_done, block,
                 cpu_init_ms_per_block, R};
    pthread_create(&disp, nullptr, gpu_dispatcher, &darg);
    for (auto& a : cargs) { pthread_t t; pthread_create(&t, nullptr, cpu_worker, &a); ths.push_back(t); }
    pthread_join(disp, nullptr);
    for (auto t : ths) pthread_join(t, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    R->cpu_blocks = cpu_done.load(std::memory_order_relaxed);
    R->gpu_blocks = nb - R->cpu_blocks;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : N_DEFAULT;
    std::string out_json = (argc > 2) ? argv[2] : "resultado_adaptive.json";
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
    { int mg = 0; cudaOccupancyMaxPotentialBlockSize(&mg, &block, (const void*)compute_batch, 0, 0); }

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

    // resposta inicial da CPU (1 bloco em 1 nucleo, escalado pelos cores)
    float* tmp_in = (float*)malloc(BLOCK_ELEMS * sizeof(float));
    for (size_t i = 0; i < BLOCK_ELEMS; ++i) tmp_in[i] = 0.5f;
    auto tc0 = std::chrono::high_resolution_clock::now();
    cpu_compute(tmp_in, h_out, alpha, 0, BLOCK_ELEMS);
    auto tc1 = std::chrono::high_resolution_clock::now();
    double cpu_init_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count()
                         / (double)phys.size();
    free(tmp_in);

    Result R;
    double best = 1e18;
    for (int k = 0; k < 3; ++k) {
        Result r;
        double t = run_adaptive(d_in, d_out, h_in, h_out, alpha, N, phys, block,
                                gpu_base_host, d_gpu_base, cpu_init_ms, &r);
        if (t < best) { best = t; R = r; }
    }

    printf("\n=== CONTROLADOR ADAPTATIVO (por resposta medida) ===\n");
    printf("CPU: %zu nucleos fisicos | CPU 1 bloco/16 cores: %.3f ms (inicial)\n",
           phys.size(), cpu_init_ms);
    printf("Blocos: %ld GPU vs %ld CPU (total %zu)\n", R.gpu_blocks, R.cpu_blocks, nb);
    printf("Resposta medida: GPU %.4f ms/bloco | CPU %.3f ms/bloco\n",
           R.gpu_ms_per_block, R.cpu_ms_per_block);
    printf("Tempo total adaptativo: %.3f ms (gpu_lag=%.1f)\n", best, g_lag);

    // JSON
    std::ostringstream js;
    js << "{\n";
    js << "  \"experimento\": \"Controlador adaptativo por tempo de resposta\",\n";
    js << "  \"gpu\": {\"nome\": \"" << prop.name << "\", \"cc\": " << prop.major << "."
       << prop.minor << ", \"sms\": " << prop.multiProcessorCount
       << ", \"bloco_otimo\": " << block << "},\n";
    js << "  \"cpu\": {\"workers_fisicos\": " << phys.size() << "},\n";
    js << "  \"n_elementos\": " << N << ",\n";
    js << "  \"fma_por_elemento\": " << g_fma << ",\n";
    js << "  \"gpu_lag\": " << g_lag << ",\n";
    js << "  \"tempo_total_ms\": " << best << ",\n";
    js << "  \"blocos\": {\"total\": " << nb << ", \"gpu\": " << R.gpu_blocks
       << ", \"cpu\": " << R.cpu_blocks << "},\n";
    js << "  \"resposta\": {\"gpu_ms_por_bloco\": " << R.gpu_ms_per_block
       << ", \"cpu_ms_por_bloco\": " << R.cpu_ms_per_block << "},\n";
    js << "  \"traco\": [\n";
    for (size_t i = 0; i < R.trace.size(); ++i)
        js << "    {\"t_ms\": " << R.trace[i][0] << ", \"frac_cpu\": " << R.trace[i][1]
           << ", \"destino\": " << (int)R.trace[i][2]
           << "}" << (i + 1 < R.trace.size() ? "," : "") << "\n";
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
