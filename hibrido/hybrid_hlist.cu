// hybrid_hlist.cu
// Fila de despacho como LISTA LIGADA HIERARQUICA COM INDICE (indice + listas por nivel):
//
//   Nivel 1 - Dispositivos: devices[] (indice: 0=LIVRE, 1=CPU, 2=GPU)
//   Nivel 2 - Periodos:     INDEX periods[pid] (O(1))  +  lista ligada dupla de periodos
//   Nivel 3 - Blocos:       INDEX blocks[idx]  (O(1))  +  lista duplamente ligada
//
//   - dispatcher (UNICO dono do pool LIVRE) popa o proximo bloco da hierarquia livre
//   - decide GPU/CPU pela resposta medida (feedback) e MOVE o no entre hierarquias O(1)
//   - workers CPU consomem a hierarquia CPU (periodos -> blocos) via indice+lista
//
// Compilar: nvcc -O3 -arch=sm_89 -o hybrid_hlist hybrid_hlist.cu -lpthread
// Rodar:    ./hybrid_hlist [N] [json] [fma] [gpu_lag]

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
constexpr int PERIOD = 4;                    // blocos por periodo (barra)
constexpr int BATCH = 8;                     // blocos por lote de decisao

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

// ------------------------ Hierarquia: Dispositivo -> Periodo -> Bloco ------------------------
struct BlockNode {
    size_t id;         // id global do bloco
    size_t idx;        // indice do bloco dentro do periodo (nivel 3)
    size_t base;       // offset em elementos
    BlockNode *prev, *next;   // lista duplamente ligada dentro do periodo
};

struct PeriodNode {
    int pid;                           // indice do periodo
    BlockNode* blocks[PERIOD];         // INDEX nivel 3: por idx (O(1))
    BlockNode *head, *tail;            // lista ligada nivel 3 (percurso)
    int count;
    PeriodNode *prev, *next;           // lista ligada nivel 2 (dentro do device)
};

struct DeviceNode {
    int dev;                           // indice nivel 1 (0 livre, 1 CPU, 2 GPU)
    PeriodNode** periods;              // INDEX nivel 2: por pid (O(1))
    PeriodNode* pool;                  // armazenamento estavel dos periodos
    PeriodNode *head, *tail;           // lista ligada nivel 2 (periodos ativos)
    long nblocks;
};

static DeviceNode livre{0, nullptr, nullptr, nullptr, nullptr, 0};
static DeviceNode cpu  {1, nullptr, nullptr, nullptr, nullptr, 0};
static DeviceNode gpu  {2, nullptr, nullptr, nullptr, nullptr, 0};

static pthread_mutex_t gmtx = PTHREAD_MUTEX_INITIALIZER;   // semaforo de I/O da estrutura
static BlockNode* g_nodes = nullptr;                        // pool estavel dos blocos

// remove um no da lista dupla do periodo
static inline void unlink_block(PeriodNode* P, BlockNode* b) {
    if (b->prev) b->prev->next = b->next; else P->head = b->next;
    if (b->next) b->next->prev = b->prev; else P->tail = b->prev;
    b->prev = b->next = nullptr;
    P->count--;
}

// remove um periodo da lista ligada do device
static inline void unlink_period(DeviceNode* D, PeriodNode* P) {
    if (P->prev) P->prev->next = P->next; else D->head = P->next;
    if (P->next) P->next->prev = P->prev; else D->tail = P->prev;
    P->prev = P->next = nullptr;
}

// move um bloco (ja removido do LIVRE) para a hierarquia do device (indice + lista)
static void move_to_device(DeviceNode* D, BlockNode* b) {
    int p = (int)(b->id / PERIOD);
    int idx = (int)(b->id % PERIOD);
    pthread_mutex_lock(&gmtx);

    PeriodNode* DP = D->periods[p];             // indice O(1) no device
    if (!DP || DP->count == 0) {
        DP = &D->pool[p];
        DP->pid = p;
        DP->head = DP->tail = nullptr;
        DP->count = 0;
        DP->prev = DP->next = nullptr;
        D->periods[p] = DP;
        if (D->head) { D->tail->next = DP; DP->prev = D->tail; D->tail = DP; }
        else D->head = D->tail = DP;
    }
    DP->blocks[idx] = b;                        // indice nivel 3 (O(1))
    DP->count++;
    if (DP->head) { DP->tail->next = b; b->prev = DP->tail; DP->tail = b; }
    else DP->head = DP->tail = b;
    D->nblocks++;
    pthread_mutex_unlock(&gmtx);
}

// popa o proximo bloco livre (dispatcher, unico dono do livre)
static BlockNode* livre_pop() {
    if (!livre.head) return nullptr;
    BlockNode* b = livre.head->head;
    unlink_block(livre.head, b);
    if (livre.head->count == 0) unlink_period(&livre, livre.head);
    livre.nblocks--;
    return b;
}

// consome o proximo bloco da hierarquia CPU (workers)
static BlockNode* cpu_pop() {
    pthread_mutex_lock(&gmtx);
    BlockNode* b = nullptr;
    while (cpu.head && cpu.head->count == 0)   // varredura defensiva da lista
        unlink_period(&cpu, cpu.head);
    if (cpu.head) {
        b = cpu.head->head;
        unlink_block(cpu.head, b);
        if (cpu.head->count == 0) unlink_period(&cpu, cpu.head);
        cpu.nblocks--;
    }
    pthread_mutex_unlock(&gmtx);
    return b;
}

// ------------------------ Workers CPU ------------------------
struct CpuArg {
    const float* in; float* out; float alpha;
    size_t be; int core;
    std::atomic<long>* cpu_done;
    std::atomic<int>* done;
};

static void* cpu_worker(void* arg_) {
    CpuArg* a = (CpuArg*)arg_;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(a->core, &set);
    sched_setaffinity(0, sizeof(set), &set);
    while (!a->done->load(std::memory_order_relaxed)) {
        BlockNode* b = cpu_pop();
        if (!b) {
            if (a->done->load(std::memory_order_relaxed)) break;
            sched_yield();
            continue;
        }
        cpu_compute(a->in, a->out, a->alpha, b->base, b->base + a->be);
        a->cpu_done->fetch_add(1, std::memory_order_relaxed);
    }
    while (true) {
        BlockNode* b = cpu_pop();
        if (!b) break;
        cpu_compute(a->in, a->out, a->alpha, b->base, b->base + a->be);
        a->cpu_done->fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

// ------------------------ Resultado ------------------------
struct Result {
    long gpu_blocks = 0, cpu_blocks = 0;
    double gpu_ms_per_block = 0, cpu_ms_per_block = 0;
    std::vector<std::vector<double>> trace;   // {t_ms, frac_cpu, destino}
};

struct DispArg {
    const float* in; float* out; float alpha; size_t be;
    size_t* gpu_base_host; size_t* d_gpu_base;
    std::atomic<long>* cpu_done;
    std::atomic<int>* done;
    int block; int nperiods;
    int cpu_workers;
    double cpu_init_ms_per_block;
    Result* r;
};

// ------------------------ Dispatcher (controlador adaptativo, dono do livre) ------------------------
static void* gpu_dispatcher(void* arg_) {
    DispArg* dp = (DispArg*)arg_;
    auto& cpu_done = *dp->cpu_done;
    auto& R = *dp->r;

    double gpu_ms_per_block = dp->cpu_init_ms_per_block;
    double cpu_ms_per_block = dp->cpu_init_ms_per_block;
    long gpu_assigned = 0, cpu_assigned = 0;
    bool gpu_measured = false;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_last = t_start;
    long cpu_last = 0;
    size_t bases[BATCH];

    // ===== Fase de SONDA: mede GPU rapidamente; CPU usa a medida inicial + refinamento =====
    {
        long n = 0;
        int take = (int)std::min<long>(BATCH, livre.nblocks);
        for (int i = 0; i < take; ++i) {
            BlockNode* b = livre_pop();
            if (!b) break;
            bases[n] = b->base;
            move_to_device(&gpu, b);
            gpu_assigned++;
            n++;
        }
        if (n > 0) {
            for (long i = 0; i < n; ++i)
                dp->gpu_base_host[i] = bases[i];
            cudaEvent_t evS, evE;
            cudaEventCreate(&evS); cudaEventCreate(&evE);
            cudaEventRecord(evS, stream);
            compute_batch<<<(int)n, dp->block, 0, stream>>>(
                dp->in, dp->out, dp->alpha, dp->d_gpu_base, (int)n, dp->be, g_lag);
            cudaEventRecord(evE, stream);
            cudaEventSynchronize(evE);
            float ms = 0; cudaEventElapsedTime(&ms, evS, evE);
            cudaEventDestroy(evS); cudaEventDestroy(evE);
            gpu_ms_per_block = ms / (double)n;      // VETOR GPU (caracteristica da GPU)
            gpu_measured = true;
            R.gpu_blocks += n;
        }
    }

    while (livre.head) {
        auto tn = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(tn - t_last).count();
        long dcpu = cpu_done.load(std::memory_order_relaxed);
        if (dt > 3.0 && dcpu > cpu_last) {
            double per = dt / (double)(dcpu - cpu_last);
            cpu_ms_per_block = (cpu_ms_per_block == 0) ? per
                              : 0.5 * cpu_ms_per_block + 0.5 * per;
            cpu_last = dcpu; t_last = tn;
        }

        int rem = livre.nblocks;
        double r_g = 1.0 / std::fmax(gpu_ms_per_block, 1e-6);
        double r_c = 1.0 / std::fmax(cpu_ms_per_block, 1e-6);
        double frac_cpu = r_c / (r_g + r_c);
        long cpu_target = (long)std::llround((double)rem * frac_cpu);
        long cpu_pending = cpu_assigned - cpu_done.load(std::memory_order_relaxed);
        long need = std::max<long>(0, cpu_target - cpu_pending);   // so o que falta
        int take = (int)std::min<long>(BATCH, rem);
        long to_cpu_n = std::min<long>(take, need);
        long to_gpu_n = take - to_cpu_n;
        bool any_cpu = to_cpu_n > 0;

        for (long i = 0; i < to_cpu_n; ++i) {
            BlockNode* b = livre_pop();
            if (!b) break;
            move_to_device(&cpu, b);
            cpu_assigned++;
        }
        long n = 0;
        for (long i = 0; i < to_gpu_n; ++i) {
            BlockNode* b = livre_pop();
            if (!b) break;
            bases[n] = b->base;
            move_to_device(&gpu, b);
            gpu_assigned++;
            n++;
        }
        if (n > 0) {
            for (long i = 0; i < n; ++i)
                dp->gpu_base_host[R.gpu_blocks + i] = bases[i];
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
            R.trace.push_back({now_ms, frac_cpu, any_cpu ? 0.0 : 1.0});
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    dp->done->store(1, std::memory_order_release);
    R.gpu_ms_per_block = gpu_ms_per_block;
    R.cpu_ms_per_block = cpu_ms_per_block;
    if (getenv("HLIST_DEBUG")) {
        pthread_mutex_lock(&gmtx);
        printf("[dbg] final: livre.nblocks=%ld cpu.nblocks=%ld gpu.nblocks=%ld cpu_done=%ld\n",
               livre.nblocks, cpu.nblocks, gpu.nblocks,
               cpu_done.load(std::memory_order_relaxed));
        pthread_mutex_unlock(&gmtx);
    }
    return nullptr;
}

// ------------------------ Execução ------------------------
static double run_hlist(const float* d_in, float* d_out, float* h_in, float* h_out,
                        float alpha, size_t total, const std::vector<int>& phys,
                        int block, size_t* gpu_base_host, size_t* d_gpu_base,
                        double cpu_init_ms_per_block, Result* R) {
    size_t nb = total / BLOCK_ELEMS;
    int nper = (int)(nb / PERIOD);

    g_nodes = new BlockNode[nb];
    livre.pool = new PeriodNode[nper];
    cpu.pool   = new PeriodNode[nper];
    gpu.pool   = new PeriodNode[nper];
    livre.periods = new PeriodNode*[nper];
    cpu.periods   = new PeriodNode*[nper];
    gpu.periods   = new PeriodNode*[nper];
    for (int p = 0; p < nper; ++p) {
        livre.periods[p] = &livre.pool[p];
        cpu.periods[p] = nullptr;
        gpu.periods[p] = nullptr;
        livre.pool[p] = PeriodNode{p, {}, nullptr, nullptr, 0, nullptr, nullptr};
    }
    livre.head = livre.tail = &livre.pool[0];
    for (int p = 1; p < nper; ++p) {
        livre.pool[p - 1].next = &livre.pool[p];
        livre.pool[p].prev = &livre.pool[p - 1];
        livre.tail = &livre.pool[p];
    }

    for (size_t i = 0; i < nb; ++i) {
        BlockNode* b = &g_nodes[i];
        b->id = i;
        b->idx = i % PERIOD;
        b->base = i * BLOCK_ELEMS;
        b->prev = b->next = nullptr;
        PeriodNode* P = &livre.pool[i / PERIOD];     // indice O(1) no livre
        P->blocks[i % PERIOD] = b;                   // indice nivel 3
        P->count++;
        if (P->head) { P->tail->next = b; b->prev = P->tail; P->tail = b; }
        else P->head = P->tail = b;
    }
    livre.nblocks = (long)nb;

    std::atomic<long> cpu_done(0);
    std::atomic<int> done(0);
    std::vector<CpuArg> cargs;
    std::vector<pthread_t> ths;
    for (size_t i = 0; i < phys.size(); ++i)
        cargs.push_back({h_in, h_out, alpha, BLOCK_ELEMS, phys[i], &cpu_done, &done});

    auto t0 = std::chrono::high_resolution_clock::now();
    pthread_t disp;
    DispArg darg{d_in, d_out, alpha, BLOCK_ELEMS, gpu_base_host, d_gpu_base,
                 &cpu_done, &done, block, nper, (int)phys.size(),
                 cpu_init_ms_per_block, R};
    pthread_create(&disp, nullptr, gpu_dispatcher, &darg);
    for (auto& a : cargs) { pthread_t t; pthread_create(&t, nullptr, cpu_worker, &a); ths.push_back(t); }
    pthread_join(disp, nullptr);
    for (auto t : ths) pthread_join(t, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();

    R->cpu_blocks = cpu_done.load(std::memory_order_relaxed);
    R->gpu_blocks = gpu.nblocks;

    delete[] g_nodes; delete[] livre.pool; delete[] cpu.pool; delete[] gpu.pool;
    delete[] livre.periods; delete[] cpu.periods; delete[] gpu.periods;
    livre.head = cpu.head = gpu.head = nullptr;
    livre.tail = cpu.tail = gpu.tail = nullptr;
    livre.nblocks = cpu.nblocks = gpu.nblocks = 0;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : N_DEFAULT;
    std::string out_json = (argc > 2) ? argv[2] : "resultado_hlist.json";
    g_fma = (argc > 3) ? atoi(argv[3]) : 128;
    g_lag = (argc > 4) ? (float)atof(argv[4]) : 1.0f;
    if (g_fma < 1) g_fma = 1;
    if (g_lag < 0.1f) g_lag = 0.1f;
    N = (N / BLOCK_ELEMS) * BLOCK_ELEMS;
    size_t nb = N / BLOCK_ELEMS;
    if (nb % PERIOD != 0) {
        fprintf(stderr, "N/BLOCK_ELEMS deve ser multiplo de PERIOD\n");
        return 1;
    }

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
        double t = run_hlist(d_in, d_out, h_in, h_out, alpha, N, phys, block,
                             gpu_base_host, d_gpu_base, cpu_init_ms, &r);
        if (t < best) { best = t; R = r; }
    }

    printf("\n=== LISTA LIGADA HIERARQUICA COM INDICE (device -> periodo -> bloco) ===\n");
    printf("CPU: %zu nucleos fisicos | CPU 1 bloco/16 cores: %.3f ms (inicial)\n",
           phys.size(), cpu_init_ms);
    printf("Estrutura: %zu periodos x %d blocos | indice O(1) por nivel + lista ligada dupla\n",
           nb / PERIOD, PERIOD);
    printf("Blocos: %ld GPU vs %ld CPU (total %zu)\n", R.gpu_blocks, R.cpu_blocks, nb);
    printf("Resposta medida: GPU %.4f ms/bloco | CPU %.3f ms/bloco\n",
           R.gpu_ms_per_block, R.cpu_ms_per_block);
    printf("Tempo total: %.3f ms (gpu_lag=%.1f)\n", best, g_lag);

    std::ostringstream js;
    js << "{\n";
    js << "  \"experimento\": \"Lista ligada hierarquica com indice (device->periodo->bloco) + feedback\",\n";
    js << "  \"gpu\": {\"nome\": \"" << prop.name << "\", \"cc\": " << prop.major << "."
       << prop.minor << ", \"sms\": " << prop.multiProcessorCount
       << ", \"bloco_otimo\": " << block << "},\n";
    js << "  \"cpu\": {\"workers_fisicos\": " << phys.size() << "},\n";
    js << "  \"n_elementos\": " << N << ",\n";
    js << "  \"fma_por_elemento\": " << g_fma << ",\n";
    js << "  \"gpu_lag\": " << g_lag << ",\n";
    js << "  \"estrutura\": {\"blocos\": " << nb << ", \"periodos\": " << nb / PERIOD
       << ", \"blocos_por_periodo\": " << PERIOD << "},\n";
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
