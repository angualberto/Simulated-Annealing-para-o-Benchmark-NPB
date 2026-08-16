// hybrid_linkedlist.cu
// Gerenciamento de blocos com LISTA LIGADA HIERARQUICA E INDICE (2 niveis):
//   Nivel 1: lista ligada de SEGMENTOS (cada segmento = grupo de blocos)
//   Nivel 2: cada segmento contem uma LISTA LIGADA de blocos (nós)
//   Indice: by_id[i] e seg_of[i]  ->  acesso/remoção de qualquer bloco em O(1)
//
// Controle de despacho (adaptativo por resposta medida):
//   - dispatcher é o único que decide; bloqueios mutex = semaforo de I/O
//   - blocos do pool vão para GPU (lote + evento + medicao) ou para a lista
//     ligada da CPU; workers CPU consomem a lista
//   - frac_cpu = r_cpu/(r_gpu+r_cpu) -> terminar junto
//
// Compilar: nvcc -O3 -arch=sm_89 -o hybrid_linkedlist hybrid_linkedlist.cu -lpthread
// Rodar:    ./hybrid_linkedlist [N] [json] [fma] [gpu_lag]

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
constexpr int SEG_BLOCKS = 8;                // blocos por segmento
constexpr int BATCH = 8;                     // blocos por lote de decisao

static int g_fma = 128;
static float g_lag = 1.0f;
__constant__ int c_fma;

// ------------------------ Kernel ------------------------
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

// ------------------------ Lista ligada dupla (com semaforo) ------------------------
struct BlockNode {
    size_t id;
    BlockNode* next;
    BlockNode* prev;
};

struct BlockList {
    BlockNode* head;
    BlockNode* tail;
    int count;
    pthread_mutex_t mtx;
    BlockList() : head(nullptr), tail(nullptr), count(0) {
        pthread_mutex_init(&mtx, nullptr);
    }
    ~BlockList() { pthread_mutex_destroy(&mtx); }
    void push_back(BlockNode* n) {
        pthread_mutex_lock(&mtx);
        n->next = nullptr; n->prev = tail;
        if (tail) tail->next = n; else head = n;
        tail = n; count++;
        pthread_mutex_unlock(&mtx);
    }
    BlockNode* pop_front() {
        pthread_mutex_lock(&mtx);
        BlockNode* n = head;
        if (n) {
            head = n->next;
            if (head) head->prev = nullptr; else tail = nullptr;
            n->next = n->prev = nullptr;
            count--;
        }
        pthread_mutex_unlock(&mtx);
        return n;
    }
    void remove(BlockNode* n) {   // O(1) com prev/next (usado com indice)
        pthread_mutex_lock(&mtx);
        if (n->prev) n->prev->next = n->next; else head = n->next;
        if (n->next) n->next->prev = n->prev; else tail = n->prev;
        n->next = n->prev = nullptr;
        count--;
        pthread_mutex_unlock(&mtx);
    }
    int size() {
        pthread_mutex_lock(&mtx);
        int c = count;
        pthread_mutex_unlock(&mtx);
        return c;
    }
};

// ------------------------ Pool hierárquico (segmentos -> blocos) + índice ------------------------
struct SegmentNode {
    int seg_id;
    BlockList blocks;      // lista ligada de blocos do segmento
    SegmentNode* next;
};

struct HierPool {
    std::vector<BlockNode> nodes;      // memoria estavel dos nós
    std::vector<SegmentNode> segs;     // memoria estavel dos segmentos
    std::vector<BlockNode*> by_id;     // INDICE: id -> nó (O(1))
    std::vector<SegmentNode*> seg_of;  // INDICE: id -> segmento (O(1))
    SegmentNode* seg_head;             // lista ligada de segmentos
    int total;
    pthread_mutex_t mtx;

    HierPool(size_t nb) : total((int)nb) {
        int nseg = (int)((nb + SEG_BLOCKS - 1) / SEG_BLOCKS);
        nodes.resize(nb);
        segs.resize(nseg);
        by_id.resize(nb);
        seg_of.resize(nb);
        pthread_mutex_init(&mtx, nullptr);

        // liga segmentos (nivel 1)
        for (int s = 0; s < nseg; ++s) {
            segs[s].seg_id = s;
            segs[s].next = (s + 1 < nseg) ? &segs[s + 1] : nullptr;
        }
        seg_head = &segs[0];

        // monta nós e liga em cada segmento (nivel 2)
        for (size_t i = 0; i < nb; ++i) {
            nodes[i].id = i;
            nodes[i].next = nodes[i].prev = nullptr;
            int s = (int)(i / SEG_BLOCKS);
            segs[s].blocks.push_back(&nodes[i]);
            by_id[i] = &nodes[i];
            seg_of[i] = &segs[s];
        }
    }
    ~HierPool() { pthread_mutex_destroy(&mtx); }

    // remove o proximo bloco disponivel caminhando pela lista de segmentos
    BlockNode* pop_next() {
        pthread_mutex_lock(&mtx);
        BlockNode* r = nullptr;
        while (seg_head && seg_head->blocks.size() == 0)
            seg_head = seg_head->next;
        if (seg_head) {
            r = seg_head->blocks.pop_front();
            if (seg_head->blocks.size() == 0) seg_head = seg_head->next;
        }
        pthread_mutex_unlock(&mtx);
        return r;
    }
    int remaining() {
        pthread_mutex_lock(&mtx);
        int c = total;
        pthread_mutex_unlock(&mtx);
        return c;
    }
    void dispatched() {   // conta um bloco processado
        pthread_mutex_lock(&mtx);
        total--;
        pthread_mutex_unlock(&mtx);
    }
};

// ------------------------ Workers CPU ------------------------
struct CpuArg {
    const float* in; float* out; float alpha;
    size_t be; int core;
    BlockList* cpuq;
    std::atomic<long>* cpu_done;
};

static void* cpu_worker(void* arg_) {
    CpuArg* a = (CpuArg*)arg_;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(a->core, &set);
    sched_setaffinity(0, sizeof(set), &set);
    BlockNode* n;
    while ((n = a->cpuq->pop_front()) != nullptr) {
        cpu_compute(a->in, a->out, a->alpha, n->id * a->be, n->id * a->be + a->be);
        a->cpu_done->fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

// ------------------------ Dispatcher (controlador adaptativo) ------------------------
struct Result {
    long gpu_blocks = 0, cpu_blocks = 0;
    double gpu_ms_per_block = 0, cpu_ms_per_block = 0;
    std::vector<std::vector<double>> trace;   // {t_ms, frac_cpu, destino}
};

struct DispArg {
    const float* in; float* out; float alpha; size_t be;
    HierPool* pool;
    BlockList* cpuq;
    size_t* gpu_base_host; size_t* d_gpu_base;
    std::atomic<long>* cpu_done;
    int block;
    double cpu_init_ms_per_block;
    Result* r;
};

static void* gpu_dispatcher(void* arg_) {
    DispArg* dp = (DispArg*)arg_;
    auto& pool = *dp->pool; auto& cpuq = *dp->cpuq;
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

    while (true) {
        if (pool.remaining() <= 0) break;

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

        int rem = pool.remaining();
        double r_g = 1.0 / std::fmax(gpu_ms_per_block, 1e-6);
        double r_c = 1.0 / std::fmax(cpu_ms_per_block, 1e-6);
        double frac_cpu = r_c / (r_g + r_c);
        long cpu_target = (long)std::llround((double)rem * frac_cpu);
        long cpu_pending = cpu_assigned - cpu_done.load(std::memory_order_relaxed);
        bool to_cpu = (cpu_pending < cpu_target);
        int take = std::min<int>(BATCH, rem);

        if (to_cpu) {
            for (int i = 0; i < take; ++i) {
                BlockNode* n = pool.pop_next();
                if (!n) break;
                cpuq.push_back(n);      // nó migra do pool para a lista da CPU
                cpu_assigned++;
                pool.dispatched();
            }
        } else {
            long n = 0;
            for (int i = 0; i < take; ++i) {
                BlockNode* b = pool.pop_next();
                if (!b) break;
                dp->gpu_base_host[R.gpu_blocks + n] = b->id * dp->be;
                gpu_assigned++;
                n++;
                pool.dispatched();
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

// ------------------------ Execução ------------------------
static double run_ll(const float* d_in, float* d_out, float* h_in, float* h_out,
                     float alpha, size_t total, const std::vector<int>& phys,
                     int block, size_t* gpu_base_host, size_t* d_gpu_base,
                     double cpu_init_ms_per_block, Result* R) {
    size_t nb = total / BLOCK_ELEMS;
    HierPool pool(nb);
    BlockList cpuq;

    std::atomic<long> cpu_done(0);
    std::vector<CpuArg> cargs;
    std::vector<pthread_t> ths;
    for (size_t i = 0; i < phys.size(); ++i)
        cargs.push_back({h_in, h_out, alpha, BLOCK_ELEMS, phys[i], &cpuq, &cpu_done});

    auto t0 = std::chrono::high_resolution_clock::now();
    pthread_t disp;
    DispArg darg{d_in, d_out, alpha, BLOCK_ELEMS, &pool, &cpuq,
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
    std::string out_json = (argc > 2) ? argv[2] : "resultado_linkedlist.json";
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
        double t = run_ll(d_in, d_out, h_in, h_out, alpha, N, phys, block,
                          gpu_base_host, d_gpu_base, cpu_init_ms, &r);
        if (t < best) { best = t; R = r; }
    }

    printf("\n=== LISTA LIGADA HIERARQUICA (segmentos -> blocos) + indice ===\n");
    printf("CPU: %zu nucleos fisicos | CPU 1 bloco/16 cores: %.3f ms (inicial)\n",
           phys.size(), cpu_init_ms);
    printf("Estrutura: %zu segmentos x %d blocos (indice by_id/seg_of O(1))\n",
           (nb + SEG_BLOCKS - 1) / SEG_BLOCKS, SEG_BLOCKS);
    printf("Blocos: %ld GPU vs %ld CPU (total %zu)\n", R.gpu_blocks, R.cpu_blocks, nb);
    printf("Resposta medida: GPU %.4f ms/bloco | CPU %.3f ms/bloco\n",
           R.gpu_ms_per_block, R.cpu_ms_per_block);
    printf("Tempo total: %.3f ms (gpu_lag=%.1f)\n", best, g_lag);

    std::ostringstream js;
    js << "{\n";
    js << "  \"experimento\": \"Despacho com lista ligada hierarquica (segmentos->blocos) + indice\",\n";
    js << "  \"gpu\": {\"nome\": \"" << prop.name << "\", \"cc\": " << prop.major << "."
       << prop.minor << ", \"sms\": " << prop.multiProcessorCount
       << ", \"bloco_otimo\": " << block << "},\n";
    js << "  \"cpu\": {\"workers_fisicos\": " << phys.size() << "},\n";
    js << "  \"n_elementos\": " << N << ",\n";
    js << "  \"fma_por_elemento\": " << g_fma << ",\n";
    js << "  \"gpu_lag\": " << g_lag << ",\n";
    js << "  \"estrutura\": {\"blocos\": " << nb << ", \"blocos_por_segmento\": " << SEG_BLOCKS
       << ", \"segmentos\": " << (nb + SEG_BLOCKS - 1) / SEG_BLOCKS << "},\n";
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
