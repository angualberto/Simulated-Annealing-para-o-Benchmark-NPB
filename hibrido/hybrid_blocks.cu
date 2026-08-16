// hybrid_blocks.cu
// Despacho híbrido POR BLOCO (chunks intercalados) com:
//   - Semáforo (mutex) controlando o acesso de I/O à fila de blocos compartilhada
//   - Modelo de decisão por bloco baseado em:
//        * BARRAmento (largura de banda de memória medida na GPU)
//        * TEMPO DE RESPOSTA da GPU (latência de launch medida)
//        * taxa real por núcleo físico da CPU
//   - Padrão intercalado: periodos de P blocos, G vao para GPU, P-G para CPU
//   - Simulated Annealing afinando o gpu_ratio (G) a partir da estimativa do modelo
//
// Compilar: nvcc -O3 -arch=sm_89 -o hybrid_blocks hybrid_blocks.cu -lpthread
// Rodar:    ./hybrid_blocks [N] [arquivo_json] [fma_por_elemento]

#include <cuda_runtime.h>
#include <cuda.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

constexpr size_t N_DEFAULT = 64ULL * 1024 * 1024;
constexpr size_t BLOCK_ELEMS = 1ULL << 20;   // tamanho de cada bloco/barra (1M elem)
constexpr int PERIOD = 4;                    // periodos do padrao intercalado
constexpr int REPEATS = 3;
constexpr int SA_ITERS = 40;

static int g_fma = 128;
__constant__ int c_fma;

// ------------------------ Kernel (1 bloco de dados por bloco CUDA) ------------------------
__global__ void compute_offsets(const float* __restrict__ in, float* __restrict__ out,
                                float alpha, const size_t* __restrict__ gpu_base,
                                int gpu_blocks, size_t be) {
    int b = blockIdx.x;
    size_t base = gpu_base[b];
    for (size_t i = base + threadIdx.x; i < base + be; i += blockDim.x) {
        float v = in[i];
        for (int j = 0; j < c_fma; ++j)
            v = fmaf(v, alpha, 0.001f * j);
        out[i] = v;
    }
}

// ------------------------ CPU (mesma carga) ------------------------
static inline void cpu_compute(const float* __restrict__ in, float* __restrict__ out,
                               float alpha, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        float v = in[i];
        for (int j = 0; j < g_fma; ++j)
            v = fmaf(v, alpha, 0.001f * j);
        out[i] = v;
    }
}

// ------------------------ Semáforo de I/O: fila de blocos compartilhada ------------------------
struct IoQueue {   // semaforo protege o acesso de I/O compartilhado
    pthread_mutex_t mtx;
    std::vector<size_t> ids;   // ids dos blocos de dados que a CPU processa
    size_t pos;
    size_t base;               // bloco de dados inicial (offset global)
    IoQueue() { pthread_mutex_init(&mtx, nullptr); pos = 0; base = 0; }
    ~IoQueue() { pthread_mutex_destroy(&mtx); }
    void init(const std::vector<size_t>& v) { ids = v; pos = 0; }
    // pop com semaforo: devolve o id do bloco OU -1 se acabou
    long pop() {
        pthread_mutex_lock(&mtx);
        long id = -1;
        if (pos < ids.size()) id = (long)ids[pos++];
        pthread_mutex_unlock(&mtx);
        return id;
    }
};

struct CpuArg {
    const float* in;
    float* out;
    float alpha;
    size_t be;            // elementos por bloco
    int core;
    IoQueue* queue;       // semaforo compartilhado
};

static void* cpu_worker(void* arg_) {
    CpuArg* a = (CpuArg*)arg_;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(a->core, &set);
    sched_setaffinity(0, sizeof(set), &set);
    long id;
    while ((id = a->queue->pop()) >= 0) {
        cpu_compute(a->in, a->out, a->alpha, (size_t)id * a->be, (size_t)id * a->be + a->be);
    }
    return nullptr;
}

// ------------------------ Medições do modelo (barramento + tempo de resposta) ------------------------
__global__ void empty_launch_kernel() {}

static double medir_latencia_gpu() {          // tempo de resposta da GPU (launch)
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 200; ++i) empty_launch_kernel<<<1, 1>>>();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 200.0;   // ms por launch
}

static double medir_banda_gpu(size_t nbytes) {  // barramento: banda de memoria da GPU
    float* a; float* b;
    cudaMalloc(&a, nbytes); cudaMalloc(&b, nbytes);
    cudaMemset(a, 0x7f, nbytes);
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    cudaMemcpy(b, a, nbytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(a); cudaFree(b);
    return (double)nbytes / (ms * 1e6);  // GB/s
}

static double medir_taxa_cpu_1core(const float* in, float* out, float alpha,
                                   size_t n) {  // elem/s por nucleo fisico
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_compute(in, out, alpha, 0, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return (double)n / (ms / 1e3);  // elem/s
}

// ------------------------ Decisão por bloco ------------------------
// t_gpu = latencia + bytes/banda ; t_cpu = elems/(taxa*nucleos)
// G (GPU por periodo) estimado = clamp(round(PERIOD * frac_gpu)) com
//   frac_gpu = t_cpu/(t_gpu+t_cpu)  ->  proporcional ao tempo de resposta/barramento
static int estimar_g_por_periodo(double lat_ms, double bw_gbps, double cpu_rate,
                                 size_t cores, size_t be) {
    double bytes = (double)be * 4 * 2;
    double t_gpu_ms = lat_ms + bytes / (bw_gbps * 1e9) * 1e3;
    double t_cpu_ms = (double)be / (cpu_rate * cores) * 1e3;
    double frac = t_cpu_ms / (t_gpu_ms + t_cpu_ms);
    int g = (int)std::lround(PERIOD * frac);
    return std::max(0, std::min(PERIOD, g));
}

// ------------------------ Execução por blocos intercalados ------------------------
static double run_blocks(const float* d_in, float* d_out, float* h_in, float* h_out,
                         float alpha, size_t total, int g_period,
                         const std::vector<int>& phys, int block,
                         size_t* gpu_base_host, size_t* d_gpu_base, int* ngpu_out) {
    size_t nb = total / BLOCK_ELEMS;
    int nperiods = (int)(nb / PERIOD);

    // Padrão intercalado: em cada periodo, primeiros G blocos -> GPU, resto -> CPU
    int G = std::max(0, std::min(PERIOD, g_period));
    int gpu_blocks = nperiods * G;
    int cpu_blocks = nb - gpu_blocks;

    // bases dos blocos GPU (contiguas no vetor, intercaladas na memoria dos dados)
    size_t* gb = gpu_base_host;
    int gcount = 0;
    for (int p = 0; p < nperiods; ++p)
        for (int k = 0; k < G; ++k)
            gb[gcount++] = (size_t)(p * PERIOD + k) * BLOCK_ELEMS;

    // fila da CPU com semaforo (ids restantes)
    IoQueue q;
    std::vector<size_t> cids;
    for (int p = 0; p < nperiods; ++p)
        for (int k = G; k < PERIOD; ++k)
            cids.push_back((size_t)(p * PERIOD + k));
    q.init(cids);

    std::vector<CpuArg> cargs;
    std::vector<pthread_t> ths;
    for (int i = 0; i < (int)phys.size(); ++i)
        cargs.push_back({h_in, h_out, alpha, BLOCK_ELEMS, phys[i], &q});

    cudaMemcpy(d_gpu_base, gb, gpu_blocks * sizeof(size_t), cudaMemcpyHostToDevice);

    auto t0 = std::chrono::high_resolution_clock::now();
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (gpu_blocks > 0)
        compute_offsets<<<gpu_blocks, block, 0, stream>>>(d_in, d_out, alpha,
                                                          d_gpu_base, gpu_blocks, BLOCK_ELEMS);
    for (auto& a : cargs) {
        pthread_t t; pthread_create(&t, nullptr, cpu_worker, &a); ths.push_back(t);
    }
    for (auto t : ths) pthread_join(t, nullptr);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (ngpu_out) *ngpu_out = gpu_blocks;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ------------------------ SA ------------------------
static double rnd01(unsigned int* seed) {
    return (double)(rand_r(seed) & 0xFFFFFF) / 16777215.0;
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : N_DEFAULT;
    std::string out_json = (argc > 2) ? argv[2] : "resultado_blocks.json";
    g_fma = (argc > 3) ? atoi(argv[3]) : 128;
    if (g_fma < 1) g_fma = 1;
    if (N % BLOCK_ELEMS) N = (N / BLOCK_ELEMS) * BLOCK_ELEMS;
    size_t nb = N / BLOCK_ELEMS;

    // nucleos fisicos
    std::vector<int> phys;
    int ncpu = std::thread::hardware_concurrency();
    for (int c = 0; c < ncpu && (int)phys.size() < 32; c += 2) phys.push_back(c);
    if (phys.empty()) phys.push_back(0);

    // GPU
    int dev = 0;
    cudaSetDevice(dev);
    cudaMemcpyToSymbol(c_fma, &g_fma, sizeof(int));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int block = 256;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)compute_offsets);
    {
        int min_grid = 0;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, (const void*)compute_offsets, 0, 0);
        if (block <= 0) block = 256;
    }

    // buffers
    float* h_in; float* h_out;
    posix_memalign((void**)&h_in, 64, N * sizeof(float));
    posix_memalign((void**)&h_out, 64, N * sizeof(float));
    for (size_t i = 0; i < N; ++i) h_in[i] = (float)(i % 1024) / 1024.0f;
    float* d_in; float* d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    float alpha = 0.5f;

    // modelo: barramento + tempo de resposta da GPU
    double lat_ms = medir_latencia_gpu();
    double bw_gbps = medir_banda_gpu(256ULL * 1024 * 1024);
    double cpu_rate = medir_taxa_cpu_1core(h_in, h_out, alpha, 1 << 20);
    double cores = (double)phys.size();
    int g_modelo = estimar_g_por_periodo(lat_ms, bw_gbps, cpu_rate, cores, BLOCK_ELEMS);

    // infra do despacho
    int max_gpu = (int)(nb / PERIOD) * PERIOD;
    size_t* gpu_base_host = (size_t*)calloc(max_gpu, sizeof(size_t));
    size_t* d_gpu_base; cudaMalloc(&d_gpu_base, max_gpu * sizeof(size_t));

    auto best_of = [&](int G) {
        double best = 1e18;
        for (int k = 0; k < REPEATS; ++k) {
            double t = run_blocks(d_in, d_out, h_in, h_out, alpha, N, G, phys, block,
                                  gpu_base_host, d_gpu_base, nullptr);
            if (t < best) best = t;
        }
        return best;
    };

    // sweep G de 0..PERIOD
    std::vector<std::vector<double>> sweep;
    for (int G = 0; G <= PERIOD; ++G) {
        double t = best_of(G);
        sweep.push_back({(double)G / PERIOD, t});
        printf("ratio %.2f (G=%d/%d) -> %8.3f ms  (%.1f M elem/s)\n",
               (double)G / PERIOD, G, PERIOD, t, N / t / 1e3);
    }

    // SA sobre G (parte do modelo + afinacao fina)
    unsigned int seed = 52;
    double T = 2.0, T_min = 1e-3, cool = 0.88;
    int g_best = g_modelo, t_best_ms = 0;
    double t_best = best_of(g_modelo);
    int g_cur = g_modelo; double t_cur = t_best;
    std::vector<std::vector<double>> sa_trace;
    for (int it = 0; it < SA_ITERS && T > T_min; ++it) {
        int g_new = std::max(0, std::min(PERIOD, g_cur + (rand_r(&seed) % 3) - 1));
        double t_new = best_of(g_new);
        double dT = t_new - t_cur;
        if (dT < 0 || rnd01(&seed) < std::exp(-dT / (T * 20.0 + 1e-9))) {
            g_cur = g_new; t_cur = t_new;
            if (t_new < t_best) { g_best = g_new; t_best = t_new; }
        }
        sa_trace.push_back({(double)it, (double)g_cur, t_cur});
        T *= cool;
    }

    // resumo do modelo
    printf("\n=== MODELO (barramento + resposta GPU) ===\n");
    printf("Latencia GPU: %.4f ms/launch | Banda GPU: %.1f GB/s | CPU: %.0f Melem/s por nucleo\n",
           lat_ms, bw_gbps, cpu_rate / 1e6);
    double t_g = lat_ms + (double)BLOCK_ELEMS * 8 / (bw_gbps * 1e9) * 1e3;
    double t_c = (double)BLOCK_ELEMS / (cpu_rate * cores) * 1e3;
    printf("Por bloco (1M elem): GPU %.4f ms vs CPU %.2f ms -> modelo propoe G=%d/%d\n",
           t_g, t_c, g_modelo, PERIOD);

    printf("\n=== RESUMO ===\n");
    printf("GPU: %s (CC %d.%d, %d SMs, regs/thread=%d, bloco=%d, ocupacao ajustada)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           attr.numRegs, block);
    int ngpu = 0;
    double t_best_rep = run_blocks(d_in, d_out, h_in, h_out, alpha, N, g_best, phys, block,
                                   gpu_base_host, d_gpu_base, &ngpu);
    double cpu_only = sweep[0][1];
    double gpu_only = sweep[PERIOD][1];
    double speedup = std::fmin(cpu_only, gpu_only) / t_best;
    printf("Blocos: %zu no total, %d atribuidos a GPU no otimo (padrao intercalado P=%d)\n",
           nb, ngpu, PERIOD);
    printf("CPU-only: %.2f ms | GPU-only: %.2f ms | SA otimo G=%d (%.2f ms)\n",
           cpu_only, gpu_only, g_best, t_best);
    printf("Speedup vs melhor sozinho: %.2fx\n", speedup);

    // padrao de despacho (primeiros 2 periodos)
    printf("Padrao de despacho (C=CPU, G=GPU): ");
    for (int p = 0; p < 2; ++p)
        for (int k = 0; k < PERIOD; ++k)
            printf("%c ", k < g_best ? 'G' : 'C');
    printf("\n");

    // JSON
    std::ostringstream js;
    js << "{\n";
    js << "  \"experimento\": \"Despacho hibrido por bloco (intercalado) + semaforo I/O\",\n";
    js << "  \"gpu\": {\"nome\": \"" << prop.name << "\", \"cc\": " << prop.major << "."
       << prop.minor << ", \"sms\": " << prop.multiProcessorCount
       << ", \"bloco_otimo\": " << block << "},\n";
    js << "  \"cpu\": {\"workers_fisicos\": " << phys.size() << ", \"nucleos\": [";
    for (size_t i = 0; i < phys.size(); ++i) js << phys[i] << (i + 1 < phys.size() ? "," : "");
    js << "]},\n";
    js << "  \"n_elementos\": " << N << ",\n";
    js << "  \"fma_por_elemento\": " << g_fma << ",\n";
    js << "  \"blocos\": " << nb << ", \"periodo\": " << PERIOD << ",\n";
    js << "  \"modelo\": {\"latencia_gpu_ms\": " << lat_ms << ", \"banda_gpu_gbps\": "
       << bw_gbps << ", \"taxa_cpu_melem_s\": " << cpu_rate / 1e6
       << ", \"t_gpu_bloco_ms\": " << t_g << ", \"t_cpu_bloco_ms\": " << t_c
       << ", \"g_proposto\": " << g_modelo << "},\n";
    js << "  \"sweep\": [\n";
    for (size_t i = 0; i < sweep.size(); ++i)
        js << "    {\"ratio\": " << sweep[i][0] << ", \"tempo_ms\": " << sweep[i][1]
           << ", \"m_elem_s\": " << N / sweep[i][1] / 1e3 << "}"
           << (i + 1 < sweep.size() ? "," : "") << "\n";
    js << "  ],\n";
    js << "  \"simulated_annealing\": {\"melhor_g\": " << g_best
       << ", \"melhor_tempo_ms\": " << t_best << ", \"iteracoes\": " << sa_trace.size()
       << ", \"traco\": [\n";
    for (size_t i = 0; i < sa_trace.size(); ++i)
        js << "      {\"iter\": " << (int)sa_trace[i][0] << ", \"g\": " << (int)sa_trace[i][1]
           << ", \"tempo_ms\": " << sa_trace[i][2] << "}"
           << (i + 1 < sa_trace.size() ? "," : "") << "\n";
    js << "    ]},\n";
    js << "  \"comparativo\": {\"cpu_only_ms\": " << cpu_only
       << ", \"gpu_only_ms\": " << gpu_only
       << ", \"hibrido_otimo_ms\": " << t_best
       << ", \"speedup_vs_melhor_sozinho\": " << speedup << "}\n";
    js << "}\n";

    std::ofstream ofs(out_json);
    ofs << js.str();
    ofs.close();
    printf("\nJSON salvo em: %s\n", out_json.c_str());

    (void)t_best_rep;
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gpu_base);
    free(h_in); free(h_out); free(gpu_base_host);
    return 0;
}
