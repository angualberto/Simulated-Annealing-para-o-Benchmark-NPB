// hybrid_benchmark.cu
// Experimento HPC híbrido CPU+GPU com auto-ajuste dinâmico:
//   - GPU: cudaOccupancyMaxPotentialBlockSize (registradores/thread vs ocupacao dos SMs)
//   - CPU: threads pthread fixadas em nucleos fisicos reais via sched_setaffinity
//   - Vetor de Despacho (dispatch_table) preenchido em O(1) antes do disparo
//   - Simulated Annealing para achar o gpu_ratio otimo
//   - Medicoes de desempenho (tempo, throughput) salvas em JSON
//
// Compilar: nvcc -O3 -arch=sm_90 -o hybrid_bench hybrid_benchmark.cu -lpthread
// Rodar:    ./hybrid_bench [N] [arquivo_json]

#include <cuda_runtime.h>
#include <cuda.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <sstream>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

// ------------------------ Config ------------------------
constexpr size_t N_DEFAULT = 64ULL * 1024 * 1024;  // 64M floats
static int g_fma = 128;                            // FMAs por elemento (argv[3])
__constant__ int c_fma;                            // mesma config para o kernel
constexpr int REPEATS = 5;
constexpr int SA_ITERS = 40;
constexpr int CPU_WORKERS_MAX = 32;

// ------------------------ Estrutura do vetor de despacho ------------------------
struct DeviceExecutionConfig {
    int device_id;          // 0 = CPU, 1 = GPU
    size_t work_offset;
    size_t work_size;
    int grid_dim;
    int block_dim;
    size_t dynamic_smem;
    int regs_per_thread;
    int target_physical_core;
};

// ------------------------ Kernel CUDA ------------------------
__global__ void compute_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               float alpha, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
#pragma unroll 8
        for (int j = 0; j < c_fma; ++j)
            v = fmaf(v, alpha, 0.001f * j);
        out[idx] = v;
    }
}

// ------------------------ Computacao CPU (mesma carga) ------------------------
static inline void cpu_compute(const float* __restrict__ in, float* __restrict__ out,
                               float alpha, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        float v = in[i];
        for (int j = 0; j < g_fma; ++j)
            v = fmaf(v, alpha, 0.001f * j);
        out[i] = v;
    }
}

struct CpuWorkerArg {
    const float* in;
    float* out;
    float alpha;
    size_t start, end;
    int core;
    double cpu_ms;
};

static void* cpu_worker(void* arg_) {
    CpuWorkerArg* a = static_cast<CpuWorkerArg*>(arg_);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(a->core, &set);
    int rc = sched_setaffinity(0, sizeof(set), &set);  // fixa no nucleo fisico
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_compute(a->in, a->out, a->alpha, a->start, a->end);
    auto t1 = std::chrono::high_resolution_clock::now();
    a->cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    (void)rc;
    return nullptr;
}

// ------------------------ Auto-ajuste híbrido ------------------------
static std::vector<DeviceExecutionConfig> auto_tune_hybrid_partition(
    size_t total, float gpu_ratio, const std::vector<int>& phys_cores,
    int block_hint = 0) {
    std::vector<DeviceExecutionConfig> cfg;
    size_t gpu_n = static_cast<size_t>(total * gpu_ratio);
    size_t cpu_n = total - gpu_n;

    if (gpu_n > 0) {
        int block = 256;
        if (block_hint <= 0) {
            int min_grid = 0;
            cudaError_t e = cudaOccupancyMaxPotentialBlockSize(
                &min_grid, &block, (const void*)compute_kernel, 0, 0);
            if (e != cudaSuccess || block <= 0) block = 256;
        } else {
            block = block_hint;
        }
        cudaFuncAttributes at;
        cudaFuncGetAttributes(&at, (const void*)compute_kernel);
        size_t grid = (gpu_n + block - 1) / block;
        cfg.push_back({1, 0, gpu_n, (int)grid, block, 0, at.numRegs, -1});
    }
    if (cpu_n > 0 && !phys_cores.empty()) {
        size_t nw = phys_cores.size();
        size_t chunk = cpu_n / nw;
        size_t base = gpu_n;
        for (size_t i = 0; i < nw; ++i) {
            size_t off = base + i * chunk;
            size_t sz = (i == nw - 1) ? (total - off) : chunk;
            cfg.push_back({0, off, sz, 0, 0, 0, 0, phys_cores[i]});
        }
    }
    return cfg;
}

// ------------------------ Execução híbrida (GPU + CPU em paralelo) ------------------------
static double run_hybrid(const float* d_in, float* d_out, float* h_in, float* h_out,
                         float alpha, size_t total, float gpu_ratio,
                         const std::vector<int>& phys_cores, int block_hint) {
    auto cfg = auto_tune_hybrid_partition(total, gpu_ratio, phys_cores, block_hint);
    size_t gpu_n = 0;
    int gpu_block = 256, gpu_grid = 0;
    std::vector<CpuWorkerArg> cargs;
    std::vector<pthread_t> ths;

    for (auto& c : cfg) {
        if (c.device_id == 1) {
            gpu_n = c.work_size;
            gpu_block = c.block_dim;
            gpu_grid = c.grid_dim;
        } else {
            cargs.push_back({h_in, h_out, alpha, c.work_offset,
                             c.work_offset + c.work_size, c.target_physical_core});
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    cudaStream_t stream;
    cudaError_t e1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (gpu_n > 0) {
        compute_kernel<<<gpu_grid, gpu_block, 0, stream>>>(d_in, d_out, alpha, gpu_n);
    }
    for (auto& a : cargs) {
        pthread_t t;
        pthread_create(&t, nullptr, cpu_worker, &a);
        ths.push_back(t);
    }
    for (auto t : ths) pthread_join(t, nullptr);
    cudaStreamSynchronize(stream);
    (void)e1;
    cudaStreamDestroy(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)d_out; (void)h_out;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ------------------------ Simulated Annealing ------------------------
static double rnd01(unsigned int* seed) {
    return (double)(rand_r(seed) & 0xFFFFFF) / 16777215.0;
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? (size_t)atoll(argv[1]) : N_DEFAULT;
    std::string out_json = (argc > 2) ? argv[2] : "resultado_hibrido.json";
    g_fma = (argc > 3) ? atoi(argv[3]) : 128;
    if (g_fma < 1) g_fma = 1;

    // ---- Topologia: nucleos fisicos reais (pares) ----
    std::vector<int> phys;
    int ncpu = std::thread::hardware_concurrency();
    int max_core = 0;
    for (int c = 0; c < ncpu && (int)phys.size() < CPU_WORKERS_MAX; c += 2)
        phys.push_back(c);
    if (phys.empty()) phys.push_back(0);
    max_core = phys.back();

    // ---- GPU ----
    int dev = 0;
    cudaSetDevice(dev);
    cudaMemcpyToSymbol(c_fma, &g_fma, sizeof(int));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)compute_kernel);
    int block = 256, min_grid = 0;
    cudaError_t occ_err = cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &block, (const void*)compute_kernel, 0, 0);
    if (occ_err != cudaSuccess || block <= 0) block = 256;
    int max_active = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active, (const void*)compute_kernel,
                                                  block, 0);
    double occupancy = (double)max_active * block / prop.maxThreadsPerMultiProcessor;

    // ---- Buffers ----
    float* h_in;  float* h_out;
    posix_memalign((void**)&h_in, 64, N * sizeof(float));
    posix_memalign((void**)&h_out, 64, N * sizeof(float));
    for (size_t i = 0; i < N; ++i) h_in[i] = (float)(i % 1024) / 1024.0f;
    float* d_in;  float* d_out;
    cudaError_t ce;
    ce = cudaMalloc(&d_in, N * sizeof(float));
    printf("cudaMalloc d_in: %s\n", cudaGetErrorString(ce));
    ce = cudaMalloc(&d_out, N * sizeof(float));
    printf("cudaMalloc d_out: %s\n", cudaGetErrorString(ce));
    ce = cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    printf("cudaMemcpy: %s\n", cudaGetErrorString(ce));
    float alpha = 0.5f;

    compute_kernel<<<64, 256>>>(d_in, d_out, alpha, 16384);
    ce = cudaDeviceSynchronize();
    printf("test launch sync: %s\n", cudaGetErrorString(ce));

    // ---- Sweep de razoes ----
    std::vector<double> ratios = {0.0, 0.25, 0.5, 0.75, 1.0};
    auto best_of = [&](float r, int block_hint) {
        double best = 1e18;
        for (int k = 0; k < REPEATS; ++k) {
            double t = run_hybrid(d_in, d_out, h_in, h_out, alpha, N, r, phys, block_hint);
            if (t < best) best = t;
        }
        return best;
    };

    std::vector<std::vector<double>> sweep;
    for (double r : ratios) {
        double t = best_of((float)r, 0);
        sweep.push_back({r, t});
        printf("ratio %.2f -> %8.3f ms  (%.1f M elem/s)\n", r, t,
               N / t / 1e3);
    }

    // ---- Simulated Annealing no gpu_ratio ----
    unsigned int seed = 52;
    double T = 0.20, T_min = 1e-4, cool = 0.90;
    double r_best = 0.75, t_best = best_of(0.75f, 0);
    double r_cur = 0.75, t_cur = t_best;
    std::vector<std::vector<double>> sa_trace;
    for (int it = 0; it < SA_ITERS && T > T_min; ++it) {
        double r_new = std::fmin(1.0, std::fmax(0.0, r_cur + (rnd01(&seed) - 0.5) * 0.30));
        double t_new = best_of((float)r_new, 0);
        double dT = t_new - t_cur;
        if (dT < 0 || rnd01(&seed) < std::exp(-dT / (T * 50.0 + 1e-9))) {
            r_cur = r_new; t_cur = t_new;
            if (t_new < t_best) { r_best = r_new; t_best = t_new; }
        }
        sa_trace.push_back({(double)it, r_cur, t_cur});
        T *= cool;
    }

    // ---- Comparativo: bloco fixo 256 vs ocupacao otima, no melhor ratio ----
    double t_tuned = t_best;
    double t_fixed = best_of((float)r_best, 256);
    double cpu_only = sweep[0][1];
    double gpu_only = sweep[4][1];
    double speedup_hybrid = std::fmin(cpu_only, gpu_only) / t_best;

    // ---- Resultados (tabela + JSON) ----
    auto now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    std::string ts = ctime(&tt); ts.pop_back();

    std::ostringstream js;
    js << "{\n";
    js << "  \"experimento\": \"Hibrido CPU+GPU auto-ajuste\",\n";
    js << "  \"data\": \"" << ts << "\",\n";
    js << "  \"gpu\": {\"nome\": \"" << prop.name << "\", \"cc\": " << prop.major << "."
       << prop.minor << ", \"sms\": " << prop.multiProcessorCount
       << ", \"max_threads_sm\": " << prop.maxThreadsPerMultiProcessor
       << ", \"regs_per_thread\": " << attr.numRegs
       << ", \"block_otimo\": " << block
       << ", \"ocupacao\": " << occupancy << "},\n";
    js << "  \"cpu\": {\"workers_fisicos\": " << phys.size() << ", \"nucleos\": [";
    for (size_t i = 0; i < phys.size(); ++i) js << phys[i] << (i + 1 < phys.size() ? "," : "");
    js << "]},\n";
    js << "  \"n_elementos\": " << N << ",\n";
    js << "  \"fma_por_elemento\": " << g_fma << ",\n";
    js << "  \"sweep\": [\n";
    for (size_t i = 0; i < sweep.size(); ++i)
        js << "    {\"ratio\": " << sweep[i][0] << ", \"tempo_ms\": " << sweep[i][1]
           << ", \"m_elem_s\": " << N / sweep[i][1] / 1e3 << "}"
           << (i + 1 < sweep.size() ? "," : "") << "\n";
    js << "  ],\n";
    js << "  \"simulated_annealing\": {\n";
    js << "    \"melhor_ratio\": " << r_best << ",\n";
    js << "    \"melhor_tempo_ms\": " << t_best << ",\n";
    js << "    \"iteracoes\": " << sa_trace.size() << ",\n";
    js << "    \"traco\": [\n";
    for (size_t i = 0; i < sa_trace.size(); ++i)
        js << "      {\"iter\": " << (int)sa_trace[i][0] << ", \"ratio\": " << sa_trace[i][1]
           << ", \"tempo_ms\": " << sa_trace[i][2] << "}"
           << (i + 1 < sa_trace.size() ? "," : "") << "\n";
    js << "    ]\n";
    js << "  },\n";
    js << "  \"comparativo\": {\"cpu_only_ms\": " << cpu_only
       << ", \"gpu_only_ms\": " << gpu_only
       << ", \"hibrido_otimo_ms\": " << t_tuned
       << ", \"speedup_hibrido_vs_melhor_sozinho\": " << speedup_hybrid
       << ", \"hibrido_bloco_fixo256_ms\": " << t_fixed << "}\n";
    js << "}\n";

    std::ofstream ofs(out_json);
    ofs << js.str();
    ofs.close();

    printf("\n=== RESUMO ===\n");
    printf("GPU: %s (regs/thread=%d, bloco otimo=%d, ocupacao=%.1f%%)\n",
           prop.name, attr.numRegs, block, 100.0 * occupancy);
    printf("CPU: %zu workers fisicos em cores %d..%d\n", phys.size(), phys.front(), phys.back());
    printf("Carga: N=%zu elementos, %d FMA/elem (g_fma=%d)\n", N, g_fma, g_fma);
    printf("CPU-only: %.2f ms | GPU-only: %.2f ms\n", cpu_only, gpu_only);
    printf("SA melhor ratio=%.2f tempo=%.2f ms\n", r_best, t_best);
    printf("Speedup hibrido vs melhor sozinho: %.2fx\n", speedup_hybrid);
    printf("Bloco fixo 256: %.2f ms | Ocupacao otima: %.2f ms\n", t_fixed, t_tuned);
    printf("JSON salvo em: %s\n", out_json.c_str());

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
