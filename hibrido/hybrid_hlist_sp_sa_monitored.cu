// hybrid_hlist_sp_sa_monitored.cu
// Solver pentadiagonal SP (Classe E) com:
//  - Despacho hierarquico O(1) + Simulated Annealing online
//  - MEGA-LOTE (1 kernel+sync por MEGA=128 blocos; validado: SA ~42,7 ms @262k)
//  - Telemetria termica e energetica via NVML (C API)
//  - Trava de protecao termica automatica (interrupcao segura)
//  - Modo multi-iteracao (RHS sin/cos gerado UMA vez; solve por iteracao)
//
// Compilar: nvcc -O3 -arch=sm_89 -fmad=false -Xcompiler "-O3 -fopenmp"
//           hybrid_hlist_sp_sa_monitored.cu -o sp_sa_monitored -lpthread -lnvml
// Rodar:    ./sp_sa_monitored [n] [json] [nb_blocos] [linhas_bloco] [iteracoes] [temp_limite]

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>

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

constexpr int PERIOD = 4;
constexpr int BATCH = 8;
constexpr int MEGA = 128;                  // blocos por lancamento GPU (mega-lote)
constexpr int N_DEFAULT_PENTA = 1020;
constexpr int NBLOCKS_DEFAULT = 1024;
constexpr int LINES_PER_BLOCK_DEFAULT = 256;

// Simulated Annealing
constexpr double SA_T0 = 1.0;
constexpr double SA_ALPHA = 0.9;
constexpr double SA_TFREEZE = 0.05;
constexpr double SA_DX = 0.03;
constexpr int SA_WIN = 16;
constexpr double SA_X_MIN = 0.05;
constexpr double SA_X_MAX = 0.95;

// Estrutura de Telemetria Termica e Energetica
struct GpuTelemetry {
    unsigned int temp_c = 0;
    double power_w = 0.0;
    bool nvml_ok = false;
};

static GpuTelemetry ler_telemetria_gpu(nvmlDevice_t dev, bool ativo) {
    GpuTelemetry t;
    if (!ativo) return t;
    t.nvml_ok = true;
    if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t.temp_c) != NVML_SUCCESS)
        t.temp_c = 0;
    unsigned int power_mw = 0;
    if (nvmlDeviceGetPowerUsage(dev, &power_mw) == NVML_SUCCESS)
        t.power_w = power_mw / 1000.0;
    return t;
}

// ------------------------ Solver pentadiagonal (SP) ------------------------
static void factor_penta(double* l1, double* l2, double* u0, double* u1, double* u2, int n) {
    const double a = -0.25, b = -1.0, c = 6.0, d = -1.0, e = -0.25;
    u0[0] = c; u1[0] = d; u2[0] = e;
    l1[1] = b / u0[0];
    u0[1] = c - l1[1]*u1[0]; u1[1] = d - l1[1]*u2[0]; u2[1] = e;
    l2[2] = a / u0[0];
    l1[2] = (b - l2[2]*u1[0]) / u0[1];
    u0[2] = c - l2[2]*u2[0] - l1[2]*u1[1];
    u1[2] = d - l1[2]*u2[1]; u2[2] = e;
    for (int i = 3; i < n; i++) {
        l2[i] = a / u0[i-2];
        l1[i] = (b - l2[i]*u1[i-2]) / u0[i-1];
        u0[i] = c - l2[i]*u2[i-2] - l1[i]*u1[i-1];
        u1[i] = d - l1[i]*u2[i-1];
        u2[i] = e;
    }
}

__global__ void solve_lines_kernel(const double* __restrict__ b, double* __restrict__ u, int n,
    const double* __restrict__ l1, const double* __restrict__ l2,
    const double* __restrict__ u0, const double* __restrict__ u1, const double* __restrict__ u2,
    const size_t* __restrict__ gpu_base, int lines_per_block, int batch) {
    int blk = blockIdx.x;
    if (blk >= batch) return;
    int t = threadIdx.x;
    if (t >= lines_per_block) return;
    long line = gpu_base[blk] / (size_t)n + t;
    long base = line * (long)n;

    double y1, y2, yy;
    yy = b[base];               u[base]    = yy; y2 = yy;
    yy = b[base+1] - l1[1]*y2;  u[base+1]  = yy; y1 = yy;
    for (int i = 2; i < n; i++) {
        yy = b[base+i] - l1[i]*y1 - l2[i]*y2;
        u[base+i] = yy;
        y2 = y1; y1 = yy;
    }
    u[base+n-1] = u[base+n-1] / u0[n-1];
    u[base+n-2] = (u[base+n-2] - u1[n-2]*u[base+n-1]) / u0[n-2];
    u[base+n-3] = (u[base+n-3] - u1[n-3]*u[base+n-2] - u2[n-3]*u[base+n-1]) / u0[n-3];
    for (int i = n-4; i >= 0; i--) {
        u[base+i] = (u[base+i] - u1[i]*u[base+i+1] - u2[i]*u[base+i+2]) / u0[i];
    }
}

static void cpu_solve_lines(const double* __restrict__ b, double* __restrict__ u,
                            const double* l1, const double* l2,
                            const double* u0, const double* u1, const double* u2,
                            int n, long line_start, long line_end) {
    std::vector<double> y(n);
    for (long line = line_start; line < line_end; ++line) {
        long base = line * (long)n;
        y[0] = b[base];
        y[1] = b[base+1] - l1[1] * y[0];
        for (int i = 2; i < n; i++)
            y[i] = b[base+i] - l1[i] * y[i-1] - l2[i] * y[i-2];
        u[base + n-1] = y[n-1] / u0[n-1];
        u[base + n-2] = (y[n-2] - u1[n-2] * u[base+n-1]) / u0[n-2];
        u[base + n-3] = (y[n-3] - u1[n-3] * u[base+n-2] - u2[n-3] * u[base+n-1]) / u0[n-3];
        for (int i = n-4; i >= 0; i--)
            u[base+i] = (y[i] - u1[i] * u[base+i+1] - u2[i] * u[base+i+2]) / u0[i];
    }
}

// ------------------------ Lista Ligada Hierarquica ------------------------
struct BlockNode {
    size_t id, idx, base;
    BlockNode *prev, *next;
};

struct PeriodNode {
    int pid;
    BlockNode* blocks[PERIOD];
    BlockNode *head, *tail;
    int count;
    PeriodNode *prev, *next;
};

struct DeviceNode {
    int dev;
    PeriodNode** periods;
    PeriodNode* pool;
    PeriodNode *head, *tail;
    long nblocks;
};

static DeviceNode livre{0, nullptr, nullptr, nullptr, nullptr, 0};
static DeviceNode cpu  {1, nullptr, nullptr, nullptr, nullptr, 0};
static DeviceNode gpu  {2, nullptr, nullptr, nullptr, nullptr, 0};

static pthread_mutex_t gmtx = PTHREAD_MUTEX_INITIALIZER;
static BlockNode* g_nodes = nullptr;

static inline void unlink_block(PeriodNode* P, BlockNode* b) {
    if (b->prev) b->prev->next = b->next; else P->head = b->next;
    if (b->next) b->next->prev = b->prev; else P->tail = b->prev;
    b->prev = b->next = nullptr;
    P->count--;
}

static inline void unlink_period(DeviceNode* D, PeriodNode* P) {
    if (P->prev) P->prev->next = P->next; else D->head = P->next;
    if (P->next) P->next->prev = P->prev; else D->tail = P->prev;
    P->prev = P->next = nullptr;
}

static void move_to_device(DeviceNode* D, BlockNode* b) {
    int p = (int)(b->id / PERIOD);
    int idx = (int)(b->id % PERIOD);
    pthread_mutex_lock(&gmtx);

    PeriodNode* DP = D->periods[p];
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
    DP->blocks[idx] = b;
    DP->count++;
    if (DP->head) { DP->tail->next = b; b->prev = DP->tail; DP->tail = b; }
    else DP->head = DP->tail = b;
    D->nblocks++;
    pthread_mutex_unlock(&gmtx);
}

static BlockNode* livre_pop() {
    if (!livre.head) return nullptr;
    BlockNode* b = livre.head->head;
    unlink_block(livre.head, b);
    if (livre.head->count == 0) unlink_period(&livre, livre.head);
    livre.nblocks--;
    return b;
}

static BlockNode* cpu_pop() {
    pthread_mutex_lock(&gmtx);
    BlockNode* b = nullptr;
    while (cpu.head && cpu.head->count == 0)
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

struct Coeffs {
    int n;
    double *h_l1, *h_l2, *h_u0, *h_u1, *h_u2;
    double *d_l1, *d_l2, *d_u0, *d_u1, *d_u2;
};

struct CpuArg {
    const double* in; double* out;
    size_t be; int core;
    Coeffs* c;
    std::atomic<long>* cpu_done;
    std::atomic<int>* done;
};

static void* cpu_worker(void* arg_) {
    CpuArg* a = (CpuArg*)arg_;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(a->core, &set);
    sched_setaffinity(0, sizeof(set), &set);
    auto* c = a->c;
    int n = c->n;
    long lpp = (long)(a->be / n);
    while (!a->done->load(std::memory_order_relaxed)) {
        BlockNode* b = cpu_pop();
        if (!b) {
            if (a->done->load(std::memory_order_relaxed)) break;
            sched_yield();
            continue;
        }
        cpu_solve_lines(a->in, a->out, c->h_l1, c->h_l2, c->h_u0, c->h_u1, c->h_u2,
                        n, b->base / n, b->base / n + lpp);
        a->cpu_done->fetch_add(1, std::memory_order_relaxed);
    }
    while (true) {
        BlockNode* b = cpu_pop();
        if (!b) break;
        cpu_solve_lines(a->in, a->out, c->h_l1, c->h_l2, c->h_u0, c->h_u1, c->h_u2,
                        n, b->base / n, b->base / n + lpp);
        a->cpu_done->fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

struct Result {
    long gpu_blocks = 0, cpu_blocks = 0;
    double gpu_ms_per_block = 0, cpu_ms_per_block = 0;
    double checksum = 0;
    std::vector<size_t> gpu_bases;
    std::vector<std::vector<double>> trace;
    unsigned int final_temp_c = 0;
    double final_power_w = 0.0;
};

struct DispArg {
    const double* in; double* out; size_t be;
    size_t* gpu_base_host; size_t* d_gpu_base;
    std::atomic<long>* cpu_done;
    std::atomic<int>* done;
    int lines_per_block; int nperiods;
    int cpu_workers;
    Coeffs* c;
    double cpu_init_ms_per_block;
    double frac_fixed;
    nvmlDevice_t nvml_dev;
    bool nvml_active;
    unsigned int temp_limit;
    Result* r;
};

static double rnd01(unsigned int* seed) {
    return (double)(rand_r(seed) & 0xFFFFFF) / 16777215.0;
}

static void* gpu_dispatcher(void* arg_) {
    DispArg* dp = (DispArg*)arg_;
    auto& cpu_done = *dp->cpu_done;
    auto& R = *dp->r;
    auto* c = dp->c;
    int n = c->n;

    double gpu_ms_per_block = dp->cpu_init_ms_per_block;
    double cpu_ms_per_block = dp->cpu_init_ms_per_block;
    long gpu_assigned = 0, cpu_assigned = 0;
    bool gpu_measured = false;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    auto t_start = std::chrono::high_resolution_clock::now();
    size_t bases[BATCH];

    // Sonda inicial
    {
        long nb2 = 0;
        int take = (dp->frac_fixed >= 1.0) ? 0 : (int)std::min<long>(BATCH, livre.nblocks);
        for (int i = 0; i < take; ++i) {
            BlockNode* b = livre_pop();
            if (!b) break;
            bases[nb2] = b->base;
            move_to_device(&gpu, b);
            gpu_assigned++;
            nb2++;
        }
        if (nb2 > 0) {
            for (long i = 0; i < nb2; ++i) dp->gpu_base_host[i] = bases[i];
            cudaMemcpyAsync(dp->d_gpu_base, dp->gpu_base_host, nb2 * sizeof(size_t),
                            cudaMemcpyHostToDevice, stream);
            cudaEvent_t evS, evE;
            cudaEventCreate(&evS); cudaEventCreate(&evE);
            cudaEventRecord(evS, stream);
            solve_lines_kernel<<<(int)nb2, dp->lines_per_block, 0, stream>>>(
                dp->in, dp->out, n, c->d_l1, c->d_l2, c->d_u0, c->d_u1, c->d_u2,
                dp->d_gpu_base, dp->lines_per_block, (int)nb2);
            cudaEventRecord(evE, stream);
            cudaEventSynchronize(evE);
            float ms = 0; cudaEventElapsedTime(&ms, evS, evE);
            cudaEventDestroy(evS); cudaEventDestroy(evE);
            gpu_ms_per_block = ms / (double)nb2;
            gpu_measured = true;
            R.gpu_blocks += nb2;
        }
    }

    double r_g = 1.0 / std::fmax(gpu_ms_per_block, 1e-6);
    double r_c = 1.0 / std::fmax(dp->cpu_init_ms_per_block, 1e-6);
    double x = std::fmax(SA_X_MIN, std::fmin(SA_X_MAX, r_c / (r_g + r_c)));
    if (dp->frac_fixed >= 0.0) x = dp->frac_fixed;
    double x_prev = x, x_best = x;
    double E_prev = -1.0, E_best = 1e18;
    double T = SA_T0;
    unsigned int seed = 52;
    long win_blocks = 0;
    double win_t0 = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    long cpu_last_rep = 0;
    auto t_last_rep = t_start;

    // ===== MEGA-LOTE: blocos GPU decididos acumulam em pending[] e sao lancados
    // em lotes de MEGA blocos (1 kernel + 1 sync). Decisao por bloco identica.
    long npend = 0;
    size_t* pending = (size_t*)malloc(MEGA * sizeof(size_t));
    auto launch_pending = [&]() {
        if (npend <= 0) return;
        // Telemetria termica + trava antes de cada lance
        if (dp->nvml_active) {
            GpuTelemetry telem = ler_telemetria_gpu(dp->nvml_dev, true);
            R.final_temp_c = telem.temp_c;
            R.final_power_w = telem.power_w;
            if (telem.temp_c >= dp->temp_limit) {
                fprintf(stderr, "\n[TRAVA TERMICA ATIVA] GPU atingiu %u°C (Limite: %u°C). "
                                "Abortando para seguranca.\n",
                        telem.temp_c, dp->temp_limit);
                exit(2);
            }
        }
        for (long i = 0; i < npend; ++i)
            dp->gpu_base_host[R.gpu_blocks + i] = pending[i];
        cudaMemcpyAsync(dp->d_gpu_base + R.gpu_blocks, dp->gpu_base_host + R.gpu_blocks,
                        npend * sizeof(size_t), cudaMemcpyHostToDevice, stream);
        cudaEvent_t evS, evE;
        cudaEventCreate(&evS); cudaEventCreate(&evE);
        cudaEventRecord(evS, stream);
        solve_lines_kernel<<<(int)npend, dp->lines_per_block, 0, stream>>>(
            dp->in, dp->out, n, c->d_l1, c->d_l2, c->d_u0, c->d_u1, c->d_u2,
            dp->d_gpu_base + R.gpu_blocks, dp->lines_per_block, (int)npend);
        cudaEventRecord(evE, stream);
        cudaEventSynchronize(evE);
        float ms = 0; cudaEventElapsedTime(&ms, evS, evE);
        cudaEventDestroy(evS); cudaEventDestroy(evE);
        R.gpu_blocks += npend;
        if (!gpu_measured) {
            gpu_ms_per_block = ms / (double)npend;
            gpu_measured = true;
        } else {
            gpu_ms_per_block = 0.5 * gpu_ms_per_block + 0.5 * (ms / (double)npend);
        }
        npend = 0;
    };

    while (livre.head) {
        auto tn = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(tn - t_last_rep).count();
        long dcpu = cpu_done.load(std::memory_order_relaxed);
        if (dt > 3.0 && dcpu > cpu_last_rep) {
            double per = dt / (double)(dcpu - cpu_last_rep);
            cpu_ms_per_block = (cpu_ms_per_block == 0) ? per : 0.5 * cpu_ms_per_block + 0.5 * per;
            cpu_last_rep = dcpu; t_last_rep = tn;
        }

        int rem = livre.nblocks;
        int take = (int)std::min<long>(BATCH, rem);
        long to_cpu_n, to_gpu_n;
        if (dp->frac_fixed >= 1.0) {
            to_cpu_n = take; to_gpu_n = 0;
        } else if (dp->frac_fixed == 0.0) {
            to_cpu_n = 0; to_gpu_n = take;
        } else {
            long cpu_target = (long)std::llround((double)rem * x);
            long cpu_pending = cpu_assigned - cpu_done.load(std::memory_order_relaxed);
            long need = std::max<long>(0, cpu_target - cpu_pending);
            to_cpu_n = std::min<long>(take, need);
            to_gpu_n = take - to_cpu_n;
        }
        bool any_cpu = to_cpu_n > 0;

        for (long i = 0; i < to_cpu_n; ++i) {
            BlockNode* b = livre_pop();
            if (!b) break;
            move_to_device(&cpu, b);
            cpu_assigned++;
        }
        long nb2 = 0;
        long to_gpu_cap = std::min<long>(to_gpu_n, MEGA - npend);   // nunca estoura pending[]
        for (long i = 0; i < to_gpu_cap; ++i) {
            BlockNode* b = livre_pop();
            if (!b) break;
            pending[npend++] = b->base;
            move_to_device(&gpu, b);
            gpu_assigned++;
            nb2++;
        }
        if (npend >= MEGA) launch_pending();
        (void)nb2;

        win_blocks += take;
        double now_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        if (win_blocks >= SA_WIN || !livre.head) {
            double E = (now_ms - win_t0) / (double)win_blocks;
            if (E < E_best) { E_best = E; x_best = x; }
            if (dp->frac_fixed >= 0.0) {
                x = dp->frac_fixed;
            } else if (T > SA_TFREEZE) {
                if (E_prev < 0) {
                    E_prev = E; x_prev = x;
                } else if (E < E_prev || rnd01(&seed) < std::exp(-(E - E_prev) / (T + 1e-9))) {
                    E_prev = E; x_prev = x;
                } else {
                    x = x_prev;
                }
                T *= SA_ALPHA;
                x = std::fmax(SA_X_MIN, std::fmin(SA_X_MAX, x + (2.0 * rnd01(&seed) - 1.0) * SA_DX));
            } else {
                x = x_best;
            }
            win_t0 = now_ms; win_blocks = 0;
        }

        if ((int)R.trace.size() < 4000)
            R.trace.push_back({now_ms, x, any_cpu ? 0.0 : 1.0});
    }
    launch_pending();                          // flush do resto (fim da varredura)
    cudaStreamSynchronize(stream);
    free(pending);
    cudaStreamDestroy(stream);
    dp->done->store(1, std::memory_order_release);
    R.gpu_ms_per_block = gpu_ms_per_block;
    R.cpu_ms_per_block = cpu_ms_per_block;
    return nullptr;
}

static double run_hlist(const double* d_in, double* d_out, double* h_in, double* h_out,
                        size_t total, const std::vector<int>& phys,
                        int lines_per_block, Coeffs* c,
                        size_t* gpu_base_host, size_t* d_gpu_base,
                        double cpu_init_ms_per_block, double frac_fixed,
                        nvmlDevice_t nvml_dev, bool nvml_active, unsigned int temp_limit,
                        int iter_atual, int iteracoes_total, int ck_every,
                        Result* R) {
    int n = c->n;
    size_t be = (size_t)lines_per_block * (size_t)n;
    size_t nb = total / be;
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
        b->base = i * be;
        b->prev = b->next = nullptr;
        PeriodNode* P = &livre.pool[i / PERIOD];
        P->blocks[i % PERIOD] = b;
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
        cargs.push_back({h_in, h_out, be, phys[i], c, &cpu_done, &done});

    auto t0 = std::chrono::high_resolution_clock::now();
    pthread_t disp;
    DispArg darg{d_in, d_out, be, gpu_base_host, d_gpu_base,
                 &cpu_done, &done, lines_per_block, nper, (int)phys.size(),
                 c, cpu_init_ms_per_block, frac_fixed, nvml_dev, nvml_active, temp_limit, R};
    pthread_create(&disp, nullptr, gpu_dispatcher, &darg);
    for (auto& a : cargs) { pthread_t t; pthread_create(&t, nullptr, cpu_worker, &a); ths.push_back(t); }
    pthread_join(disp, nullptr);
    for (auto t : ths) pthread_join(t, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();

    R->cpu_blocks = cpu_done.load(std::memory_order_relaxed);
    R->gpu_blocks = gpu.nblocks;
    R->gpu_bases.assign(gpu_base_host, gpu_base_host + R->gpu_blocks);

    long nlines = (long)(total / n);
    std::vector<char> gpu_line(nlines, 0);
    for (size_t k = 0; k < R->gpu_bases.size(); ++k) {
        long gs = (long)(R->gpu_bases[k] / n);
        for (long ln = gs; ln < gs + lines_per_block && ln < nlines; ++ln)
            gpu_line[ln] = 1;
    }
    bool ck_faz = (ck_every > 0) && (iter_atual == 1 || iter_atual == iteracoes_total ||
                                     (iter_atual % ck_every) == 0);
    double checksum = 0.0;
    if (ck_faz) {
        #pragma omp parallel for reduction(+:checksum) schedule(static)
        for (size_t i = 0; i < total; ++i)
            if (!gpu_line[i / n]) checksum += h_out[i];
        cudaMemcpy(h_out, d_out, total * sizeof(double), cudaMemcpyDeviceToHost);
        #pragma omp parallel for reduction(+:checksum) schedule(static)
        for (size_t i = 0; i < total; ++i)
            if (gpu_line[i / n]) checksum += h_out[i];
    }
    R->checksum = checksum;

    delete[] g_nodes; delete[] livre.pool; delete[] cpu.pool; delete[] gpu.pool;
    delete[] livre.periods; delete[] cpu.periods; delete[] gpu.periods;
    livre.head = cpu.head = gpu.head = nullptr;
    livre.tail = cpu.tail = gpu.tail = nullptr;
    livre.nblocks = cpu.nblocks = gpu.nblocks = 0;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : N_DEFAULT_PENTA;
    std::string out_json = (argc > 2) ? argv[2] : "resultado_monitored.json";
    long nb = (argc > 3) ? atol(argv[3]) : NBLOCKS_DEFAULT;
    int lines_per_block = (argc > 4) ? atoi(argv[4]) : LINES_PER_BLOCK_DEFAULT;
    int iteracoes = (argc > 5) ? atoi(argv[5]) : 1;
    unsigned int temp_limit = (argc > 6) ? (unsigned int)atoi(argv[6]) : 84;

    int ck_every = 100;
    if (const char* env = getenv("HSP_CK_EVERY")) ck_every = atoi(env);

    if (n < 8) n = 8;
    if (nb < 8) nb = 8;
    if (lines_per_block < 1 || lines_per_block > 1024) lines_per_block = LINES_PER_BLOCK_DEFAULT;
    if (nb % PERIOD != 0) nb = (nb / PERIOD) * PERIOD;

    // Inicializacao da NVML (device 0)
    nvmlDevice_t nvml_dev;
    bool nvml_ok = (nvmlInit() == NVML_SUCCESS);
    if (nvml_ok) nvmlDeviceGetHandleByIndex(0, &nvml_dev);

    std::vector<int> phys;
    int ncpu = std::thread::hardware_concurrency();
    for (int c = 0; c < ncpu && (int)phys.size() < 32; c += 2) phys.push_back(c);
    if (phys.empty()) phys.push_back(0);

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    Coeffs c;
    c.n = n;
    c.h_l1 = (double*)malloc(n * sizeof(double));
    c.h_l2 = (double*)malloc(n * sizeof(double));
    c.h_u0 = (double*)malloc(n * sizeof(double));
    c.h_u1 = (double*)malloc(n * sizeof(double));
    c.h_u2 = (double*)malloc(n * sizeof(double));
    factor_penta(c.h_l1, c.h_l2, c.h_u0, c.h_u1, c.h_u2, n);
    cudaMalloc(&c.d_l1, n * sizeof(double));
    cudaMalloc(&c.d_l2, n * sizeof(double));
    cudaMalloc(&c.d_u0, n * sizeof(double));
    cudaMalloc(&c.d_u1, n * sizeof(double));
    cudaMalloc(&c.d_u2, n * sizeof(double));
    cudaMemcpy(c.d_l1, c.h_l1, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_l2, c.h_l2, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_u0, c.h_u0, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_u1, c.h_u1, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_u2, c.h_u2, n * sizeof(double), cudaMemcpyHostToDevice);

    size_t be = (size_t)lines_per_block * (size_t)n;
    size_t total = (size_t)nb * be;

    double* h_in; double* h_out;
    posix_memalign((void**)&h_in, 64, total * sizeof(double));
    posix_memalign((void**)&h_out, 64, total * sizeof(double));
    // fill OMP (mesma aritmetica do fill serial -> RHS bit-identico; gerado UMA vez)
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < (long long)total; ++idx) {
        long i = (long)(idx % (size_t)n);
        long j = (long)((idx / (size_t)n) % (size_t)n);
        long k = (long)(idx / ((size_t)n * (size_t)n));
        h_in[idx] = sin(0.001 * i) * cos(0.001 * j) + 0.5 * sin(0.001 * k);
    }
    double* d_in; double* d_out;
    cudaMalloc(&d_in, total * sizeof(double));
    cudaMalloc(&d_out, total * sizeof(double));
    cudaMemcpy(d_in, h_in, total * sizeof(double), cudaMemcpyHostToDevice);

    size_t* gpu_base_host = (size_t*)malloc(nb * sizeof(size_t));
    size_t* d_gpu_base; cudaMalloc(&d_gpu_base, nb * sizeof(size_t));

    double* tmp_in = (double*)malloc(be * sizeof(double));
    for (size_t i = 0; i < be; ++i) tmp_in[i] = 0.5;
    auto tc0 = std::chrono::high_resolution_clock::now();
    cpu_solve_lines(tmp_in, h_out, c.h_l1, c.h_l2, c.h_u0, c.h_u1, c.h_u2, n, 0, lines_per_block);
    auto tc1 = std::chrono::high_resolution_clock::now();
    double cpu_init_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / (double)phys.size();
    free(tmp_in);

    double frac_fixed = -1.0;
    if (const char* env = getenv("HSP_FIXED_FRAC")) frac_fixed = atof(env);

    double pontos_por_iter = (double)total;
    double sistemas_por_iter = (double)(total / (size_t)n);
    printf("====================================================================\n");
    printf("  NPB SP PENTADIAGONAL HIBRIDO + TELEMETRIA NVML (%s)\n", prop.name);
    printf("  Malha: %ld linhas (%zu pontos/iter) | %d iteracoes | Trava Termica: %u C\n",
           (long)(total / n), total, iteracoes, temp_limit);
    printf("  Total previsto: %.3e pontos | %.3e sistemas lineares (n=%d)\n",
           pontos_por_iter * iteracoes, sistemas_por_iter * iteracoes, n);
    printf("====================================================================\n");

    Result R;
    double t_total = 0.0;
    auto t_wall0 = std::chrono::high_resolution_clock::now();
    for (int iter = 1; iter <= iteracoes; ++iter) {
        Result r_iter;
        double t = run_hlist(d_in, d_out, h_in, h_out, total, phys, lines_per_block, &c,
                             gpu_base_host, d_gpu_base, cpu_init_ms, frac_fixed,
                             nvml_dev, nvml_ok, temp_limit, iter, iteracoes, ck_every, &r_iter);
        t_total += t;
        R = r_iter;
        if (iteracoes > 1) {
            if (iter <= 10 || iter % 25 == 0 || iter == iteracoes) {
                double el = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t_wall0).count();
                double eta = (iteracoes > iter) ? el / iter * (iteracoes - iter) : 0.0;
                bool ck_f0 = (ck_every > 0) && (iter == 1 || iter == iteracoes || (iter % ck_every) == 0);
                printf("[Iter %4d/%d] t=%.1f ms | ck=%s | GPU=%uC %.1fW | G%ld/C%ld | ETA %.0fs\n",
                       iter, iteracoes, t,
                       ck_f0 ? std::to_string(r_iter.checksum).c_str() : "-",
                       r_iter.final_temp_c, r_iter.final_power_w,
                       r_iter.gpu_blocks, r_iter.cpu_blocks, eta);
                fflush(stdout);
            }
        }
    }
    double wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();

    double t_medio = t_total / iteracoes;
    double gflops = ((double)total * 28.0) / (t_medio * 1e6);

    printf("\n=== RESULTADO CONSOLIDADO ===\n");
    printf("Pontos totais resolvidos: %.3e (marca 1,06e12 %s)\n",
           pontos_por_iter * iteracoes,
           (pontos_por_iter * iteracoes >= 1.0e12) ? "ATINGIDA" : "NAO atingida");
    printf("Tempo medio/iteracao: %.3f ms | Wall total: %.1f s | Vazao: %.3e pontos/s\n",
           t_medio, wall, pontos_por_iter * iteracoes / wall);
    printf("Vazao: %.2f GFLOPS (estimado 28 flops/ponto) | Checksum: %.8e\n",
           gflops, R.checksum);
    printf("Telemetria Final: Temperatura = %u C | Consumo = %.1f W\n",
           R.final_temp_c, R.final_power_w);

    std::ostringstream js;
    js << "{\n"
       << "  \"gpu\": \"" << prop.name << "\",\n"
       << "  \"tempo_medio_ms\": " << t_medio << ",\n"
       << "  \"wall_total_s\": " << wall << ",\n"
       << "  \"gflops\": " << gflops << ",\n"
       << "  \"checksum\": " << R.checksum << ",\n"
       << "  \"telemetria\": {\"temp_c\": " << R.final_temp_c << ", \"power_w\": " << R.final_power_w << "},\n"
       << "  \"blocos\": {\"total\": " << nb << ", \"gpu\": " << R.gpu_blocks << ", \"cpu\": " << R.cpu_blocks << "},\n"
       << "  \"contagem\": {\"iteracoes\": " << iteracoes << ", \"pontos_por_iter\": " << (long long)total
       << ", \"pontos_totais\": " << (long long)(pontos_por_iter * iteracoes)
       << ", \"sistemas_por_iter\": " << (long long)sistemas_por_iter
       << ", \"sistemas_totais\": " << (long long)(sistemas_por_iter * iteracoes) << "}\n"
       << "}\n";
    std::ofstream ofs(out_json);
    ofs << js.str();
    ofs.close();

    if (nvml_ok) nvmlShutdown();
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gpu_base);
    cudaFree(c.d_l1); cudaFree(c.d_l2); cudaFree(c.d_u0); cudaFree(c.d_u1); cudaFree(c.d_u2);
    free(h_in); free(h_out); free(gpu_base_host);
    free(c.h_l1); free(c.h_l2); free(c.h_u0); free(c.h_u1); free(c.h_u2);
    return 0;
}