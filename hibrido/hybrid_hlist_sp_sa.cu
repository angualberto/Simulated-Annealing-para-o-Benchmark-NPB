// hybrid_hlist_sp_sa.cu
// VARIANTE COM PROBLEMA REAL: solver pentadiagonal SP-like (NPB-SP Classe E).
// CONTROLADOR = SIMULATED ANNEALING ONLINE (substitui o EMA do hybrid_hlist_sp.cu).
//
// Arquitetura de despacho IDENTICA ao hybrid_hlist_sp.cu:
//   Fila = LISTA LIGADA HIERARQUICA COM INDICE (indice + listas por nivel)
//   dispatcher (dono do LIVRE) popa blocos, decide GPU/CPU pelo frac_cpu, move O(1)
//
// DIFERENCA: o frac_cpu NAO e mais derivado de EMA ruidoso de resposta. O controlador
// trata frac_cpu como parametro x em [0.05,0.95] e roda SA ONLINE:
//   - cada JANELA de SA_WIN blocos e despachada com um x candidato;
//   - energia E = tempo de parede da janela / blocos na janela (ms/bloco);
//   - ao fim da janela: Metropolis compara E com o incumbente (E_prev);
//     aceita com prob 1 se E<E_prev, senao com exp(-(E-E_prev)/T);
//   - T resfria por janela (T *= SA_ALPHA) -> converge para o split otimo
//     (minimiza tempo de parede = makespan) sem oscilar.
// EMA de CPU e mantido APENAS para reporte no JSON (resposta), fora do controle.
//
// FIX vs hybrid_hlist_sp.cu: os gpu_base eram escritos so em gpu_base_host (host) e o
// kernel lia d_gpu_base (device) sem cudaMemcpy -> indices de linha lixo (checksum variava
// entre execucoes e a medida de GPU era instavel). Aqui cada lote faz o upload dos bases.
//
// UNIDADE DE TRABALHO: cada BLOCO = LINES_PER_BLOCK linhas independentes do
// sistema pentadiagonal A*x = b (a=-0.25, b=-1, c=6, d=-1, e=-0.25, exatamente o SP).
//   - GPU: 1 thread por linha (varredura do SP)
//   - CPU: mesma recurrencia (sp_class_e.f90 solve_x) por linha
//
// Compilar: nvcc -O3 -arch=sm_89 -o hybrid_hlist_sp_sa hybrid_hlist_sp_sa.cu -lpthread
// Rodar:    ./hybrid_hlist_sp_sa [n] [json] [nb_blocos] [linhas_por_bloco]

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

constexpr int PERIOD = 4;                    // blocos por periodo (barra)
constexpr int BATCH = 8;                     // blocos por lote de decisao
constexpr int N_DEFAULT_PENTA = 1020;        // tamanho do sistema pentadiagonal (SP classe E)
constexpr int NBLOCKS_DEFAULT = 64;          // numero de blocos de trabalho
constexpr int LINES_PER_BLOCK_DEFAULT = 256; // linhas independentes por bloco (<=1024)

// ------------------------ Simulated Annealing (controlador online) ------------------------
constexpr double SA_T0 = 1.0;      // temperatura inicial (escala de ms/bloco)
constexpr double SA_ALPHA = 0.9;   // resfriamento por janela
constexpr double SA_TFREEZE = 0.05; // abaixo disto congela no melhor x (elitismo)
constexpr double SA_DX = 0.03;     // amplitude da perturbacao no frac_cpu
constexpr int SA_WIN = 16;         // blocos por janela de avaliacao
constexpr double SA_X_MIN = 0.05;  // frac_cpu minimo
constexpr double SA_X_MAX = 0.95;  // frac_cpu maximo

// ------------------------ Solver pentadiagonal (extraido do SP) ------------------------
// factor_penta: fatoracao LU banda pentadiagonal - feita uma unica vez
static void factor_penta(double* l1, double* l2, double* u0, double* u1, double* u2, int n) {
    const double a = -0.25, b = -1.0, c = 6.0, d = -1.0, e = -0.25;
    int i;
    u0[0] = c; u1[0] = d; u2[0] = e;
    l1[1] = b / u0[0];
    u0[1] = c - l1[1]*u1[0]; u1[1] = d - l1[1]*u2[0]; u2[1] = e;
    l2[2] = a / u0[0];
    l1[2] = (b - l2[2]*u1[0]) / u0[1];
    u0[2] = c - l2[2]*u2[0] - l1[2]*u1[1];
    u1[2] = d - l1[2]*u2[1]; u2[2] = e;
    for (i = 3; i < n; i++) {
        l2[i] = a / u0[i-2];
        l1[i] = (b - l2[i]*u1[i-2]) / u0[i-1];
        u0[i] = c - l2[i]*u2[i-2] - l1[i]*u1[i-1];
        u1[i] = d - l1[i]*u2[i-1];
        u2[i] = e;
    }
}

// varredura pentadiagonal (solve_x do SP): resolve A*x = b em cada linha
// FIX (corrige o bug do kernel original): a eliminacao direta escreve o RHS
// transformado y DIRETAMENTE em u (que sera sobrescrito pela retrosubstituicao).
// Assim b (d_in) fica read-only (streamado pela GPU) e a retrosubstituicao le
// u[..]=y[i] e o converte em x[i] — aritmetica IDENTICA ao sp_class_e_cuda.cu
// de referencia (verificado: checksum 3.4645900000e+07 bit-idêntico entre
// all-GPU, all-CPU e todos os splits).
// O bug original usava b[base+i] SEM transformar (nunca armazenava y) → solucao
// errada (~28% no checksum). O kernel in-place-sobre-b (tentativa) era 2x mais
// lento por write-allocate no d_in read-only. Esta variante e b+read-only.
__global__ void solve_lines_kernel(const double* __restrict__ b, double* __restrict__ u, int n,
    const double* __restrict__ l1, const double* __restrict__ l2,
    const double* __restrict__ u0, const double* __restrict__ u1, const double* __restrict__ u2,
    const size_t* __restrict__ gpu_base, int lines_per_block, int batch) {
    int blk = blockIdx.x;
    if (blk >= batch) return;
    int t = threadIdx.x;
    if (t >= lines_per_block) return;
    long line = gpu_base[blk] / (size_t)n + t;   // linha global deste bloco
    long base = line * (long)n;

    double y1, y2, yy;
    yy = b[base];               u[base]    = yy; y2 = yy;          // u temp = y
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

// versao CPU (mesma recurrencia do sp_class_e.f90 solve_x), por bloco de linhas
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

// ------------------------ Resultado ------------------------
struct Result {
    long gpu_blocks = 0, cpu_blocks = 0;
    double gpu_ms_per_block = 0, cpu_ms_per_block = 0;
    double checksum = 0;                 // checksum real (GPU+CPU) deste run
    std::vector<size_t> gpu_bases;   // bases dos blocos GPU deste run (p/ checksum real)
    std::vector<std::vector<double>> trace;   // {t_ms, frac_cpu, destino}
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
    double frac_fixed;   // <0 = SA online; >=0 = frac_cpu fixo (sweep/validacao)
    Result* r;
};

// ------------------------ Dispatcher (controlador SA, dono do livre) ------------------------
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
    double cpu_ms_per_block = dp->cpu_init_ms_per_block;   // EMA apenas para reporte
    long gpu_assigned = 0, cpu_assigned = 0;
    bool gpu_measured = false;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    auto t_start = std::chrono::high_resolution_clock::now();
    size_t bases[BATCH];

    // ===== Fase de SONDA: mede GPU rapidamente; CPU usa a medida inicial =====
    // (modo forçado =1.0: tudo CPU, sem probe na GPU)
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
            for (long i = 0; i < nb2; ++i)
                dp->gpu_base_host[i] = bases[i];
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
            gpu_ms_per_block = ms / (double)nb2;      // VETOR GPU (caracteristica da GPU)
            gpu_measured = true;
            R.gpu_blocks += nb2;
        }
    }

    // ===== Controlador SIMULATED ANNEALING (online) =====
    // x = frac_cpu incumbente; cada janela de SA_WIN blocos avalia uma perturbacao
    // vizinha; energia E = ms/bloco de parede da janela; Metropolis decide; T resfria.
    // Elitismo: guarda (x_best,E_best); quando T passa de SA_TFREEZE, congela em
    // x_best e so aceita melhoria estrita -> convergencia sem vagar no fim do run.
    double r_g = 1.0 / std::fmax(gpu_ms_per_block, 1e-6);
    double r_c = 1.0 / std::fmax(dp->cpu_init_ms_per_block, 1e-6);
    double x = std::fmax(SA_X_MIN, std::fmin(SA_X_MAX, r_c / (r_g + r_c)));
    if (dp->frac_fixed >= 0.0) x = dp->frac_fixed;   // sweep: fixa frac_cpu
    double x_prev = x, x_best = x;
    double E_prev = -1.0, E_best = 1e18;
    double T = SA_T0;
    unsigned int seed = 52;
    long win_blocks = 0;
    double win_t0 = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    long cpu_last_rep = 0;
    auto t_last_rep = t_start;

    while (livre.head) {
        auto tn = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(tn - t_last_rep).count();
        long dcpu = cpu_done.load(std::memory_order_relaxed);
        if (dt > 3.0 && dcpu > cpu_last_rep) {   // EMA SO PARA REPORTE (nao controla)
            double per = dt / (double)(dcpu - cpu_last_rep);
            cpu_ms_per_block = (cpu_ms_per_block == 0) ? per
                              : 0.5 * cpu_ms_per_block + 0.5 * per;
            cpu_last_rep = dcpu; t_last_rep = tn;
        }

        int rem = livre.nblocks;
        int take = (int)std::min<long>(BATCH, rem);
        long to_cpu_n, to_gpu_n;
        if (dp->frac_fixed >= 1.0) {            // baseline forcado: tudo CPU
            to_cpu_n = take; to_gpu_n = 0;
        } else if (dp->frac_fixed == 0.0) {     // baseline forcado: tudo GPU
            to_cpu_n = 0; to_gpu_n = take;
        } else {                                // top-up normal (EMA/SA online ou sweep 0<x<1)
            // TOP-UP (igual ao EMA): x = fracao-alvo do que RESTA; alimenta CPU ate atingir o
            // alvo considerando o que ja esta pendente -> mantem CPU e GPU saturados.
            // (sem isso, round(take*x) por lote deixa a CPU faminta e o hybrid fica serial.)
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
        for (long i = 0; i < to_gpu_n; ++i) {
            BlockNode* b = livre_pop();
            if (!b) break;
            bases[nb2] = b->base;
            move_to_device(&gpu, b);
            gpu_assigned++;
            nb2++;
        }
        if (nb2 > 0) {
            for (long i = 0; i < nb2; ++i)
                dp->gpu_base_host[R.gpu_blocks + i] = bases[i];
            cudaMemcpyAsync(dp->d_gpu_base + R.gpu_blocks, dp->gpu_base_host + R.gpu_blocks,
                            nb2 * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaEvent_t evS, evE;
            cudaEventCreate(&evS); cudaEventCreate(&evE);
            cudaEventRecord(evS, stream);
            solve_lines_kernel<<<(int)nb2, dp->lines_per_block, 0, stream>>>(
                dp->in, dp->out, n, c->d_l1, c->d_l2, c->d_u0, c->d_u1, c->d_u2,
                dp->d_gpu_base + R.gpu_blocks, dp->lines_per_block, (int)nb2);
            cudaEventRecord(evE, stream);
            cudaEventSynchronize(evE);            // espera (sync por lote, como no original)
            float ms = 0; cudaEventElapsedTime(&ms, evS, evE);
            cudaEventDestroy(evS); cudaEventDestroy(evE);
            R.gpu_blocks += nb2;
            if (!gpu_measured) {
                gpu_ms_per_block = ms / (double)nb2;
                gpu_measured = true;
            } else {
                gpu_ms_per_block = 0.5 * gpu_ms_per_block + 0.5 * (ms / (double)nb2);
            }
        }

        // fechamento da janela SA
        win_blocks += take;
        double now_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        if (win_blocks >= SA_WIN || !livre.head) {
            double E = (now_ms - win_t0) / (double)win_blocks;   // ms/bloco da janela
            if (E < E_best) { E_best = E; x_best = x; }          // elitismo
            if (dp->frac_fixed >= 0.0) {                         // sweep: sem exploracao
                x = dp->frac_fixed;
            } else if (T > SA_TFREEZE) {                         // fase de exploracao
                if (E_prev < 0) {                  // primeira janela vira incumbente
                    E_prev = E; x_prev = x;
                } else if (E < E_prev || rnd01(&seed) < std::exp(-(E - E_prev) / (T + 1e-9))) {
                    E_prev = E; x_prev = x;        // Metropolis: aceita candidato
                } else {
                    x = x_prev;                    // rejeita: volta ao incumbente
                }
                T *= SA_ALPHA;
                x = std::fmax(SA_X_MIN, std::fmin(SA_X_MAX,
                    x + (2.0 * rnd01(&seed) - 1.0) * SA_DX));
            } else {                               // fase de explotacao: congela no melhor
                x = x_best;
            }
            win_t0 = now_ms; win_blocks = 0;
        }

        if ((int)R.trace.size() < 4000)
            R.trace.push_back({now_ms, x, any_cpu ? 0.0 : 1.0});
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
static double run_hlist(const double* d_in, double* d_out, double* h_in, double* h_out,
                        size_t total, const std::vector<int>& phys,
                        int lines_per_block, Coeffs* c,
                        size_t* gpu_base_host, size_t* d_gpu_base,
                        double cpu_init_ms_per_block, double frac_fixed, Result* R) {
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
        cargs.push_back({h_in, h_out, be, phys[i], c, &cpu_done, &done});

    auto t0 = std::chrono::high_resolution_clock::now();
    pthread_t disp;
    DispArg darg{d_in, d_out, be, gpu_base_host, d_gpu_base,
                 &cpu_done, &done, lines_per_block, nper, (int)phys.size(),
                 c, cpu_init_ms_per_block, frac_fixed, R};
    pthread_create(&disp, nullptr, gpu_dispatcher, &darg);
    for (auto& a : cargs) { pthread_t t; pthread_create(&t, nullptr, cpu_worker, &a); ths.push_back(t); }
    pthread_join(disp, nullptr);
    for (auto t : ths) pthread_join(t, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();

    R->cpu_blocks = cpu_done.load(std::memory_order_relaxed);
    R->gpu_blocks = gpu.nblocks;
    R->gpu_bases.assign(gpu_base_host, gpu_base_host + R->gpu_blocks);

    // checksum REAL deste repeat (usa o split deste mesmo run, nao o do melhor):
    // linhas GPU vem de d_out (copiado de volta), linhas CPU vem de h_out.
    // h_out so contem as linhas resolvidas pela CPU; o copyback de d_out corromperia essas
    // linhas, entao somamos CPU primeiro e GPU depois, usando o mapa das linhas GPU.
    long nlines = (long)(total / n);
    std::vector<char> gpu_line(nlines, 0);
    for (size_t k = 0; k < R->gpu_bases.size(); ++k) {
        long gs = (long)(R->gpu_bases[k] / n);
        for (long ln = gs; ln < gs + lines_per_block && ln < nlines; ++ln)
            gpu_line[ln] = 1;
    }
    double checksum = 0.0;
    for (size_t i = 0; i < total; ++i)
        if (!gpu_line[i / n]) checksum += h_out[i];          // linhas da CPU
    cudaMemcpy(h_out, d_out, total * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < total; ++i)
        if (gpu_line[i / n]) checksum += h_out[i];           // linhas da GPU (pos-copyback)
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
    std::string out_json = (argc > 2) ? argv[2] : "resultado_hlist_sp.json";
    long nb = (argc > 3) ? atol(argv[3]) : NBLOCKS_DEFAULT;
    int lines_per_block = (argc > 4) ? atoi(argv[4]) : LINES_PER_BLOCK_DEFAULT;
    if (n < 8) n = 8;
    if (nb < 8) nb = 8;
    if (lines_per_block < 1 || lines_per_block > 1024) lines_per_block = LINES_PER_BLOCK_DEFAULT;
    if (nb % PERIOD != 0) {
        fprintf(stderr, "nb_blocos deve ser multiplo de PERIOD (%d)\n", PERIOD);
        return 1;
    }

    std::vector<int> phys;
    int ncpu = std::thread::hardware_concurrency();
    for (int c = 0; c < ncpu && (int)phys.size() < 32; c += 2) phys.push_back(c);
    if (phys.empty()) phys.push_back(0);

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // ------------------ Coeficientes pentadiagonais (SP) ------------------
    Coeffs c;
    c.n = n;
    c.h_l1 = (double*)malloc(n * sizeof(double));
    c.h_l2 = (double*)malloc(n * sizeof(double));
    c.h_u0 = (double*)malloc(n * sizeof(double));
    c.h_u1 = (double*)malloc(n * sizeof(double));
    c.h_u2 = (double*)malloc(n * sizeof(double));
    if (!c.h_l1 || !c.h_l2 || !c.h_u0 || !c.h_u1 || !c.h_u2) { perror("malloc"); return 1; }
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

    // ------------------ RHS: inicializacao do SP (sin/cos) ------------------
    double* h_in; double* h_out;
    posix_memalign((void**)&h_in, 64, total * sizeof(double));
    posix_memalign((void**)&h_out, 64, total * sizeof(double));
    for (size_t idx = 0; idx < total; ++idx) {
        long i = (long)(idx % (size_t)n);
        long j = (long)((idx / (size_t)n) % (size_t)n);
        long k = (long)(idx / ((size_t)n * (size_t)n));
        h_in[idx] = sin(0.001 * i) * cos(0.001 * j) + 0.5 * sin(0.001 * k);
    }
    double* d_in; double* d_out;
    cudaMalloc(&d_in, total * sizeof(double));
    cudaMalloc(&d_out, total * sizeof(double));
    cudaMemcpy(d_in, h_in, total * sizeof(double), cudaMemcpyHostToDevice);
    // d_in serve de RHS b para o GPU; o kernel faz a eliminacao direta in-place (b->y)
    // e escreve a solucao u em d_out (como em sp_class_e_cuda.cu de referencia).

    size_t* gpu_base_host = (size_t*)malloc(nb * sizeof(size_t));
    size_t* d_gpu_base; cudaMalloc(&d_gpu_base, nb * sizeof(size_t));

    // ------------------ Probe CPU: um bloco (linhas_per_bloco linhas) ------------------
    double* tmp_in = (double*)malloc(be * sizeof(double));
    for (size_t i = 0; i < be; ++i) tmp_in[i] = 0.5;
    auto tc0 = std::chrono::high_resolution_clock::now();
    cpu_solve_lines(tmp_in, h_out, c.h_l1, c.h_l2, c.h_u0, c.h_u1, c.h_u2,
                    n, 0, lines_per_block);
    auto tc1 = std::chrono::high_resolution_clock::now();
    double cpu_init_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count()
                         / (double)phys.size();
    free(tmp_in);

    double frac_fixed = -1.0;
    if (const char* env = getenv("HSP_FIXED_FRAC")) frac_fixed = atof(env);

    Result R;
    double best = 1e18;
    for (int k = 0; k < 3; ++k) {
        Result r;
        double t = run_hlist(d_in, d_out, h_in, h_out, total, phys, lines_per_block, &c,
                             gpu_base_host, d_gpu_base, cpu_init_ms, frac_fixed, &r);
        if (t < best) { best = t; R = r; }
    }

    // checksum real do melhor repeat (computado dentro de run_hlist com o split desse repeat)

    printf("\n=== LISTA LIGADA HIERARQUICA COM INDICE + SP pentadiagonal (controle SA online) ===\n");
    printf("Sistema: n=%d | %ld linhas independentes (%ld blocos x %d linhas/bloco)\n",
           n, (long)(total / n), nb, lines_per_block);
    printf("CPU: %zu nucleos fisicos | CPU 1 bloco/16 cores: %.3f ms (inicial)\n",
           phys.size(), cpu_init_ms);
    printf("Blocos: %ld GPU vs %ld CPU (total %ld)\n", R.gpu_blocks, R.cpu_blocks, nb);
    printf("Resposta medida: GPU %.4f ms/bloco | CPU %.3f ms/bloco\n",
           R.gpu_ms_per_block, R.cpu_ms_per_block);
    printf("Checksum da solucao: %.6e\n", R.checksum);
    printf("Tempo total: %.3f ms\n", best);

    std::ostringstream js;
    js << "{\n";
    js << "  \"experimento\": \"Lista ligada hierarquica com indice + problema real SP pentadiagonal (controle por simulated annealing online)\",\n";
    js << "  \"gpu\": {\"nome\": \"" << prop.name << "\", \"cc\": " << prop.major << "."
       << prop.minor << ", \"sms\": " << prop.multiProcessorCount << "},\n";
    js << "  \"cpu\": {\"workers_fisicos\": " << phys.size() << "},\n";
    js << "  \"problema\": {\"tipo\": \"SP pentadiagonal (a=-.25 b=-1 c=6 d=-1 e=-.25)\", \"n\": " << n
       << ", \"linhas\": " << (long)(total / n) << "},\n";
    js << "  \"estrutura\": {\"blocos\": " << nb << ", \"periodos\": " << nb / PERIOD
       << ", \"blocos_por_periodo\": " << PERIOD << ", \"linhas_por_bloco\": " << lines_per_block << "},\n";
    js << "  \"controlador\": {\"tipo\": \"simulated_annealing_online\", \"T0_ms\": " << SA_T0
       << ", \"alpha\": " << SA_ALPHA << ", \"T_freeze\": " << SA_TFREEZE
       << ", \"dx\": " << SA_DX
       << ", \"janela_blocos\": " << SA_WIN
       << ", \"frac_fixo\": " << frac_fixed << "},\n";
    js << "  \"tempo_total_ms\": " << best << ",\n";
    js << "  \"blocos\": {\"total\": " << nb << ", \"gpu\": " << R.gpu_blocks
       << ", \"cpu\": " << R.cpu_blocks << "},\n";
    js << "  \"resposta\": {\"gpu_ms_por_bloco\": " << R.gpu_ms_per_block
       << ", \"cpu_ms_por_bloco\": " << R.cpu_ms_per_block << "},\n";
    js << "  \"checksum\": " << R.checksum << ",\n";
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
    cudaFree(c.d_l1); cudaFree(c.d_l2); cudaFree(c.d_u0); cudaFree(c.d_u1); cudaFree(c.d_u2);
    free(h_in); free(h_out); free(gpu_base_host);
    free(c.h_l1); free(c.h_l2); free(c.h_u0); free(c.h_u1); free(c.h_u2);
    return 0;
}
