#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>

#define N 1020
#define BLOCK_LINES 256

const double A_ = -0.25;  // sub-sub-diagonal (ds, lower 2 away)
const double B_ = -1.0;   // sub-diagonal     (dl, lower 1 away)
const double C_ =  6.0;   // main diagonal    (d)
const double D_ = -1.0;   // super-diagonal   (du, upper 1 away)
const double E_ = -0.25;  // super-super      (dw, upper 2 away)

#include "cuPentBatch.cu"

// ------------------------------------------------------------------
// kernel de referencia: SOLVER EXATO de producao (sp_class_e / dispatcher)
// LU banda completa (l1,l2,u0,u1,u2), fatores precomutados NO CPU 1x,
// layout natural (linha*N+i), 1 thread/linha - NAO coalescido
// ------------------------------------------------------------------
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

__global__ void kernel_prod_exact(double* __restrict__ r, size_t total_lines,
    const double* __restrict__ l1, const double* __restrict__ l2,
    const double* __restrict__ u0, const double* __restrict__ u1,
    const double* __restrict__ u2, int n) {
    size_t line_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_id >= total_lines) return;
    size_t base = line_id * (size_t)n;
    double y1, y2, yy;
    yy = r[base];               r[base]   = yy; y2 = yy;
    yy = r[base+1] - l1[1]*y2;  r[base+1] = yy; y1 = yy;
    for (int i = 2; i < n; i++) {
        yy = r[base+i] - l1[i]*y1 - l2[i]*y2;
        r[base+i] = yy;
        y2 = y1; y1 = yy;
    }
    r[base+n-1] = r[base+n-1] / u0[n-1];
    r[base+n-2] = (r[base+n-2] - u1[n-2]*r[base+n-1]) / u0[n-2];
    r[base+n-3] = (r[base+n-3] - u1[n-3]*r[base+n-2] - u2[n-3]*r[base+n-1]) / u0[n-3];
    for (int i = n-4; i >= 0; i--)
        r[base+i] = (r[base+i] - u1[i]*r[base+i+1] - u2[i]*r[base+i+2]) / u0[i];
}

// ------------------------------------------------------------------
// preenche fatoracoes com coefs constantes do SP (limites = 0)
// ------------------------------------------------------------------
__global__ void fill_penta(double* ds, double* dl, double* d, double* du, double* dw,
                           int m, int bc) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bc) return;
    for (int i = 0; i < m; ++i) {
        int k = i * bc + t;
        ds[k] = (i >= 2) ? A_ : 0.0;
        dl[k] = (i >= 1) ? B_ : 0.0;
        d [k] = C_;
        du[k] = (i < m - 1) ? D_ : 0.0;
        dw[k] = (i < m - 2) ? E_ : 0.0;
    }
}

__global__ void set_one(double* v, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = 1.0;
}

template <typename F>
double time_kernel(F launch) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    launch();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)ms;
}

int main(int argc, char** argv) {
    size_t total_lines = (argc > 1) ? std::stoul(argv[1]) : (1024 * BLOCK_LINES);
    int R = (argc > 2) ? std::atoi(argv[2]) : 5;
    int m = N, bc = (int)total_lines;
    size_t nelem = (size_t)m * bc;
    size_t bytes = nelem * sizeof(double);

    std::printf("=== cuPentBatch vs baseline 1-thread/linha (SP, n=%d) ===\n", N);
    std::printf("linhas=%zu  pontos=%zu  bytes/b=%zu MB (RHS), fator=%zu MB\n",
                total_lines, nelem, bytes / (1024*1024), 5*bytes / (1024*1024));

    double *ds, *dl, *d, *du, *dw, *b_il, *b_ni;
    cudaMalloc(&ds, bytes); cudaMalloc(&dl, bytes); cudaMalloc(&d, bytes);
    cudaMalloc(&du, bytes); cudaMalloc(&dw, bytes);
    cudaMalloc(&b_il, bytes); cudaMalloc(&b_ni, bytes);

    double h_l1[N], h_l2[N], h_u0[N], h_u1[N], h_u2[N];
    double *d_l1, *d_l2, *d_u0, *d_u1, *d_u2;
    cudaMalloc(&d_l1, N * 8); cudaMalloc(&d_l2, N * 8); cudaMalloc(&d_u0, N * 8);
    cudaMalloc(&d_u1, N * 8); cudaMalloc(&d_u2, N * 8);
    factor_penta(h_l1, h_l2, h_u0, h_u1, h_u2, N);
    cudaMemcpy(d_l1, h_l1, N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l2, h_l2, N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u0, h_u0, N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u1, h_u1, N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2, h_u2, N * 8, cudaMemcpyHostToDevice);

    double* h_b_il = (double*)malloc(bytes);
    double* h_b_ni = (double*)malloc(bytes);
    double* h_out  = (double*)malloc(bytes);
    for (size_t i = 0; i < nelem; ++i) { h_b_il[i] = 1.0; h_b_ni[i] = 1.0; }
    cudaMemcpy(b_il, h_b_il, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ni, h_b_ni, bytes, cudaMemcpyHostToDevice);

    int tpb = 256;
    int grid_b = (int)((total_lines + tpb - 1) / tpb);

    fill_penta<<<grid_b, tpb>>>(ds, dl, d, du, dw, m, bc);
    cudaDeviceSynchronize();

    // ---- aquecimento termico ----
    for (int w = 0; w < 2; ++w) {
        kernel_prod_exact<<<grid_b, tpb>>>(b_ni, total_lines, d_l1, d_l2, d_u0, d_u1, d_u2, N);
        pentFactorBatch<<<grid_b, tpb>>>(ds, dl, d, du, dw, m, bc);
        pentSolveBatch<<<grid_b, tpb>>>(ds, dl, d, du, dw, b_il, m, bc);
        set_one<<<(size_t)((nelem + tpb - 1) / tpb), tpb>>>(b_il, nelem);
        set_one<<<(size_t)((nelem + tpb - 1) / tpb), tpb>>>(b_ni, nelem);
        fill_penta<<<grid_b, tpb>>>(ds, dl, d, du, dw, m, bc);
    }
    cudaDeviceSynchronize();

    double best_b = 1e30, best_f = 1e30, best_s = 1e30, sum_b = 0, sum_f = 0, sum_s = 0;
    for (int k = 0; k < R; ++k) {
        double tb = time_kernel([&]{ kernel_prod_exact<<<grid_b, tpb>>>(b_ni, total_lines, d_l1, d_l2, d_u0, d_u1, d_u2, N); });
        fill_penta<<<grid_b, tpb>>>(ds, dl, d, du, dw, m, bc);
        cudaDeviceSynchronize();
        double tf = time_kernel([&]{ pentFactorBatch<<<grid_b, tpb>>>(ds, dl, d, du, dw, m, bc); });
        double ts = time_kernel([&]{ pentSolveBatch<<<grid_b, tpb>>>(ds, dl, d, du, dw, b_il, m, bc); });
        best_b = std::min(best_b, tb); best_f = std::min(best_f, tf); best_s = std::min(best_s, ts);
        sum_b += tb; sum_f += tf; sum_s += ts;
        set_one<<<(size_t)((nelem + tpb - 1) / tpb), tpb>>>(b_il, nelem);
        set_one<<<(size_t)((nelem + tpb - 1) / tpb), tpb>>>(b_ni, nelem);
    }
    cudaDeviceSynchronize();
    double avg_b = sum_b / R, avg_f = sum_f / R, avg_s = sum_s / R;

    // ---- validacao (rodada limpa; nao restaurar apos) ----
    fill_penta<<<grid_b, tpb>>>(ds, dl, d, du, dw, m, bc);
    set_one<<<(size_t)((nelem + tpb - 1) / tpb), tpb>>>(b_il, nelem);
    set_one<<<(size_t)((nelem + tpb - 1) / tpb), tpb>>>(b_ni, nelem);
    cudaDeviceSynchronize();
    pentFactorBatch<<<grid_b, tpb>>>(ds, dl, d, du, dw, m, bc);
    pentSolveBatch<<<grid_b, tpb>>>(ds, dl, d, du, dw, b_il, m, bc);
    kernel_prod_exact<<<grid_b, tpb>>>(b_ni, total_lines, d_l1, d_l2, d_u0, d_u1, d_u2, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_b_il, b_il, bytes, cudaMemcpyDeviceToHost);   // cuPentBatch: interleaved (i*bc+t)
    cudaMemcpy(h_b_ni, b_ni, bytes, cudaMemcpyDeviceToHost);   // producao: natural (t*m+i)

    auto x_cpb = [&](int t, int i) -> double {
        if (i < 0 || i >= m) return 0.0;
        return h_b_il[(size_t)i * bc + t];
    };
    auto x_prd = [&](int t, int i) -> double {
        if (i < 0 || i >= m) return 0.0;
        return h_b_ni[(size_t)t * m + i];
    };
    double ck_cpb = 0.0, ck_base = 0.0;
    double maxdiff = 0.0;
    for (size_t t = 0; t < (size_t)bc; ++t) {
        for (int i = 0; i < m; ++i) {
            double xc = x_cpb((int)t, i);
            double xb = x_prd((int)t, i);
            ck_cpb += xc; ck_base += xb;
            maxdiff = std::max(maxdiff, std::fabs(xc - xb));
        }
    }

    double rinf_cpb = 0.0, rinf_prd = 0.0;
    for (int t : {0, 777, bc - 1}) {
        for (int i = 0; i < m; ++i) {
            double r = 1.0
                - A_ * x_cpb(t, i - 2) - B_ * x_cpb(t, i - 1)
                - C_ * x_cpb(t, i)
                - D_ * x_cpb(t, i + 1) - E_ * x_cpb(t, i + 2);
            rinf_cpb = std::max(rinf_cpb, std::fabs(r));
            r = 1.0
                - A_ * x_prd(t, i - 2) - B_ * x_prd(t, i - 1)
                - C_ * x_prd(t, i)
                - D_ * x_prd(t, i + 1) - E_ * x_prd(t, i + 2);
            rinf_prd = std::max(rinf_prd, std::fabs(r));
        }
    }

    // ---- custo DMA do RHS (2,14 GB H2D, uma vez) ----
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    cudaMemcpyAsync(b_il, h_b_il, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms_dma = 0; cudaEventElapsedTime(&ms_dma, e0, e1);

    std::printf("\n--- tempos de KERNEL (min / media, %d runs, janela termica) ---\n", R);
    std::printf("[B] producao 1-thread/linha (LU exata, natural): %.2f ms  (med %.2f)\n", best_b, avg_b);
    std::printf("[F] cuPentBatch pentFactorBatch .............: %.2f ms  (med %.2f)\n", best_f, avg_f);
    std::printf("[S] cuPentBatch pentSolveBatch ..............: %.2f ms  (med %.2f)\n", best_s, avg_s);
    std::printf("[F+S] cuPentBatch total .....................: %.2f ms\n", best_f + best_s);
    std::printf("speedup solve vs producao: %.2fx   |  factor+solve vs producao: %.2fx\n",
                best_b / best_s, best_b / (best_f + best_s));
    std::printf("\n--- validacao (rodada limpa) ---\n");
    std::printf("checksum cuPentBatch: %.8e   producao: %.8e\n", ck_cpb, ck_base);
    std::printf("max |cuPentBatch - producao| (mesmo sistema) = %.3e\n", maxdiff);
    std::printf("residuo max ||A x - b||inf: cuPentBatch = %.3e   producao = %.3e\n",
                rinf_cpb, rinf_prd);
    std::printf("DMA RHS H2D (2,14 GB): %.1f ms\n", ms_dma);

    cudaFree(ds); cudaFree(dl); cudaFree(d); cudaFree(du); cudaFree(dw);
    cudaFree(b_il); cudaFree(b_ni);
    free(h_b_il); free(h_b_ni); free(h_out);
    return 0;
}
