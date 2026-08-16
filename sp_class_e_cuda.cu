#include <cuda_runtime.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1020
#define NITER 400

#define CK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while(0)

/* ---------- Kernels ---------- */

/* RHS difusivo 7-pontos */
__global__ void compute_b_kernel(const double* __restrict__ u, double* __restrict__ b, int n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n * n * n;
    if (idx >= total) return;
    long n2 = (long)n * n;
    long i = idx % n;
    long j = (idx / n) % n;
    long k = idx / n2;
    double acc = 6.0 * u[idx];
    if (i > 0)     acc -= u[idx - 1];
    if (i < n-1)   acc -= u[idx + 1];
    if (j > 0)     acc -= u[idx - n];
    if (j < n-1)   acc -= u[idx + n];
    if (k > 0)     acc -= u[idx - n2];
    if (k < n-1)   acc -= u[idx + n2];
    b[idx] = acc;
}

/* solve pentadiagonal em x: 1 thread por linha (j,k) */
__global__ void solve_x_kernel(double* __restrict__ b, double* __restrict__ u, int n,
    const double* __restrict__ l1, const double* __restrict__ l2,
    const double* __restrict__ u0, const double* __restrict__ u1, const double* __restrict__ u2) {
    long lin = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long nlines = (long)n * n;
    if (lin >= nlines) return;
    long j = lin % n;
    long k = lin / n;
    long base = j * n + k * (long)n * n;
    double y1 = 0.0, y2 = 0.0, yy;

    yy = b[base];
    b[base] = yy;
    y2 = yy;
    yy = b[base+1] - l1[1] * y2;
    b[base+1] = yy;
    y1 = yy;
    for (int i = 2; i < n; i++) {
        yy = b[base+i] - l1[i] * y1 - l2[i] * y2;
        b[base+i] = yy;
        y2 = y1; y1 = yy;
    }
    u[base + n-1] = b[base+n-1] / u0[n-1];
    u[base + n-2] = (b[base+n-2] - u1[n-2] * u[base+n-1]) / u0[n-2];
    u[base + n-3] = (b[base+n-3] - u1[n-3] * u[base+n-2] - u2[n-3] * u[base+n-1]) / u0[n-3];
    for (int i = n-4; i >= 0; i--) {
        u[base+i] = (b[base+i] - u1[i] * u[base+i+1] - u2[i] * u[base+i+2]) / u0[i];
    }
}

/* solve pentadiagonal em y: 1 thread por linha (i,k) */
__global__ void solve_y_kernel(double* __restrict__ b, double* __restrict__ u, int n,
    const double* __restrict__ l1, const double* __restrict__ l2,
    const double* __restrict__ u0, const double* __restrict__ u1, const double* __restrict__ u2) {
    long lin = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long nlines = (long)n * n;
    if (lin >= nlines) return;
    long i = lin % n;
    long k = lin / n;
    long base = i + k * (long)n * n;
    double y1 = 0.0, y2 = 0.0, yy;

    yy = b[base];
    b[base] = yy;
    y2 = yy;
    yy = b[base+n] - l1[1] * y2;
    b[base+n] = yy;
    y1 = yy;
    for (int j = 2; j < n; j++) {
        yy = b[base + j*n] - l1[j] * y1 - l2[j] * y2;
        b[base + j*n] = yy;
        y2 = y1; y1 = yy;
    }
    u[base + (n-1)*n] = b[base + (n-1)*n] / u0[n-1];
    u[base + (n-2)*n] = (b[base + (n-2)*n] - u1[n-2] * u[base + (n-1)*n]) / u0[n-2];
    u[base + (n-3)*n] = (b[base + (n-3)*n] - u1[n-3] * u[base + (n-2)*n] - u2[n-3] * u[base + (n-1)*n]) / u0[n-3];
    for (int j = n-4; j >= 0; j--) {
        u[base + j*n] = (b[base + j*n] - u1[j] * u[base + (j+1)*n] - u2[j] * u[base + (j+2)*n]) / u0[j];
    }
}

/* solve pentadiagonal em z: 1 thread por linha (i,j) */
__global__ void solve_z_kernel(double* __restrict__ b, double* __restrict__ u, int n,
    const double* __restrict__ l1, const double* __restrict__ l2,
    const double* __restrict__ u0, const double* __restrict__ u1, const double* __restrict__ u2) {
    long lin = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long nlines = (long)n * n;
    if (lin >= nlines) return;
    long i = lin % n;
    long j = lin / n;
    long base = i + j * n;
    long n2 = (long)n * n;
    double y1 = 0.0, y2 = 0.0, yy;

    yy = b[base];
    b[base] = yy;
    y2 = yy;
    yy = b[base+n2] - l1[1] * y2;
    b[base+n2] = yy;
    y1 = yy;
    for (int k = 2; k < n; k++) {
        yy = b[base + k*n2] - l1[k] * y1 - l2[k] * y2;
        b[base + k*n2] = yy;
        y2 = y1; y1 = yy;
    }
    u[base + (n-1)*n2] = b[base + (n-1)*n2] / u0[n-1];
    u[base + (n-2)*n2] = (b[base + (n-2)*n2] - u1[n-2] * u[base + (n-1)*n2]) / u0[n-2];
    u[base + (n-3)*n2] = (b[base + (n-3)*n2] - u1[n-3] * u[base + (n-2)*n2] - u2[n-3] * u[base + (n-1)*n2]) / u0[n-3];
    for (int k = n-4; k >= 0; k--) {
        u[base + k*n2] = (b[base + k*n2] - u1[k] * u[base + (k+1)*n2] - u2[k] * u[base + (k+2)*n2]) / u0[k];
    }
}

/* reducao (checksum) */
__global__ void sum_kernel(const double* __restrict__ v, double* __restrict__ out, long total) {
    __shared__ double sm[256];
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    double acc = 0.0;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = idx; i < total; i += stride) acc += v[i];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sm[0];
}

/* ---------- Host ---------- */

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

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : N;
    int niter = (argc > 2) ? atoi(argv[2]) : NITER;
    long total = (long)n * n * n;
    size_t sz = (size_t)total * sizeof(double);

    double *h_u = NULL, *h_l1, *h_l2, *h_u0, *h_u1, *h_u2, *h_sum = NULL;
    double *d_u, *d_b, *d_l1, *d_l2, *d_u0, *d_u1, *d_u2, *d_sum;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    CK(cudaMallocHost(&h_u, sz));
    CK(cudaMallocHost(&h_sum, blocks * sizeof(double)));
    CK(cudaMalloc(&d_u, sz));
    CK(cudaMalloc(&d_b, sz));
    CK(cudaMalloc(&d_sum, blocks * sizeof(double)));

    h_l1 = (double*)malloc(n * sizeof(double));
    h_l2 = (double*)malloc(n * sizeof(double));
    h_u0 = (double*)malloc(n * sizeof(double));
    h_u1 = (double*)malloc(n * sizeof(double));
    h_u2 = (double*)malloc(n * sizeof(double));
    if (!h_l1 || !h_l2 || !h_u0 || !h_u1 || !h_u2) { perror("malloc"); exit(1); }

    factor_penta(h_l1, h_l2, h_u0, h_u1, h_u2, n);
    CK(cudaMalloc(&d_l1, n * sizeof(double)));
    CK(cudaMalloc(&d_l2, n * sizeof(double)));
    CK(cudaMalloc(&d_u0, n * sizeof(double)));
    CK(cudaMalloc(&d_u1, n * sizeof(double)));
    CK(cudaMalloc(&d_u2, n * sizeof(double)));
    CK(cudaMemcpy(d_l1, h_l1, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_l2, h_l2, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_u0, h_u0, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_u1, h_u1, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_u2, h_u2, n * sizeof(double), cudaMemcpyHostToDevice));

    double t0 = omp_get_wtime();

    /* Inicializacao hibrida: OpenMP no host */
    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < total; idx++) {
        long i = idx % n, j = (idx / n) % n, k = idx / ((long)n * n);
        h_u[idx] = sin(0.001 * i) * cos(0.001 * j) + 0.5 * sin(0.001 * k);
    }
    CK(cudaMemcpy(d_u, h_u, sz, cudaMemcpyHostToDevice));

    printf("SP-like Classe E (CUDA+OpenMP) | Malha: %dx%dx%d (%ld pts) | Iters: %d\n", n, n, n, total, niter);

    for (int it = 1; it <= niter; it++) {
        compute_b_kernel<<<blocks, threads>>>(d_u, d_b, n);
        CK(cudaGetLastError());
        solve_x_kernel<<<(long)n*n/threads + 1, threads>>>(d_b, d_u, n, d_l1, d_l2, d_u0, d_u1, d_u2);
        CK(cudaGetLastError());
        compute_b_kernel<<<blocks, threads>>>(d_u, d_b, n);
        CK(cudaGetLastError());
        solve_y_kernel<<<(long)n*n/threads + 1, threads>>>(d_b, d_u, n, d_l1, d_l2, d_u0, d_u1, d_u2);
        CK(cudaGetLastError());
        compute_b_kernel<<<blocks, threads>>>(d_u, d_b, n);
        CK(cudaGetLastError());
        solve_z_kernel<<<(long)n*n/threads + 1, threads>>>(d_b, d_u, n, d_l1, d_l2, d_u0, d_u1, d_u2);
        CK(cudaGetLastError());

        if (it % 50 == 0) {
            CK(cudaDeviceSynchronize());
            printf("Iteracao %5d | tempo: %8.2f s\n", it, omp_get_wtime() - t0);
            fflush(stdout);
        }
    }

    CK(cudaDeviceSynchronize());
    sum_kernel<<<blocks, threads>>>(d_u, d_sum, total);
    CK(cudaMemcpy(h_sum, d_sum, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < blocks; i++) sum += h_sum[i];

    double t1 = omp_get_wtime();
    double gflops = (double)niter * (double)total * 40.0 / 1.0e9;
    printf("Tempo total (s): %.4f\n", t1 - t0);
    printf("Checksum: %.6e\n", sum);
    printf("Rendimento: %.2f GFlops\n", gflops / (t1 - t0));

    cudaFree(d_u); cudaFree(d_b); cudaFree(d_sum);
    cudaFree(d_l1); cudaFree(d_l2); cudaFree(d_u0); cudaFree(d_u1); cudaFree(d_u2);
    cudaFreeHost(h_u); cudaFreeHost(h_sum);
    free(h_l1); free(h_l2); free(h_u0); free(h_u1); free(h_u2);
    return 0;
}
