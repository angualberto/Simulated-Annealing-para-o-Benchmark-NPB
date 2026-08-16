#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <cuda_runtime.h>


#define N 1020
#define BLOCK_LINES 256
#define WARP_SIZE 32


// Coeficientes escalares do NPB SP
__constant__ double d_a = -0.25;
__constant__ double d_b = -1.0;
__constant__ double d_c =  6.0;
__constant__ double d_d = -1.0;
__constant__ double d_e = -0.25;


const double h_a = -0.25;
const double h_b = -1.0;
const double h_c =  6.0;
const double h_d = -1.0;
const double h_e = -0.25;


double soma_device(double* d_ptr, size_t n, double* h_buf) {
    cudaMemcpy(h_buf, d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost);
    return std::accumulate(h_buf, h_buf + n, 0.0);
}


// =========================================================================
// 1. TÉCNICA BASELINE: 1 Thread por Linha (Varredura Serial Descoalescida)
// =========================================================================
__global__ void kernel_baseline_serial(double* __restrict__ rhs, size_t total_lines) {
    size_t line_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_id >= total_lines) return;


    double* u = rhs + (line_id * N);


    for (int i = 2; i < N; ++i) {
        u[i] = u[i] - (d_a / d_c) * u[i-2] - (d_b / d_c) * u[i-1];
    }
    for (int i = N - 3; i >= 0; --i) {
        u[i] = (u[i] - d_d * u[i+1] - d_e * u[i+2]) / d_c;
    }
}


// =========================================================================
// 2. TÉCNICA AVANÇADA (SOTA): Coalescência por Tiling + Warp Parallel
// =========================================================================
__global__ void kernel_advanced_coalesced(double* __restrict__ rhs, size_t total_lines) {
    // Shared memory para transposição local e acesso 100% coalescido à DRAM
    __shared__ double tile[32][33]; // +1 para evitar bank conflicts


    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    size_t global_line = warp_id;


    if (global_line >= total_lines) return;


    double* u = rhs + (global_line * N);


    // Processamento em blocos com leitura coalescida
    for (int chunk = 0; chunk < N; chunk += WARP_SIZE) {
        int idx = chunk + lane_id;
        if (idx < N) {
            tile[lane_id][0] = u[idx];
        }
        __syncwarp();


        // Eliminação local acelerada via registradores / intra-warp shuffle
        #pragma unroll 4
        for (int step = 1; step < WARP_SIZE; step *= 2) {
            double val = tile[lane_id][0];
            double up = __shfl_up_sync(0xFFFFFFFF, val, step);
            if (lane_id >= step) {
                tile[lane_id][0] -= (d_b / d_c) * up;
            }
            __syncwarp();
        }


        if (idx < N) {
            u[idx] = tile[lane_id][0] / d_c;
        }
        __syncwarp();
    }
}


// =========================================================================
// 3. KERNEL CPU OPENMP (Vetorizado nos Núcleos Físicos)
// =========================================================================
void solve_cpu(double* rhs, size_t start_line, size_t num_lines) {
    #pragma omp parallel for schedule(static)
    for (size_t l = 0; l < num_lines; ++l) {
        double* u = rhs + ((start_line + l) * N);
        for (int i = 2; i < N; ++i) {
            u[i] = u[i] - (h_a / h_c) * u[i-2] - (h_b / h_c) * u[i-1];
        }
        for (int i = N - 3; i >= 0; --i) {
            u[i] = (u[i] - h_d * u[i+1] - h_e * u[i+2]) / h_c;
        }
    }
}


int main(int argc, char** argv) {
    size_t num_blocks = (argc > 1) ? std::stoul(argv[1]) : 1024;
    size_t total_lines = num_blocks * BLOCK_LINES;
    size_t total_elements = total_lines * N;
    size_t bytes = total_elements * sizeof(double);


    std::cout << "===============================================================\n";
    std::cout << " BENCHMARK COMPARATIVO: SOTA AVANÇADO vs BASELINE vs HÍBRIDO  \n";
    std::cout << "===============================================================\n";
    std::cout << "Malha: " << total_lines << " linhas (" << total_elements << " pontos, n=" << N << ")\n\n";


    double *h_data, *d_data, *h_ck;
    cudaMallocHost(&h_data, bytes);
    cudaMalloc(&d_data, bytes);
    cudaMallocHost(&h_ck, bytes);


    for (size_t i = 0; i < total_elements; ++i) h_data[i] = 1.0;


    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);


    // -------------------------------------------------------------
    // TESTE 1: BASELINE GPU (1 Thread / Linha)
    // -------------------------------------------------------------
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    auto t0 = std::chrono::high_resolution_clock::now();

    int tpb = 256;
    int blocks_baseline = (total_lines + tpb - 1) / tpb;
    kernel_baseline_serial<<<blocks_baseline, tpb, 0, stream>>>(d_data, total_lines);
    cudaStreamSynchronize(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_baseline = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ck_baseline = soma_device(d_data, total_elements, h_ck);
    std::cout << "[1] GPU Baseline (1 thread/linha) : " << ms_baseline << " ms  checksum=" << ck_baseline << "\n";


    // -------------------------------------------------------------
    // TESTE 2: GPU AVANÇADO SOTA (Warp Shuffle + Coalescência)
    // -------------------------------------------------------------
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    auto t2 = std::chrono::high_resolution_clock::now();

    int threads_advanced = 128; // 4 warps por bloco
    int blocks_advanced = ((total_lines * WARP_SIZE) + threads_advanced - 1) / threads_advanced;
    kernel_advanced_coalesced<<<blocks_advanced, threads_advanced, 0, stream>>>(d_data, total_lines);
    cudaStreamSynchronize(stream);

    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_advanced = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double ck_advanced = soma_device(d_data, total_elements, h_ck);
    std::cout << "[2] GPU Avançado (Warp Shuffles)   : " << ms_advanced << " ms (Speedup: "
              << (ms_baseline / ms_advanced) << "x)  checksum=" << ck_advanced << "\n";


    // -------------------------------------------------------------
    // TESTE 3: SEU MÉTODO HÍBRIDO ADAPTATIVO (CPU + GPU Sobrepostos)
    // -------------------------------------------------------------
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    double frac_cpu = 0.44;
    size_t cpu_lines = total_lines * frac_cpu;
    size_t gpu_lines = total_lines - cpu_lines;
    size_t gpu_bytes = gpu_lines * N * sizeof(double);


    auto t4 = std::chrono::high_resolution_clock::now();


    // GPU processa sua fatia de forma assíncrona
    int blocks_gpu = (gpu_lines + tpb - 1) / tpb;
    kernel_baseline_serial<<<blocks_gpu, tpb, 0, stream>>>(d_data, gpu_lines);


    // CPU processa sua fatia em paralelo
    solve_cpu(h_data, gpu_lines, cpu_lines);


    cudaStreamSynchronize(stream);
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_hybrid = std::chrono::duration<double, std::milli>(t5 - t4).count();

    double ck_gpu_slice = soma_device(d_data, gpu_lines * N, h_ck);
    double ck_cpu_slice = std::accumulate(h_data + gpu_lines * N, h_data + total_elements, 0.0);
    std::cout << "[3] Seu Método Híbrido Adaptativo : " << ms_hybrid << " ms  checksum(gpu+ cpu)="
              << (ck_gpu_slice + ck_cpu_slice) << "\n";
    std::cout << "===============================================================\n";
    std::cout << "Referencia (checksum esperado com RHS=1.0, solver correto): ver soma do baseline\n";
    std::cout << "Se checksum do [2] != checksum do [1] -> kernel 'avancado' NAO resolve o mesmo sistema.\n";


    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaFreeHost(h_ck);
    cudaStreamDestroy(stream);
    return 0;
}