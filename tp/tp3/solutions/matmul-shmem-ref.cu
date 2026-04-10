#include <cstdio>  
#include <iostream>
#include <cmath>
#include "cuda.h"  

#define N 1024
#define BSXY 32

float *A, *B, *C;
float *dA, *dB, *dC;

__global__ void multiplyMatrixGPUByBlocksThreads2DNonMultipleSharedMemory(float *dA, float *dB, float *dC, int n)
{
    __shared__ float shA[BSXY][BSXY];
    __shared__ float shB[BSXY][BSXY];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * BSXY + ty;
    int col = blockIdx.y * BSXY + tx;

    float c = 0.0;

    for (int m = 0; m < (n + BSXY - 1) / BSXY; ++m) {
        // Load tile of A (row-major)
        if (row < n && (m * BSXY + tx) < n)
            shA[ty][tx] = dA[row * n + (m * BSXY + tx)];
        else
            shA[ty][tx] = 0.0;

        // Load tile of B (column-major: B[row + col * N])
        if (col < n && (m * BSXY + ty) < n)
            shB[ty][tx] = dB[(m * BSXY + ty) + col * n];
        else
            shB[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < BSXY; k++) {
            c += shA[ty][k] * shB[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        dC[row * n + col] = c;
    }
}

void verifyResults()
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += A[i * N + k] * B[k + j * N];
            }
            if (std::abs(C[i * N + j] - expected) > 1e-4) {
                std::cout << "Multiplication is incorrect for the element C[" << i << "][" << j << "]" << std::endl;
                std::cout << "GPU: " << C[i * N + j] << " CPU: " << expected << std::endl;
                return;
            }
        }
    }
    std::cout << "Multiplication is correct!" << std::endl;
}

int main(int argc, char **argv)
{
    size_t size = N * N * sizeof(float);
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    for (int j = 0; j < N; j++) { 
        for (int i = 0; i < N; i++) { 
            A[i * N + j] = (float)(i + j); 
            B[i + j * N] = 1.0f; 
        }
    }

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BSXY, BSXY);
    dim3 dimGrid((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    multiplyMatrixGPUByBlocksThreads2DNonMultipleSharedMemory<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
    cudaEventRecord(stop);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double flops = 2.0 * N * N * N;
    double gflops = (flops / (milliseconds / 1000.0)) / 1e9;

    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFlops/s" << std::endl;

    verifyResults();

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);

    return 0;
}
