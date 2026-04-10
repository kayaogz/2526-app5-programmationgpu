/**
 * In this exercise, we will implement GPU kernels for computing the average of 9 points on a 2D array.
 * Dans cet exercice, nous implantons un kernel GPU pour un calcul de moyenne de 9 points sur un tableau 2D.
 *
 * Kernel 1: Use 1D grid of blocks (only blockIdx.x), no additional threads (1 thread per block)
 * Kernel 1: Utiliser grille 1D de blocs (seulement blockIdx.x), pas de threads (1 thread par bloc)
 *
 * Kernel 2: Use 2D grid of blocks (blockIdx.x/.y), no additional threads (1 thread per block)
 * Kernel 2: Utiliser grille 2D de blocs (blockIdx.x/.y), pas de threads (1 thread par bloc)
 *
 * Kernel 3: Use 2D grid of blocks and 2D threads (BSXY x BSXY), each thread computing 1 element of Aavg
 * Kernel 3: Utiliser grille 2D de blocs, threads de 2D (BSXY x BSXY), chaque thread calcule 1 element de Aavg
 *
 * Kernel 4: Use 2D grid of blocks and 2D threads, each thread computing 1 element of Aavg, use shared memory. Each block should load BSXY x BSXY elements of A, then compute (BSXY - 2) x (BSXY - 2) elements of Aavg. Borders of tiles loaded by different blocks must overlap to be able to compute all elements of Aavg.
 * Kernel 4: Utiliser grille 2D de blocs, threads de 2D, chaque thread calcule 1 element de Aavg, avec shared memory. Chaque bloc doit lire BSXY x BSXY elements de A, puis calculer avec ceci (BSXY - 2) x (BSXY - 2) elements de Aavg. Les bords des tuiles chargees par de differents blocs doivent chevaucher afin de pouvoir calculer tous les elements de Aavg.
 *
 * Kernel 5: Use 2D grid of blocks and 2D threads, use shared memory, each thread computes KxK elements of Aavg
 * Kernel 5: Utiliser grille 2D de blocs, threads de 2D, avec shared memory et KxK ops par thread
 *
 * For all kernels: Make necessary memory allocations/deallocations and memcpy in the main.
 * Pour tous les kernels: Effectuer les allocations/desallocations et memcpy necessaires dans le main.
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include "cuda.h"

#define N 1024
#define K 2
#define BSXY 32

float *A;

// Reference CPU implementation
// Implémentation CPU de référence
void ninePointAverageCPU(const float *A, float *Aavg) {
  for (int j = 1; j < N - 1; j++) {
    for (int i = 1; i < N - 1; i++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
                         A[i + (j - 1) * N]     + A[i + (j) * N]     + A[i + (j + 1) * N] +
                         A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0f / 9.0f);
    }
  }
}

// Verification function
// Fonction de vérification
void verifyResults(const float *cpu_res, const float *gpu_res, const char* kernel_name) {
  for (int j = 1; j < N - 1; j++) {
    for (int i = 1; i < N - 1; i++) {
      int idx = i + j * N;
      float cpu = cpu_res[idx];
      float gpu = gpu_res[idx];
      float diff = std::fabs(cpu - gpu);
      float rel_diff = diff / (std::fabs(cpu) + 1e-5f);
      
      if (rel_diff > 1e-6f) {
        std::printf("Mismatch in %s at (i=%d, j=%d): CPU=%f, GPU=%f, RelDiff=%e\n", 
                    kernel_name, i, j, cpu, gpu, rel_diff);
        return; 
      }
    }
  }
  std::printf("%s verified successfully!\n", kernel_name);
}

// Kernel 1: 1D grid of blocks, 1 thread per block
// Kernel 1: Grille 1D de blocs, 1 thread par bloc
__global__ void kernel1(const float *A, float *Aavg) {
  int idx = blockIdx.x;
  int i = idx % N;
  int j = idx / N;
  
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
    Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
                       A[i + (j - 1) * N]     + A[i + (j) * N]     + A[i + (j + 1) * N] +
                       A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0f / 9.0f);
  }
}

// Kernel 2: 2D grid of blocks, 1 thread per block
// Kernel 2: Grille 2D de blocs, 1 thread par bloc
__global__ void kernel2(const float *A, float *Aavg) {
  int i = blockIdx.x;
  int j = blockIdx.y;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
    Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
                       A[i + (j - 1) * N]     + A[i + (j) * N]     + A[i + (j + 1) * N] +
                       A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0f / 9.0f);
  }
}

// Kernel 3: 2D grid of blocks, 2D threads
// Kernel 3: Grille 2D de blocs, threads 2D
__global__ void kernel3(const float *A, float *Aavg) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
    Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
                       A[i + (j - 1) * N]     + A[i + (j) * N]     + A[i + (j + 1) * N] +
                       A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0f / 9.0f);
  }
}

// Kernel 4: 2D grid, 2D threads, overlapping shared memory
// Kernel 4: Grille 2D, threads 2D, mémoire partagée avec chevauchement
__global__ void kernel4(const float *A, float *Aavg) {
  __shared__ float sA[BSXY][BSXY];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int i = blockIdx.x * (BSXY - 2) + tx;
  int j = blockIdx.y * (BSXY - 2) + ty;

  if (i < N && j < N) {
    sA[tx][ty] = A[i + j * N];
  }
  __syncthreads();

  if (tx >= 1 && tx < BSXY - 1 && ty >= 1 && ty < BSXY - 1 && i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
    Aavg[i + j * N] = (sA[tx - 1][ty - 1] + sA[tx - 1][ty] + sA[tx - 1][ty + 1] +
                       sA[tx][ty - 1]     + sA[tx][ty]     + sA[tx][ty + 1] +
                       sA[tx + 1][ty - 1] + sA[tx + 1][ty] + sA[tx + 1][ty + 1]) * (1.0f / 9.0f);
  }
}

// Kernel 5: Each block loads (K*BSXY) x (K*BSXY) elements and computes the interior
// Kernel 5: Chaque bloc charge (K*BSXY) x (K*BSXY) éléments et calcule l'intérieur
__global__ void kernel5(const float *A, float *Aavg) {
  const int s_dim = K * BSXY;
  __shared__ float sA[K * BSXY][K * BSXY];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int bStartX = blockIdx.x * (s_dim - 2); 
  int bStartY = blockIdx.y * (s_dim - 2);

  for (int j = ty; j < s_dim; j += BSXY) {
    for (int i = tx; i < s_dim; i += BSXY) {
      int gi = bStartX + i;
      int gj = bStartY + j;
      sA[i][j] = (gi < N && gj < N) ? A[gi + gj * N] : 0.0f;
    }
  }
  __syncthreads();

  for (int k_j = 0; k_j < K; k_j++) {
    for (int k_i = 0; k_i < K; k_i++) {
      int si = tx * K + k_i + 1; 
      int sj = ty * K + k_j + 1;
      
      if (si < s_dim - 1 && sj < s_dim - 1) {
        int gi = bStartX + si;
        int gj = bStartY + sj;

        if (gi >= 1 && gi < N - 1 && gj >= 1 && gj < N - 1) {
          Aavg[gi + gj * N] = (sA[si - 1][sj - 1] + sA[si - 1][sj] + sA[si - 1][sj + 1] +
                               sA[si][sj - 1]     + sA[si][sj]     + sA[si][sj + 1] +
                               sA[si + 1][sj - 1] + sA[si + 1][sj] + sA[si + 1][sj + 1]) * (1.0f / 9.0f);
        }
      }
    }
  }
}

int main() {
  size_t bytes = N * N * sizeof(float);
  A = (float *) malloc(bytes);
  float *Aavg_ref = (float *) malloc(bytes);
  float *Aavg_gpu = (float *) malloc(bytes);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
    }
  }

  // Run CPU Reference
  // Exécuter la référence CPU
  ninePointAverageCPU(A, Aavg_ref);

  // Allocate Device Memory and copy 'A' to device
  // Allouer la mémoire sur le Device et y copier 'A'
  float *d_A, *d_Aavg;
  cudaMalloc((void**)&d_A, bytes);
  cudaMalloc((void**)&d_Aavg, bytes);
  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

  // --- Kernel 1 Execution ---
  // --- Exécution du Kernel 1 ---
  cudaMemset(d_Aavg, 0, bytes);
  kernel1<<<N * N, 1>>>(d_A, d_Aavg);
  
  cudaDeviceSynchronize();
  cudaMemcpy(Aavg_gpu, d_Aavg, bytes, cudaMemcpyDeviceToHost);
  verifyResults(Aavg_ref, Aavg_gpu, "Kernel 1");

  // --- Kernel 2 Execution ---
  // --- Exécution du Kernel 2 ---
  cudaMemset(d_Aavg, 0, bytes);
  dim3 grid2(N, N);
  kernel2<<<grid2, 1>>>(d_A, d_Aavg);
  
  cudaDeviceSynchronize();
  cudaMemcpy(Aavg_gpu, d_Aavg, bytes, cudaMemcpyDeviceToHost);
  verifyResults(Aavg_ref, Aavg_gpu, "Kernel 2");

  // --- Kernel 3 Execution ---
  // --- Exécution du Kernel 3 ---
  cudaMemset(d_Aavg, 0, bytes);
  dim3 block3(BSXY, BSXY);
  dim3 grid3((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);
  kernel3<<<grid3, block3>>>(d_A, d_Aavg);
  
  cudaDeviceSynchronize();
  cudaMemcpy(Aavg_gpu, d_Aavg, bytes, cudaMemcpyDeviceToHost);
  verifyResults(Aavg_ref, Aavg_gpu, "Kernel 3");

  // --- Kernel 4 Execution ---
  // --- Exécution du Kernel 4 ---
  cudaMemset(d_Aavg, 0, bytes);
  dim3 block4(BSXY, BSXY);
  dim3 grid4((N + (BSXY - 2) - 1) / (BSXY - 2), (N + (BSXY - 2) - 1) / (BSXY - 2));
  kernel4<<<grid4, block4>>>(d_A, d_Aavg);
  
  cudaDeviceSynchronize();
  cudaMemcpy(Aavg_gpu, d_Aavg, bytes, cudaMemcpyDeviceToHost);
  verifyResults(Aavg_ref, Aavg_gpu, "Kernel 4");

  // --- Kernel 5 Execution ---
  // --- Exécution du Kernel 5 ---
  cudaMemset(d_Aavg, 0, bytes);
  dim3 block5(BSXY, BSXY); 
  int outTileDim = K * BSXY - 2;
  dim3 grid5((N + outTileDim - 1) / outTileDim, (N + outTileDim - 1) / outTileDim);
  kernel5<<<grid5, block5>>>(d_A, d_Aavg);
  
  cudaDeviceSynchronize();
  cudaMemcpy(Aavg_gpu, d_Aavg, bytes, cudaMemcpyDeviceToHost);
  verifyResults(Aavg_ref, Aavg_gpu, "Kernel 5");

  // Cleanup
  // Nettoyage
  cudaFree(d_A);
  cudaFree(d_Aavg);
  free(A);
  free(Aavg_ref);
  free(Aavg_gpu);

  return 0;
}
