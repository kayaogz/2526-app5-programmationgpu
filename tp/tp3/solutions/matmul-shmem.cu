#include <cstdio>  
#include <iostream>
#include "cuda.h"  

#define N 513
#define BSXY 32
float A[N][N], B[N][N], C[N][N];

__device__ float dA[N][N], dB[N][N], dC[N][N];


// Creer un bloc pour le calcul de chaque element C[i][j], calculer avec 1 thread par bloc
__global__ void multiplyMatrixGPUByBlocks(int n)
{
  // A FAIRE ...
  int i = blockIdx.x;
  int j = blockIdx.y;
  float c = 0.0;
  for (int k = 0; k < n; k++) { c += dA[i][k] * dB[k][j]; }
  dC[i][j] = c;
}


// Creer un bloc pour le calcul de blockDim.x elements de C, calculer avec blockDim.x threads par bloc.
// Chaque thread calcule un element de C.
__global__ void multiplyMatrixGPUByBlocksThreads1D(int n)
{
  // A FAIRE ...
  int i = blockIdx.x;
  int j = threadIdx.x + blockIdx.y * blockDim.y;
  float c = 0.0;
  for (int k = 0; k < n; k++) { c += dA[i][k] * dB[k][j]; }
  dC[i][j] = c;
}


// Creer un bloc pour le calcul de blockDim.x elements de C, calculer avec blockDim.x threads par bloc.
// Chaque thread calcule un element de C.
// Faire marcher pour N n'est pas multiple de blockDim.x.
__global__ void multiplyMatrixGPUByBlocksThreads1DNonMultiple(int n)
{
  // A FAIRE ...
  int i = blockIdx.x;
  int j = threadIdx.x + blockIdx.y * blockDim.y;
  if (j < n) { 
    float c = 0.0;
    for (int k = 0; k < n; k++) { c += dA[i][k] * dB[k][j]; }
    dC[i][j] = c;
  }
}


// Creer un bloc pour le calcul de blockDim.x * blockDim.y elements de C, calculer avec blockDim.x * blockDim.y threads par bloc.
// Chaque thread calcule un element de C.
__global__ void multiplyMatrixGPUByBlocksThreads2D(int n)
{
  // A FAIRE ...
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  int j = threadIdx.x + blockIdx.y * blockDim.x;
  float c = 0.0;
  for (int k = 0; k < n; k++) { c += dA[i][k] * dB[k][j]; }
  dC[i][j] = c;
}


// Creer un bloc pour le calcul de blockDim.x * blockDim.y elements de C, calculer avec blockDim.x * blockDim.y threads par bloc.
// Chaque thread calcule un element de C.
// Faire marcher pour N n'est pas multiple de ni blockDim.x ni blockDim.y.
__global__ void multiplyMatrixGPUByBlocksThreads2DNonMultiple(int n)
{
  // A FAIRE ...
  int i = threadIdx.y + blockIdx.x * blockDim.y;
  int j = threadIdx.x + blockIdx.y * blockDim.x;
  if (i < n && j < n) {
    float c = 0.0;
    for (int k = 0; k < n; k++) { c += dA[i][k] * dB[k][j]; }
    dC[i][j] = c;
  }
}


// Utiliser BSXY == blockDim.x == blockDim.y (blocs carres)
// Creer un bloc pour le calcul de blockDim.x * blockDim.x elements de C, calculer avec blockDim.x * blockDim.x threads par bloc.
// Chaque thread calcule un element de C.
// Faire marcher pour N n'est pas multiple de ni blockDim.x;
// Operer par des blocs de matrices de taille blockDim.x * blockDim.x en utilisant la shared memory.
//   A savoir, a chaque etape, recuperer un bloc de taille blockDim.x * blockDim.x de A et B, multiplier-les, puis passer aux blocs suivants
__global__ void multiplyMatrixGPUByBlocksThreads2DNonMultipleSharedMemory(int n)
{
  // A FAIRE ...
  int row = threadIdx.y + blockIdx.x * BSXY;
  int col = threadIdx.x + blockIdx.y * BSXY;
  __shared__ float shA[BSXY][BSXY];
  __shared__ float shB[BSXY][BSXY];
  float c = 0.0;
  const int nsteps = (n - 1) / BSXY + 1;
  for (int step = 0; step < nsteps; step++) {
    int offset = step * BSXY;
    int stepRowB = offset + threadIdx.y;
    int stepColA = offset + threadIdx.x;
    if (row < n && stepColA < n) {
      shA[threadIdx.y][threadIdx.x] = dA[row][stepColA];
    } else {
      shA[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (stepRowB < n && col < n) {
      shB[threadIdx.y][threadIdx.x] = dB[stepRowB][col];
    } else {
      shB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (int k = 0; k < BSXY; k++) {
      c += shA[threadIdx.y][k] * shB[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < n && col < n) { dC[row][col] = c; }
}


// Code reference de CPU pour effectuer la multiplication de matrices C = AB
void multiplyMatrixCPU()
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0.0f;
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void verifyResults()
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float c = 0.0f;
      for (int k = 0; k < N; k++) {
        c += A[i][k] * B[k][j];
      }
      if (std::abs(C[i][j] - c) > 1e-6) {
        std::cout << "Multiplication is incorrect for the element C[" << i << "][" << j << "]" << std::endl;
        return;
      }
    }
  }
  std::cout << "Multiplication is correct!" << std::endl;
}

int main(int argc, char **argv)
{
  // Initialisation
  for (int i = 0; i < N; i++) { 
    for (int j = 0; j < N; j++) { 
      A[i][j] = i + j;
      B[i][j] = i - j;
    }
  }

  // Copier les tableaux A et B vers le GPU
  // A FAIRE ...
  cudaMemcpyToSymbol(dA, A, N * N * sizeof(float), 0,
      cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dB, B, N * N * sizeof(float), 0,
      cudaMemcpyHostToDevice);

  // Appeler chaque kernel GPU de maniere appropriee pour multiplier les matrices A et B
  // A FAIRE ...
  dim3 dimGrid;
  dimGrid.x = (N - 1) / BSXY + 1;
  dimGrid.y = (N - 1) / BSXY + 1;
  dimGrid.z = 1;
  dim3 dimBlock;
  dimBlock.x = BSXY;
  dimBlock.y = BSXY;
  dimBlock.z = 1;
  // multiplyMatrixGPUByBlocks<<<dimGrid, dimBlock>>>(N);
  // multiplyMatrixGPUByBlocksThreads1D<<<dimGrid, dimBlock>>>(N);
  // multiplyMatrixGPUByBlocksThreads1DNonMultiple<<<dimGrid, dimBlock>>>(N);
  // multiplyMatrixGPUByBlocksThreads2D<<<dimGrid, dimBlock>>>(N);
  // multiplyMatrixGPUByBlocksThreads2DNonMultiple<<<dimGrid, dimBlock>>>(N);
  multiplyMatrixGPUByBlocksThreads2DNonMultipleSharedMemory<<<dimGrid, dimBlock>>>(N);

  // Recopier le tableau dC vers le CPU
  // A FAIRE ...
  cudaMemcpyFromSymbol(C, dC, N * N * sizeof(float), 0,
      cudaMemcpyDeviceToHost);

  // Verifier les resultats
  // multiplyMatrixCPU();
  verifyResults();
  return 0;
}
