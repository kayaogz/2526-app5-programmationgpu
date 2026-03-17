#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

__global__ void cudaCopyByBlocks(float *tab0, const float *tab1, int size)
{
  int idx;
  // Compute the correct idx
  // Calculer le bon idx
  // TODO / A FAIRE ...
  // idx = ?
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

__global__ void cudaCopyByBlocksThreads(float *tab0, const float *tab1, int size)
{
  int idx;
  // Compute the correct idx in terms of blockIdx.x, threadIdx.x, and blockDim.x
  // Calculer le bon idx en fonction du blockIdx.x, threadIdx.x, et blockDim.x
  // TODO / A FAIRE ...
  // idx = ?
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

int main(int argc, char **argv) {
  float *A, *B, *dA, *dB;
  int N, i;

  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 0;
  }
  N = atoi(argv[1]);

  // Initialization
  // Initialisation
  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) { 
    A[i] = (float)i;
    B[i] = 0.0f;
  }
  
  // Allocate dynamic arrays dA and dB of size N on the GPU with cudaMalloc
  // Allouer les tableau dA et dB dynamiques de size N sur le GPU avec cudaMalloc 
  // TODO / A FAIRE ...

  // Copy A into dA and B into dB
  // Copier A dans dA et B dans dB
  // TODO / A FAIRE ...

  // Copy dA into dB using the kernel cudaCopyByBlocks
  // Copier dA dans dB avec le kernel cudaCopyByBlocks
  // TODO / A FAIRE ...
  // cudaCopyByBlocks<<<...,...>>>(...) ???

  // Wait for kernel cudaCopyByBlocks to finish
  // Attendre que le kernel cudaCopyByBlocks termine
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // Copier dB dans B pour la verification
  // TODO / A FAIRE ...

  // Verify the results on the CPU by comparing B with A
  // Verifier le resultat en CPU en comparant B avec A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Reinitialize B to zero, then copy B into dB again to test the second copy kernel
  // Remettre B a zero puis recopier dans dB tester le deuxieme kernel de copie
  for (int i = 0; i < N; i++) { B[i] = 0.0f; }
  // TODO / A FAIRE ...

  // Copy dA into dB with the kernel cudaCopyByBlocksThreads
  // Copier dA dans dB avec le kernel cudaCopyByBlocksThreads
  // TODO / A FAIRE ...
  // cudaCopyByBlocksThreads<<<...,...>>>(...) ???

  // Wait for the kernel cudaCopyByBlocksThreads to finish
  // Attendre que le kernel cudaCopyByBlocksThreads termine
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // Copier dB dans B pour la verification
  // TODO / A FAIRE ...

  // Verify the results on the CPU by comparing B with A
  // Verifier le resultat en CPU en comparant B avec A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Deallocate arrays dA[N] and dB[N] on the GPU
  // Desaollouer le tableau dA[N] et dB[N] sur le GPU
  // TODO / A FAIRE ...

  // Deallocate A and B
  // Desallouer A et B
  free(A);
  free(B);

  return 0;
}
