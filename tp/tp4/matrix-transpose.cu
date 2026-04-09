/**
  * Kernel 1: Implement a first kernel where each block (using BSXY x BSXY threads) transposes a BSXY x BSXY tile of A, and writes it into the corresponding location in At. Do without using shared memory.
  *
  * Kernel 2: In the second kernel, do the same, but using the shared memory. Each block should load a tile of BSXY x BSXY of A into the shared memory, then perform the transposition using this tile in the shared memory into At. Test the difference in speedup. Test the performance using shared memory without padding and with padding (to avoid shared memory bank conflicts).
  *
  * Kernel 3: In this kernel, perform the transpose in-place on the matrix A (do not use At). A block should be transpose two tiles simultenously to be able to do this.
  *
  */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>

#define BSXY 32

void transposeCPU(float *A, float *At, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      At[i * N + j] = A[j * N + i];
    }
  }
}

int main()
{
  // Allocate A and At
  // A is an N * N matrix stored by rows, i.e. A(i, j) = A[i * N + j]
  // At is also stored by rows and is the transpose of A, i.e., At(i, j) = A(j, i)
  int N = 1024;
  float *A = (float *) malloc(N * N * sizeof(A[0]));
  float *At = (float *) malloc(N * N * sizeof(At[0]));
  
  // Allocate dA and dAt, and call the corresponding matrix transpose kernel
  // TODO / A FAIRE ...


  // Deallocate dA and dAt
  // TODO / A FAIRE ...

  // Deallocate A and At
  free(A);
  free(At);
  return 0;
}
