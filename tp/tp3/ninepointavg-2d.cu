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
#include "cuda.h"
#include "omp.h"

#define N 1024
#define K 4
#define BSXY 32

// The matrix is stored by columns, that is A(i, j) = A[i + j * N]. The average should be computed on Aavg array.
// La matrice A est stockee par colonnes, a savoir A(i, j) = A[i + j * N]
float *A;
float *Aavg;

// Reference CPU implementation
// Code de reference pour le CPU
void ninePointAverageCPU(const float *A, float *Aavg)
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0 / 9.0);
    }
  }
}


int main()
{
  A = (float *) malloc (N * N * sizeof(float));
  Aavg = (float *) malloc (N * N * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
    }
  }

  free(A);
  free(Am);

  return 0;

