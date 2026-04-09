#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>

#define BLOCKSIZE 1024

/**
  * Step 1: Write a 1D (blocks and threads) GPU kernel that finds the minimum element in an array dA[N] in each block, then writes the minimum in dAmin[blockIdx.x]. CPU should take this array and find the global minimum by iterating over this array.
  * Etape 1: Ecrire un kernel GPU 1D (blocs et threads) qui trouve l'element minimum d'un tableau dA[N] pour chaque bloc et ecrit le minimum de chaque bloc dans dAmin[blockIdx.x]. En suite, CPU reprend dAmin et calcul le minimum global en sequentiel sur ce petit tableau.
  *
  * Step 2: The first call to findMinimum reduces the size of the array to N/BLOCKSIZE. In this version, use findMinimum a second time on this resulting array, in order to reduce the size to N/(BLOCKSIZE*BLOCKSIZE) so that computation on the CPU to find the global minimum becomes negligible.
  * Etape 2: Le premier appel au findMinimum reduit la taille du tableau a parcourir en sequentiel a N/BLOCKSIZE. Dans cette version, utiliser findMinimum une deuxieme fois afin de reduire la taille du tableau a  N/(BLOCKSIZE*BLOCKSIZE) pour que le calcul cote CPU pour trouver le minimum global devienne negligeable.
  *
  * To find the minimum of two floats on a GPU, use the function fminf(x, y).
  * Pour trouver le minimum des deux flottants en GPU, utiliser la fonction fminf(x, y).
  */

__global__ void findMinimum(float *dA, float *dAmin, int N)
{
  __shared__ volatile float buff[BLOCKSIZE];
  int idx = threadIdx.x + blockIdx.x * BLOCKSIZE;
  // TODO / A FAIRE ...
}

using namespace std;

int main()
{
  srand(1234);
  int N = 100000000;
  int numBlocks;// = ???; (TODO / A FAIRE ...)
  float *A, *dA; // Le tableau dont minimum on va chercher
  float *Amin, *dAmin; // Amin contiendra en suite le tableau reduit par un facteur de BLOCKSIZE apres l'execution du kernel GPU

  // Allocate arrays A[N] and Amin[numBlocks} on the CPU
  // Allocate arrays dA[N] and dAmin[numBlocks} on the GPU
  // Allour les tableaux A[N] et Amin[numBlocks] sur le CPU
  // Allouer les tableaux dA[N] et dAmin[numBlocks] sur le GPU
  // TODO / A FAIRE ...

  // Initialize the array A, set the minimum to -1
  // Initialiser le tableau A, mettre le minimum a -1.
  for (int i = 0; i < N; i++) { A[i] = (float)(rand() % 1000); }
  A[rand() % N] = -1.0; 

  // Transfer A on the GPU (dA) with cudaMemcpy
  // Transferer A sur le GPU (dA) avec cudaMemcpy
  // TODO / A FAIRE ...

  // Put maximum attainable value to minA.
  // Affecter la valeur maximum atteignable dans minA
  float minA = FLT_MAX; 

  // Find the minimum of the array dA for each thread block, put it in dAMin[...] and transfer to the CPU, then find the global minimum of this smaller array and put it in minA.
  // Trouver le minimum du tableau dA pour chaque bloc de threads, mettre dans dAmin[...] puis transferer vers le CPU, puis trouver le minimum global de ce petit tableau et l'affecter dans la variable minA.
  // TODO / A FAIRE ...
  // findMinimum<<<...>>>(...)
  // ...

  // Verify the result
  // Verifier le resultat
  if (minA == -1) { cout << "The minimum is correct!" << endl; }
  else { cout << "The minimum found (" << minA << ") is incorrect (it should have been -1)!" << endl; }

  return 0;
}
