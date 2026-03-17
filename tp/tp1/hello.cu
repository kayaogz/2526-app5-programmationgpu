#include <cstdio>
#include "cuda.h"

__global__ void cudaHello(){
  // Afficher le message Hello World ainsi que blockidx et threadidx depuis chaque thread
  // A FAIRE ...
}

int main() {
  int numBlocks = 64;
  int blockSize = 1;
  // Experimenter avec de differents blockSize (nombre de threads par block) pour les puissances de 2
  // tout en gardant le nombre total de threads egale a 64
  // A FAIRE ...
  cudaHello<<<numBlocks, blockSize>>>(); 

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  return 0;
  }
