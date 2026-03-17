#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>

using namespace std;


// Calcul de convolution 1D en utilisant 1 thread par bloc
__global__
void conv1DBlocs(const int N, const float *x, float *y)
{
  // A FAIRE ...
}


// Calcul de convolution 1D en utilisant blockSize thread par bloc
__global__
void conv1DBlocsThreads(const int N, const float *x, float *y)
{
  // A FAIRE ...
}


// Calcul de convolution 1D en utilisant blockSize thread par bloc et effectuant k operation par thread dans un bloc
__global__
void conv1DBlocsThreadsKops(const int N, const float *x, float *y, const int k)
{
  // A FAIRE ...
}


// Verifier si le resultat dans res[N] correspond a la convolution 1D de x
void verifyConv1D(float *x, float *res, int N)
{
  int i;
  for (i = 0; i < N; i++) {
    float temp = (i == 0 || i == N - 1) ? x[i] : (x[i - 1] + x[i] + x[i + 1]) / 3.0f;
    if (std::abs(res[i] - temp) > 1e-6) { 
      cout << res[i] << " " << temp << endl;
      break;
    }
  }
  if (i == N) {
    cout << "convolution on GPU is correct." << endl;
  } else {
    cout << "convolution on GPU is incorrect on element " << i << "." << endl;
  }
}


int main(int argc, char **argv)
{
  int blockSize;
  int k;
  float *x, *y, *res, *dx, *dy;

  int N;

  if (argc < 2) {
    printf("Utilisation: ./conv1d N\n");
    return 0;
  }
  N = atoi(argv[1]);

  // Allouer et initialiser les vecteurs x, y et res sur le CPU
  x = (float *) malloc(N * sizeof(float));
  y = (float *) malloc(N * sizeof(float));
  res = (float *) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = (float)i;
    y[i] = 0.0f;
  }

  // Allouer les vecteurs dx[N] et dy[N] sur le GPU, puis copier x et y dans dx et dy.
  // A FAIRE ...


  // Lancer le kernel conv1DBlocs avec un nombre de bloc approprie
  // A FAIRE ...


  // Copier dy[N] dans res[N] pour la verification sur CPU
  // A FAIRE ...


  // Verifier le resultat
  verifyConv1D(x, res, N);

  // Re-initialiser dy[N] en recopiant y[N] la-dedans
  // A FAIRE ...


  // Lancer le kernel conv1DBlocsThreads avec un certain blockSize et nombre de bloc
  // A FAIRE ...
  // blockSize = 32, 64, 128, 256, 512, 1024
  blockSize = 1024;


  // Copier dy[N] dans res[N] pour la verification sur CPU
  // A FAIRE ...


  // Verifier le resultat
  verifyConv1D(x, res, N);

  // Re-initialiser dy[N] en recopiant y[N] la-dedans
  // A FAIRE ...


  // Lancer le kernel conv1DBlocsThreadsKops avec un certain blockSize, nombre de bloc, et nombre d'operations par thread (variable k)
  // A FAIRE ...
  // blockSize = 32, 64, 128, 256, 512, 1024
  // k = 1, 2, 4, 8, 16, ...
  blockSize = 1024;
  k = 8; 


  // Copier dy[N] dans res[N] pour la verification sur CPU
  // A FAIRE ...


  // Verifier le resultat
  verifyConv1D(x, res, N);

  // Desallouer les tableau GPU
  // A FAIRE ...


  // Desallouer les tableaux CPU
  free(x);
  free(y);
  free(res);

  return 0;
}
