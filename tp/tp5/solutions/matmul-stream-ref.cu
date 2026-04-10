#include <cstdio>
#include <cuda.h>
#include <math.h>

int N = 1024;
const int nStreams = 4;
float *A, *B, *C;
float *dA, *dB, *dC;
cudaStream_t streams[nStreams];

void verify_result(float *A, float *B, float *C, int N) {
  printf("Verifying on CPU...\n");
  bool errorFound = false;
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      float expected = 0.0f;
      for (int k = 0; k < N; k++) {
        expected += A[i * N + k] * B[k + j * N]; 
      }
      
      float actual = C[i + j * N];
      float diff = fabs(actual - expected);
      float relError = (expected != 0.0f) ? (diff / fabs(expected)) : diff;
      
      if (relError > 1e-6f) {
        printf("Mismatch at indices (%d, %d)! Expected: %f, Got: %f\n", i, j, expected, actual);
        errorFound = true;
        break; 
      }
    }
    if (errorFound) {
      break; 
    }
  }
  
  if (!errorFound) {
    printf("Verification passed successfully.\n");
  }
}

__global__ void matvec(float *A, float *x, float *b, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
      sum += A[i * n + j] * x[j];
    }
    b[i] = sum;
  }
}

int main()
{
  // Allocate pinned host memory
  cudaMallocHost((void**)&A, N * N * sizeof(float));
  cudaMallocHost((void**)&B, N * N * sizeof(float));
  cudaMallocHost((void**)&C, N * N * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j; 
      B[i + j * N] = i - j; 
      C[i + j * N] = 0; 
    }
  }
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dB, N * nStreams * sizeof(float));
  cudaMalloc(&dC, N * nStreams * sizeof(float));

  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  for (int j = 0; j < N; j++) {
    int s = j % nStreams;
    int colOffset = j * N;
    int streamOffset = s * N;

    cudaMemcpyAsync(&dB[streamOffset], &B[colOffset], N * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
    matvec<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(dA, &dB[streamOffset], &dC[streamOffset], N);
    cudaMemcpyAsync(&C[colOffset], &dC[streamOffset], N * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
  }
  
  cudaDeviceSynchronize();

  verify_result(A, B, C, N);

  for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }

  // Free pinned memory
  cudaFreeHost(A); 
  cudaFreeHost(B); 
  cudaFreeHost(C);
  cudaFree(dA); 
  cudaFree(dB); 
  cudaFree(dC);
  
  return 0;
}
