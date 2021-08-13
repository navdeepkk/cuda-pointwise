//  nvcc -O3 -std=c++11 -use_fast_math -ccbin g++ -arch=compute_75 -code=sm_75 -expt-relaxed-constexpr

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>
#include "common.h"

#define M 1024 
#define N 1024 
#define NUM_THREADS_PER_BLOCK 1024
#define vec 4
using namespace std;

__host__ void init_host_matrices(float *c){
  for (int t = 0; t < M * N; t++) {
    c[t] = (float) 0.0f;
  }
}

__host__ void printMatrixFloat(float* matrix, int m, int n){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      printf("%f ", (float)matrix[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void pwAdd(float *c, int m, int n){
  float cst = 5;

  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n; i += (((m * n) + (NUM_THREADS_PER_BLOCK * vec) - 1) / (NUM_THREADS_PER_BLOCK * vec)) * blockDim.x * vec) {
    // Calculate this block's starting address.
    float *base = c + (i * vec);
    float4 *cGmem = (float4*)base;
    float4 cData = *(cGmem);

    cData.w = cData.w + cst;
    cData.x = cData.x + cst;
    cData.y = cData.y + cst;
    cData.z = cData.z + cst;

    *(cGmem) = cData;
    //printf("%f\n",(float)cData.w);
  }
}

int main() {
  float *d_c, *h_c, *h_c_gpu_res;
  int m, n;

  m = M;
  n = N;

  h_c = (float*) malloc(m * n * sizeof(float));
  h_c_gpu_res = (float*) malloc(m * n * sizeof(float));
  check_cuda_error(cudaMalloc(&d_c, m * n * sizeof(float)));

  assert(((unsigned long long)d_c) % 128 == 0);

  init_host_matrices(h_c);
  check_cuda_error(cudaMemcpy(d_c, h_c, m * n * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
  dim3 grid(((m * n) + (NUM_THREADS_PER_BLOCK * vec) - 1) / (NUM_THREADS_PER_BLOCK * vec), 1, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  pwAdd<<<grid, block>>>(d_c, m , n);
  cudaEventRecord(stop, NULL);

  cudaEventSynchronize(stop);
  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);
  check_cuda_error(cudaPeekAtLastError());
  //cout<<"time: "<<msecTotal<<"ms \n";

#ifdef PRINT_HOST
  check_cuda_error(cudaDeviceSynchronize());
  cudaMemcpy(h_c_gpu_res, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  check_cuda_error(cudaDeviceSynchronize());
  printMatrixFloat(h_c_gpu_res, m, n);
#endif

  free(h_c);
  free(h_c_gpu_res);
  cudaFree(d_c);

  return 0;
}
