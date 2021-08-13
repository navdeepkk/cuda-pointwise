// Usage: nvcc -lcublas -DPRINT_RES -DPRINT_PERF gemm.cu
#include "cublas_v2.h"
#include "library_types.h"
#include <stdio.h>
#include "iostream"

#define NUM_THREADS_PER_BLOCK 1024
#define vec 4

using namespace std;

__global__ void init_a_b(half *a, half *b, int M, int N, int K) {
  int c16 = 5;
  for(int i = 0; i < M; i++){
    for(int j = 0; j < K; j++){
      int im = i % c16;
      int jm = j % c16;
      int add = im + jm;
      int am = add % c16;
      float resf = (float) am;
      half sum = __float2half_rd(resf);
      a[i * K + j] = sum;
    }
  }

  for(int i = 0; i < K; i++){
    for (int j = 0; j < N; j++){
      int im = i % c16;
      int jm = j % c16;
      int add = im + jm;
      int am = add % c16;
      float resf = (float) am;
      half sum = __float2half_rd(resf);
      b[i * N + j] = sum;
    }
  }
}

__global__ void init_c(float *c_float, int M, int N) {
  for (int t = 0; t < M * N; t++) {
    c_float[t] =  0.0f;
  }
}

__global__ void init_c_cst(float *c_float, int M, int N) {
  int c16 = 5;
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      int im = i % c16;
      int jm = j % c16;
      int add = im + jm;
      int am = add % c16;
      float resf = (float) am;
      c_float[i * N + j] = resf;
    }
  }
}

void print_res_host(float * arr, int m, int n) {
  std::cout << "[";
  for(int i = 0; i < m; i++){
    if(i == 0)
      std::cout << "[";
    else
      std::cout << " [";
    for (int j = 0; j < n; j++){
      if(j == 0)
        std::cout << (arr[i * n + j]);
      else
        std::cout<<",   "<< (arr[i * n + j]);
    }
    if(i == m - 1)
      std::cout << "]";
    else
      std::cout << "], "<<std::endl;
  }
  std::cout << "]";
}

__global__ void matAdd(float *c, float* c_cst, int m, int n){
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n; i += (((m * n) + (NUM_THREADS_PER_BLOCK * vec) - 1) / (NUM_THREADS_PER_BLOCK * vec)) * blockDim.x * vec) {
    // Calculate this block's starting address.
    float *base = c + (i * vec);
    float4 *cGmem = (float4*)base;
    float4 cData = *(cGmem);

    float *cst_base = c_cst + (i * vec);
    float4 *cst_cGmem = (float4*)cst_base;
    float4 cst_cData = *(cst_cGmem);

    cData.w = cData.w + cst_cData.w;
    cData.x = cData.x + cst_cData.x;
    cData.y = cData.y + cst_cData.y;
    cData.z = cData.z + cst_cData.z;

    *(cGmem) = cData;
  }
}

__global__ void pwRelu(float *c, int m, int n){
  float cutoff = 0;
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n; i += (((m * n) + (NUM_THREADS_PER_BLOCK * vec) - 1) / (NUM_THREADS_PER_BLOCK * vec)) * blockDim.x * vec) {
    // Calculate this block's starting address.
    float *base = c + (i * vec);
    float4 *cGmem = (float4*)base;
    float4 cData = *(cGmem);

    if(cData.w < cutoff)
      cData.w *= 0;
    if(cData.x < cutoff)
      cData.x *= 0;
    if(cData.y < cutoff)
      cData.y *= 0;
    if(cData.z < cutoff)
      cData.z *= 0;

    *(cGmem) = cData;
  }
}

int main(int argc, char **argv)
{
  if(argc != 5){
    printf("Specify problem sizes as ./gemm m n k num_iters\n");
    return 0;
  }

  int M = std::atoi(argv[1]);
  int N = std::atoi(argv[2]);
  int K = std::atoi(argv[3]);
  int num_iters = std::atoi(argv[4]);
  cublasHandle_t handle;
  cublasCreate(&handle);

  half *A, *B;
  float *C, *C_cst;

  cudaMalloc(&A, M * K * sizeof(half));
  cudaMalloc(&B, K * N * sizeof(half));
  cudaMalloc(&C, M * N * sizeof(float));
  cudaMalloc(&C_cst, M * N * sizeof(float));

  float alpha = 1.0;
  float beta = 1.0;

  init_a_b<<<1, 1>>>(A, B, M, N, K);
  init_c<<<1, 1>>>(C, M, N);
  init_c_cst<<<1, 1>>>(C_cst, M, N);


  // Warmup iterations.
  for(int i = 0; i < 5; ++i){
    if (CUBLAS_STATUS_SUCCESS != cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)) {
      printf("cublasGemmEx failed\n");
      exit(-1);
    }
  }

  // Profiling iterations.
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
  float aggregateTime = 0.0f;
  for(int i = 0; i < num_iters; ++i){
    float ms = 0.0f;
    init_c<<<1, 1>>>(C, M, N);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    if (CUBLAS_STATUS_SUCCESS != cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)) {
      printf("cublasGemmEx failed\n");
      exit(-1);
    }
    dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 grid(((M * N) + (NUM_THREADS_PER_BLOCK * vec) - 1) / (NUM_THREADS_PER_BLOCK * vec), 1, 1);
    matAdd<<<grid, block>>>(C, C_cst, M, N);
    pwRelu<<<grid, block>>>(C, M, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    aggregateTime += ms;
  }

#ifdef PRINT_PERF
  float avg_time = ((aggregateTime / num_iters) / 1000.0f);
  float ops = (float)M * (float)N * (float)K * 2.0f;
  float tflops = (ops * 1.0e-12f) / (avg_time);
  fprintf(stderr, "m:%d, n:%d, k:%d, ", M, N, K);
  fprintf(stderr, "%f TFLOPS\n", tflops);
#endif

#ifdef PRINT_RES
  float * C_host;
  C_host = (float*)malloc(M * N * sizeof(float));
  cudaDeviceSynchronize();
  cudaMemcpy(C_host, C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  print_res_host(C_host, M, N);
  free(C_host);
#endif

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}
