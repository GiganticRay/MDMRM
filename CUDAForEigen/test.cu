#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

__global__ void helloGPU()
{
  printf("Hello from the GPU.\n");
}

int main()
{
  helloCPU();
  helloGPU<<<1, 1>>>();
  cudaDeviceSynchronize();
}
