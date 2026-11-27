#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main(int argc, char ** argv) { size_t gb = argc > 1 ? atoll(argv[1]) : 88; void * p; if (cudaMalloc(&p, gb << 30) != cudaSuccess) { printf("alloc failed\n"); return 1; } cudaMemset(p, 1, gb << 30); cudaDeviceSynchronize(); printf("ballast %zu GB held\n", gb); fflush(stdout); sleep(argc > 2 ? atoi(argv[2]) : 150); return 0; }
