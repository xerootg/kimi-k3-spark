// Read-bandwidth roofline for the GB10: sum a big buffer with 128-bit loads, several grid shapes.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const int4 *p, size_t n, int *out) {
  int4 acc = {0,0,0,0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = __ldcs(p + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) *out = 1;
}
int main() {
  size_t bytes = 2ull << 30, n = bytes / 16; int4 *p; int *o;
  cudaMalloc(&p, bytes); cudaMalloc(&o, 4); cudaMemset(p, 1, bytes);
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int mult : {2, 4, 8, 16, 32}) for (int bs : {256, 512, 1024}) {
    rd<<<sms * mult, bs>>>(p, n, o); cudaEventRecord(a);
    for (int r = 0; r < 10; r++) rd<<<sms * mult, bs>>>(p, n, o);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    printf("grid %d x %d: %.1f GB/s\n", sms * mult, bs, 10.0 * bytes / (ms * 1e6));
  }
  printf("SMs %d\n", sms);
}
