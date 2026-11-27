// Does a cudaMemcpyAsync on stream B complete while stream A is blocked in cuStreamWaitValue32 (mode 0) or in a
// polling kernel (mode 1)? Prints ok/deadlock (3 s timeout on the host side).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
__global__ void k_spin_wait(volatile unsigned * flag) { while (*flag != 1) { __nanosleep(200); } *flag = 0; __threadfence_system(); }
__global__ void k_touch(float * p) { p[threadIdx.x] += 1.f; }
int main(int argc, char ** argv) {
    const int mode = argc > 1 ? atoi(argv[1]) : 0;
    cudaStream_t a, b; cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
    unsigned * flag_h; cudaHostAlloc((void **) &flag_h, 4096, cudaHostAllocMapped | cudaHostAllocPortable); *flag_h = 0;
    unsigned * flag_d; cudaHostGetDevicePointer((void **) &flag_d, flag_h, 0);
    float * dev; cudaMalloc(&dev, 64 << 20);
    void * pinned; cudaMallocHost(&pinned, 64 << 20);
    k_touch<<<1, 32, 0, a>>>(dev);
    if (mode == 0) { CUresult r = cuStreamWaitValue32((CUstream) a, (CUdeviceptr) (uintptr_t) flag_d, 1, CU_STREAM_WAIT_VALUE_EQ); if (r) { printf("wait value err %d\n", r); return 1; } }
    else k_spin_wait<<<1, 1, 0, a>>>(flag_d);
    k_touch<<<1, 32, 0, a>>>(dev);
    cudaGetLastError();
    // stream B: a 64 MB H2D copy while A waits
    std::atomic<bool> done{false};
    std::thread t([&] { cudaMemcpyAsync(dev, pinned, 64 << 20, cudaMemcpyHostToDevice, b); cudaError_t e = cudaStreamSynchronize(b); printf("copy on B: %s\n", cudaGetErrorString(e)); done = true; });
    auto t0 = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3)) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool completed = done; // before releasing A: the join below always ends with done == true
    printf("mode %d (%s): copy on another stream %s while A waits\n", mode, mode ? "spin kernel" : "cuStreamWaitValue32", completed ? "COMPLETED" : "DEADLOCKED");
    *flag_h = 1; // release A
    cudaStreamSynchronize(a); t.join();
    printf("A released: %s\n", cudaGetErrorString(cudaGetLastError()));
    return completed ? 0 : 2;
}
