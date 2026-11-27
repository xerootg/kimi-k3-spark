// Channel sharing: N streams exist (like ggml's 8 per device); stream 0 blocks in cuStreamWaitValue32 and a host
// thread also keeps launching kernels on stream 0 until the launch queue is full (like the eager prefill); then a
// copy on stream `cs` (created last, optionally high priority) + sync: completes or deadlocks (4 s timeout).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
__global__ void k_touch(float * p) { p[threadIdx.x] += 1.f; }
int main(int argc, char ** argv) {
    const int nstreams = argc > 1 ? atoi(argv[1]) : 8, prio = argc > 2 ? atoi(argv[2]) : 0, nlaunch = argc > 3 ? atoi(argv[3]) : 0;
    cudaStream_t s[64]; for (int i = 0; i < nstreams; i++) cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
    int lo, hi; cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStream_t cs; if (prio) cudaStreamCreateWithPriority(&cs, cudaStreamNonBlocking, hi); else cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
    unsigned * flag_h; cudaHostAlloc((void **) &flag_h, 4096, cudaHostAllocMapped | cudaHostAllocPortable); *flag_h = 0;
    unsigned * flag_d; cudaHostGetDevicePointer((void **) &flag_d, flag_h, 0);
    float * dev; cudaMalloc(&dev, 64 << 20); void * pinned; cudaMallocHost(&pinned, 64 << 20);
    k_touch<<<1, 32, 0, s[0]>>>(dev);
    cuStreamWaitValue32((CUstream) s[0], (CUdeviceptr) (uintptr_t) flag_d, 1, CU_STREAM_WAIT_VALUE_EQ);
    std::atomic<bool> done{false}, launched{false};
    std::thread launcher([&] { for (int i = 0; i < nlaunch; i++) { k_touch<<<1, 32, 0, s[0]>>>(dev); if (i == 2000) launched = true; } launched = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::thread t([&] { cudaMemcpyAsync(dev, pinned, 64 << 20, cudaMemcpyHostToDevice, cs); cudaError_t e = cudaStreamSynchronize(cs); if (e) printf("copy err %s\n", cudaGetErrorString(e)); done = true; });
    auto t0 = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(4)) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool completed = done; // before releasing the flag: the join below always ends with done == true
    printf("streams=%d prio=%d launches=%d: copy %s (launcher %s)\n", nstreams, prio, nlaunch, completed ? "COMPLETED" : "DEADLOCKED", launched ? "done/far" : "blocked");
    *flag_h = 1; cudaDeviceSynchronize(); t.join(); launcher.join();
    return completed ? 0 : 2;
}
