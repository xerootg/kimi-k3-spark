#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <sys/mman.h>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void * pop_alloc(size_t n) {
    void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, n, MADV_POPULATE_WRITE);
    memset(p, 1, n);
    return p;
}

// read kernel: each thread loads 16B, unroll U independent loads per iteration, sums to defeat DCE
template <int U>
__global__ void rd(const uint4 * __restrict__ p, size_t n16, unsigned * out) {
    size_t tid = blockIdx.x*(size_t) blockDim.x + threadIdx.x, stride = (size_t) gridDim.x*blockDim.x;
    unsigned acc = 0;
    for (size_t i = tid; i < n16; i += stride*U) {
        uint4 v[U];
#pragma unroll
        for (int k = 0; k < U; k++) { size_t j = i + k*stride; v[k] = j < n16 ? __ldcs(p + j) : make_uint4(0,0,0,0); }
#pragma unroll
        for (int k = 0; k < U; k++) acc += v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    }
    if (acc == 0x12345678) out[0] = acc;
}

template <int U>
static double kread(const void * p, size_t n, int blocks, int threads, unsigned * d_out) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    rd<U><<<blocks, threads>>>((const uint4 *) p, n/16, d_out); CK(cudaDeviceSynchronize());
    int reps = 5;
    cudaEventRecord(a);
    for (int r = 0; r < reps; r++) rd<U><<<blocks, threads>>>((const uint4 *) p, n/16, d_out);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return (double) n*reps/(ms/1000)/1e9;
}

static double copy_bw(const char * src, size_t total, size_t chunk, int ns, char * dst, size_t dstsz) {
    std::vector<cudaStream_t> s(ns);
    for (auto & x : s) CK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    CK(cudaDeviceSynchronize());
    double best = 0;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        size_t nch = total/chunk;
        for (size_t i = 0; i < nch; i++) {
            CK(cudaMemcpyAsync(dst + (i*chunk) % dstsz, src + i*chunk, chunk, cudaMemcpyHostToDevice, s[i % ns]));
        }
        CK(cudaDeviceSynchronize());
        double bw = (double) total/(now() - t0)/1e9;
        if (bw > best) best = bw;
    }
    for (auto & x : s) cudaStreamDestroy(x);
    return best;
}

int main() {
    size_t N = 8ull << 30;
    char * reg = (char *) pop_alloc(N);
    CK(cudaHostRegister(reg, N, cudaHostRegisterPortable));
    char * plain = (char *) pop_alloc(N);
    char * pinned; CK(cudaMallocHost((void **) &pinned, N)); memset(pinned, 1, N);
    char * hbm; CK(cudaMalloc((void **) &hbm, N));
    char * dst; CK(cudaMalloc((void **) &dst, 2ull << 30));
    unsigned * d_out; CK(cudaMalloc((void **) &d_out, 64));
    CK(cudaMemset(hbm, 1, N));
    struct { const char * n; char * p; } bufs[3] = { {"registered-malloc", reg}, {"plain-malloc", plain}, {"cudaMallocHost", pinned} };

    printf("== a: cudaMemcpyAsync H2D GB/s (best of 3, 4 GiB moved)\n");
    printf("%-18s %6s %8s %8s\n", "source", "chunk", "1 strm", "4 strm");
    for (auto & b : bufs) for (size_t c : {2ul, 4ul, 8ul, 16ul}) {
        size_t chunk = c << 20;
        printf("%-18s %4zuMB %8.1f %8.1f\n", b.n, c, copy_bw(b.p, 4ull << 30, chunk, 1, dst, 2ull << 30), copy_bw(b.p, 4ull << 30, chunk, 4, dst, 2ull << 30));
        fflush(stdout);
    }
    printf("== b: read kernel GB/s (16B loads, 4 GiB, 132 SMs)\n");
    printf("%-18s %-22s %8s\n", "source", "config", "GB/s");
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    struct { const char * n; const void * p; } rb[4] = { {"registered-malloc", reg}, {"plain-malloc", plain}, {"cudaMallocHost", pinned}, {"HBM cudaMalloc", hbm} };
    size_t n = 4ull << 30;
    for (auto & b : rb) {
        for (int bps : {2, 4, 8}) {
            int blocks = sms*bps;
            printf("%-18s blk=%dx%d thr=1024 U=1 %8.1f\n", b.n, sms, bps, kread<1>(b.p, n, blocks, 1024, d_out));
            printf("%-18s blk=%dx%d thr=1024 U=4 %8.1f\n", b.n, sms, bps, kread<4>(b.p, n, blocks, 1024, d_out));
            fflush(stdout);
        }
        printf("%-18s blk=%dx8 thr=256  U=8 %8.1f\n", b.n, sms, kread<8>(b.p, n, sms*8, 256, d_out));
        fflush(stdout);
    }
    // small-grid case like MMQ tile reads: fewer blocks
    printf("== b2: registered, 16 blocks/ 64 blocks (low occupancy)\n");
    printf("blocks=16 %8.1f\nblocks=64 %8.1f\n", kread<4>(reg, n, 16, 256, d_out), kread<4>(reg, n, 64, 256, d_out));
    return 0;
}
