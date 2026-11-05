// Patch 0046: staged prefill gather. One kernel per (layer, part) copies the used experts' slices from the registered
// host cache slots straight into the layer's HBM expert tensor (16-byte coalesced loads over all SMs, no per-expert
// copies, no host sync), and zero-fills the 512 bytes the MMQ reads past a used expert's neighbour.
#pragma once

#define GATHER_MAX_JOBS 1024
#define GATHER_RING     4

struct ggml_cuda_gather_jobs {
    const char * src[GATHER_MAX_JOBS]; // host slot base of the record (the part offset is added by the kernel)
    int32_t      x[GATHER_MAX_JOBS];   // destination expert index
    int32_t      zx[GATHER_MAX_JOBS];  // experts whose first 512 bytes are zeroed
    int32_t      n;
    int32_t      nz;
};

static ggml_cuda_gather_jobs * g_gather_host[GATHER_RING];
static ggml_cuda_gather_jobs * g_gather_dev[GATHER_RING];
static int                     g_gather_next = 0;
static cudaEvent_t             g_gather_ev0 = nullptr, g_gather_ev1 = nullptr;

static __global__ void ggml_cuda_gather_experts_kernel(char * __restrict__ dst, const size_t msize, const size_t part_off,
        const ggml_cuda_gather_jobs * __restrict__ jobs) {
    const size_t upj    = msize/16;
    const size_t total  = (size_t) jobs->n*upj;
    const size_t nthr   = (size_t) gridDim.x*blockDim.x;
    const size_t gtid   = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    constexpr int U = 4;

    for (size_t idx = gtid; idx < total; idx += U*nthr) {
        uint4  v[U];
        size_t j[U], o[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
            const size_t i = idx + k*nthr;
            if (i < total) {
                j[k] = i/upj;
                o[k] = i - j[k]*upj;
                v[k] = __ldg((const uint4 *) (jobs->src[j[k]] + part_off) + o[k]);
            }
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
            if (idx + k*nthr < total) {
                ((uint4 *) (dst + (size_t) jobs->x[j[k]]*msize))[o[k]] = v[k];
            }
        }
    }

    const size_t ztotal = (size_t) jobs->nz*32;
    for (size_t i = gtid; i < ztotal; i += nthr) {
        const size_t z = i/32;
        ((uint4 *) (dst + (size_t) jobs->zx[z]*msize))[i - z*32] = make_uint4(0, 0, 0, 0);
    }
}

// a pinned job list the caller fills once per layer (the ring keeps a list alive until its kernels have run)
static ggml_cuda_gather_jobs * ggml_cuda_gather_begin() {
    if (g_gather_host[0] == nullptr) {
        for (int i = 0; i < GATHER_RING; ++i) {
            CUDA_CHECK(cudaMallocHost((void **) &g_gather_host[i], sizeof(ggml_cuda_gather_jobs)));
            CUDA_CHECK(cudaMalloc((void **) &g_gather_dev[i], sizeof(ggml_cuda_gather_jobs)));
        }
        CUDA_CHECK(cudaEventCreate(&g_gather_ev0));
        CUDA_CHECK(cudaEventCreate(&g_gather_ev1));
    }
    g_gather_next = (g_gather_next + 1) % GATHER_RING;
    return g_gather_host[g_gather_next];
}

// launches on the backend's stream; with timing it waits for the kernel and returns its time in ms (else -1)
static double ggml_cuda_gather_launch(ggml_backend_t backend, void * dst, size_t msize, size_t part_off, const ggml_cuda_gather_jobs * jobs, int timing) {
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    int slot = -1;
    for (int i = 0; i < GATHER_RING; ++i) {
        if (g_gather_host[i] == jobs) {
            slot = i;
        }
    }
    GGML_ASSERT(slot >= 0);
    cudaStream_t s = ctx->stream();
    if (timing) {
        CUDA_CHECK(cudaEventRecord(g_gather_ev0, s));
    }
    CUDA_CHECK(cudaMemcpyAsync(g_gather_dev[slot], jobs, sizeof(ggml_cuda_gather_jobs), cudaMemcpyHostToDevice, s));
    const int nsm = ggml_cuda_info().devices[ctx->device].nsm;
    ggml_cuda_gather_experts_kernel<<<nsm*4, 512, 0, s>>>((char *) dst, msize, part_off, g_gather_dev[slot]);
    CUDA_CHECK(cudaGetLastError());
    static const bool verify = getenv("LLAMA_EXPERT_STORE_PREFILL_GATHER_VERIFY") != nullptr;
    if (verify) {
        CUDA_CHECK(cudaStreamSynchronize(s));
        static std::vector<char> tmp;
        tmp.resize(msize);
        int bad = 0;
        for (int j = 0; j < jobs->n; ++j) {
            CUDA_CHECK(cudaMemcpy(tmp.data(), (const char *) dst + (size_t) jobs->x[j]*msize, msize, cudaMemcpyDeviceToHost));
            if (memcmp(tmp.data(), jobs->src[j] + part_off, msize) != 0) {
                bad++;
            }
        }
        if (bad) {
            fprintf(stderr, "GATHER-VERIFY mismatch: %d of %d experts\n", bad, jobs->n);
        }
    }
    if (!timing) {
        return -1.0;
    }
    CUDA_CHECK(cudaEventRecord(g_gather_ev1, s));
    CUDA_CHECK(cudaEventSynchronize(g_gather_ev1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_gather_ev0, g_gather_ev1));
    return ms;
}
