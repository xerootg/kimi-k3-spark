// Q6_K MMVQ (ncols_dst=1) microbenchmark: stock llama.cpp kernel vs memory-level variants.
// Built against the real ggml-cuda headers so vec_dot_q6_K_q8_1{,_impl_mmvq} are the shipped code.
#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

constexpr int NW = 4;      // nwarps of the GENERIC table for ncols_dst=1
constexpr int WS = 32;
constexpr int QK = QK_K;   // 256
constexpr int QI = QI6_K;  // 32
constexpr int VDR = VDR_Q6_K_Q8_1_MMVQ; // 2
constexpr int BPI = VDR*NW*WS/QI;       // 8 blocks per loop iteration

// ---------------------------------------------------------------- stock (verbatim logic of mul_mat_vec_q, ncols_dst=1, no fusion)
template <int R>
__launch_bounds__(NW*WS, 1)
static __global__ void k_stock(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
                               const int ncols_x, const int stride_row_x, const int nrows) {
    const int tid  = WS*threadIdx.y + threadIdx.x;
    const int row0 = R*blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK;
    ggml_cuda_pdl_sync();
    float tmp[R] = {0.0f};
    const int kbx_offset = row0*stride_row_x;
    for (int kbx = tid / (QI/VDR); kbx < blocks_per_row_x; kbx += BPI) {
        const int kby = kbx * (QK/QK8_1);
        const int kqs = VDR * (tid % (QI/VDR));
#pragma unroll
        for (int i = 0; i < R; ++i) {
            tmp[i] += vec_dot_q6_K_q8_1(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }
    __shared__ float tmp_shared[NW-1][R][WS];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < R; ++i) tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    dst += row0;
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int l = 0; l < NW-1; ++l) tmp[i] += tmp_shared[l][i][threadIdx.x];
        tmp[i] = warp_reduce_sum<WS>(tmp[i]);
        if (threadIdx.x == i && (R == 1 || uint32_t(row0 + i) < uint32_t(nrows))) dst[i] = tmp[i];
    }
}

// ---------------------------------------------------------------- load helpers
// H: 0 = plain ld.global, 1 = ld.global.nc, 2 = nc + L1::no_allocate, 3 = nc + L1::no_allocate + L2 evict_first policy
template <int H> static __device__ __forceinline__ uint32_t ldw16(const void * p, uint64_t pol) {
    if constexpr (H == 0) { return *(const uint16_t *) p; }
    else if constexpr (H == 1) { uint16_t v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v; }
    else if constexpr (H == 2) { uint16_t v; asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v; }
    else { uint16_t v; asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u16 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol)); return v; }
}
template <int H> static __device__ __forceinline__ int8_t ldw8(const void * p, uint64_t pol) {
    if constexpr (H == 0) { return *(const int8_t *) p; }
    else if constexpr (H == 1) { int16_t v; asm volatile("ld.global.nc.s8 %0, [%1];" : "=h"(v) : "l"(p)); return (int8_t) v; }
    else if constexpr (H == 2) { int16_t v; asm volatile("ld.global.nc.L1::no_allocate.s8 %0, [%1];" : "=h"(v) : "l"(p)); return (int8_t) v; }
    else { int16_t v; asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.s8 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol)); return (int8_t) v; }
}

struct q6raw {
    int vl[VDR];
    int vh[VDR];   // unshifted
    int8_t sc0, sc1;
    half d;
};
struct q8raw {
    int u[VDR*QR6_K];
    float d8[QR6_K];
};

template <int H>
static __device__ __forceinline__ void load_q6(q6raw & r, const block_q6_K * b, const int iqs, uint64_t pol) {
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int qh_idx = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);
#pragma unroll
    for (int j = 0; j < VDR; ++j) {
        const uint8_t * pl = b->ql + 4*(iqs + j);
        const uint8_t * ph = b->qh + 4*(qh_idx + j);
        r.vl[j] = (int) (ldw16<H>(pl, pol) | (ldw16<H>(pl + 2, pol) << 16));
        r.vh[j] = (int) (ldw16<H>(ph, pol) | (ldw16<H>(ph + 2, pol) << 16));
    }
    r.sc0 = ldw8<H>(b->scales + scale_offset, pol);
    r.sc1 = ldw8<H>(b->scales + scale_offset + 4, pol);
    const uint16_t dr = ldw16<H>(&b->d, pol);
    r.d = __ushort_as_half(dr);
}

static __device__ __forceinline__ void load_q8(q8raw & r, const block_q8_1 * bq8_1, const int iqs) {
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR; ++j) {
            r.u[VDR*i + j] = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1 + j);
        }
        r.d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }
}

static __device__ __forceinline__ float compute_q6(const q6raw & a, const q8raw & b, const int iqs) {
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
    int vh[VDR];
#pragma unroll
    for (int j = 0; j < VDR; ++j) vh[j] = a.vh[j] >> vh_shift;
    int8_t sc[5];
    sc[0] = a.sc0; sc[4] = a.sc1; sc[1] = sc[2] = sc[3] = 0;
    const float d = a.d;
    return vec_dot_q6_K_q8_1_impl_mmvq(a.vl, vh, b.u, sc, d, b.d8);
}


// Same integers and the same float expression as vec_dot_q6_K_q8_1_impl_mmvq, fewer instructions:
//  - the -32 offset term dp4a(0xE0E0E0E0, u) does not depend on the weights, it is computed once per q8_1 slice (m[i])
//    and reused for every row (integer addition is exact, so the order of the dp4a terms does not matter)
//  - ((vh >> sh) << 4) & 0x30303030 == rotl(vh, 4 - sh) & 0x30303030 for sh in {0,2,4,6}
static __device__ __forceinline__ void q8_offsets(int m[QR6_K], const q8raw & b) {
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        int t = 0;
#pragma unroll
        for (int j = 0; j < VDR; ++j) t = ggml_cuda_dp4a(0xE0E0E0E0, b.u[VDR*i + j], t);
        m[i] = t;
    }
}
static __device__ __forceinline__ float compute_q6_fast(const q6raw & a, const q8raw & b, const int m[QR6_K], const int iqs) {
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = i == 0 ? a.sc0 : a.sc1;
        int sumi = m[i];
#pragma unroll
        for (int j = 0; j < VDR; ++j) {
            const int vil = (a.vl[j] >> (4*i)) & 0x0F0F0F0F;
            const int vih = i == 0 ? (int) (__funnelshift_l(a.vh[j], a.vh[j], (4 - vh_shift) & 31) & 0x30303030)
                                   : (int) (__funnelshift_r(a.vh[j], a.vh[j], vh_shift) & 0x30303030);
            sumi = ggml_cuda_dp4a(vil | vih, b.u[VDR*i + j], sumi);
        }
        sumf += b.d8[i] * (sumi * sc);
    }
    const float d = a.d;
    return d*sumf;
}

// ---------------------------------------------------------------- variant: batched loads (P iterations in flight per thread), R rows per block
template <int R, int P, int H, int MINB>
__launch_bounds__(NW*WS, MINB)
static __global__ void k_pipe(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
                              const int ncols_x, const int stride_row_x, const int nrows) {
    const int tid  = WS*threadIdx.y + threadIdx.x;
    const int row0 = R*blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK;
    uint64_t pol = 0;
    if constexpr (H == 3) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    ggml_cuda_pdl_sync();
    float tmp[R] = {0.0f};
    const block_q6_K * x = (const block_q6_K *) vx + row0*stride_row_x;
    const int kqs = VDR * (tid % (QI/VDR));
    for (int kbx0 = tid / (QI/VDR); kbx0 < blocks_per_row_x; kbx0 += P*BPI) {
        q6raw a[P][R];
        q8raw b[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const int kbx = kbx0 + p*BPI;
            if (p == 0 || kbx < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < R; ++i) load_q6<H>(a[p][i], x + i*stride_row_x + kbx, kqs, pol);
                load_q8(b[p], &y[kbx*(QK/QK8_1)], kqs);
            }
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const int kbx = kbx0 + p*BPI;
            if (p == 0 || kbx < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < R; ++i) tmp[i] += compute_q6(a[p][i], b[p], kqs);
            }
        }
    }
    __shared__ float tmp_shared[NW-1][R][WS];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < R; ++i) tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    dst += row0;
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int l = 0; l < NW-1; ++l) tmp[i] += tmp_shared[l][i][threadIdx.x];
        tmp[i] = warp_reduce_sum<WS>(tmp[i]);
        if (threadIdx.x == i && (R == 1 || uint32_t(row0 + i) < uint32_t(nrows))) dst[i] = tmp[i];
    }
}


// ---------------------------------------------------------------- variant: persistent, TMA bulk copies of R-row tiles into an S-stage smem ring
static __device__ __forceinline__ void mbar_init(uint64_t * bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((uint32_t) __cvta_generic_to_shared(bar)), "r"(count));
}
static __device__ __forceinline__ void tma_load(void * sdst, const void * gsrc, uint32_t bytes, uint64_t * bar, uint64_t pol) {
    const uint32_t b = (uint32_t) __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"((uint32_t) __cvta_generic_to_shared(sdst)), "l"(gsrc), "r"(bytes), "r"(b), "l"(pol) : "memory");
}
static __device__ __forceinline__ void mbar_wait(uint64_t * bar, uint32_t phase) {
    const uint32_t b = (uint32_t) __cvta_generic_to_shared(bar);
    asm volatile("{\n\t.reg .pred p;\n"
                 "W%=:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra W%=;\n}" :: "r"(b), "r"(phase) : "memory");
}

template <int R>
static __device__ __forceinline__ void load_q6_smem(q6raw & r, const uint8_t * b, const int iqs) {
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int qh_idx = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);
#pragma unroll
    for (int j = 0; j < VDR; ++j) {
        const uint16_t * pl = (const uint16_t *) (b + offsetof(block_q6_K, ql) + 4*(iqs + j));
        const uint16_t * ph = (const uint16_t *) (b + offsetof(block_q6_K, qh) + 4*(qh_idx + j));
        r.vl[j] = (int) (pl[0] | (pl[1] << 16));
        r.vh[j] = (int) (ph[0] | (ph[1] << 16));
    }
    r.sc0 = (int8_t) b[offsetof(block_q6_K, scales) + scale_offset];
    r.sc1 = (int8_t) b[offsetof(block_q6_K, scales) + scale_offset + 4];
    r.d = __ushort_as_half(*(const uint16_t *) (b + offsetof(block_q6_K, d)));
}

// YS: 1 = stage the q8_1 vector in smem, 0 = read it from global like stock
template <int R, int YS, int MINB, int NOCOMP = 0>
__launch_bounds__(NW*WS, MINB)
static __global__ void k_tma(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
                             const int ncols_x, const int stride_row_x, const int nrows, const int S) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ uint64_t full[16];
    __shared__ float tmp_shared[2][NW-1][R][WS];

    const int tid = WS*threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = ncols_x / QK;
    const uint32_t row_bytes  = blocks_per_row_x * sizeof(block_q6_K);
    const uint32_t tile_bytes = R * row_bytes;
    const int ntiles = nrows / R;
    const uint32_t tile_stride_smem = (tile_bytes + 127) & ~127u;
    uint8_t * ybuf = smem + S*tile_stride_smem;

    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

    if (tid == 0) {
        for (int s = 0; s < S; ++s) mbar_init(&full[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    // weights do not depend on the previous kernel: start streaming them before the grid dependency sync
    if (tid == 0) {
        for (int s = 0; s < S; ++s) {
            const int t = blockIdx.x + s*gridDim.x;
            if (t < ntiles) tma_load(smem + s*tile_stride_smem, (const char *) vx + (size_t) t*tile_bytes, tile_bytes, &full[s], pol);
        }
    }
    ggml_cuda_pdl_sync();
    const block_q8_1 * ys = y;
    if constexpr (YS) {
        const int nyb = ncols_x / QK8_1;
        const int4 * src = (const int4 *) y;
        int4 * d4 = (int4 *) ybuf;
        for (int i = tid; i < nyb*(int)sizeof(block_q8_1)/16; i += NW*WS) d4[i] = src[i];
        __syncthreads();
        ys = (const block_q8_1 *) ybuf;
    }

    const int kqs = VDR * (tid % (QI/VDR));
    int k = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++k) {
        const int s = k % S;
        mbar_wait(&full[s], (k / S) & 1);
        const uint8_t * tile = smem + s*tile_stride_smem;
        float tmp[R] = {0.0f};
        if (!NOCOMP) for (int kbx = tid / (QI/VDR); kbx < blocks_per_row_x; kbx += BPI) {
            q8raw b;
            load_q8(b, &ys[kbx*(QK/QK8_1)], kqs);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                q6raw a;
                load_q6_smem<R>(a, tile + i*row_bytes + kbx*sizeof(block_q6_K), kqs);
                tmp[i] += compute_q6(a, b, kqs);
            }
        }
        if (threadIdx.y > 0) {
#pragma unroll
            for (int i = 0; i < R; ++i) tmp_shared[k & 1][threadIdx.y-1][i][threadIdx.x] = tmp[i];
        }
        __syncthreads();
        if (threadIdx.y == 0) {
            if (threadIdx.x == 0) {
                const int tn = t + S*gridDim.x;
                if (tn < ntiles) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_load(smem + s*tile_stride_smem, (const char *) vx + (size_t) tn*tile_bytes, tile_bytes, &full[s], pol);
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int l = 0; l < NW-1; ++l) tmp[i] += tmp_shared[k & 1][l][i][threadIdx.x];
                tmp[i] = warp_reduce_sum<WS>(tmp[i]);
                if (threadIdx.x == i) dst[t*R + i] = tmp[i];
            }
        }
    }
}


static __global__ void k_readbw(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
                                const int ncols_x, const int stride_row_x, const int nrows) {
    const size_t n16 = (size_t) nrows * (ncols_x/QK) * sizeof(block_q6_K) / 16;
    const int4 * p = (const int4 *) vx;
    int acc = 0;
    for (size_t i = blockIdx.x*(size_t)blockDim.x*blockDim.y + WS*threadIdx.y + threadIdx.x; i < n16; i += (size_t) gridDim.x*blockDim.x*blockDim.y) {
        int4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) dst[0] = acc;
}

// 8 bytes starting at a 2-byte aligned smem address, as two ints, via three aligned 32-bit loads
static __device__ __forceinline__ void lds_8b_a2(const uint8_t * p, int & lo, int & hi) {
    const uint32_t a  = (uint32_t) __cvta_generic_to_shared(p);
    const uint32_t a4 = a & ~3u;
    const uint32_t sh = (a & 3u) * 8;
    uint32_t w0, w1, w2;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w0) : "r"(a4));
    asm volatile("ld.shared.u32 %0, [%1+4];" : "=r"(w1) : "r"(a4));
    asm volatile("ld.shared.u32 %0, [%1+8];" : "=r"(w2) : "r"(a4));
    lo = (int) __funnelshift_r(w0, w1, sh);
    hi = (int) __funnelshift_r(w1, w2, sh);
}

template <int ALN>
static __device__ __forceinline__ void load_q6_smem2(q6raw & r, const uint8_t * b, const int iqs) {
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int qh_idx = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);
    static_assert(VDR == 2, "");
    if constexpr (ALN) {
        lds_8b_a2(b + offsetof(block_q6_K, ql) + 4*iqs,    r.vl[0], r.vl[1]);
        lds_8b_a2(b + offsetof(block_q6_K, qh) + 4*qh_idx, r.vh[0], r.vh[1]);
    } else {
#pragma unroll
        for (int j = 0; j < VDR; ++j) {
            const uint16_t * pl = (const uint16_t *) (b + offsetof(block_q6_K, ql) + 4*(iqs + j));
            const uint16_t * ph = (const uint16_t *) (b + offsetof(block_q6_K, qh) + 4*(qh_idx + j));
            r.vl[j] = (int) (pl[0] | (pl[1] << 16));
            r.vh[j] = (int) (ph[0] | (ph[1] << 16));
        }
    }
    r.sc0 = (int8_t) b[offsetof(block_q6_K, scales) + scale_offset];
    r.sc1 = (int8_t) b[offsetof(block_q6_K, scales) + scale_offset + 4];
    r.d = __ushort_as_half(*(const uint16_t *) (b + offsetof(block_q6_K, d)));
}

// G row groups of NW warps per CUDA block; a tile of R rows, group g handles rows g, g+G, ...
template <int R, int G, int YS, int ALN, int CONLY = 0, int F = 0>
__launch_bounds__(NW*WS*G, 1)
static __global__ void k_tma2(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
                              const int ncols_x, const int stride_row_x, const int nrows, const int S) {
    static_assert(R % G == 0, "");
    constexpr int RG = R / G;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ uint64_t full[16];
    __shared__ float tmp_shared[2][G][NW-1][RG][WS];

    const int grp = threadIdx.y / NW;
    const int wy  = threadIdx.y % NW;
    const int tid = WS*wy + threadIdx.x;               // virtual thread index within the row group, as in stock
    const int btid = WS*threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = ncols_x / QK;
    const uint32_t row_bytes  = blocks_per_row_x * sizeof(block_q6_K);
    const uint32_t tile_bytes = R * row_bytes;
    const int ntiles = nrows / R;
    const uint32_t tile_stride_smem = (tile_bytes + 127) & ~127u;
    uint8_t * ybuf = smem + S*tile_stride_smem;

    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

    if (btid == 0) {
        for (int s = 0; s < S; ++s) mbar_init(&full[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (btid == 0) {
        for (int s = 0; s < S; ++s) {
            const int t = blockIdx.x + s*gridDim.x;
            if (t < ntiles) tma_load(smem + s*tile_stride_smem, (const char *) vx + (size_t) t*tile_bytes, tile_bytes, &full[s], pol);
        }
    }
    ggml_cuda_pdl_sync();
    const block_q8_1 * ys = y;
    if constexpr (YS) {
        const int nyb = ncols_x / QK8_1;
        const int4 * src = (const int4 *) y;
        int4 * d4 = (int4 *) ybuf;
        for (int i = btid; i < nyb*(int)sizeof(block_q8_1)/16; i += NW*WS*G) d4[i] = src[i];
        __syncthreads();
        ys = (const block_q8_1 *) ybuf;
    }

    const int kqs = VDR * (tid % (QI/VDR));
    int k = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++k) {
        const int s = k % S;
        if (!CONLY || k < S) mbar_wait(&full[s], (k / S) & 1);
        const uint8_t * tile = smem + s*tile_stride_smem;
        float tmp[RG] = {0.0f};
        for (int kbx = tid / (QI/VDR); kbx < blocks_per_row_x; kbx += BPI) {
            q8raw b;
            load_q8(b, &ys[kbx*(QK/QK8_1)], kqs);
            int m[QR6_K];
            if constexpr (F) q8_offsets(m, b);
#pragma unroll
            for (int i = 0; i < RG; ++i) {
                q6raw a;
                load_q6_smem2<ALN>(a, tile + (grp + i*G)*row_bytes + kbx*sizeof(block_q6_K), kqs);
                if constexpr (F) tmp[i] += compute_q6_fast(a, b, m, kqs);
                else             tmp[i] += compute_q6(a, b, kqs);
            }
        }
        if (wy > 0) {
#pragma unroll
            for (int i = 0; i < RG; ++i) tmp_shared[k & 1][grp][wy-1][i][threadIdx.x] = tmp[i];
        }
        __syncthreads();
        if (wy == 0) {
            if (btid == 0) {
                const int tn = t + S*gridDim.x;
                if (tn < ntiles && !CONLY) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_load(smem + s*tile_stride_smem, (const char *) vx + (size_t) tn*tile_bytes, tile_bytes, &full[s], pol);
                }
            }
#pragma unroll
            for (int i = 0; i < RG; ++i) {
#pragma unroll
                for (int l = 0; l < NW-1; ++l) tmp[i] += tmp_shared[k & 1][grp][l][i][threadIdx.x];
                tmp[i] = warp_reduce_sum<WS>(tmp[i]);
                if (threadIdx.x == 0) dst[t*R + grp + i*G] = tmp[i];
            }
        }
    }
}


static __device__ __forceinline__ void mbar_arrive(uint64_t * bar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"((uint32_t) __cvta_generic_to_shared(bar)) : "memory");
}
static __device__ __forceinline__ void mbar_wait_acq(uint64_t * bar, uint32_t phase) {
    const uint32_t b = (uint32_t) __cvta_generic_to_shared(bar);
    asm volatile("{\n\t.reg .pred p;\n"
                 "W%=:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra W%=;\n}" :: "r"(b), "r"(phase) : "memory");
}

// no __syncthreads in the main loop: per-stage "empty" mbarrier (all 128 threads arrive after writing partials and
// finishing their reads of the stage); warp 0 waits for it, reduces, and refills the stage.
template <int R, int ALN>
__launch_bounds__(NW*WS, 1)
static __global__ void k_tma3(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
                              const int ncols_x, const int stride_row_x, const int nrows, const int S) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ uint64_t full[16];
    __shared__ uint64_t empty[16];
    __shared__ float partial[8][NW-1][R][WS];

    const int tid = WS*threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = ncols_x / QK;
    const uint32_t row_bytes  = blocks_per_row_x * sizeof(block_q6_K);
    const uint32_t tile_bytes = R * row_bytes;
    const int ntiles = nrows / R;
    const uint32_t tile_stride_smem = (tile_bytes + 127) & ~127u;

    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

    if (tid == 0) {
        for (int s = 0; s < S; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], NW*WS); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) {
        for (int s = 0; s < S; ++s) {
            const int t = blockIdx.x + s*gridDim.x;
            if (t < ntiles) tma_load(smem + s*tile_stride_smem, (const char *) vx + (size_t) t*tile_bytes, tile_bytes, &full[s], pol);
        }
    }
    ggml_cuda_pdl_sync();

    const int kqs = VDR * (tid % (QI/VDR));
    int k = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++k) {
        const int s = k % S;
        const uint32_t ph = (k / S) & 1;
        mbar_wait(&full[s], ph);
        const uint8_t * tile = smem + s*tile_stride_smem;
        float tmp[R] = {0.0f};
        for (int kbx = tid / (QI/VDR); kbx < blocks_per_row_x; kbx += BPI) {
            q8raw b;
            load_q8(b, &y[kbx*(QK/QK8_1)], kqs);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                q6raw a;
                load_q6_smem2<ALN>(a, tile + i*row_bytes + kbx*sizeof(block_q6_K), kqs);
                tmp[i] += compute_q6(a, b, kqs);
            }
        }
        if (threadIdx.y > 0) {
#pragma unroll
            for (int i = 0; i < R; ++i) partial[s][threadIdx.y-1][i][threadIdx.x] = tmp[i];
        }
        mbar_arrive(&empty[s]);
        if (threadIdx.y == 0) {
            mbar_wait_acq(&empty[s], ph);
            if (threadIdx.x == 0) {
                const int tn = t + S*gridDim.x;
                if (tn < ntiles) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_load(smem + s*tile_stride_smem, (const char *) vx + (size_t) tn*tile_bytes, tile_bytes, &full[s], pol);
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int l = 0; l < NW-1; ++l) tmp[i] += partial[s][l][i][threadIdx.x];
                tmp[i] = warp_reduce_sum<WS>(tmp[i]);
                if (threadIdx.x == 0) dst[t*R + i] = tmp[i];
            }
        }
    }
}

// ---------------------------------------------------------------- harness
struct Shape { int K, N, count; const char * name; };

typedef void (*kfn_t)(const void *, const block_q8_1 *, float *, int, int, int);
typedef void (*kfn_tma_t)(const void *, const block_q8_1 *, float *, int, int, int, int);

#include <functional>
typedef std::function<bool(cudaStream_t, const void *, const block_q8_1 *, float *, int, int)> launch_t;
struct Variant { std::string name; launch_t fn; };

static int g_nsm = 132;

static launch_t L_simple(kfn_t fn, int R) {
    return [=](cudaStream_t s, const void * x, const block_q8_1 * y, float * d, int K, int N) {
        fn<<<(N + R - 1)/R, dim3(WS, NW), 0, s>>>(x, y, d, K, K/QK, N);
        return true;
    };
}
// R must make a tile 16-byte sized; S stages; B blocks per SM
static launch_t L_tma(kfn_tma_t fn, int R, int YS, int S, int B, int G = 1) {
    return [=](cudaStream_t s, const void * x, const block_q8_1 * y, float * d, int K, int N) {
        const size_t tile = (size_t) R * (K/QK) * sizeof(block_q6_K);
        if (tile % 16 != 0 || N % R != 0) return false;
        const size_t tstride = (tile + 127) & ~(size_t)127;
        int SS = S;
        const size_t ybytes = YS ? (size_t) (K/QK8_1)*sizeof(block_q8_1) : 0;
        while (SS > 1 && SS*tstride + ybytes > (size_t) (200*1024)/B) --SS;
        if (SS > 8) SS = 8;
        const size_t smem = SS*tstride + ybytes;
        if (smem > 220*1024) return false;
        CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        const int ntiles = N / R;
        const int grid = std::min(ntiles, g_nsm * B);
        fn<<<grid, dim3(WS, NW*G), smem, s>>>(x, y, d, K, K/QK, N, SS);
        return true;
    };
}


// the configuration selected in mmvq.cu: R = 2 if a 2-row tile is 16-byte sized else 4, q8_1 in smem, 2 stages,
// blocks/SM = min(4, 200 KiB / smem); stock path for < 2048 rows
static launch_t L_int() {
    return [=](cudaStream_t s, const void * x, const block_q8_1 * y, float * d, int K, int N) {
        const size_t row = (size_t) (K/QK) * sizeof(block_q6_K);
        const int R = (2*row) % 16 == 0 ? 2 : ((4*row) % 16 == 0 ? 4 : 0);
        if (N < 2048 || R == 0 || N % R) {
            const int Rs = (K/QK) < BPI ? 4 : 1;
            if (Rs == 4) k_stock<4><<<(N+3)/4, dim3(WS, NW), 0, s>>>(x, y, d, K, K/QK, N);
            else         k_stock<1><<<N, dim3(WS, NW), 0, s>>>(x, y, d, K, K/QK, N);
            return true;
        }
        const size_t tstride = (R*row + 127) & ~(size_t)127;
        const size_t smem = 2*tstride + (size_t) (K/QK8_1)*sizeof(block_q8_1);
        if (smem > 200*1024) return false;
        const int B = std::min<int>(4, (200*1024) / smem);
        kfn_tma_t fn = R == 2 ? (kfn_tma_t) k_tma2<2,1,1,0> : (kfn_tma_t) k_tma2<4,1,1,0>;
        CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024));
        fn<<<std::min(N/R, g_nsm*B), dim3(WS, NW), smem, s>>>(x, y, d, K, K/QK, N, 2);
        return true;
    };
}
static launch_t L_stockdefault() {
    return [=](cudaStream_t s, const void * x, const block_q8_1 * y, float * d, int K, int N) {
        if ((K/QK) < BPI) k_stock<4><<<(N+3)/4, dim3(WS, NW), 0, s>>>(x, y, d, K, K/QK, N);
        else              k_stock<1><<<N, dim3(WS, NW), 0, s>>>(x, y, d, K, K/QK, N);
        return true;
    };
}

static float bench_one(const launch_t & fn, const std::vector<void *> & xs, const block_q8_1 * y, float * dst, int K, int N, int iters) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    bool ok = true;
    for (int it = 0; it < iters; ++it) {
        ok = fn(s, xs[it % xs.size()], y, dst, K, N) && ok;
    }
    CK(cudaStreamEndCapture(s, &g));
    if (!ok) { CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s)); return -1.0f; }
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s)); // warmup
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> ts;
    for (int rep = 0; rep < 5; ++rep) {
        CK(cudaEventRecord(e0, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        ts.push_back(ms * 1000.0f / iters);
    }
    std::sort(ts.begin(), ts.end());
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaStreamDestroy(s));
    return ts[ts.size()/2];
}

int main(int argc, char ** argv) {
    std::vector<Shape> shapes = {
        {7168, 12288, 300, "attn_q/k/v/gate,ssm_g"},
        {12288, 7168, 93,  "attn_output"},
        {7168, 6144, 184,  "ffn_gate/up_shexp"},
        {6144, 7168, 92,   "ffn_down_shexp"},
        {7168, 3584, 92,   "ffn_routed_down"},
        {3584, 7168, 92,   "ffn_routed_up"},
        {1536, 18432, 24,  "attn_q_b"},
        {7168, 1536, 24,   "attn_q_a"},
        {7168, 576, 24,    "attn_kv_a_mqa"},
        {512, 12288, 24,   "attn_v_b(96x128 as N)"},
        {7168, 128, 69,    "ssm_f_a"},
        {7168, 33792, 2,   "ffn_gate/up(dense)"},
        {33792, 7168, 1,   "ffn_down(dense)"},
        {7168, 163840, 1,  "output"},
    };
    const char * only  = argc > 1 && strcmp(argv[1], "-") ? argv[1] : nullptr;
    const char * vonly = argc > 2 ? argv[2] : nullptr;
    const int bitcheck_seeds = 3;
    { cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); g_nsm = p.multiProcessorCount; }

    std::vector<Variant> vars;
#define V(name, L) vars.push_back({name, L})
    V("stock R1",            L_simple(k_stock<1>, 1));
    V("STOCK(default)",      L_stockdefault());
    V("INTEGRATED",          L_int());
    V("stock R2",            L_simple(k_stock<2>, 2));
    V("stock R4",            L_simple(k_stock<4>, 4));
    V("pipe R1 P1 H0",       L_simple(k_pipe<1,1,0,1>, 1));
    V("pipe R1 P2 H0",       L_simple(k_pipe<1,2,0,1>, 1));
    V("pipe R1 P4 H0",       L_simple(k_pipe<1,4,0,1>, 1));
    V("pipe R1 P2 H1",       L_simple(k_pipe<1,2,1,1>, 1));
    V("pipe R1 P2 H2",       L_simple(k_pipe<1,2,2,1>, 1));
    V("pipe R1 P2 H3",       L_simple(k_pipe<1,2,3,1>, 1));
    V("pipe R1 P4 H2",       L_simple(k_pipe<1,4,2,1>, 1));
    V("pipe R2 P2 H0",       L_simple(k_pipe<2,2,0,1>, 2));
    V("pipe R2 P2 H2",       L_simple(k_pipe<2,2,2,1>, 2));
    V("pipe R2 P1 H2",       L_simple(k_pipe<2,1,2,1>, 2));
    V("pipe R4 P1 H2",       L_simple(k_pipe<4,1,2,1>, 4));
    V("pipe R1 P2 H2 m8",    L_simple(k_pipe<1,2,2,8>, 1));
    V("pipe R1 P4 H2 m8",    L_simple(k_pipe<1,4,2,8>, 1));
    V("probe readbw", ([](cudaStream_t s, const void * x, const block_q8_1 * y, float * d, int K, int N) {
        k_readbw<<<g_nsm*8, dim3(WS, NW), 0, s>>>(x, y, d, K, K/QK, N); return true; }));
    V("probe tma-nocomp R2 B4 S4", L_tma(k_tma<2,0,1,1>, 2, 0, 4, 4));
    V("probe tma-nocomp R2 B2 S8", L_tma(k_tma<2,0,1,1>, 2, 0, 8, 2));
    V("probe tma-nocomp R4 B2 S4", L_tma(k_tma<4,0,1,1>, 4, 0, 4, 2));
#define T2(R,G,YS,ALN,S,B) { char nm[64]; snprintf(nm, sizeof nm, "tma2 R%d G%d Y%d A%d S%d B%d", R,G,YS,ALN,S,B); V(nm, L_tma(k_tma2<R,G,YS,ALN>, R, YS, S, B, G)); }
#define T3(R,ALN,S,B) { char nm[64]; snprintf(nm, sizeof nm, "tma3 R%d A%d S%d B%d", R,ALN,S,B); V(nm, L_tma(k_tma3<R,ALN>, R, 0, S, B, 1)); }
    T3(2,1,2,4) T3(2,1,3,4) T3(2,1,4,4) T3(2,0,4,4) T3(2,1,4,3) T3(2,1,6,3) T3(2,1,4,6) T3(2,1,3,6) T3(2,1,2,8) T3(2,1,8,2)
    T3(4,1,2,4) T3(4,1,3,3) T3(4,1,4,2) T3(1,1,4,4) T3(1,1,8,4) T3(1,1,4,8)
#define TF(R,YS,S,B) { char nm[64]; snprintf(nm, sizeof nm, "tmaF R%d Y%d S%d B%d", R,YS,S,B); V(nm, L_tma(k_tma2<R,1,YS,1,0,1>, R, YS, S, B, 1)); }
    TF(1,0,2,4) TF(1,0,2,6) TF(1,0,3,4) TF(1,1,2,4) TF(4,0,2,4) TF(4,1,2,4) TF(4,1,2,3)
    TF(2,0,2,4) TF(2,1,2,4) TF(2,0,2,3) TF(2,0,3,4) TF(4,0,2,2) TF(4,1,2,2) TF(4,0,2,3) TF(2,1,2,6) TF(1,0,2,8) TF(1,0,4,4) TF(4,1,3,3)
    V("probe computeonly-F R2 S2 B4", L_tma(k_tma2<2,1,0,1,1,1>, 2, 0, 2, 4, 1));
    V("probe computeonly R2 S2 B4", L_tma(k_tma2<2,1,0,1,1>, 2, 0, 2, 4, 1));
    V("probe computeonly R2 S2 B8", L_tma(k_tma2<2,1,0,1,1>, 2, 0, 2, 8, 1));
    T2(2,1,0,1,2,3) T2(2,1,0,1,3,3) T2(2,1,0,1,2,5) T2(2,1,0,1,2,6) T2(2,1,0,1,3,4) T2(2,1,1,1,2,6) T2(2,1,0,1,2,8)
    T2(2,1,0,0,2,4) T2(2,1,0,1,2,4) T2(2,1,1,1,2,4) T2(4,1,0,1,2,2) T2(4,1,0,1,2,4)
    T2(2,2,0,0,2,2) T2(2,2,0,1,2,2) T2(2,2,0,1,4,2) T2(2,2,1,1,4,2) T2(2,2,0,1,2,4)
    T2(4,2,0,1,2,2) T2(4,2,0,1,4,2) T2(4,4,0,1,2,2) T2(4,4,0,1,4,1) T2(4,4,0,1,8,1) T2(4,4,1,1,4,2)
    T2(8,4,0,1,2,2) T2(8,4,0,1,4,1) T2(8,8,0,1,4,1) T2(8,2,0,1,2,2)
    for (int R : {1, 2, 4}) for (int YS : {0, 1}) for (int B : {2, 4, 8}) for (int S : {2, 4, 8}) {
        kfn_tma_t f = nullptr;
        if (R == 1) f = YS ? k_tma<1,1,1> : k_tma<1,0,1>;
        if (R == 2) f = YS ? k_tma<2,1,1> : k_tma<2,0,1>;
        if (R == 4) f = YS ? k_tma<4,1,1> : k_tma<4,0,1>;
        char nm[64]; snprintf(nm, sizeof nm, "tma R%d Y%d B%d S%d", R, YS, B, S);
        V(nm, L_tma(f, R, YS, S, B));
    }
#undef V
    if (vonly) {
        std::vector<Variant> keep;
        for (auto & v : vars) {
            bool m = v.name.find("stock R1") == 0;
            std::string alts = vonly; size_t p0 = 0;
            while (!m) { size_t p1 = alts.find('|', p0); std::string a = alts.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
                if (!a.empty() && v.name.find(a) != std::string::npos) m = true; if (p1 == std::string::npos) break; p0 = p1 + 1; }
            if (m) keep.push_back(v);
        }
        vars = keep;
    }

    std::mt19937_64 rng(1234);
    const size_t target_bytes = 512ull << 20; // rotate through >= 512 MiB of weights (L2 is 50 MB)

    printf("%-24s %6s %6s %5s | %-20s %9s %7s %s\n", "shape", "K", "N", "cnt", "variant", "us", "TB/s", "bitident");
    for (const Shape & sh : shapes) {
        if (only && !strstr(sh.name, only)) continue;
        const int nb = sh.K / QK;
        const size_t xbytes = (size_t) sh.N * nb * sizeof(block_q6_K);
        const int ncopies = std::max<size_t>(2, (target_bytes + xbytes - 1) / xbytes);
        std::vector<void *> xs(ncopies);
        std::vector<uint8_t> hx(xbytes);
        auto fill_x = [&](uint64_t seed) {
            std::mt19937_64 r(seed);
            for (size_t i = 0; i < xbytes; ++i) hx[i] = (uint8_t) r();
            block_q6_K * b = (block_q6_K *) hx.data();
            for (size_t i = 0; i < (size_t) sh.N*nb; ++i) {
                float d = (float) ((r() % 2000) + 1) * 1e-6f * ((r() & 1) ? -1.f : 1.f);
                half hd = __float2half(d);
                memcpy(&b[i].d, &hd, 2);
            }
        };
        const int nyb = sh.K / QK8_1;
        std::vector<block_q8_1> hy(nyb);
        auto fill_y = [&](uint64_t seed) {
            std::mt19937_64 r(seed);
            for (int i = 0; i < nyb; ++i) {
                int sum = 0;
                for (int j = 0; j < QK8_1; ++j) { hy[i].qs[j] = (int8_t) (r() % 255 - 127); sum += hy[i].qs[j]; }
                float d = (float) ((r() % 1000) + 1) * 1e-4f;
                hy[i].ds = __floats2half2_rn(d, d*sum);
            }
        };
        for (int c = 0; c < ncopies; ++c) CK(cudaMalloc(&xs[c], xbytes));
        block_q8_1 * dy; CK(cudaMalloc(&dy, nyb*sizeof(block_q8_1)));
        float * ddst; CK(cudaMalloc(&ddst, (size_t) sh.N*sizeof(float)));
        float * dref; CK(cudaMalloc(&dref, (size_t) sh.N*sizeof(float)));
        fill_x(42); fill_y(43);
        for (int c = 0; c < ncopies; ++c) CK(cudaMemcpy(xs[c], hx.data(), xbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dy, hy.data(), nyb*sizeof(block_q8_1), cudaMemcpyHostToDevice));

        // bit-identity check vs stock R(default) over several seeds (copy 0 is reseeded, then restored)
        const int R_default = nb < BPI ? NW : 1; // small_k path uses rows_per_block = nwarps
        std::vector<int> ident(vars.size(), 1);
        std::vector<float> href(sh.N), hv(sh.N);
        for (int seed = 0; seed < bitcheck_seeds; ++seed) {
            fill_x(1000 + seed); fill_y(2000 + seed);
            CK(cudaMemcpy(xs[0], hx.data(), xbytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dy, hy.data(), nyb*sizeof(block_q8_1), cudaMemcpyHostToDevice));
            kfn_t ref = R_default == 1 ? (kfn_t) k_stock<1> : (kfn_t) k_stock<4>;
            ref<<<(sh.N + R_default - 1)/R_default, dim3(WS, NW)>>>(xs[0], dy, dref, sh.K, nb, sh.N);
            CK(cudaMemcpy(href.data(), dref, sh.N*sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t v = 0; v < vars.size(); ++v) {
                CK(cudaMemset(ddst, 0xff, sh.N*sizeof(float)));
                if (!vars[v].fn(0, xs[0], dy, ddst, sh.K, sh.N)) { ident[v] = -1; continue; }
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hv.data(), ddst, sh.N*sizeof(float), cudaMemcpyDeviceToHost));
                if (memcmp(hv.data(), href.data(), sh.N*sizeof(float)) != 0) ident[v] = 0;
            }
        }
        fill_x(42); CK(cudaMemcpy(xs[0], hx.data(), xbytes, cudaMemcpyHostToDevice));

        const int iters = std::max(50, std::min(2000, (int) (4e9 / xbytes)));
        for (size_t v = 0; v < vars.size(); ++v) {
            if (ident[v] < 0) continue;
            const float us = bench_one(vars[v].fn, xs, dy, ddst, sh.K, sh.N, iters);
            printf("%-24s %6d %6d %5d | %-20s %9.2f %7.3f %s\n", sh.name, sh.K, sh.N, sh.count, vars[v].name.c_str(), us,
                   xbytes / (us * 1e-6) / 1e12, ident[v] ? "yes" : "NO");
            fflush(stdout);
        }
        for (void * p : xs) CK(cudaFree(p));
        CK(cudaFree(dy)); CK(cudaFree(ddst)); CK(cudaFree(dref));
    }
    return 0;
}
