#include "g53parallel.h"
// G53X_EXACT=2. Two kernels per gate/up or down call, both functors with pointer-sized fields only (static SLM), so
// serveidx pre-records them as Level Zero command lists like modes 0 and 1 (allocation-free request path):
// ParPrepK: one work-item per 32-value block, CUDA q8_1 (or the given q8_1) -> Q8 + float(half(d)) scratch, once per
// distinct activation block, no subgroup RCP broadcast;
// ParMatK: 16 rows per work-group, one SIMD16 sub-group per row (CUDA's virtual-thread partials, XOR-tree reduction).
// The matvec reads the scratch the preceding prepare wrote (in-order queue / a barrier inside a recorded list).
template<class T> static inline T*par_lptr(T*p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space,sycl::access::decorated::no>(p).get();
}
static inline const uint8_t*par_gptr(const uint8_t*p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(p).get();
}
struct ParPrepK {
    static constexpr int NARGS = 10, WG = 256;
    const float*xf; const int8_t*xq; const float*xd; const float*xs; g53exact::Q8*prepared; float*scales;
    const uint32_t*rcp; const uint32_t*ex2p; const uint32_t*ex2n; size_t blocks;
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
        const size_t b=it.get_global_id(0);if(b>=blocks)return;
        const mufu_tables mt{rcp,nullptr,ex2p,ex2n};
        g53exact::Q8 v;
        if(xf)g53exact::quant(xf+32*b,v,mt);
        else {
            for(int i=0;i<32;i++)v.q[i]=xq[32*b+i];
            v.d=g53exact::tohalf(xd[b]);v.s=g53exact::tohalf(xs[b]);
        }
        prepared[b]=v;scales[b]=g53exact::storage_float(v.d);
    }
};
static size_t par_prep_groups(size_t blocks) { return (blocks+ParPrepK::WG-1)/ParPrepK::WG; }
static sycl::event parallel_prepare(sycl::queue&q,size_t blocks,const float*xf,
    const int8_t*xq,const float*xd,const float*xs,g53exact::Q8*prepared,
    float*scales,mufu_tables mt) {
    return q.parallel_for(sycl::nd_range<1>(par_prep_groups(blocks)*ParPrepK::WG,ParPrepK::WG),
                          ParPrepK{xf,xq,xd,xs,prepared,scales,mt.rcp,mt.ex2p,mt.ex2n,blocks});
}
// T,U: row types; K: row length; N: rows per expert. GU: gate+up+SwiGLU (prepared = one shared vector), else down
// (prepared = k vectors of K/32 blocks, expert e at e*K/32).
template<int T,int U,int K,int N,bool GU> struct ParMatK {
    // down uses neither gu nor the MUFU tables: the compiler drops those trailing arguments
    static constexpr int NARGS = GU ? 8 : 4, WG = 256, R = 16;
    static_assert(N%R==0,"whole work-groups");
    static_assert(!GU||T==U,"matching gate/up types");
    const uint8_t*const*W; const g53exact::Q8*prepared; const float*scales; float*out; float*gu;
    const uint32_t*rcp; const uint32_t*ex2p; const uint32_t*ex2n;
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> it) const {
        constexpr int NB=K/256;
        syclex::work_group_static<g53exact::Q8[K/32]> ys;
        syclex::work_group_static<float[K/32]> yds;
        syclex::work_group_static<uint64_t[T==17?512:1]> g2s;
        syclex::work_group_static<uint32_t[T==18?256:1]> g3s;
        syclex::work_group_static<int8_t[T==23?16:1]> k4s;
        g53exact::Q8*y=par_lptr(&ys[0]);float*yd=par_lptr(&yds[0]);
        uint64_t*g2=par_lptr(&g2s[0]);uint32_t*g3=par_lptr(&g3s[0]);int8_t*kval=par_lptr(&k4s[0]);
        const int e=it.get_group(0),l=it.get_local_id(1),off=GU?0:e*K;
        auto subgroup=it.get_sub_group();
        const int lane=subgroup.get_local_id()[0];
        // Coalesced cooperative copy only: no quantization, half conversion,
        // or MUFU table access inside the row-group staging / dot loops.
        auto dst=reinterpret_cast<uint8_t*>(y);
        auto src=reinterpret_cast<const uint8_t*>(prepared+off/32);
        for(int i=l;i<int((K/32)*sizeof(g53exact::Q8));i+=WG)dst[i]=src[i];
        for(int b=l;b<K/32;b+=WG)yd[b]=scales[off/32+b];
        if constexpr(T==17)for(int i=l;i<512;i+=WG)g2[i]=iq2xs_grid[i];
        if constexpr(T==18)for(int i=l;i<256;i+=WG)g3[i]=iq3xxs_grid[i];
        if constexpr(T==23)for(int i=l;i<16;i+=WG)kval[i]=kvalues_iq4nl[i];
        sycl::group_barrier(it.get_group());
        const int row=it.get_group(1)*R+subgroup.get_group_id()[0];
        const uint8_t*a=par_gptr(W[3*e+(GU?0:2)])+size_t(row)*NB*g53exact::blockbytes(T);
        float g=g53parallel::lane16<T,NB>(a,lane,y,yd,g2,g3,kval);
        float u=0;
        if constexpr(GU) {
            const uint8_t*b=par_gptr(W[3*e+1])+size_t(row)*NB*g53exact::blockbytes(U);
            u=g53parallel::lane16<U,NB>(b,lane,y,yd,g2,g3,kval);
        }
#pragma unroll
        for(int offxor=8;offxor;offxor>>=1) {
            g=g53parallel::rowadd(g,sycl::permute_group_by_xor(subgroup,g,offxor));
            if constexpr(GU)u=g53parallel::rowadd(u,sycl::permute_group_by_xor(subgroup,u,offxor));
        }
        if(lane==0) {
            if constexpr(GU) {
                const mufu_tables mt{rcp,nullptr,ex2p,ex2n};
                out[e*N+row]=g53exact::silu(g,u,mt);
                if(gu){gu[2*e*N+row]=g;gu[(2*e+1)*N+row]=u;}
            } else out[e*N+row]=g;
        }
    }
};
// The scratch: grown here only for bench / check / tests (never per request); serveidx reserves its whole layout at
// startup (parallel_reserve) so this never reallocates while it serves.
static void parallel_reserve(sycl::queue&q,size_t blocks) {
    if(blocks<=exact_device.capacity)return;
    q.wait_and_throw();
    sycl::free(exact_device.prepared,q);sycl::free(exact_device.scales,q);
    exact_device.prepared=sycl::malloc_device<g53exact::Q8>(blocks,q);
    exact_device.scales=sycl::malloc_device<float>(blocks,q);
    if(!exact_device.prepared||!exact_device.scales) {
        fprintf(stderr,"exact activation allocation failed\n");exit(1);
    }
    exact_device.capacity=blocks;
}
// SYCL-queue form (bench, check, tests, serveidx with G53X_SYCL_LAUNCH): the vector(s) at the start of the scratch.
template<int T,int U,int K,int N,bool GU>
static sycl::event parallel_exact(sycl::queue&q,const uint8_t*const*W,int k,
                                  const float*xf,const int8_t*xq,const float*xd,const float*xs,
                                  float*out,float*gu) {
    using MK=ParMatK<T,U,K,N,GU>;
    const auto mt=exact_device.mt;
    const size_t blocks=size_t(GU?1:k)*(K/32);
    parallel_reserve(q,blocks);
    auto prepared=exact_device.prepared;auto scales=exact_device.scales;
    parallel_prepare(q,blocks,xf,xq,xd,xs,prepared,scales,mt);
    return q.parallel_for(sycl::nd_range<2>({size_t(k),size_t(N/MK::R*MK::WG)},{1,size_t(MK::WG)}),
                          MK{W,prepared,scales,out,gu,mt.rcp,mt.ex2p,mt.ex2n});
}
