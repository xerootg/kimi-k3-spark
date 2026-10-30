#include "g53parallel.h"
// One work-item owns the entire CUDA butterfly. Pin the same SIMD16 execution
// shape as the passing v2 path; retain one quantization per distinct block.
static sycl::event parallel_prepare(sycl::queue&q,size_t blocks,const float*xf,
    const int8_t*xq,const float*xd,const float*xs,g53exact::Q8*prepared,
    float*scales,mufu_tables mt) {
    constexpr size_t WG=256;
    // One work-item per 32-value block, no subgroup RCP broadcast. In-order
    // submission orders preparation, all consumers, then reuse by the next call.
    return q.parallel_for(sycl::nd_range<1>(((blocks+WG-1)/WG)*WG,WG),
        [=](sycl::nd_item<1>it) [[sycl::reqd_sub_group_size(16)]] {
        const size_t b=it.get_global_id(0);if(b>=blocks)return;
        g53exact::Q8 v;
        if(xf)g53exact::quant(xf+32*b,v,mt);
        else {
            for(int i=0;i<32;i++)v.q[i]=xq[32*b+i];
            v.d=g53exact::tohalf(xd[b]);v.s=g53exact::tohalf(xs[b]);
        }
        prepared[b]=v;scales[b]=g53exact::storage_float(v.d);
    });
}
// 16 rows per work-group, one SIMD16 subgroup per row. Prepare Q8 once per
// activation vector, shared by all row groups (and all gate/up experts).
// The public float-input path also uses this prepass; G53X_FUSEQ=1 no longer
// embeds quantization in mode 2's matvec. Keep half(float sum) for CUDA parity.
template<int T,int U,int K,int N,bool GU>
static sycl::event parallel_exact(sycl::queue&q,const uint8_t*const*W,int k,
                                  const float*xf,const int8_t*xq,const float*xd,const float*xs,
                                  float*out,float*gu) {
    constexpr int NB=K/256, WG=256, R=16;
    static_assert(N%R==0,"whole work-groups");
    static_assert(!GU||T==U,"matching gate/up types");
    const auto mt=exact_device.mt;
    const size_t blocks=size_t(GU?1:k)*(K/32);
    // Growth happens only on warm-up / a new largest shape. Never allocate or
    // free per request; wait before replacing storage still used by this queue.
    if(blocks>exact_device.capacity) {
        q.wait_and_throw();
        sycl::free(exact_device.prepared,q);sycl::free(exact_device.scales,q);
        exact_device.prepared=sycl::malloc_device<g53exact::Q8>(blocks,q);
        exact_device.scales=sycl::malloc_device<float>(blocks,q);
        if(!exact_device.prepared||!exact_device.scales) {
            fprintf(stderr,"exact activation allocation failed\n");exit(1);
        }
        exact_device.capacity=blocks;
    }
    auto prepared=exact_device.prepared;auto scales=exact_device.scales;
    parallel_prepare(q,blocks,xf,xq,xd,xs,prepared,scales,mt);
    return q.submit([&](sycl::handler&cgh) {
        sycl::local_accessor<g53exact::Q8,1> y(K/32,cgh);
        sycl::local_accessor<float,1> yd(K/32,cgh);
        sycl::local_accessor<uint64_t,1> g2(T==17?512:1,cgh);
        sycl::local_accessor<uint32_t,1> g3(T==18?256:1,cgh);
        sycl::local_accessor<int8_t,1> kval(T==23?16:1,cgh);
        cgh.parallel_for(sycl::nd_range<2>({size_t(k),size_t(N/R*WG)},{1,WG}),
            [=](sycl::nd_item<2>it) [[sycl::reqd_sub_group_size(16)]] {
            const int e=it.get_group(0),l=it.get_local_id(1),off=GU?0:e*K;
            auto subgroup=it.get_sub_group();
            const int lane=subgroup.get_local_id()[0];
            // Coalesced cooperative copy only: no quantization, half conversion,
            // or MUFU table access inside the row-group staging / dot loops.
            auto dst=reinterpret_cast<uint8_t*>(&y[0]);
            auto src=reinterpret_cast<const uint8_t*>(prepared+off/32);
            for(int i=l;i<int((K/32)*sizeof(g53exact::Q8));i+=WG)dst[i]=src[i];
            for(int b=l;b<K/32;b+=WG)yd[b]=scales[off/32+b];
            if constexpr(T==17)for(int i=l;i<512;i+=WG)g2[i]=iq2xs_grid[i];
            if constexpr(T==18)for(int i=l;i<256;i+=WG)g3[i]=iq3xxs_grid[i];
            if constexpr(T==23)for(int i=l;i<16;i+=WG)kval[i]=kvalues_iq4nl[i];
            sycl::group_barrier(it.get_group());
            const int row=it.get_group(1)*R+subgroup.get_group_id()[0];
            const uint8_t *wa=sycl::address_space_cast<sycl::access::address_space::global_space,
                sycl::access::decorated::no>(W[3*e+(GU?0:2)]).get();
            const uint8_t *a=wa+size_t(row)*NB*g53exact::blockbytes(T);
            float g=g53parallel::lane16<T,NB>(a,lane,&y[0],&yd[0],&g2[0],&g3[0],&kval[0]);
            float u=0;
            if constexpr(GU) {
                const uint8_t *wb=sycl::address_space_cast<sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(W[3*e+1]).get();
                const uint8_t *b=wb+size_t(row)*NB*g53exact::blockbytes(U);
                u=g53parallel::lane16<U,NB>(b,lane,&y[0],&yd[0],&g2[0],&g3[0],&kval[0]);
            }
#pragma unroll
            for(int offxor=8;offxor;offxor>>=1) {
                g=g53parallel::rowadd(g,sycl::permute_group_by_xor(subgroup,g,offxor));
                if constexpr(GU)u=g53parallel::rowadd(u,sycl::permute_group_by_xor(subgroup,u,offxor));
            }
            if(lane==0) {
                if constexpr(GU) {
                    out[e*N+row]=g53exact::silu(g,u,mt);
                    if(gu){gu[2*e*N+row]=g;gu[(2*e+1)*N+row]=u;}
                } else out[e*N+row]=g;
            }
        });
    });
}
