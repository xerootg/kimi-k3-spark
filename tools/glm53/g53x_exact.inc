#ifdef __clang__   // exact arithmetic, whatever the TU's FP flags (g53x keeps icpx's defaults for its fast kernels)
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#endif
// Deliberately scalar-per-row SYCL oracle. Same shared arithmetic as CPU; no
// unspecified subgroup reductions or compiler-selected exp/div approximations.
static int exact_version() { const char*s=getenv("G53X_EXACT");return s?atoi(s):2; }
static bool exact_mode() { const char *s=getenv("G53X_EXACT");return !s||atoi(s)!=0; }
struct ExactDevice { mufu_tables mt{};uint64_t*g2=nullptr;uint32_t*g3=nullptr;int8_t*k4=nullptr;
    // Reused on the single in-order production queue, including streamed down calls.
    g53exact::Q8 *prepared=nullptr;float *scales=nullptr;size_t capacity=0;
};
static ExactDevice exact_device;
static void exact_init(sycl::queue&q) {
    if(!exact_mode()||exact_device.g2)return;
    if(exact_version()!=1&&exact_version()!=2) {fprintf(stderr,"G53X_EXACT must be 0 (fast), 1 (scalar exact) or 2 (fast exact)\n");exit(1);}
    const char*dir=getenv("G53X_MUFU");
    if(!dir) {fprintf(stderr,"exact mode requires G53X_MUFU (verified GB10 rcp/ex2p/ex2n tables); G53X_EXACT=0 selects non-exact mode\n");exit(1);}
    auto load=[&](const char*name,size_t n)->const uint32_t* {
        std::string p=std::string(dir)+"/"+name;FILE*f=fopen(p.c_str(),"rb");
        if(!f){perror(p.c_str());exit(1);}auto v=sycl::malloc_host<uint32_t>(n/4,q);
        if(!v||fread(v,1,n,f)!=n||fgetc(f)!=EOF){fprintf(stderr,"bad MUFU table %s\n",p.c_str());exit(1);}fclose(f);return v;
    };
    exact_device.mt.rcp=load("rcp.bin",size_t(1)<<25);
    exact_device.mt.ex2p=load("ex2p.bin",size_t(31)<<25);
    exact_device.mt.ex2n=load("ex2n.bin",size_t(33)<<25);
    exact_device.g2=sycl::malloc_shared<uint64_t>(512,q);
    exact_device.g3=sycl::malloc_shared<uint32_t>(256,q);
    exact_device.k4=sycl::malloc_shared<int8_t>(16,q);
    if(!exact_device.g2||!exact_device.g3||!exact_device.k4){fprintf(stderr,"exact tables allocation failed\n");exit(1);}
    memcpy(exact_device.g2,iq2xs_grid,4096);memcpy(exact_device.g3,iq3xxs_grid,1024);memcpy(exact_device.k4,kvalues_iq4nl,16);
    fprintf(stderr,"G53X_EXACT=%d: CUDA IQ arithmetic (%s)\n",exact_version(),exact_version()==2?"fast exact":"scalar oracle");
}
static inline void exact_qrow(const float*f,const int8_t*q,const float*d,const float*s,int n,
                              g53exact::Q8*y,const mufu_tables&t) {
    for(int b=0;b<n/32;b++) {
        if(f)g53exact::quant(f+32*b,y[b],t);
        else {for(int i=0;i<32;i++)y[b].q[i]=q[32*b+i];y[b].d=g53exact::tohalf(d[b]);y[b].s=g53exact::tohalf(s[b]);}
    }
}
// The exact kernels are functors (all 8-byte pointer fields, the MUFU and grid tables passed directly) so serveidx
// can pre-record them as Level Zero command lists like the fast kernels (ze_kernel, Kind::zl): the request path stays
// allocation-free in exact mode too. Grid: dim 0 = expert, dim 1 = output row, work-group {1, EXACT_WG}; every work-item
// computes one row exactly as before (no cross-lane arithmetic, so the work-group shape does not change any bit).
static constexpr int EXACT_WG = 64;
template<int TG,int TU,int DD,int FF> struct ExactGateUpK {
    static constexpr int NARGS = 13, WG = EXACT_WG;
    const uint8_t*const*W; const float*xf; const int8_t*xq; const float*xd; const float*xs; float*h; float*gu;
    const uint32_t*rcp; const uint32_t*ex2p; const uint32_t*ex2n; const uint64_t*g2; const uint32_t*g3; const int8_t*k4;
    void operator()(sycl::nd_item<2> it) const {
        const mufu_tables mt{rcp,nullptr,ex2p,ex2n};
        const int e=it.get_global_id(0),j=it.get_global_id(1);g53exact::Q8 y[DD/32];exact_qrow(xf,xq,xd,xs,DD,y,mt);
        float g=g53exact::row(TG,W[3*e]+size_t(j)*(DD/256)*g53exact::blockbytes(TG),DD/256,y,true,g2,g3,k4);
        float u=g53exact::row(TU,W[3*e+1]+size_t(j)*(DD/256)*g53exact::blockbytes(TU),DD/256,y,true,g2,g3,k4);
        h[e*FF+j]=g53exact::silu(g,u,mt);
        if(gu){gu[(2*e)*FF+j]=g;gu[(2*e+1)*FF+j]=u;}
    }
};
template<int TD,int DD,int FF> struct ExactDownK {
    static constexpr int NARGS = 12, WG = EXACT_WG;
    const uint8_t*const*W; const float*hf; const int8_t*hq; const float*hd; const float*hs; float*out;
    const uint32_t*rcp; const uint32_t*ex2p; const uint32_t*ex2n; const uint64_t*g2; const uint32_t*g3; const int8_t*k4;
    void operator()(sycl::nd_item<2> it) const {
        const mufu_tables mt{rcp,nullptr,ex2p,ex2n};
        const int e=it.get_global_id(0),j=it.get_global_id(1);g53exact::Q8 y[FF/32];
        exact_qrow(hf?hf+e*FF:nullptr,hq?hq+e*FF:nullptr,hd?hd+e*FF/32:nullptr,hs?hs+e*FF/32:nullptr,FF,y,mt);
        out[e*DD+j]=g53exact::row(TD,W[3*e+2]+size_t(j)*(FF/256)*g53exact::blockbytes(TD),FF/256,y,true,g2,g3,k4);
    }
};
#include "g53x_parallel.inc"
template<int TG,int TU,int DD,int FF>
static sycl::event exact_gateup(sycl::queue&q,const uint8_t*const*W,int k,const float*xf,const int8_t*xq,
                               const float*xd,const float*xs,float*h,float*gu) {
    if(exact_version()==2)return parallel_exact<TG,TU,DD,FF,true>(q,W,k,xf,xq,xd,xs,h,gu);
    const auto&t=exact_device;
    return q.parallel_for(sycl::nd_range<2>({size_t(k),size_t(FF)},{1,size_t(EXACT_WG)}),
                          ExactGateUpK<TG,TU,DD,FF>{W,xf,xq,xd,xs,h,gu,t.mt.rcp,t.mt.ex2p,t.mt.ex2n,t.g2,t.g3,t.k4});
}
template<int TD,int DD,int FF>
static sycl::event exact_down(sycl::queue&q,const uint8_t*const*W,int k,const float*hf,const int8_t*hq,
                             const float*hd,const float*hs,float*out) {
    if(exact_version()==2)return parallel_exact<TD,TD,FF,DD,false>(q,W,k,hf,hq,hd,hs,out,nullptr);
    const auto&t=exact_device;
    return q.parallel_for(sycl::nd_range<2>({size_t(k),size_t(DD)},{1,size_t(EXACT_WG)}),
                          ExactDownK<TD,DD,FF>{W,hf,hq,hd,hs,out,t.mt.rcp,t.mt.ex2p,t.mt.ex2n,t.g2,t.g3,t.k4});
}
static void exact_quantize(sycl::queue&q,const float*x,int8_t*xq,float*xd,float*xs,int nblk) {
    const auto t=exact_device.mt;
    q.parallel_for(sycl::range<1>(nblk),[=](sycl::id<1>id){
        int b=id[0];g53exact::Q8 y;g53exact::quant(x+32*b,y,t);
        for(int i=0;i<32;i++)xq[32*b+i]=y.q[i];
        xd[b]=g53exact::half(y.d);xs[b]=g53exact::half(y.s);
    });
}
#ifdef __clang__
#pragma float_control(pop)
#endif
