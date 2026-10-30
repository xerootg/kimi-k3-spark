// Deliberately scalar-per-row SYCL oracle. Same shared arithmetic as CPU; no
// unspecified subgroup reductions or compiler-selected exp/div approximations.
static bool exact_mode() { const char *s=getenv("G53X_EXACT");return !s||atoi(s)!=0; }
struct ExactDevice { mufu_tables mt{};uint64_t*g2=nullptr;uint32_t*g3=nullptr;int8_t*k4=nullptr; };
static ExactDevice exact_device;
static void exact_init(sycl::queue&q) {
    if(!exact_mode()||exact_device.g2)return;
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
    fprintf(stderr,"G53X_EXACT=1: source-derived CUDA IQ arithmetic; GLM SASS validation pending\n");
}
static inline void exact_qrow(const float*f,const int8_t*q,const float*d,const float*s,int n,
                              g53exact::Q8*y,const mufu_tables&t) {
    for(int b=0;b<n/32;b++) {
        if(f)g53exact::quant(f+32*b,y[b],t);
        else {for(int i=0;i<32;i++)y[b].q[i]=q[32*b+i];y[b].d=g53exact::tohalf(d[b]);y[b].s=g53exact::tohalf(s[b]);}
    }
}
template<int TG,int TU,int DD,int FF>
static sycl::event exact_gateup(sycl::queue&q,const uint8_t*const*W,int k,const float*xf,const int8_t*xq,
                               const float*xd,const float*xs,float*h,float*gu) {
    const auto t=exact_device;
    return q.parallel_for(sycl::range<1>(size_t(k)*FF),[=](sycl::id<1>id){
        const int e=id[0]/FF,j=id[0]%FF;g53exact::Q8 y[DD/32];exact_qrow(xf,xq,xd,xs,DD,y,t.mt);
        float g=g53exact::row(TG,W[3*e]+size_t(j)*(DD/256)*g53exact::blockbytes(TG),DD/256,y,true,t.g2,t.g3,t.k4);
        float u=g53exact::row(TU,W[3*e+1]+size_t(j)*(DD/256)*g53exact::blockbytes(TU),DD/256,y,true,t.g2,t.g3,t.k4);
        h[e*FF+j]=g53exact::silu(g,u,t.mt);
        if(gu){gu[(2*e)*FF+j]=g;gu[(2*e+1)*FF+j]=u;}
    });
}
template<int TD,int DD,int FF>
static sycl::event exact_down(sycl::queue&q,const uint8_t*const*W,int k,const float*hf,const int8_t*hq,
                             const float*hd,const float*hs,float*out) {
    const auto t=exact_device;
    return q.parallel_for(sycl::range<1>(size_t(k)*DD),[=](sycl::id<1>id){
        const int e=id[0]/DD,j=id[0]%DD;g53exact::Q8 y[FF/32];
        exact_qrow(hf?hf+e*FF:nullptr,hq?hq+e*FF:nullptr,hd?hd+e*FF/32:nullptr,hs?hs+e*FF/32:nullptr,FF,y,t.mt);
        out[e*DD+j]=g53exact::row(TD,W[3*e+2]+size_t(j)*(FF/256)*g53exact::blockbytes(TD),FF/256,y,true,t.g2,t.g3,t.k4);
    });
}
