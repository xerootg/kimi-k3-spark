// Reuse the scalar oracle's read-only slice loader, not its compute/reduction code.
#define main g53ref_exact_main
#include "g53ref_exact.cpp"
#undef main
#include "g53parallel.h"

static std::vector<uint8_t> pack(int t,const uint8_t *raw,int nb) {
    const int off[3][4]={{2,66,0,0},{2,66,0,0},{8,4,2,0}};
    const int fs[3][4]={{64,8,2,0},{64,32,2,0},{128,4,2,2}};
    int ti=t==17?0:t==18?1:2,pre=0;std::vector<uint8_t>p(nb*blockbytes(t));
    for(int f=0;f<(t==23?4:3);f++) {
        for(int b=0;b<nb;b++)memcpy(p.data()+nb*pre+b*fs[ti][f],raw+b*blockbytes(t)+off[ti][f],fs[ti][f]);
        pre+=fs[ti][f];
    }
    return p;
}
template<int T,int NB> static float emulated(const uint8_t*r,const Q8*y) {
    auto p=pack(T,r,NB);float yd[NB*8],v[16],next[16];
    for(int i=0;i<NB*8;i++)yd[i]=half(y[i].d);
    // Also compare the independently packed DP4A integer implementation per invocation.
    for(int b=0;b<NB;b++)for(int j=0;j<8;j++) {
        auto a=g53parallel::dot<T,NB>(p.data(),b,j,y[b*8+j],yd[b*8+j],iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
        auto c=g53exact::dot(T,r,NB,b,j,y[b*8+j],false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
        require(a.s==c.s&&bits(a.d)==bits(c.d),"DP4A dot differs");
    }
    for(int l=0;l<16;l++)v[l]=g53parallel::lane16<T,NB>(p.data(),l,y,yd,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    for(int off=8;off;off>>=1) {
        for(int l=0;l<16;l++)next[l]=g53parallel::rowadd(v[l],v[l^off]);
        memcpy(v,next,sizeof v);
    }
    float ref=row(T,r,NB,y,false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    require(bits(v[0])==bits(ref),"SIMD16 vs g53ref_exact row differs");
    return ref;
}
static float compare(int t,int nb,const uint8_t*r,const Q8*y) {
#define RUN(T) if(t==T) {if(nb==8)return emulated<T,8>(r,y);return emulated<T,24>(r,y);}
    RUN(17) RUN(18) RUN(23)
#undef RUN
    throw std::runtime_error("type");
}
static std::vector<Q8> inputs(int nb,std::mt19937&rng,bool edge=false) {
    std::vector<Q8>y(nb*8);
    const uint16_t edges[]={0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0xfbff,0x7c00,0xfc00,0x7e01};
    for(auto&v:y) {
        for(auto&q:v.q)q=int8_t(rng());
        v.d=edge?edges[rng()%11]:tohalf(std::ldexp(float(1+rng()%127),-int(7+rng()%16)));
        v.s=uint16_t(rng());
    }
    return y;
}
static void quantizer(std::mt19937&rng) {
    for(int c=0;c<4096;c++) {
        float x[32],a[16],s[16],nexta[16],nexts[16];
        float scale=std::ldexp(1.f,int(rng()%19)-10);
        for(float&v:x)v=float(int(rng()%255)-127)*scale;
        x[31]=127*scale;
        if(c%3==0) {x[0]=127*scale;x[16]=-x[0];x[1]=0x1p-20f*scale;}
        if(c%5==0) {x[2]=value(1);x[3]=value(0x80000001);}
        for(int l=0;l<16;l++) {
            float a0=ftz(value(bits(x[l])&0x7fffffff)),a1=ftz(value(bits(x[l+16])&0x7fffffff));
            a[l]=a0>a1?a0:a1;s[l]=add(x[l],x[l+16]);
        }
        for(int ox=8;ox;ox>>=1) {
            for(int l=0;l<16;l++) {nexta[l]=a[l]>a[l^ox]?a[l]:a[l^ox];nexts[l]=add(s[l],s[l^ox]);}
            memcpy(a,nexta,sizeof a);memcpy(s,nexts,sizeof s);
        }
        Q8 got{},ref{};float d=mul(a[0],value(0x3c010204)),r=rcp({},d);
        got.d=tohalf(d);got.s=tohalf(s[0]);
        for(int i=0;i<32;i++)got.q[i]=int8_t(uint8_t(roundaway(mul(r,x[i]))));
        quant(x,ref,{});
        require(memcmp(&got,&ref,sizeof got)==0,"SIMD16 q8_1 differs");
    }
    std::cout<<"quantizer: 4096 SIMD16 float-sum/half-scale/q8 blocks bit-identical PASS (power-of-two RCP subset)\n";
}
// Compare the original subgroup quantizer against the scalar block quantizer
// now used by the fused SYCL path. The pipeline consumes the latter result.
static size_t quantized_blocks=0;
static void quant16(const float*x,Q8&q,const mufu_tables&t) {
    float a[16],s[16],na[16],ns[16];
    for(int l=0;l<16;l++) {
        float a0=ftz(value(bits(x[l])&0x7fffffff)),a1=ftz(value(bits(x[l+16])&0x7fffffff));
        a[l]=a0>a1?a0:a1;s[l]=add(x[l],x[l+16]);
    }
    for(int ox=8;ox;ox>>=1) {
        for(int l=0;l<16;l++){na[l]=a[l]>a[l^ox]?a[l]:a[l^ox];ns[l]=add(s[l],s[l^ox]);}
        memcpy(a,na,sizeof a);memcpy(s,ns,sizeof s);
    }
    float d=mul(a[0],value(0x3c010204)),r=a[0]==0?0:rcp(t,d);
    q.d=tohalf(d);q.s=tohalf(s[0]);
    for(int l=0;l<32;l++)q.q[l]=int8_t(uint8_t(roundaway(mul(r,x[l]))));
    Q8 ref;quant(x,ref,t);require(memcmp(&q,&ref,sizeof q)==0,"full MUFU SIMD16 quantizer differs");
    q=ref;quantized_blocks++;
}
template<int T,int NB> static float packed_row(const uint8_t*r,const Q8*y,const float*yd) {
    float v[16],next[16];
    for(int l=0;l<16;l++)v[l]=g53parallel::lane16<T,NB>(r,l,y,yd,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    for(int ox=8;ox;ox>>=1){for(int l=0;l<16;l++)next[l]=g53parallel::rowadd(v[l],v[l^ox]);memcpy(v,next,sizeof v);}
    return v[0];
}
static float packed_row(int t,int nb,const uint8_t*r,const Q8*y,const float*yd) {
#define RUN(T) if(t==T){if(nb==8)return packed_row<T,8>(r,y,yd);return packed_row<T,24>(r,y,yd);}
    RUN(17) RUN(18) RUN(23)
#undef RUN
    throw std::runtime_error("type");
}
static void dumps(const std::string&records,const std::string&input,const std::string&mufu) {
    Tables tables(mufu);size_t total=0,bad=0;
    const int layers[]={3,8,75},tg[]={17,18,17},td[]={18,23,23};
    for(int li=0;li<3;li++) {
        std::string label="L"+std::to_string(layers[li]),dir=input+"/"+label;
        Weights w(true,records+"/"+label+".records",8,tg[li],tg[li],td[li]);
        std::vector<uint8_t> packed[8][3];
        for(int e=0;e<8;e++)for(int pr=0;pr<3;pr++) {
            int nb=pr==2?8:24,n=pr==2?D:F,t=pr==2?td[li]:tg[li],sz=nb*blockbytes(t);
            auto&dst=packed[e][pr];dst.resize(size_t(n)*sz);
            for(int j=0;j<n;j++){auto r=pack(t,w.ptr(e,pr)+size_t(j)*sz,nb);memcpy(dst.data()+size_t(j)*sz,r.data(),sz);}
        }
        Map x(dir+"/x.bin");require(x.n%(D*4)==0,"dump x size");size_t cases=x.n/(D*4);
        Map ids(dir+"/ids.bin",cases*K*4);
        const char*names[]={"g","u","h","dn"};std::unique_ptr<Map> want[4];
        auto o=outputs();size_t differing[4]={};
        for(int s=0;s<4;s++)want[s].reset(new Map(dir+"/"+names[s]+".bin",cases*o[s].size()*4));
        for(size_t c=0;c<cases;c++) {
            Q8 qx[D/32],qh[F/32];float xd[D/32],hd[F/32];
            for(int b=0;b<D/32;b++){quant16((const float*)x.p+c*D+32*b,qx[b],tables.t);xd[b]=half(qx[b].d);}
            for(int e=0;e<K;e++) {
                int id=((const int32_t*)ids.p)[c*K+e];require(id>=0&&id<8,"dump expert id");
                for(int j=0;j<F;j++) {
                    for(int pr=0;pr<2;pr++)o[pr][e*F+j]=packed_row(tg[li],24,packed[id][pr].data()+size_t(j)*24*blockbytes(tg[li]),qx,xd);
                    o[2][e*F+j]=silu(o[0][e*F+j],o[1][e*F+j],tables.t);
                }
                for(int b=0;b<F/32;b++){quant16(o[2].data()+e*F+32*b,qh[b],tables.t);hd[b]=half(qh[b].d);}
                for(int j=0;j<D;j++)o[3][e*D+j]=packed_row(td[li],8,packed[id][2].data()+size_t(j)*8*blockbytes(td[li]),qh,hd);
            }
            for(int s=0;s<4;s++) {
                auto ref=(const uint32_t*)want[s]->p+c*o[s].size();size_t diff=0;
                for(size_t j=0;j<o[s].size();j++)if(bits(o[s][j])!=ref[j]) {
                    if(differing[s]+diff==0)std::cout<<label<<" first "<<names[s]<<" mismatch case="<<c<<" index="<<j<<" got="<<std::hex<<bits(o[s][j])<<" want="<<ref[j]<<std::dec<<std::endl;
                    diff++;
                }
                differing[s]+=diff;
            }
            std::cout<<label<<" case "<<c<<" complete"<<std::endl;
        }
        std::cout<<label<<": "<<cases<<" cases, "<<cases*K<<" expert rows; differing words g/u/h/dn=";
        for(int s=0;s<4;s++){std::cout<<(s?"/":"")<<differing[s];bad+=differing[s];}
        std::cout<<std::endl;total+=cases;
    }
    require(bad==0,"SIMD16 CUDA dump comparison failed");
    std::cout<<"quantizer: "<<quantized_blocks<<" real-MUFU blocks, scalar/subgroup bit-identical PASS"<<std::endl;
    std::cout<<"PASS: "<<total<<" CUDA dump cases, all recorded ids, all g/u/h/dn words bit-identical with real MUFU tables"<<std::endl;
}
static void storage_conversions() {
    for(unsigned h=0;h<65536;h++) {
        if((h&0x7c00)==0x7c00 && (h&1023))continue; // NaN payloads are tested below.
        require(bits(storage_float(h))==bits(half(h)),"binary16 expansion differs");
        // Every finite half, every midpoint, and the FP32 values on either
        // side of each midpoint. Include the overflow tie and both signs.
        if(h<0x7c00) {
            float lo=half(h),hi=h==0x7bff?65536.f:half(h+1);
            uint32_t mid=bits((lo+hi)*0.5f);
            for(uint32_t u:{bits(lo),mid-1,mid,mid+1})for(uint32_t sign:{0u,0x80000000u}) {
                float x=value(u|sign);
                require(storage_half(x)==tohalf(x),"binary16 midpoint rounding differs");
            }
        }
    }
    std::mt19937 rng(0x5316);
    for(int i=0;i<1000000;i++) {
        float x=value(rng());
        require(storage_half(x)==tohalf(x),"binary16 random rounding/NaN differs");
    }
    std::cout<<"binary16: all expansions, signed rounding boundaries, 1000000 random FP32 values PASS\n";
}
int main(int argc,char**argv)try {
    storage_conversions();
    if(argc==4){dumps(argv[1],argv[2],argv[3]);return 0;}
    require(argc==2,"usage: test-parallel DATA | RECORDS DUMPS MUFU");std::mt19937 rng(0x5316);
    // Mixed signs/exponents, odd integer divisions, half subnormals, infinities,
    // canonical NaNs and signed zero. Test both production reduction shapes.
    quantizer(rng);
    size_t count=0;
    for(int t:{17,18,23})for(int nb:{8,24})for(int c=0;c<128;c++) {
        std::vector<uint8_t>r(nb*blockbytes(t));for(auto&v:r)v=uint8_t(rng());
        for(int b=0;b<nb;b++) {
            uint16_t d=c%2?uint16_t(rng()):tohalf(std::ldexp(1.f,-int(rng()%24)));
            memcpy(r.data()+b*blockbytes(t),&d,2);
        }
        auto y=inputs(nb,rng,c%2);compare(t,nb,r.data(),y.data());count++;
    }
    std::cout<<"synthetic: "<<count<<" rows; IQ2/3/4 DP4A + SIMD16 reduction bit-identical PASS\n";
    int layer[]={3,8,75},tg[]={17,18,17},td[]={18,23,23};
    for(int li=0;li<3;li++) {
        Weights w(false,std::string(argv[1])+"/UD-Q2_K_XL_L"+std::to_string(layer[li]),8,tg[li],tg[li],td[li]);
        size_t nr=0;
        for(int proj=0;proj<3;proj++) {
            int nb=proj==2?8:24,n=proj==2?D:F,t=proj==2?td[li]:tg[li];
            auto y=inputs(nb,rng);
            for(int e=0;e<8;e++)for(int r=0;r<n;r++) {
                compare(t,nb,w.ptr(e,proj)+size_t(r)*nb*blockbytes(t),y.data());nr++;
            }
        }
        std::cout<<"L"<<layer[li]<<": "<<nr<<" extracted gate/up/down rows, all 8 experts bit-identical PASS\n";
    }
    std::cout<<"PASS: scheduling/integer emulation; GPU compilation, MUFU and performance require owner checks\n";
    return 0;
} catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
