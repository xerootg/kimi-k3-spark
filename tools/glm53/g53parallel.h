// Shared by the SIMD16 kernel and CPU scheduling emulation. Oracle stays in g53exact.h.
#pragma once
#include "g53exact.h"
#if defined(__SYCL_DEVICE_ONLY__)
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int, int, int, bool);
#endif
namespace g53parallel {
// IQ row arithmetic starts from two binary16 scales and integer dots. Every finite
// value is a multiple of 2^-48, including cancellations and rounded sums: FP32
// subnormals cannot occur. Skip redundant input/output FTZ tests in this domain,
// but canonicalize every NaN. Quantization and SwiGLU still use full FTZ helpers.
inline float canon(float x) {
    return (g53exact::bits(x)&0x7fffffff)>0x7f800000 ? g53exact::value(0x7fffffff) : x;
}
inline float rowadd(float a,float b) { return canon(a+b); }
inline float rowfma(float a,float b,float c) {
#ifdef G53EX_SYCL
    return canon(sycl::fma(a,b,c));
#else
    return canon(std::fma(a,b,c));
#endif
}
inline int dp4(uint32_t a, const int8_t *q, int s) {
#if defined(__SYCL_DEVICE_ONLY__)
    // Byte assembly is alias-safe; IGC can combine these aligned loads.
    return __builtin_IB_dp4a_ss(s, int(a), int(g53exact::u32(reinterpret_cast<const uint8_t *>(q))), false);
#else
    return g53exact::dot4(a,q,s);
#endif
}
inline uint32_t sign4(uint32_t g, unsigned s) {
    unsigned b=(s&1)|((s&2)<<7)|((s&4)<<14)|((s&8)<<21);
    unsigned m=(b<<8)-b;
    return (g^m)+(m&0x01010101u);
}
template<int T,int NB>
inline g53exact::Dot dot(const uint8_t *r,int b,int j,const g53exact::Q8 &y,float yd,
                         const uint64_t*g2,const uint32_t*g3,const int8_t*k4) {
    using namespace g53exact;
    constexpr int SZ=T==17?74:T==18?98:136;
    int sum=0;
    if constexpr(T==17) {
        const uint8_t *qs=r+64*b+8*j;
        unsigned sc=r[NB*64+b*8+j]; int a=0,c=0;
#ifdef G53EX_SYCL
#pragma unroll
#endif
        for(int l=0;l<4;l++) {
            unsigned q=u16(qs+2*l),sg=signs(q>>9);uint64_t g=g2[q&511];
            int &v=l<2?a:c;
            v=dp4(sign4(uint32_t(g),sg),y.q+8*l,v);
            v=dp4(sign4(uint32_t(g>>32),sg>>4),y.q+8*l+4,v);
        }
        sum=(a*int(sc&15)+c*int(sc>>4)+(a+c)/2)/4;
    } else if constexpr(T==18) {
        const uint8_t *qs=r+64*b+8*j;unsigned aux=u32(r+NB*64+32*b+4*j);
#ifdef G53EX_SYCL
#pragma unroll
#endif
        for(int l=0;l<4;l++) {
            unsigned sg=signs(aux>>(7*l));
            sum=dp4(sign4(g3[qs[2*l]],sg),y.q+8*l,sum);
            sum=dp4(sign4(g3[qs[2*l+1]],sg>>4),y.q+8*l+4,sum);
        }
        sum=(int(aux>>28)*sum+sum/2)/2;
    } else {
        static_assert(T==23,"IQ types only");
        const uint8_t *qs=r+128*b+16*j;
#ifdef G53EX_SYCL
#pragma unroll
#endif
        for(int l=0;l<4;l++) {
            uint32_t lo=0,hi=0;
#ifdef G53EX_SYCL
#pragma unroll
#endif
            for(int i=0;i<4;i++) {
                unsigned v=qs[4*l+i];
                lo|=uint32_t(uint8_t(k4[v&15]))<<(8*i);
                hi|=uint32_t(uint8_t(k4[v>>4]))<<(8*i);
            }
            sum=dp4(lo,y.q+4*l,sum);sum=dp4(hi,y.q+16+4*l,sum);
        }
        unsigned lo=r[NB*128+4*b+j/2],hi=u16(r+NB*132+2*b);
        sum*=int(((lo>>(4*(j&1)))&15)|(((hi>>(2*j))&3)<<4))-32;
    }
    return {canon(half(u16(r+NB*(SZ-2)+2*b))*yd),sum};
}
// Each Intel lane owns CUDA lanes lane and lane+16, in all four warps.
// Finish offset 16 locally; the caller must then perform XOR 8,4,2,1 simultaneously.
template<int T,int NB>
inline float lane16(const uint8_t *r,int lane,const g53exact::Q8 *y,const float *yd,
                    const uint64_t*g2,const uint32_t*g3,const int8_t*k4) {
    using namespace g53exact;
    float v[2];
#ifdef G53EX_SYCL
#pragma unroll
#endif
    for(int halfwarp=0;halfwarp<2;halfwarp++) {
        float w[4];
#ifdef G53EX_SYCL
#pragma unroll
#endif
        for(int warp=0;warp<4;warp++) {
            const int t=32*warp+lane+16*halfwarp;float a=0;
#ifdef G53EX_SYCL
#pragma unroll
#endif
            for(int b=t/8;b<NB;b+=16) {
                const int u=b*8+t%8;
                Dot d=dot<T,NB>(r,b,t%8,y[u],yd[u],g2,g3,k4);
                a=rowfma(d.d,float(d.s),a);
            }
            w[warp]=a;
        }
        v[halfwarp]=rowadd(rowadd(rowadd(w[0],w[1]),w[2]),w[3]);
    }
    return rowadd(v[0],v[1]);
}
}
