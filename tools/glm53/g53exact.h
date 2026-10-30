// Source-derived GB10 arithmetic. Pending GLM SASS/dump validation: SPEC-glm.md.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#ifdef __clang__   // exact arithmetic, whatever the TU's FP flags (g53x keeps icpx's defaults for its fast kernels)
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#endif
#include "mufu/mufu_emu.h"
namespace g53exact {
#ifdef G53EX_SYCL
inline uint32_t bits(float x) { return sycl::bit_cast<uint32_t>(x); }
inline float value(uint32_t x) { return sycl::bit_cast<float>(x); }
inline float half(uint16_t x) { return float(sycl::bit_cast<sycl::half>(x)); }
#else
inline uint32_t bits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
inline float value(uint32_t u) { float x; memcpy(&x,&u,4); return x; }
inline float half(uint16_t x) { _Float16 h; memcpy(&h,&x,2); return float(h); }
#endif
// Make the binary16 boundary explicit even when immediately widened in a
// prepass: no dependence on native FP16 denormal modes or conversion folding.
// Round to nearest/even, preserving signed zero and half subnormals explicitly.
inline uint16_t storage_half(float x) {
    uint32_t u=bits(x), a=u&0x7fffffff, sign=(u>>16)&0x8000;
    if(a>=0x7f800000) return uint16_t(sign|0x7c00|(a>0x7f800000 ? ((a>>13)&1023)|512 : 0));
    if(a>=0x47800000) return uint16_t(sign|0x7c00);
    if(a<0x33000000) return uint16_t(sign);
    if(a<0x38800000) {
        unsigned shift=126-(a>>23), m=(a&0x7fffff)|0x800000;
        unsigned n=m>>shift, rem=m&((1u<<shift)-1), tie=1u<<(shift-1);
        return uint16_t(sign| (n+(rem>tie || (rem==tie && (n&1)))));
    }
    unsigned n=(a-0x38000000)>>13, rem=a&8191;
    return uint16_t(sign|(n+(rem>4096 || (rem==4096 && (n&1)))));
}
inline float storage_float(uint16_t h) {
    unsigned sign=unsigned(h&0x8000)<<16, e=(h>>10)&31, m=h&1023;
    if(e==31)return value(sign|0x7f800000|(m<<13));
    if(e)return value(sign|((e+112)<<23)|(m<<13));
    if(!m)return value(sign);
    unsigned exp=113;
    while(!(m&1024)){m<<=1;--exp;}
    return value(sign|(exp<<23)|((m&1023)<<13));
}
inline uint16_t tohalf(float x) {
#ifdef G53EX_SYCL
    return storage_half(x);
#else
    _Float16 h = (_Float16)x; uint16_t u; memcpy(&u,&h,2); return u;
#endif
}
// NVIDIA FP32 units return one canonical NaN (0x7fffffff) whatever the inputs; x86, Arm and Intel GPUs return
// 0x7fc00000 or propagate payloads. Every op result passes through here, so canonicalise NaN here too (measured:
// the GB10 dumps' only difference from this path was that payload, case 7, sigma 1000).
inline float ftz(float x) { uint32_t u=bits(x); return (u&0x7f800000)==0 ? value(u&0x80000000) : (u&0x7fffffff)>0x7f800000 ? value(0x7fffffffu) : x; }
inline float add(float a,float b) { return ftz(ftz(a)+ftz(b)); }
inline float mul(float a,float b) { return ftz(ftz(a)*ftz(b)); }
inline float fma(float a,float b,float c) {
#ifdef G53EX_SYCL
    return ftz(sycl::fma(ftz(a),ftz(b),ftz(c)));
#else
    return ftz(std::fma(ftz(a),ftz(b),ftz(c)));
#endif
}
inline float rcp(const mufu_tables &t,float x) {
    // Exact closed-form power-of-two subset, also permits table-free zero tests.
    uint32_t u=bits(x), e=(u>>23)&255;
    if (!(u&0x7fffff) && e>0 && e<255) return value((u&0x80000000) | (e<254 ? (254-e)<<23 : 0));
    return value(mufu_rcp_bits(&t,u));
}
inline float silu(float g,float u,const mufu_tables &t) {
    g=add(g,0.f); u=add(u,0.f); // fused CUDA bias placeholders (SPEC §5)
    float e=value(mufu_ex2_bits(&t,bits(mul(g,value(0xbfb8aa3b)))));
    return mul(mul(g,rcp(t,add(1.f,e))),u);
}
// roundf(v) followed by CUDA's saturating F2I and low-byte store. Avoid fp64 on Arc.
inline int roundaway(float v) {
    uint32_t u=bits(v), a=u&0x7fffffff; int e=int(a>>23)-127;
    if (a>0x7f800000) return 0;
    if (e>=31) return (u>>31) ? INT32_MIN : INT32_MAX;
    if (e< -1) return 0;
    uint32_t m=(a&0x7fffff)|0x800000, n;
    if (e>=23) n=m<<(e-23);
    else { int s=23-e; n=(m>>s)+((m&((1u<<s)-1)) >= (1u<<(s-1))); }
    return (u>>31) ? -int(n) : int(n);
}
struct Q8 { int8_t q[32]; uint16_t d,s; };
inline void quant(const float *x,Q8 &q,const mufu_tables &t) {
    float a[32],s[32];
    for(int i=0;i<32;i++) { a[i]=ftz(value(bits(x[i])&0x7fffffff)); s[i]=x[i]; }
    for(int off=16;off;off>>=1) for(int i=0;i<off;i++) {
        a[i]=a[i]>a[i+off]?a[i]:a[i+off]; s[i]=add(s[i],s[i+off]);
    }
    float d=mul(a[0],value(0x3c010204)), r=a[0]==0 ? 0 : rcp(t,d);
    for(int i=0;i<32;i++) q.q[i]=int8_t(uint8_t(roundaway(mul(r,x[i]))));
    q.d=tohalf(d); q.s=tohalf(s[0]);
}
inline uint16_t u16(const uint8_t*p) { return uint16_t(p[0]) | uint16_t(p[1])<<8; }
inline uint32_t u32(const uint8_t*p) { return uint32_t(u16(p)) | uint32_t(u16(p+2))<<16; }
inline int dot4(uint32_t a,const int8_t*b,int s) {
    for(int i=0;i<4;i++) s+=int(int8_t(a>>(8*i)))*int(b[i]);
    return s;
}
inline uint32_t signed4(uint32_t g,unsigned s) {
    uint32_t r=0; for(int i=0;i<4;i++) { int v=(g>>(8*i))&255; if(s&(1u<<i))v=-v; r|=uint32_t(uint8_t(v))<<(8*i); } return r;
}
inline unsigned signs(unsigned s) { s&=127; unsigned p=s; p^=p>>4;p^=p>>2;p^=p>>1; return s|((p&1)<<7); }
inline int blockbytes(int t) { return t==17?74:t==18?98:t==23?136:0; }
// Address a field in raw ggml blocks or g53x's existing per-row field runs.
inline const uint8_t *field(const uint8_t *r,int nb,int b,int sz,int off,int pre,int fs,bool packed) {
    return packed?r+nb*pre+b*fs:r+b*sz+off;
}
struct Dot { float d; int s; };
// One CUDA vec_dot invocation: 32 values, iqs=2*j (IQ2/3), 4*j (IQ4).
inline Dot dot(int type,const uint8_t*r,int nb,int b,int j,const Q8 &y,bool packed,
               const uint64_t *g2,const uint32_t*g3,const int8_t*k4) {
    int sz=blockbytes(type),sum=0;
    float dw=half(u16(field(r,nb,b,sz,0,type==17?72:type==18?96:134,2,packed)));
    if(type==17) {
        const uint8_t *qs=field(r,nb,b,sz,2,0,64,packed)+8*j;
        unsigned sc=field(r,nb,b,sz,66,64,8,packed)[j]; int s[2]={0,0};
        for(int l=0;l<4;l++) {
            unsigned q=u16(qs+2*l),sg=signs(q>>9); uint64_t g=g2[q&511];
            s[l/2]=dot4(signed4(uint32_t(g),sg),y.q+8*l,s[l/2]);
            s[l/2]=dot4(signed4(uint32_t(g>>32),sg>>4),y.q+8*l+4,s[l/2]);
        }
        sum=(s[0]*int(sc&15)+s[1]*int(sc>>4)+(s[0]+s[1])/2)/4;
    } else if(type==18) {
        const uint8_t *qs=field(r,nb,b,sz,2,0,64,packed)+8*j;
        unsigned aux=u32(field(r,nb,b,sz,66,64,32,packed)+4*j);
        for(int l=0;l<4;l++) {
            unsigned sg=signs(aux>>(7*l));
            sum=dot4(signed4(g3[qs[2*l]],sg),y.q+8*l,sum);
            sum=dot4(signed4(g3[qs[2*l+1]],sg>>4),y.q+8*l+4,sum);
        }
        sum=(int(aux>>28)*sum+sum/2)/2;
    } else {
        const uint8_t *qs=field(r,nb,b,sz,8,0,128,packed)+16*j;
        for(int i=0;i<16;i++) sum+=int(k4[qs[i]&15])*y.q[i]+int(k4[qs[i]>>4])*y.q[i+16];
        unsigned lo=field(r,nb,b,sz,4,128,4,packed)[j/2];
        unsigned hi=u16(field(r,nb,b,sz,2,132,2,packed));
        sum*=int(((lo>>(4*(j&1)))&15)|(((hi>>(2*j))&3)<<4))-32;
    }
    return {mul(dw,half(y.d)),sum};
}
// 128 virtual CUDA threads: kbx=t/8+16*i; identical for the three IQ types.
inline float row(int type,const uint8_t*r,int nb,const Q8*y,bool packed,
                 const uint64_t*g2,const uint32_t*g3,const int8_t*k4) {
    float s[32];
    for(int l=0;l<32;l++) {
        float w[4];
        for(int warp=0;warp<4;warp++) {
            int t=32*warp+l; float a=0;
            for(int b=t/8;b<nb;b+=16) { Dot d=dot(type,r,nb,b,t%8,y[b*8+t%8],packed,g2,g3,k4); a=fma(d.d,float(d.s),a); }
            w[warp]=a;
        }
        s[l]=add(add(add(w[0],w[1]),w[2]),w[3]);
    }
    for(int off=16;off;off>>=1) for(int i=0;i<off;i++) s[i]=add(s[i],s[i+off]);
    return s[0];
}
} // namespace g53exact
#ifdef __clang__
#pragma float_control(pop)
#endif
