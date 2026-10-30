// g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -Iref -Itools/bitexact tools/glm53/g53ref_exact.cpp -o tools/glm53/g53ref_exact
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include "g53exact.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace g53exact;
static constexpr int D=6144,F=2048,K=8;
static void require(bool b,const std::string&s) { if(!b)throw std::runtime_error(s); }
struct Map {
    void*p=MAP_FAILED; size_t n=0;
    explicit Map(const std::string&s,size_t expected=0) {
        int fd=open(s.c_str(),O_RDONLY); require(fd>=0,"open "+s);
        struct stat st{}; if(fstat(fd,&st)) {close(fd);throw std::runtime_error("stat "+s);}
        n=st.st_size;
        if(!n || (expected && n!=expected)) {close(fd);throw std::runtime_error("size "+s);}
        p=mmap(nullptr,n,PROT_READ,MAP_PRIVATE,fd,0);close(fd);require(p!=MAP_FAILED,"mmap "+s);
    }
    ~Map(){if(p!=MAP_FAILED)munmap(p,n);}
    Map(const Map&)=delete; Map&operator=(const Map&)=delete;
};
struct Tables {
    std::vector<std::unique_ptr<Map>> m; mufu_tables t{};
    explicit Tables(const std::string&dir) {
        if(dir=="-")return; // only closed-form selftests; never substitutes libm for MUFU
        const char*names[]={"rcp.bin","ex2p.bin","ex2n.bin"};
        size_t sizes[]={size_t(1)<<25,size_t(31)<<25,size_t(33)<<25};
        for(int i=0;i<3;i++)m.emplace_back(new Map(dir+"/"+names[i],sizes[i]));
        t.rcp=(const uint32_t*)m[0]->p;t.ex2p=(const uint32_t*)m[1]->p;t.ex2n=(const uint32_t*)m[2]->p;
    }
};
static void write(const std::string&p,const void*b,size_t n) {
    std::ofstream f(p,std::ios::binary);require(bool(f.write((const char*)b,n)),"write "+p);
}
struct Weights {
    int tg,tu,td,E;size_t gs,us,ds,rec=0;
    std::vector<std::unique_ptr<Map>> m;
    Weights(bool records,const std::string&p,int e,int a,int b,int c):tg(a),tu(b),td(c),E(e) {
        for(int t:{a,b,c})require(t==17||t==18||t==23,"exact supports IQ2_XS/IQ3_XXS/IQ4_XS; unsupported type "+std::to_string(t));
        require(e>=8,"need >=8 experts"); gs=size_t(D/256)*F*blockbytes(a);us=size_t(D/256)*F*blockbytes(b);ds=size_t(F/256)*D*blockbytes(c);
        if(records) {m.emplace_back(new Map(p));require(m[0]->n%size_t(e)==0,"record alignment");rec=m[0]->n/e;require(rec>=gs+us+ds,"record too short");}
        else {const char*pr[]={"gate","up","down"};size_t sz[]={gs,us,ds};for(int i=0;i<3;i++)m.emplace_back(new Map(p+"_"+pr[i]+".bin",sz[i]*e));}
    }
    const uint8_t*ptr(int e,int proj)const {
        if(rec)return (const uint8_t*)m[0]->p+rec*e+(proj==0?0:proj==1?gs:gs+us);
        return (const uint8_t*)m[proj]->p+size_t(e)*(proj==0?gs:proj==1?us:ds);
    }
};
static void compute(const Weights&w,const float*x,const int32_t*ids,const mufu_tables&t,
                    std::array<std::vector<float>,4>&o,bool gate_only=false) {
    std::vector<Q8>qx(D/32),qh(F/32);
    for(int b=0;b<D/32;b++)quant(x+32*b,qx[b],t);
    for(int e=0;e<K;e++) {
        require(ids[e]>=0&&ids[e]<w.E,"id out of range");
        for(int j=0;j<F;j++) {
            o[0][e*F+j]=row(w.tg,w.ptr(ids[e],0)+size_t(j)*(D/256)*blockbytes(w.tg),D/256,qx.data(),false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
            o[1][e*F+j]=row(w.tu,w.ptr(ids[e],1)+size_t(j)*(D/256)*blockbytes(w.tu),D/256,qx.data(),false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
            if(!gate_only)o[2][e*F+j]=silu(o[0][e*F+j],o[1][e*F+j],t);
        }
        if(gate_only)continue;
        for(int b=0;b<F/32;b++)quant(o[2].data()+e*F+32*b,qh[b],t);
        for(int j=0;j<D;j++)o[3][e*D+j]=row(w.td,w.ptr(ids[e],2)+size_t(j)*(F/256)*blockbytes(w.td),F/256,qh.data(),false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    }
}
static std::array<std::vector<float>,4> outputs() {return {std::vector<float>(K*F),std::vector<float>(K*F),std::vector<float>(K*F),std::vector<float>(K*D)};}
static void unit() {
    mufu_tables t{}; float x[32];Q8 q;
    for(float &v:x)v=127;
    quant(x,q,t);require(q.d==tohalf(1)&&q.s==tohalf(4064),"q8 half scale/sum");for(auto v:q.q)require(v==127,"q8 integer");
    // Cancellation distinguishes CUDA butterfly from a sequential or subgroup-selected tree.
    for(float&v:x)v=0;
    x[0]=127;x[1]=0x1p-20f;x[16]=-127;quant(x,q,t);require(q.s==tohalf(0x1p-20f),"q8 float sum");
    require(roundaway(0.49999997f)==0&&roundaway(.5f)==1&&roundaway(-.5f)==-1,"roundaway tie");
    require(bits(mul(value(1),1.f))==0,"FTZ");
    require(bits(g53exact::fma(1.f,1.f,-1.f))==0,"fma zero");
    require(bits(silu(0,3,t))==0,"SwiGLU zero");
    // Deliberately odd IQ integer sums exercise truncation before conversion to float.
    std::vector<uint8_t>b(136); Q8 y{};y.d=tohalf(1);y.q[0]=1;
    uint16_t d=tohalf(1);memcpy(b.data(),&d,2);
    b[2]=1; // IQ2 grid 1, first element 43 (odd)
    auto z=dot(17,b.data(),1,0,0,y,false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    require(z.s==5,"IQ2 positive odd truncation");
    y.q[0]=-1;z=dot(17,b.data(),1,0,0,y,false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    require(z.s==-5,"IQ2 negative odd truncation");
    b[2]=5; // IQ3 grid 5, first element 62: (62/2)/2 = 15, not 15.5
    z=dot(18,b.data(),1,0,0,y,false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    require(z.s==-15,"IQ3 negative truncation");
    y.q[0]=1;z=dot(18,b.data(),1,0,0,y,false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
    require(z.s==15,"IQ3 positive truncation");
    // Exercise the same repacked field layout consumed by SYCL, with random grids,
    // signs, scales, q8 ints and half scales, at both production K lengths.
    std::mt19937 rng(91);
    for(int ty:{17,18,23}) for(int nb:{8,24}) {
        int sz=blockbytes(ty);std::vector<uint8_t>raw(nb*sz),packed(nb*sz);std::vector<Q8>ys(nb*8);
        for(auto &v:raw)v=uint8_t(rng());
        for(int k=0;k<nb;k++) {uint16_t d=tohalf(0.125f);memcpy(raw.data()+k*sz,&d,2);}
        for(auto &v:ys) {for(auto &q:v.q)q=int8_t(int(rng()%255)-127);v.d=tohalf(0.0625f);v.s=0;}
        const int off[3][4]={{2,66,0,0},{2,66,0,0},{8,4,2,0}};
        const int fs[3][4]={{64,8,2,0},{64,32,2,0},{128,4,2,2}};
        int ti=ty==17?0:ty==18?1:2,pre=0;
        for(int f=0;f<(ty==23?4:3);f++) {for(int k=0;k<nb;k++)memcpy(packed.data()+nb*pre+k*fs[ti][f],raw.data()+k*sz+off[ti][f],fs[ti][f]);pre+=fs[ti][f];}
        float a=row(ty,raw.data(),nb,ys.data(),false,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
        float c=row(ty,packed.data(),nb,ys.data(),true,iq2xs_grid,iq3xxs_grid,kvalues_iq4nl);
        require(bits(a)==bits(c),"raw vs SYCL packed row");
    }
    std::cout<<"synthetic: q8 half scale/sum, rounding, FTZ, FMA, SwiGLU zero, IQ signed truncation, raw/repacked K=2048/6144 PASS\n";
}
int main(int argc,char**argv)try {
    require(argc>=2,"usage: selftest DATA OUT | slices|records MUFU PATH N TG TU TD INPUT OUT");
    if(std::string(argv[1])=="selftest") {
        require(argc==4,"selftest DATA OUT");unit();std::filesystem::create_directories(argv[3]);
        int layer[]={3,8,75},tg[]={17,18,17},td[]={18,23,23};
        for(int l=0;l<3;l++) {
            Weights w(false,std::string(argv[2])+"/UD-Q2_K_XL_L"+std::to_string(layer[l]),8,tg[l],tg[l],td[l]);
            std::string out=std::string(argv[3])+"/L"+std::to_string(layer[l]);std::filesystem::create_directories(out);
            std::vector<float>x(2*D);for(int i=D;i<2*D;i++)x[i]=value((i&1?0x80000000:0)|1);
            int32_t ids[16];for(int i=0;i<16;i++)ids[i]=i%8;
            const char *kind="0 zeros\n1 signed denormals\n";write(out+"/kind.txt",kind,strlen(kind));
            write(out+"/x.bin",x.data(),x.size()*4);write(out+"/ids.bin",ids,sizeof ids);
            auto o=outputs();const char*names[]={"g","u","h","dn"};std::ofstream f[4];for(int s=0;s<4;s++)f[s].open(out+"/"+names[s]+".bin",std::ios::binary);
            for(int c=0;c<2;c++) {compute(w,x.data()+c*D,ids+c*K,{},o);for(int s=0;s<4;s++) {for(float v:o[s])require(bits(v)==0,"zero/denormal stage "+std::string(names[s]));f[s].write((char*)o[s].data(),o[s].size()*4);}}
            // Nonzero extracted weights through all gate/up rows with a table-free power-of-two quantizer scale.
            std::fill(x.begin(),x.end(),127.f);compute(w,x.data(),ids,{},o,true);
            size_t nz=0;for(int s=0;s<2;s++)for(float v:o[s]) {require(std::isfinite(v),"finite gate/up");nz+=v!=0;}
            require(nz>0,"nonzero gate/up");
            std::cout<<"L"<<layer[l]<<": zero + signed denormal g/u/h/dn PASS; nonzero gate/up "<<nz<<"/32768 finite nonzero\n";
        }
        std::cout<<"PASS (table-free subset only; random end-to-end requires GB10 MUFU tables and CUDA dumps)\n";return 0;
    }
    require(argc==10,"slices|records MUFU PATH N TG TU TD INPUT OUT");
    std::string mode=argv[1];require(mode=="slices"||mode=="records","mode");require(std::string(argv[2])!="-","full dump requires verified MUFU tables");
    Tables tables(argv[2]);Weights w(mode=="records",argv[3],std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]));
    std::string in=argv[8],out=argv[9];
    Map x(in+"/x.bin");require(x.n%(D*4)==0,"x size");size_t cases=x.n/(D*4);Map ids(in+"/ids.bin",cases*K*4);
    require(std::filesystem::weakly_canonical(in)!=std::filesystem::weakly_canonical(out),"input and output must differ");
    std::filesystem::create_directories(out);write(out+"/x.bin",x.p,x.n);write(out+"/ids.bin",ids.p,ids.n);
    const char*names[]={"g","u","h","dn"};std::ofstream f[4];for(int s=0;s<4;s++){f[s].open(out+"/"+names[s]+".bin",std::ios::binary);require(bool(f[s]),"output stage");}
    auto o=outputs();for(size_t c=0;c<cases;c++){compute(w,(const float*)x.p+c*D,(const int32_t*)ids.p+c*K,tables.t,o);for(int s=0;s<4;s++)require(bool(f[s].write((char*)o[s].data(),o[s].size()*4)),"write stage");}
    std::cout<<"wrote "<<cases<<" cases (8 experts each), g/u/h/dn\n";return 0;
} catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
