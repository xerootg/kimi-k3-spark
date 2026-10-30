// Direct stage checks, without serveidx's documented zero-row wire sentinels.
// Include the production kernels and dispatch, so the test cannot silently drift.
#define main g53x_main
#include "g53x.cpp"
#undef main
#include <fstream>
#include <stdexcept>

static void need(bool ok,const std::string&msg) {if(!ok)throw std::runtime_error(msg);}
static std::vector<uint8_t> read_blob(const std::string&p) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"open "+p);
    auto n=f.tellg();need(n>0,"empty "+p);std::vector<uint8_t>b(static_cast<size_t>(n));
    f.seekg(0);need(bool(f.read(reinterpret_cast<char*>(b.data()),b.size())),"read "+p);return b;
}
// Compare the actual prepass output before scratch can be reused. The CPU
// oracle uses the same verified tables, without GPU half conversion/lowering.
static void check_prepared(sycl::queue&q,const float*x,size_t blocks,const std::string&where) {
    std::vector<g53exact::Q8> got(blocks);std::vector<float> scales(blocks);
    q.memcpy(got.data(),exact_device.prepared,blocks*sizeof(got[0]));
    q.memcpy(scales.data(),exact_device.scales,blocks*sizeof(float)).wait_and_throw();
    for(size_t b=0;b<blocks;b++) {
        g53exact::Q8 want;g53exact::quant(x+32*b,want,exact_device.mt);
        auto a=reinterpret_cast<const uint8_t*>(&got[b]);
        auto w=reinterpret_cast<const uint8_t*>(&want);
        for(size_t j=0;j<sizeof(want);j++)if(a[j]!=w[j]) {
            char msg[320];snprintf(msg,sizeof msg,"%s first prepare block=%zu field=%s byte=%zu got=%02x want=%02x",
                where.c_str(),b,j<32?"q8":j<34?"half(d)":"half(sum)",j,a[j],w[j]);
            throw std::runtime_error(msg);
        }
        if(g53exact::bits(scales[b])!=g53exact::bits(g53exact::half(want.d))) {
            char msg[320];snprintf(msg,sizeof msg,"%s first prepare scale block=%zu got=%08x want=%08x",
                where.c_str(),b,g53exact::bits(scales[b]),g53exact::bits(g53exact::half(want.d)));
            throw std::runtime_error(msg);
        }
    }
}
static void preparation_edges(sycl::queue&q) {
    constexpr size_t blocks=256;
    auto x=sycl::malloc_host<float>(32*blocks,q);need(x,"edge input allocation");
    exact_device.prepared=sycl::malloc_device<g53exact::Q8>(blocks,q);
    exact_device.scales=sycl::malloc_device<float>(blocks,q);
    need(exact_device.prepared&&exact_device.scales,"edge scratch allocation");
    exact_device.capacity=blocks;
    // Half subnormal scales, rounding ties, cancellation, signed zeros, FP32
    // subnormals/FTZ, overflow, infinity, NaNs and non-power-of-two RCP inputs.
    const uint32_t special[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
        0x7f800000,0xff800000,0x7fc12345,0x3f800001};
    for(size_t b=0;b<blocks;b++)for(int j=0;j<32;j++) {
        float v=std::ldexp(float(127+(j%7)),int(b%50)-30);
        if(j&1)v=-v;
        if(b<10)v=g53exact::value(special[b]);
        x[32*b+j]=v;
    }
    parallel_prepare(q,blocks,x,nullptr,nullptr,nullptr,exact_device.prepared,exact_device.scales,exact_device.mt);
    check_prepared(q,x,blocks,"synthetic");
    sycl::free(x,q);
    printf("prepare: 256 edge blocks, all q8/half(d)/half(sum)/scale bits PASS\n");fflush(stdout);
}
int main(int argc,char**argv)try {
    need(argc==4,"usage: test-parallel-sycl MUFU RECORDS DUMPS");
    setenv("G53X_EXACT","2",1);setenv("G53X_MUFU",argv[1],1);
    sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order()};exact_init(q);
    preparation_edges(q);
    auto x=sycl::malloc_host<float>(D,q),gu=sycl::malloc_host<float>(2*NE*F,q);
    auto h=sycl::malloc_device<float>(NE*F,q),dn=sycl::malloc_host<float>(NE*D,q);
    auto W=sycl::malloc_host<const uint8_t*>(3*NE,q);
    auto xq=sycl::malloc_device<int8_t>(D,q),hq=sycl::malloc_device<int8_t>(NE*F,q);
    auto xd=sycl::malloc_device<float>(D/32,q),xs=sycl::malloc_device<float>(D/32,q);
    auto hd=sycl::malloc_device<float>(NE*F/32,q),hs=sycl::malloc_device<float>(NE*F/32,q);
    need(x&&gu&&h&&dn&&W&&xq&&hq&&xd&&xs&&hd&&hs,"USM allocation");
    size_t checked=0,bad=0;int layers[]={3,8,75},tgs[]={17,18,17},tds[]={18,23,23};
    for(int li=0;li<3;li++) {
        std::string label="L"+std::to_string(layers[li]),dir=std::string(argv[3])+"/"+label;
        auto raw=read_blob(std::string(argv[2])+"/"+label+".records");
        size_t gs=row_bytes(tgs[li],D)*F,ds=row_bytes(tds[li],F)*D,rec=2*gs+ds;
        need(raw.size()==NE*rec,"record size "+label);
        auto arena=sycl::malloc_host<uint8_t>(raw.size(),q);need(arena,"weight allocation");
        for(int e=0;e<NE;e++) {
            repack(tgs[li],raw.data()+e*rec,arena+e*rec,F,D);
            repack(tgs[li],raw.data()+e*rec+gs,arena+e*rec+gs,F,D);
            repack(tds[li],raw.data()+e*rec+2*gs,arena+e*rec+2*gs,D,F);
        }
        auto xb=read_blob(dir+"/x.bin"),ids=read_blob(dir+"/ids.bin");
        need(xb.size()%(D*4)==0,"x size");size_t cases=xb.size()/(D*4);
        need(ids.size()==cases*NE*4,"ids size");
        const char*names[]={"g","u","h","dn"};std::vector<uint8_t>want[4];
        for(int s=0;s<4;s++){want[s]=read_blob(dir+"/"+names[s]+".bin");need(want[s].size()==cases*NE*(s==3?D:F)*4,"stage size");}
        auto gate=pick_gu(tgs[li],tgs[li],D,F);auto down=pick_dn(tds[li],D,F);
        std::vector<float>hosth(NE*F),got[4];got[0].resize(NE*F);got[1].resize(NE*F);
        for(int fused:{0,1})for(int stream:{0,1}) {
            size_t diffs[4]={};
            for(size_t c=0;c<cases;c++) {
                memcpy(x,xb.data()+c*D*4,D*4);
                for(int e=0;e<NE;e++) {
                    int32_t id;memcpy(&id,ids.data()+(c*NE+e)*4,4);need(id>=0&&id<NE,"id range");
                    W[3*e]=arena+id*rec;W[3*e+1]=W[3*e]+gs;W[3*e+2]=W[3*e]+2*gs;
                }
                if(!fused)quantize(q,x,xq,xd,xs,D/32);
                gate(q,W,NE,fused?x:nullptr,xq,xd,xs,h,gu);
                std::string context=label+" fused="+std::to_string(fused)+" stream="+std::to_string(stream)+" case="+std::to_string(c);
                check_prepared(q,x,D/32,context+" x");
                q.memcpy(hosth.data(),h,NE*F*4).wait_and_throw();
                if(!fused)quantize(q,h,hq,hd,hs,NE*F/32);
                if(stream)for(int e=0;e<NE;e++) {
                    down(q,W+3*e,1,fused?h+e*F:nullptr,hq+e*F,hd+e*F/32,hs+e*F/32,dn+e*D);
                    check_prepared(q,hosth.data()+e*F,F/32,context+" h expert="+std::to_string(e));
                } else {
                    down(q,W,NE,fused?h:nullptr,hq,hd,hs,dn);
                    check_prepared(q,hosth.data(),NE*F/32,context+" h");
                }
                for(int e=0;e<NE;e++)for(int j=0;j<F;j++){got[0][e*F+j]=gu[2*e*F+j];got[1][e*F+j]=gu[(2*e+1)*F+j];}
                const float*stages[]={got[0].data(),got[1].data(),hosth.data(),dn};
                for(int s=0;s<4;s++)for(size_t j=0,n=NE*(s==3?D:F);j<n;j++) {
                    uint32_t word;memcpy(&word,want[s].data()+(c*n+j)*4,4);
                    if(g53exact::bits(stages[s][j])!=word) {
                        if(diffs[s]==0)fprintf(stderr,"%s fused=%d stream=%d first %s case=%zu index=%zu got=%08x want=%08x\n",label.c_str(),fused,stream,names[s],c,j,g53exact::bits(stages[s][j]),word);
                        diffs[s]++;
                    }
                }
                checked++;
            }
            printf("%s fused=%d stream=%d cases=%zu differing words g/u/h/dn=%zu/%zu/%zu/%zu\n",label.c_str(),fused,stream,cases,diffs[0],diffs[1],diffs[2],diffs[3]);fflush(stdout);
            for(auto d:diffs)bad+=d;
        }
        sycl::free(arena,q);
    }
    printf("%s: %zu SYCL case/configurations, differing stage words=%zu\n",bad?"FAIL":"PASS",checked,bad);
    return bad?1:0;
} catch(const std::exception&e){fprintf(stderr,"error: %s\n",e.what());return 1;}
