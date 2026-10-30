#include "trace.h"
#include <cassert>
#include <iostream>
#define PFDUMP_CALLBACK_TEST
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)
enum { GGML_TYPE_F32, GGML_TYPE_I32 };
struct ggml_tensor { const char * name; int type; int64_t n; const void * data; };
static int64_t ggml_nelements(ggml_tensor * t) { return t->n; }
static bool ggml_is_contiguous(ggml_tensor *) { return true; }
static const char * ggml_type_name(int) { return "test"; }
static void ggml_backend_tensor_get(ggml_tensor * t, void * dst, size_t, size_t n) { memcpy(dst,t->data,n); }
#include "k3-pfdump.cpp"
static std::vector<char> bytes(FILE * f) {
    fflush(f); long n=ftell(f); rewind(f); std::vector<char> b(n);
    assert(fread(b.data(),1,b.size(),f)==b.size()); return b;
}
int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--check") return check_trace(argv[2]);
    for (auto layout : {trace_layout{93,896,16,1}, trace_layout{78,256,8,3}}) {
        dump_state st; st.layout=layout; st.f=tmpfile(); FILE * old=tmpfile();
        std::vector<float> scores(layout.experts); std::iota(scores.begin(),scores.end(),0.f);
        std::vector<int32_t> ids(layout.topk);
        for (int j=0;j<layout.topk;++j) ids[j]=layout.experts-1-j;
        for (int step=-1;step<2;++step) {
            st.step=step;
            for (int il=layout.first;il<layout.layers;++il) {
                for (auto kind : {1,2,3,4,6,7,8,9}) {
                    const char * prefixes[]={"", "moe_prefetch_probs-", "ffn_moe_probs_biased-", "ffn_moe_topk-", "ffn_moe_probs-", "", "pfv2_probs-", "pfv3_probs-", "pfv4_probs-", "pfv5_probs-"};
                    std::string name=std::string(prefixes[kind])+std::to_string(il);
                    ggml_tensor t{name.c_str(), kind==3 ? GGML_TYPE_I32 : GGML_TYPE_F32,
                        kind==3 ? layout.topk : layout.experts, kind==3 ? (void*)ids.data() : (void*)scores.data()};
                    assert(cb_eval(&t,true,&st)==(step>=0)); cb_eval(&t,false,&st);
                    if (step>=0) { // Historical callback's exact native int32 header and payload.
                        int32_t h[]={step,il,kind,(int32_t)t.n};
                        fwrite(h,sizeof(h),1,old); fwrite(t.data,4,t.n,old);
                    }
                }
            }
        }
        auto output = bytes(st.f);
        assert(output==bytes(old));
        if (argc == 2 && layout.experts == 256) {
            FILE * saved=fopen(argv[1], "wb"); assert(saved);
            assert(fwrite(output.data(),1,output.size(),saved)==output.size()); fclose(saved);
        }
        fclose(st.f); fclose(old);
        st.step=0;
        for (int il : {layout.first-1, layout.layers}) {
            std::string name="ffn_moe_topk-"+std::to_string(il);
            ggml_tensor t{name.c_str(),GGML_TYPE_I32,layout.topk,ids.data()};
            assert(!cb_eval(&t,true,&st)); assert(cb_eval(&t,false,&st));
        }
        std::cout << "PASS " << layout.experts << " experts: callback stream equals legacy bytes; prefill/dense/MTP excluded\n";
    }
    int il; assert(!kind_of("moe_prefetch-4",il)); assert(!kind_of("moe_prefetch_self-4",il));
    assert(kind_of("moe_prefetch_probs-4",il)==1 && il==4);
    assert(!kind_of("ffn_moe_topk-3junk",il));
    std::cout << "PASS predictor lookup/slot tensors are not real IDs; destination layer preserved\n";
    return 0;
}
