// Public ggml API probe; no model, llama context, server, or private CUDA headers.
// Evidence: glm-op-profile.log: heads=64, KV heads=1, kv_lora=512,
// n_rot=64, unabsorbed key dim=256, freq_scale=1, yarn_log_mul=0.
// glm-dsa.cpp absorbs Q to 576; llama-graph.cpp aliases V to K[:512].
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * s) { if (!ok) throw std::runtime_error(s); }
static uint64_t hash(const void * p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    auto b = static_cast<const unsigned char *>(p);
    for (size_t i=0;i<n;++i) { h ^= b[i]; h *= UINT64_C(1099511628211); }
    return h;
}
// Counter-based input: the prefix is identical regardless of allocation extent.
static float value(uint32_t i, uint32_t salt) {
    uint32_t x = i ^ salt; x ^= x>>16; x *= 0x7feb352du;
    x ^= x>>15; x *= 0x846ca68bu; x ^= x>>16;
    return (float(int(x & 65535)-32768))/32768.0f;
}
struct Result { std::vector<float> out; uint64_t membership; };
static Result run(ggml_backend_t backend, int E, int valid, int limit, const char * mode) {
    ggml_init_params ip = {16*1024*1024, nullptr, true};
    ggml_context * ctx = ggml_init(ip); require(ctx, "ggml_init failed");
    auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 576, 1, 64, 1);
    auto k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 576, E, 1, 1);
    auto v = ggml_view_4d(ctx, k, 512, E, 1, 1, k->nb[1], k->nb[2], k->nb[3], 0);
    auto mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, E, 1, 1, 1);
    auto scores = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, E, 1, 1, 1);
    auto ids = ggml_cont(ctx, ggml_top_k(ctx, scores, limit));
    // Exact singleton DSA mask construction from llama-graph.cpp.
    auto all = ggml_fill(ctx, mask, -INFINITY);
    all = ggml_view_4d(ctx, all, 1, E, 1, 1, all->nb[0], all->nb[1], all->nb[2], 0);
    auto zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, limit, 1, 1), 0);
    auto selected = ggml_set_rows(ctx, all, zeros, ids);
    selected = ggml_view_4d(ctx, selected, E, 1, 1, 1,
                            selected->nb[2], selected->nb[3], selected->nb[3], 0);
    auto final_mask = ggml_add(ctx, selected, mask);
    auto out = ggml_flash_attn_ext(ctx, q, k, v, final_mask, 1.0f/16.0f, 0, 0);
    ggml_flash_attn_ext_set_n_kv_max(out, limit);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_set_name(out, "fageom_flash_attn"); ggml_set_output(out);
    ggml_set_output(ids); ggml_set_output(final_mask);
    auto graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, ids);
    ggml_build_forward_expand(graph, final_mask);
    ggml_build_forward_expand(graph, out);
    require(ggml_backend_supports_op(backend, out), "CUDA backend does not support this FA op");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "backend allocation failed");
    std::vector<float> qbytes(576*64), scorebytes(E, -INFINITY);
    std::vector<ggml_fp16_t> kbytes(size_t(576)*E), maskbytes(E, ggml_fp32_to_fp16(-INFINITY));
    for (size_t i=0;i<qbytes.size();++i) qbytes[i] = value(uint32_t(i), 0x12345678);
    for (size_t i=0;i<kbytes.size();++i) kbytes[i] = ggml_fp32_to_fp16(value(uint32_t(i), 0x87654321));
    for (int i=0;i<valid;++i) {
        maskbytes[i] = ggml_fp32_to_fp16(0);
        // Unique finite scores: padding ties may change raw IDs, never valid membership.
        scorebytes[i] = float((uint32_t(i)*73u)%1024u) + float(i)/1024.0f;
    }
    ggml_backend_tensor_set(q, qbytes.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kbytes.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(mask, maskbytes.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(scores, scorebytes.data(), 0, ggml_nbytes(scores));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    ggml_backend_synchronize(backend);
    std::vector<int32_t> idbytes(limit), members;
    std::vector<ggml_fp16_t> actual_mask(E);
    Result result; result.out.resize(512*64);
    ggml_backend_tensor_get(ids, idbytes.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_get(final_mask, actual_mask.data(), 0, ggml_nbytes(final_mask));
    ggml_backend_tensor_get(out, result.out.data(), 0, ggml_nbytes(out));
    for (auto id : idbytes) {
        require(id>=0 && id<E, "invalid TOP_K id");
        if (id<valid) members.push_back(id);
    }
    std::sort(members.begin(), members.end());
    require(int(members.size())==valid, "TOP_K omitted valid keys");
    for (int i=0;i<valid;++i) require(members[i]==i, "duplicate/missing valid key");
    for (int i=0;i<E;++i) require(i<valid ? ggml_fp16_to_fp32(actual_mask[i])==0 :
        ggml_fp16_to_fp32(actual_mask[i])==-INFINITY, "final mask membership changed");
    for (float x : result.out) require(std::isfinite(x), "nonfinite FA output");
    result.membership = hash(members.data(), members.size()*sizeof(int32_t));
    auto sorted = idbytes; std::sort(sorted.begin(), sorted.end());
    // Prediction for supplied fattn.cu/.cuh on sm_121, not a runtime kernel trace.
    // 576/512 + GQA64 selects ncols2=16, ncols1=1; sparse iff E>=max(4096,2*K).
    std::printf("FA E=%d valid=%d K=%d mode=%s predicted=mma_f16_%s_tile1x16 effective_KV=%d "
        "ids=%016llx sorted_ids=%016llx valid_members=%016llx output=%016llx\n",
        E, valid, limit, mode, E>=std::max(4096,2*limit)?"sparse":"dense",
        E>=std::max(4096,2*limit)?limit:E,
        (unsigned long long)hash(idbytes.data(), idbytes.size()*4),
        (unsigned long long)hash(sorted.data(), sorted.size()*4),
        (unsigned long long)result.membership,
        (unsigned long long)hash(result.out.data(), result.out.size()*4));
    // Repeat same graph/bytes to distinguish geometry from run-to-run instability.
    require(ggml_backend_graph_compute(backend, graph)==GGML_STATUS_SUCCESS, "repeat failed");
    ggml_backend_synchronize(backend);
    std::vector<float> again(result.out.size());
    ggml_backend_tensor_get(out, again.data(), 0, ggml_nbytes(out));
    require(std::memcmp(again.data(),result.out.data(),ggml_nbytes(out))==0, "same geometry is nondeterministic");
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return result;
}
int main() try {
    auto backend = ggml_backend_cuda_init(0); require(backend, "CUDA device 0 unavailable");
    std::printf("FAGEOM backend=%s seed=12345678/87654321 Q=[576,1,64,1] K=f16[576,E,1,1] V=K[:512] scale=0.0625\n", ggml_backend_name(backend));
    std::puts("Prediction assumes supplied production dispatcher on sm_121; partition/fixup depends on SM occupancy (not traced).");
    std::map<int, Result> baseline; int configs=0, differences=0;
    for (int valid : {10,138,300}) for (int E : {256,512,2048,4096,32768,65536}) {
        if (valid>E) { std::printf("SKIP E=%d valid=%d reason=valid_exceeds_extent\n",E,valid); continue; }
        for (int mode=0; mode<2; ++mode) {
            int limit = mode==0 ? std::min(E,2048) : ((valid+255)/256)*256;
            auto r = run(backend,E,valid,limit,mode==0?"production":"valid_pad256");
            if (!baseline.count(valid)) baseline.emplace(valid,r);
            auto & b=baseline.at(valid);
            require(b.membership==r.membership,"membership differs across geometry");
            bool equal = std::memcmp(b.out.data(),r.out.data(),r.out.size()*4)==0;
            double maxdiff=0; for(size_t i=0;i<r.out.size();++i) maxdiff=std::max(maxdiff,std::abs(double(r.out[i])-b.out[i]));
            std::printf("COMPARE valid=%d E=%d K=%d baseline_E=%d equal=%d max_abs=%.9g\n",valid,E,limit,valid<=256?256:512,equal,maxdiff);
            ++configs; differences += !equal;
        }
    }
    std::printf("SUMMARY configs=%d differing=%d membership=identical repeats=identical verdict=%s\n", configs,differences,
        differences?"GEOMETRY_SENSITIVE":"NO_DIFFERENCE_FOR_TESTED_INPUTS");
    ggml_backend_free(backend);
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"FAGEOM ERROR: %s\n",e.what()); return 1; }
