// k3-pfdump: decode a chat prompt greedily and dump, per decode step and layer, the speculative prefetch
// predictor's scores, the real router's selection scores and the real selected expert ids, for an offline
// evaluation of prediction precision (kimi-k3-spark expert store).
//
// Output file records: int32 step, int32 layer, int32 kind, int32 n, then n elements
//   kind 1: moe_prefetch_probs (f32)   kind 2: ffn_moe_probs_biased (f32)   kind 3: ffn_moe_topk (i32)
//   kind 4: ffn_moe_probs (f32)        kind 5: generated token id (i32, n = 1, layer = -1)
// Usage: llama-k3-pfdump <common args> -p "<user prompt>" -n <decode tokens> --out <file>

#include "trace.h"
#ifndef PFDUMP_CALLBACK_TEST
#include "arg.h"
#include "common.h"
#include "chat.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"
#endif

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    FILE * f = nullptr;
    int step = -1; // -1 during the prefill: nothing is recorded
    std::vector<uint8_t> buf;
    trace_layout layout;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    int il = -1;
    const int kind = kind_of(t->name, il);
    const bool eligible = kind && st->layout.contains(il);
    if (ask) {
        return st->step >= 0 && eligible;
    }
    if (st->step < 0 || !eligible) {
        return true;
    }
    const int64_t n = ggml_nelements(t);
    const bool ids = kind == 3;
    if (n != (ids ? st->layout.topk : st->layout.experts) ||
        t->type != (ids ? GGML_TYPE_I32 : GGML_TYPE_F32) || !ggml_is_contiguous(t)) {
        LOG_ERR("%s: unexpected tensor %s type %s\n", __func__, t->name, ggml_type_name(t->type));
        throw std::runtime_error("unexpected decode routing tensor");
    }
    st->buf.resize((size_t) n * 4);
    ggml_backend_tensor_get(t, st->buf.data(), 0, st->buf.size());
    write_record(st->f, st->step, il, kind, (int) n, st->buf.data());
    return true;
}

#ifndef PFDUMP_CALLBACK_TEST
// Metadata uses the architecture namespace; block_count includes MTP blocks.
static int meta_int(const llama_model * model, const std::string & key, int fallback = -1) {
    char value[64];
    if (llama_model_meta_val_str(model, key.c_str(), value, sizeof(value)) < 0) return fallback;
    char * end;
    long n = strtol(value, &end, 10);
    if (*end || n < 0 || n > INT32_MAX) throw std::runtime_error("invalid model metadata: " + key);
    return (int) n;
}
int main(int argc, char ** argv) try {
    if (argc == 3 && strcmp(argv[1], "--check") == 0) return check_trace(argv[2]);
    std::string out_path = "pfdump.bin";
    bool ignore_eos = false;
    std::vector<char *> args;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--ignore-eos") == 0) {
            ignore_eos = true;
            continue;
        }
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
            continue;
        }
        args.push_back(argv[i]);
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    dump_state st;
    st.f = fopen(out_path.c_str(), "wb");
    if (st.f == nullptr) {
        LOG_ERR("cannot open %s\n", out_path.c_str());
        return 1;
    }
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }
    char arch[128];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0)
        throw std::runtime_error("missing architecture");
    const std::string prefix = std::string(arch) + ".";
    st.layout = {meta_int(model, prefix + "block_count") - meta_int(model, prefix + "nextn_predict_layers", 0),
                 meta_int(model, prefix + "expert_count"), meta_int(model, prefix + "expert_used_count"),
                 meta_int(model, prefix + "leading_dense_block_count", 0)};
    if (st.layout.layers <= st.layout.first || st.layout.experts <= 0 ||
        st.layout.topk <= 0 || st.layout.topk > st.layout.experts)
        throw std::runtime_error("invalid MoE model dimensions");
    LOG_INF("routing: layers %d..%d, %d experts, top-%d (dense/MTP excluded)\n",
            st.layout.first, st.layout.layers - 1, st.layout.experts, st.layout.topk);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the chat template, like the server with --jinja
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs in;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = params.prompt;
    in.messages.push_back(msg);
    in.add_generation_prompt = true;
    in.use_jinja = true;
    std::string text;
    for (const auto & part : common_chat_templates_apply(tmpls.get(), in).prompt.parts) {
        text += part.text;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
    LOG_INF("prompt: %zu tokens, decoding %d\n", tokens.size(), params.n_predict);

    st.step = -1;
    {
        common_batch batch = common_batch_get_one(ctx, tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("prefill failed\n");
            return 1;
        }
    }
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string out_text;
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < params.n_predict; i++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (!ignore_eos && llama_vocab_is_eog(vocab, tok)) {
            LOG_INF("eog after %d tokens\n", i);
            break;
        }
        out_text += common_token_to_piece(ctx, tok);
        st.step = i;
        const int32_t hdr[4] = { i, -1, 5, 1 };
        fwrite(hdr, sizeof(hdr), 1, st.f);
        fwrite(&tok, 4, 1, st.f);
        std::vector<llama_token> one = { tok };
        common_batch batch = common_batch_get_one(ctx, one);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("decode failed at step %d\n", i);
            return 1;
        }
    }
    const double dt = (ggml_time_us() - t0) / 1e6;
    LOG_INF("decode: %d steps in %.1f s (%.2f tok/s, with the dump callback)\n", st.step + 1, dt, (st.step + 1) / dt);
    if (fclose(st.f) != 0) throw std::runtime_error("trace close failed");
    FILE * ft = fopen((out_path + ".txt").c_str(), "wb");
    if (ft != nullptr) {
        fwrite(out_text.data(), 1, out_text.size(), ft);
        fclose(ft);
    }
    llama_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
 catch (const std::exception & e) {
    fprintf(stderr, "pfdump: %s\n", e.what());
    return 1;
}
#endif
