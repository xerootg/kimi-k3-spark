#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// No architecture constants: the callback adapter supplies model dimensions.
struct trace_layout {
    int layers = 0, experts = 0, topk = 0, first = 0;
    bool contains(int il) const { return il >= first && il < layers; }
};
inline int kind_of(const char * name, int & il) {
    static const struct { const char * prefix; int kind; } kinds[] = {
        {"moe_prefetch_probs-", 1}, {"ffn_moe_probs_biased-", 2},
        {"ffn_moe_topk-", 3}, {"ffn_moe_probs-", 4},
        {"pfv2_probs-", 6}, {"pfv3_probs-", 7},
        {"pfv4_probs-", 8}, {"pfv5_probs-", 9},
    };
    for (const auto & k : kinds) {
        size_t n = strlen(k.prefix);
        if (strncmp(name, k.prefix, n) != 0) continue;
        char * end;
        long l = strtol(name + n, &end, 10);
        if (end == name + n || *end || l < 0 || l > INT32_MAX) return 0;
        il = (int) l;
        return k.kind;
    }
    return 0;
}
inline void write_record(FILE * f, int step, int il, int kind, int n, const void * data) {
    // GGML platforms used here are little endian; preserve the historical words exactly.
    const int32_t hdr[] = {step, il, kind, n};
    if (fwrite(hdr, sizeof(hdr), 1, f) != 1 || fwrite(data, 4, n, f) != (size_t) n)
        throw std::runtime_error("trace write failed");
}
inline int check_trace(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) throw std::runtime_error("cannot open trace");
    struct counts { int real=0, pred=0, pairs=0, hits=0, ids=0; };
    std::map<int, counts> stats;
    std::map<int, std::vector<int32_t>> actual;
    std::map<int, std::vector<float>> predicted;
    int step = -1, steps = 0;
    auto finish = [&]() {
        if (step < 0) return;
        ++steps;
        for (const auto & a : actual) {
            auto it = predicted.find(a.first);
            if (it == predicted.end()) continue;
            const auto & scores = it->second;
            if (scores.size() < a.second.size()) throw std::runtime_error("too few scores");
            std::vector<int> order(scores.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
            auto & c = stats[a.first];
            ++c.pairs;
            c.ids += a.second.size();
            for (size_t j=0; j<a.second.size(); ++j)
                c.hits += std::find(a.second.begin(), a.second.end(), order[j]) != a.second.end();
            for (int id : a.second) if (id < 0 || id >= (int) scores.size())
                throw std::runtime_error("expert id outside predictor vector");
        }
        actual.clear(); predicted.clear();
    };
    try {
        while (true) {
            int32_t h[4];
            size_t got = fread(h, 1, sizeof(h), f);
            if (!got) break;
            if (got != sizeof(h) || h[0] < 0 || h[3] < 0 || h[3] > 1048576)
                throw std::runtime_error("invalid/truncated header");
            if (h[0] != step) {
                if (h[0] < step) throw std::runtime_error("steps out of order");
                finish(); step = h[0];
            }
            std::vector<int32_t> words(h[3]);
            if (fread(words.data(), 4, words.size(), f) != words.size())
                throw std::runtime_error("truncated payload");
            if (h[2] == 3) {
                if (h[1] < 0 || words.empty() || actual.count(h[1])) throw std::runtime_error("invalid/duplicate real ids");
                auto sorted = words; std::sort(sorted.begin(), sorted.end());
                if (sorted.front() < 0 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
                    throw std::runtime_error("invalid real ids");
                actual[h[1]] = words; ++stats[h[1]].real;
            } else if (h[2] == 1) {
                if (h[1] < 0 || words.empty() || predicted.count(h[1])) throw std::runtime_error("invalid/duplicate scores");
                std::vector<float> scores(words.size());
                memcpy(scores.data(), words.data(), words.size()*4);
                for (float x : scores) if (!std::isfinite(x)) throw std::runtime_error("nonfinite score");
                predicted[h[1]] = scores; ++stats[h[1]].pred;
            }
        }
        if (ferror(f)) throw std::runtime_error("trace read failed");
        finish();
        if (stats.empty()) throw std::runtime_error("no routing records");
        counts total;
        for (const auto & kv : stats) {
            const auto & c = kv.second;
            printf("layer %d real=%d predictor=%d pairs=%d precision=", kv.first, c.real, c.pred, c.pairs);
            if (c.ids) printf("%.3f%% (%d/%d)\n", 100.0*c.hits/c.ids, c.hits, c.ids);
            else printf("n/a\n");
            total.hits += c.hits; total.ids += c.ids; total.pairs += c.pairs;
        }
        printf("steps=%d layers=%zu pairs=%d precision=", steps, stats.size(), total.pairs);
        if (total.ids) printf("%.3f%% (%d/%d)\n", 100.0*total.hits/total.ids, total.hits, total.ids);
        else printf("n/a\n");
        fclose(f); return 0;
    } catch (...) { fclose(f); throw; }
}
