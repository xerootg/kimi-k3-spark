#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "llama-moe-cache.h"
#include <fstream>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <initializer_list>

struct load { size_t offset, width; };
struct layout { ggml_type type; size_t block, alignment; std::initializer_list<load> loads; };
#define L(t, f, w) {offsetof(block_##t, f), w}
#define T(t, e, a, ...) {GGML_TYPE_##e, sizeof(block_##t), a, {__VA_ARGS__}}
static const layout layouts[] = {
    T(iq1_s, IQ1_S, 2, L(iq1_s,d,2), L(iq1_s,qs,2), L(iq1_s,qh,2)),
    T(iq1_m, IQ1_M, 4, L(iq1_m,qs,4), L(iq1_m,qh,1), L(iq1_m,scales,2)),
    T(iq2_xxs, IQ2_XXS, 2, L(iq2_xxs,d,2), L(iq2_xxs,qs,2)),
    T(iq2_xs, IQ2_XS, 2, L(iq2_xs,d,2), L(iq2_xs,qs,2), L(iq2_xs,scales,1)),
    T(iq2_s, IQ2_S, 2, L(iq2_s,d,2), L(iq2_s,qs,2), L(iq2_s,qh,1), L(iq2_s,scales,1)),
    T(iq3_xxs, IQ3_XXS, 2, L(iq3_xxs,d,2), L(iq3_xxs,qs,2), {66,2}),
    T(iq3_s, IQ3_S, 2, L(iq3_s,d,2), L(iq3_s,qs,2), L(iq3_s,qh,1), L(iq3_s,signs,2), L(iq3_s,scales,1)),
    T(iq4_xs, IQ4_XS, 4, L(iq4_xs,d,2), L(iq4_xs,scales_h,2), L(iq4_xs,scales_l,1), L(iq4_xs,qs,4)),
    T(iq4_nl, IQ4_NL, 2, L(iq4_nl,d,2), L(iq4_nl,qs,2)),
    T(q2_K, Q2_K, 4, L(q2_K,scales,4), L(q2_K,qs,4), L(q2_K,data.dm,4)),
    T(q3_K, Q3_K, 2, L(q3_K,hmask,2), L(q3_K,qs,2), L(q3_K,scales,2), L(q3_K,d,2)),
    T(q4_K, Q4_K, 4, L(q4_K,data.dm,4), L(q4_K,scales,2), L(q4_K,qs,4)),
    T(q5_K, Q5_K, 4, L(q5_K,data.dm,4), L(q5_K,scales,2), L(q5_K,qh,4), L(q5_K,qs,4)),
    T(q6_K, Q6_K, 2, L(q6_K,ql,2), L(q6_K,qh,2), L(q6_K,scales,2), L(q6_K,d,2)),
    T(q4_0, Q4_0, 2, L(q4_0,d,2), L(q4_0,qs,2)),
    T(q4_1, Q4_1, 4, L(q4_1,data.dm,4), L(q4_1,qs,4)),
    T(q5_0, Q5_0, 2, L(q5_0,d,2), L(q5_0,qh,2), L(q5_0,qs,2)),
    T(q5_1, Q5_1, 4, L(q5_1,data.dm,4), L(q5_1,qh,4), L(q5_1,qs,4)),
    T(q8_0, Q8_0, 2, L(q8_0,d,2), L(q8_0,qs,2)),
};

static void alignment_tests() {
    for (const auto & t : layouts) {
        assert(llama_moe_cache_cuda_alignment(t.type) == t.alignment);
        // Independently enumerate field addresses for every offset/row residue.
        for (size_t off = 0; off < 8; ++off) {
            for (size_t row = 0; row < 8; ++row) {
                bool safe = true;
                for (size_t j = 0; j < 8; ++j) {
                    for (size_t r = 0; r < 3; ++r) {
                        for (auto f : t.loads) {
                            safe &= (4096 + off + r*row + j*t.block + f.offset) % f.width == 0;
                        }
                    }
                }
                assert(safe == llama_moe_cache_byte_stride_aligned(t.alignment, t.block, off, row));
            }
        }
        // Both stores' evidenced part offsets, crossed with every audited type.
        for (size_t off : {0UL,2150400UL,4300800UL,2838528UL,5677056UL,3637248UL,7274496UL}) {
            assert(llama_moe_cache_byte_stride_aligned(t.alignment, t.block, off, 14*t.block));
            for (size_t slot : {0UL,1UL,7000UL}) {
                for (size_t j : {0UL,1UL,43007UL,49151UL}) {
                    for (auto f : t.loads) {
                        assert((slot*12091392ULL + off + j*t.block + f.offset) % f.width == 0);
                    }
                }
            }
        }
    }
    assert(llama_moe_cache_cuda_alignment(GGML_TYPE_F32) == 0);
    assert(llama_moe_cache_cuda_alignment(GGML_TYPE_Q8_K) == 0);
    assert(!llama_moe_cache_byte_stride_aligned(0, 50, 0, 700));
    assert(!llama_moe_cache_byte_stride_aligned(2, 51, 0, 714));
    assert(!llama_moe_cache_byte_stride_aligned(4, 66, 0, 924));
    assert(!llama_moe_cache_byte_stride_aligned(2, 50, 1, 700));
    assert(!llama_moe_cache_byte_stride_aligned(4, 56, 2, 784));
    assert(!llama_moe_cache_byte_stride_aligned(4, 56, 0, 786));
    // Complete layer records extracted from the truncated K3 index by k3-capacity.py.
    std::ifstream fixtures("validation/store-parts.tsv");
    assert(fixtures.good());
    size_t count = 0, off, row;
    int type;
    while (fixtures >> type >> off >> row) {
        bool found = false;
        for (const auto & t : layouts) {
            if (int(t.type) == type) {
                assert(llama_moe_cache_byte_stride_aligned(t.alignment, t.block, off, row));
                found = true;
            }
        }
        assert(found);
        ++count;
    }
    assert(fixtures.eof() && count >= 81);
    std::printf("PASS: 19 CUDA type layouts, exhaustive field alignment, odd-block rejection, %zu store parts\n", count);
}

int main() {
    alignment_tests();
    static_assert(sizeof(block_iq2_xs) == 74 && sizeof(block_iq3_xxs) == 98 && sizeof(block_iq4_xs) == 136);
    static_assert(offsetof(block_iq2_xs, qs) == 2 && offsetof(block_iq3_xxs, qs) == 2);
    static_assert(offsetof(block_iq4_xs, qs) == 8);
    const int64_t record = 12091392;
    const int64_t stride = (record + 4095)/4096*4096;
    const int64_t padded = (record + std::lcm(4096LL, std::lcm(74LL, 98LL)) - 1)/std::lcm(4096LL, std::lcm(74LL, 98LL))*std::lcm(4096LL, std::lcm(74LL, 98LL));
    assert(stride == record && padded == 14852096);
    for (const int64_t bs : {74, 98, 136}) {
        const int64_t ordinary_stride = std::lcm(4096LL, (long long) bs)*100;
        for (int64_t sample = 0; sample < 3; ++sample) {
            for (int64_t slot = 0; slot <= 7000; ++slot) {
                const int64_t nb03 = ordinary_stride*7001;
                const int64_t old_offset = (sample*(nb03/bs) + slot*(ordinary_stride/bs))*bs;
                const int64_t new_offset = sample*nb03 + slot*ordinary_stride;
                assert(old_offset == new_offset);
                for (const int64_t block : {0, 1, 31, 49151}) {
                    const int64_t part = bs == 98 ? 7274496 : (bs == 74 ? 3637248 : 0);
                    const int64_t address = slot*stride + part + block*bs;
                    assert(address % (bs == 136 ? 4 : 2) == 0);
                    if (slot == 7000) { assert(address > INT32_MAX); }
                }
            }
        }
    }
    assert((stride/74)*74 != stride && (stride/98)*98 != stride);
    const int64_t lo_budget = (5288LL + 1)*padded;
    const int64_t hi_budget = (5288LL + 2)*padded - 1;
    assert(lo_budget/stride - 1 == 6495 && hi_budget/stride - 1 == 6496);
    std::puts("PASS: block layouts, load alignment, ordinary pointer equivalence, >32-bit offsets, zero-slot capacity (6495..6496)");
}
