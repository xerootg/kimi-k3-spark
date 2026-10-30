// Standalone K3XS v1/v2 expert-store MUL_MAT_ID bit comparison. See README.md.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static uint32_t u32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
static uint64_t u64(const uint8_t * p) {
    return uint64_t(u32(p)) | uint64_t(u32(p + 4)) << 32;
}
static uint64_t number(const std::string & s) {
    check(!s.empty() && s.find_first_not_of("0123456789") == std::string::npos, "invalid unsigned integer");
    size_t end;
    auto n = std::stoull(s, &end);
    check(end == s.size(), "invalid integer");
    return n;
}
static size_t product(uint64_t a, uint64_t b) {
    check(b == 0 || a <= std::numeric_limits<size_t>::max()/b, "size overflow");
    return size_t(a*b);
}
struct store {
    std::ifstream file;
    uint64_t length;
    uint32_t experts;
    uint32_t types[3];
    uint64_t sizes[3], base, stride, record;
    void read(uint64_t offset, void * dst, size_t size) {
        check(offset <= length && size <= length - offset && size <= uint64_t(INT64_MAX), "read outside file");
        file.clear();
        file.seekg(std::streamoff(offset));
        file.read(static_cast<char *>(dst), std::streamsize(size));
        check(bool(file), "short read");
    }
    store(const char * path, uint32_t layer) : file(path, std::ios::binary | std::ios::ate) {
        check(bool(file), "cannot open store");
        auto end = file.tellg();
        check(end >= 64, "short store");
        length = uint64_t(end);
        uint8_t h[64];
        read(0, h, sizeof(h));
        const auto version = u32(h + 4), count = u32(h + 8), align = u32(h + 16);
        experts = u32(h + 12);
        check(std::memcmp(h, "K3XS", 4) == 0 && (version == 1 || version == 2) &&
              count && experts && align && u32(h + 20) == 64, "unsupported store header");
        const auto index = u64(h + 24);
        check(index <= length && count <= (length - index)/64, "index outside file");
        bool found = false;
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t p[64];
            read(index + uint64_t(i)*64, p, sizeof(p));
            if (u32(p) != layer) { continue; }
            check(!found, "duplicate layer");
            found = true;
            record = u64(p + 40); base = u64(p + 48);
            stride = version == 1 ? record : u64(p + 56);
            uint64_t remaining = record;
            for (int j = 0; j < 3; ++j) {
                types[j] = u32(p + 4 + 4*j); sizes[j] = u64(p + 16 + 8*j);
                check(sizes[j] <= remaining, "components exceed record");
                remaining -= sizes[j];
            }
            check(record && stride >= record && base % align == 0 && stride % align == 0 &&
                  (version == 1 || (align % 4096 == 0 && record % 4096 == 0)), "bad record alignment");
            check(base <= length && record <= length - base &&
                  experts - 1 <= (length - base - record)/stride, "records outside file");
        }
        check(found, "layer not found");
    }
};
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
static context_ptr context() {
    auto p = ggml_init({8*1024*1024, nullptr, true});
    check(p != nullptr, "ggml_init failed");
    return context_ptr(p, ggml_free);
}
static buffer_ptr allocate(ggml_context * ctx, ggml_backend_t backend) {
    auto p = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(p != nullptr, "CUDA allocation failed");
    return buffer_ptr(p, ggml_backend_buffer_free);
}
static void dump(const std::string & path, const std::vector<float> & values) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size()*sizeof(float)));
    check(bool(f), "cannot write dump");
}
static std::vector<float> run(ggml_backend_t backend, ggml_tensor * weight, ggml_tensor * gate,
                             int k, int rows, int used, int channels, int cols,
                             const float * input, const int32_t * routes) {
    auto ctx = context();
    auto x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, channels, cols);
    auto ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, cols);
    auto out = ggml_mul_mat_id(ctx.get(), weight, x, ids);
    if (gate) {
        auto g = ggml_mul_mat_id(ctx.get(), gate, x, ids);
        out = ggml_swiglu_split(ctx.get(), g, out);
    }
    check(ggml_backend_supports_op(backend, out), "CUDA does not support output op");
    auto graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    auto buffer = allocate(ctx.get(), backend);
    ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids, routes, 0, ggml_nbytes(ids));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA compute failed");
    ggml_backend_synchronize(backend);
    std::vector<float> result(product(product(rows, used), cols));
    check(result.size()*sizeof(float) == ggml_nbytes(out), "output shape mismatch");
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    return result;
}
int main(int argc, char ** argv) {
    try {
        if (argc != 12) {
            std::fprintf(stderr, "usage: %s STORE LAYER gate|up|down|swiglu K ROWS EXPERT_CSV USED CHANNELS seed|X.f32 off|on DUMP_PREFIX\n", argv[0]);
            return 2;
        }
        const auto layer = number(argv[2]), k64 = number(argv[4]), rows64 = number(argv[5]);
        const auto used64 = number(argv[7]), channels64 = number(argv[8]);
        check(layer <= INT32_MAX && k64 && k64 <= INT32_MAX && rows64 && rows64 <= INT32_MAX &&
              used64 && used64 <= 256 && (channels64 == 1 || channels64 == used64), "invalid dimensions");
        const int k = int(k64), rows = int(rows64), used = int(used64), channels = int(channels64);
        const std::string part = argv[3], expect = argv[10];
        check(part == "gate" || part == "up" || part == "down" || part == "swiglu", "invalid component");
        check(expect == "off" || expect == "on", "invalid expectation");
        const char * knob = std::getenv("GGML_CUDA_MMVQ_EXACT_COLS");
        check((knob && std::strcmp(knob, "1") == 0) == (expect == "on"), "knob and expectation disagree");
        store s(argv[1], uint32_t(layer));
        std::vector<uint32_t> records;
        std::istringstream csv(argv[6]);
        std::string item;
        while (std::getline(csv, item, ',')) {
            auto id = number(item);
            check(id < s.experts && records.size() < 256, "expert out of range or more than 256 slots");
            check(std::find(records.begin(), records.end(), id) == records.end(), "duplicate expert record");
            records.push_back(uint32_t(id));
        }
        check(records.size() >= size_t(used), "need at least USED records");
        const int component = part == "gate" ? 0 : part == "down" ? 2 : 1;
        backend_ptr backend(ggml_backend_cuda_init(0), ggml_backend_free);
        check(bool(backend), "CUDA backend unavailable");
        auto weights_ctx = context();
        const auto load = [&](int c) {
            check(s.types[c] < GGML_TYPE_COUNT, "invalid ggml type");
            const auto type = ggml_type(s.types[c]);
            check(type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS ||
                  type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K, "type outside exact-cols coverage");
            check(k % ggml_blck_size(type) == 0, "K is not block aligned");
            check(product(ggml_row_size(type, k), rows) == s.sizes[c], "K/ROWS disagree with store component size");
            return ggml_new_tensor_3d(weights_ctx.get(), type, k, rows, int64_t(records.size()));
        };
        auto weight = load(component);
        auto gate = part == "swiglu" ? load(0) : nullptr;
        if (gate) { check(gate->type == weight->type, "fusion requires matching types"); }
        auto weights_buffer = allocate(weights_ctx.get(), backend.get());
        ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const auto upload = [&](ggml_tensor * tensor, int c) {
            std::vector<uint8_t> bytes(size_t(s.sizes[c]));
            const uint64_t component_offset = c == 0 ? 0 : c == 1 ? s.sizes[0] : s.sizes[0] + s.sizes[1];
            for (size_t slot = 0; slot < records.size(); ++slot) {
                s.read(s.base + uint64_t(records[slot])*s.stride + component_offset, bytes.data(), bytes.size());
                ggml_backend_tensor_set(tensor, bytes.data(), slot*bytes.size(), bytes.size());
            }
        };
        upload(weight, component);
        if (gate) { upload(gate, 0); }
        std::vector<float> x(product(product(k, channels), 4));
        if (std::string(argv[9]) == "seed") {
            uint32_t state = 0x53ba7c;
            for (auto & value : x) {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                value = (int32_t(state & 0xffff) - 32768)/8192.0f;
            }
        } else {
            std::ifstream f(argv[9], std::ios::binary | std::ios::ate);
            check(bool(f) && f.tellg() == std::streamoff(x.size()*sizeof(float)), "X.f32 must contain exactly 4*K*CHANNELS floats");
            f.seekg(0); f.read(reinterpret_cast<char *>(x.data()), std::streamsize(x.size()*sizeof(float)));
            check(bool(f), "cannot read activations");
        }
        bool passed = true;
        bool witnessed[5] = {};
        // Same, overlapping and disjoint routes (disjoint requires at least 4*USED records).
        for (const int shift : {0, 1, used}) {
            std::vector<int32_t> ids(size_t(4*used));
            for (int col = 0; col < 4; ++col) {
                for (int slot = 0; slot < used; ++slot) {
                    ids[size_t(col*used + slot)] = (col*shift + slot) % int(records.size());
                }
            }
            std::vector<float> singles;
            for (int col = 0; col < 4; ++col) {
                auto one = run(backend.get(), weight, gate, k, rows, used, channels, 1,
                               x.data() + size_t(col)*k*channels, ids.data() + col*used);
                singles.insert(singles.end(), one.begin(), one.end());
            }
            const auto stem = std::string(argv[11]) + ".shift" + std::to_string(shift);
            dump(stem + ".single.f32", singles);
            for (int n = 2; n <= 4; ++n) {
                auto batch = run(backend.get(), weight, gate, k, rows, used, channels, n, x.data(), ids.data());
                const bool equal = std::memcmp(singles.data(), batch.data(), batch.size()*sizeof(float)) == 0;
                size_t different = 0, first = batch.size();
                uint32_t a = 0, b = 0;
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (std::memcmp(&singles[i], &batch[i], sizeof(float)) != 0) {
                        if (first == batch.size()) {
                            first = i; std::memcpy(&a, &singles[i], 4); std::memcpy(&b, &batch[i], 4);
                        }
                        ++different;
                    }
                }
                dump(stem + ".batch" + std::to_string(n) + ".f32", batch);
                std::printf("shift=%d batch=%d unequal=%zu/%zu first=%zu single=%08x batch=%08x %s\n",
                            shift, n, different, batch.size(), first, a, b, equal ? "IDENTICAL" : "DIFFERENT");
                // Off mode needs a witness for every batch size, across the route patterns below.
                if (expect == "on" && !equal) { passed = false; }
                if (!equal) { witnessed[n] = true; }
            }
        }
        if (expect == "off") {
            for (int n = 2; n <= 4; ++n) { passed = passed && witnessed[n]; }
        }
        std::printf("%s: %s batch 2..4 vs singleton columns\n", passed ? "PASS" : "FAIL",
                    expect == "on" ? "bit-identical" : "difference witnessed for each of");
        return passed ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "batchdump: %s\n", e.what());
        return 2;
    }
}
