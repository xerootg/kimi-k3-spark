// what kpool_build_state does per llama_decode call: a fresh vector of 4 bytes per cell of the whole cache, zeroed
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
struct st_t { std::vector<uint32_t> is_new, rep_gen; };
int main(int argc, char ** argv) {
    const size_t n = argc > 1 ? strtoull(argv[1], nullptr, 10) : 5242880;
    const int reps = 300;
    volatile uint64_t sink = 0;
    for (int round = 0; round < 3; ++round) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            auto st = std::make_unique<st_t>();
            st->rep_gen.resize(n, 0);
            st->rep_gen[(size_t) i * 7919 % n] = 1;
            sink += st->rep_gen[n / 2];
        }
        auto t1 = std::chrono::steady_clock::now();
        printf("n=%zu (%.1f MiB) round %d: %.3f ms per call\n", n, n * 4 / 1048576.0, round,
               std::chrono::duration<double, std::milli>(t1 - t0).count() / reps);
    }
    return 0;
}
