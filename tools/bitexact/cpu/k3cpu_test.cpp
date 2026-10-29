// k3cpu_test: validation and benchmark driver for the k3cpu library (k3cpu.h).
//
//   k3cpu_test validate MUFU_DIR RECORDS N TG TU TD DUMP_DIR CPUS [ncases]
//       every stage (g, u, h, dn) of every case and all 16 experts per case vs agent A's dumps (../FORMAT.md)
//   k3cpu_test check MUFU_DIR RECORDS N TG TU TD CPUS [ntokens]
//       random tokens (edge cases first) vs the scalar host reference of ../sycl/k3x_math.h, all 4 stages
//   k3cpu_test bench MUFU_DIR RECORDS N TG TU TD CPUS ITERS [DUMP_DIR]
//       k = 1..4 (K3B_KS=1,2,..): per token p50/p90/min ms, GB/s of weights; records are used round robin so each is
//       cold (not in L3) when reused; x from the dumps if given. K3B_GAP_US sleeps between tokens (engine-like
//       pacing), K3B_SPIN_US sets the workers' spin before sleeping, K3B_INTERLEAVE=1 the streaming task order.
// CPUS: a list like 0-3,4-11 (worker i pinned to the i-th cpu).
#include "k3cpu.h"
#include "../sycl/k3x_act.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL
#include "../sycl/ggml-common.h"

static const int D = 3584, F = 3072;

static const void * map_file(const std::string & p, size_t * n, bool populate) {
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) { perror(p.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED | (populate ? MAP_POPULATE : 0), fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);
    if (n) *n = st.st_size;
    return m;
}
static std::vector<int> parse_cpus(const char * s) {
    std::vector<int> v; std::string t(s); size_t i = 0;
    while (i < t.size()) {
        size_t j = t.find(',', i); if (j == std::string::npos) j = t.size();
        std::string r = t.substr(i, j - i); size_t d = r.find('-');
        if (d == std::string::npos) v.push_back(atoi(r.c_str()));
        else for (int c = atoi(r.substr(0, d).c_str()); c <= atoi(r.substr(d + 1).c_str()); c++) v.push_back(c);
        i = j + 1;
    }
    return v;
}
static inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// ---- scalar host reference (k3x_math.h, the code the k3x SYCL server's own check uses)
static size_t rowb(int t, int k) { return (size_t) (k / 256) * (t == 19 ? 50 : 66); }
static float ref_row(int t, const uint8_t * row, int nb, const k3_q8_1 * q8) {
    float p[128];
    for (int i = 0; i < 128; i++) {
        const int kb = i / 8, l = i % 8;
        p[i] = kb < nb ? k3_vdot(t, row + kb * (t == 19 ? 50 : 66), q8 + kb * 8, l, iq1s_grid_gpu, iq2xxs_grid) : 0.0f;
    }
    return k3_row_reduce(p);
}
static void ref_expert(const uint8_t * rec, int tg, int tu, int td, const float * x, const k3_mufu_tabs & M, float * g, float * u, float * h, float * dn) {
    std::vector<k3_q8_1> qx(D / 32), qh(F / 32);
    for (int b = 0; b < D / 32; b++) k3_quantize_block(x + 32 * b, &qx[b], M);
    const uint8_t * gate = rec, * up = gate + rowb(tg, D) * F, * down = up + rowb(tu, D) * F;
    for (int i = 0; i < F; i++) {
        g[i] = ref_row(tg, gate + i * rowb(tg, D), D / 256, qx.data());
        u[i] = ref_row(tu, up + i * rowb(tu, D), D / 256, qx.data());
        h[i] = k3_act(g[i], u[i], M);
    }
    for (int b = 0; b < F / 32; b++) k3_quantize_block(h + 32 * b, &qh[b], M);
    for (int i = 0; i < D; i++) dn[i] = ref_row(td, down + i * rowb(td, F), F / 256, qh.data());
}

int main(int argc, char ** argv) {
    if (argc < 9) { fprintf(stderr, "usage: see the head of k3cpu_test.cpp\n"); return 1; }
    const std::string mode = argv[1], mdir = argv[2];
    const int N = atoi(argv[4]), tg = atoi(argv[5]), tu = atoi(argv[6]), td = atoi(argv[7]);
    size_t fsz; const uint8_t * base = (const uint8_t *) map_file(argv[3], &fsz, true);
    const size_t rec = k3cpu_rec_bytes(tg, tu, td);
    if (fsz < rec * (size_t) N) { fprintf(stderr, "records file too small: %zu < %d x %zu\n", fsz, N, rec); return 1; }
    const int ai = mode == "validate" ? 9 : 8;   // CPUS argument
    if (argc <= ai) { fprintf(stderr, "missing CPUS\n"); return 1; }
    const std::vector<int> cpus = parse_cpus(argv[ai]);
    auto t0 = std::chrono::steady_clock::now();
    k3_mufu_tabs mt;
    if (!k3cpu_load_mufu(mdir.c_str(), &mt)) return 1;
    fprintf(stderr, "tables in %.1f s; %zu workers on cpus %s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), cpus.size(), argv[ai]);
    k3cpu_opts o; o.cpus = cpus.data(); o.ncpu = (int) cpus.size();
    if (getenv("K3B_SPIN_US")) o.spin_us = atoi(getenv("K3B_SPIN_US"));
    o.interleave = getenv("K3B_INTERLEAVE") && atoi(getenv("K3B_INTERLEAVE"));
    if (getenv("K3B_LOOKUP")) o.lookup = atoi(getenv("K3B_LOOKUP"));
    k3cpu_pool * P = k3cpu_create(&mt, o);
    if (!P) return 1;

    if (mode == "validate") {
        const std::string dd = argv[8];
        size_t nx; const float * X = (const float *) map_file(dd + "/x.bin", &nx, true);
        int ncase = (int) (nx / (4 * D));
        if (argc > 10) ncase = std::min(ncase, atoi(argv[10]));
        const int32_t * IDS = (const int32_t *) map_file(dd + "/ids.bin", nullptr, true);
        const float * REF[4] = {(const float *) map_file(dd + "/g.bin", nullptr, true), (const float *) map_file(dd + "/u.bin", nullptr, true),
                                (const float *) map_file(dd + "/h.bin", nullptr, true), (const float *) map_file(dd + "/dn.bin", nullptr, true)};
        const int K = 16;
        std::vector<float> g(F * K), u(F * K), h(F * K), dn(D * K);
        unsigned long long bad[4] = {0, 0, 0, 0}, tot[4] = {0, 0, 0, 0}, badrows = 0, rows = 0;
        int firstbad[4] = {-1, -1, -1, -1};
        for (int c = 0; c < ncase; c++) {
            k3cpu_expert e[16];
            for (int k = 0; k < K; k++) {
                if (IDS[c * K + k] < 0 || IDS[c * K + k] >= N) { fprintf(stderr, "bad id\n"); return 1; }
                e[k] = k3cpu_expert{base + (size_t) IDS[c * K + k] * rec, tg, tu, td, &dn[k * D], &g[k * F], &u[k * F], &h[k * F]};
            }
            std::fill(dn.begin(), dn.end(), -12345.0f);
            k3cpu_run(P, X + (size_t) c * D, e, K);
            const float * mine[4] = {g.data(), u.data(), h.data(), dn.data()};
            const size_t n[4] = {(size_t) F * K, (size_t) F * K, (size_t) F * K, (size_t) D * K};
            for (int s = 0; s < 4; s++) {
                const float * r = REF[s] + (size_t) c * n[s];
                for (size_t i = 0; i < n[s]; i++) if (fbits(mine[s][i]) != fbits(r[i])) { bad[s]++; if (firstbad[s] < 0) firstbad[s] = c; }
                tot[s] += n[s];
            }
            for (int k = 0; k < K; k++) { rows++; if (memcmp(&dn[k * D], REF[3] + (size_t) c * D * K + (size_t) k * D, 4 * D)) badrows++; }
        }
        const char * nm[4] = {"g", "u", "h", "dn"};
        for (int s = 0; s < 4; s++) printf("%-2s: %llu / %llu values bit-identical (%llu differ, first case %d)\n", nm[s], tot[s] - bad[s], tot[s], bad[s], firstbad[s]);
        printf("dn rows: %llu / %llu bit-identical; %d cases\n", rows - badrows, rows, ncase);
        k3cpu_destroy(P);
        return (bad[0] | bad[1] | bad[2] | bad[3]) ? 2 : 0;
    }
    if (mode == "check") {
        const int ntok = argc > 9 ? atoi(argv[9]) : 16;
        std::mt19937 rng(1234);
        std::vector<float> x(D), g(F), u(F), h(F), dn(D), rg(F), ru(F), rh(F), rd(D);
        unsigned long long bad = 0, badtok = 0;
        for (int t = 0; t < ntok; t++) {
            // edge cases first: zeros, constants, tiny normals, denormals, huge values, blocks with a single spike
            const float sig[] = {0.0f, 1e-30f, 1e-38f, 1e-3f, 0.05f, 0.3f, 1.0f, 3.0f, 30.0f, 1000.0f};
            std::normal_distribution<float> nd(0.f, sig[t % 10] == 0.0f ? 1.0f : sig[t % 10]);
            for (int i = 0; i < D; i++) x[i] = sig[t % 10] == 0.0f ? 0.0f : nd(rng);
            if (t % 7 == 3) for (int i = 0; i < D; i += 97) x[i] = 1e4f;
            if (t % 11 == 5) for (int i = 0; i < D; i++) if (i % 64 < 32) x[i] = 0.0f;
            if (t % 13 == 6) for (int i = 0; i < D; i++) x[i] = k3_u2f(0x00000400u + i);   // denormals
            const int slot = (int) (rng() % N);
            const uint8_t * r = base + (size_t) slot * rec;
            k3cpu_expert e{r, tg, tu, td, dn.data(), g.data(), u.data(), h.data()};
            k3cpu_run(P, x.data(), &e, 1);
            ref_expert(r, tg, tu, td, x.data(), mt, rg.data(), ru.data(), rh.data(), rd.data());
            unsigned long long b = 0;
            for (int i = 0; i < F; i++) b += (fbits(g[i]) != fbits(rg[i])) + (fbits(u[i]) != fbits(ru[i])) + (fbits(h[i]) != fbits(rh[i]));
            for (int i = 0; i < D; i++) b += fbits(dn[i]) != fbits(rd[i]);
            if (b) { badtok++; fprintf(stderr, "token %d (slot %d): %llu values differ\n", t, slot, b); }
            bad += b;
        }
        printf("check: %d tokens, %llu tokens differ, %llu values differ\n", ntok, badtok, bad);
        k3cpu_destroy(P);
        return bad ? 2 : 0;
    }
    if (mode == "bench") {
        const int iters = argc > 9 ? atoi(argv[9]) : 200;
        std::vector<float> xs;
        int nxs = 0;
        if (argc > 10) {
            size_t nx; const float * X = (const float *) map_file(std::string(argv[10]) + "/x.bin", &nx, true);
            nxs = (int) (nx / (4 * D)); xs.assign(X, X + (size_t) nxs * D);
        } else {
            std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
            nxs = 16; xs.resize((size_t) nxs * D); for (auto & v : xs) v = nd(rng);
        }
        const int gap = getenv("K3B_GAP_US") ? atoi(getenv("K3B_GAP_US")) : 0;
        const bool gapspin = getenv("K3B_GAP_SPIN") && atoi(getenv("K3B_GAP_SPIN"));
        std::vector<int> ks = {1, 2, 3, 4};
        if (getenv("K3B_KS")) { ks.clear(); for (int c : parse_cpus(getenv("K3B_KS"))) ks.push_back(c); }
        std::vector<float> out((size_t) 16 * D);
        FILE * dumpf = getenv("K3B_TRACE") ? fopen(getenv("K3B_TRACE"), "w") : nullptr;
        int slot = 0, xi = 0;
        printf("records %d x %.2f MB, types %d/%d/%d, %zu workers, gap %d us%s%s\n", N, rec / 1e6, tg, tu, td, cpus.size(), gap, gapspin ? " (caller spins)" : "", o.interleave ? ", interleaved" : "");
        for (int k : ks) {
            std::vector<double> t, tgu, tdn0, tx;
            for (int it = 0; it < iters + 5; it++) {
                k3cpu_expert e[16];
                for (int j = 0; j < k; j++) { e[j] = k3cpu_expert{base + (size_t) slot * rec, tg, tu, td, &out[(size_t) j * D], nullptr, nullptr, nullptr}; slot = (slot + 1) % N; }
                const float * x = &xs[(size_t) (xi++ % nxs) * D];
                if (gap && !gapspin) std::this_thread::sleep_for(std::chrono::microseconds(gap));
                if (gap && gapspin) { auto g0 = std::chrono::steady_clock::now(); while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - g0).count() < gap) {} }
                auto a = std::chrono::steady_clock::now();
                k3cpu_run(P, x, e, k);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
                k3cpu_times tm; k3cpu_last_times(P, &tm);
                if (it >= 5) { t.push_back(ms); tgu.push_back(tm.gu_done[k - 1] / 1e6); tdn0.push_back(tm.dn_done[0] / 1e6); tx.push_back(tm.x_done / 1e6); }
                if (dumpf) fprintf(dumpf, "%d %d %d %.4f %.4f %.4f %.4f\n", k, it, (xi - 1) % nxs, ms, tm.gu_done[k - 1] / 1e6, tm.dn_done[k - 1] / 1e6, tm.x_done / 1e6);
            }
            auto st = [](std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t) (q * v.size()))]; };
            const double med = st(t, 0.5);
            printf("k=%d: p50 %.3f ms  p90 %.3f  min %.3f | %.3f ms/expert, %.1f GB/s | all gate/up done %.3f, expert 0 done %.3f (p50) | p10 %.3f p25 %.3f p75 %.3f p99 %.3f | x posted %.4f p90 %.4f\n",
                   k, med, st(t, 0.9), st(t, 0.0), med / k, k * rec / med / 1e6, st(tgu, 0.5), st(tdn0, 0.5), st(t, 0.1), st(t, 0.25), st(t, 0.75), st(t, 0.99), st(tx, 0.5), st(tx, 0.9));
            if (const char * dump = getenv("K3B_DUMP")) { FILE * f = fopen((std::string(dump) + ".k" + std::to_string(k)).c_str(), "w"); for (double v : t) fprintf(f, "%.4f\n", v); fclose(f); }
        }
        if (dumpf) fclose(dumpf);
        k3cpu_destroy(P);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
