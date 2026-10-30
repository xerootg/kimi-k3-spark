// k3cs: CPU-only bit-exact remote expert server (P, 2026-10-29). Serves the same wire protocol as k3x serveidx
// (tools/distributed/rrclient.c: int32 header {k, gid1..gid16}, 3584 f32 x; reply k rows of 3584 f32, streamed in
// order) from the k3cpu library alone: no SYCL, no iGPU, records kept in their GGUF layout (k3cpu is 20-27 % faster per
// expert on GGUF than on k3x's v7 layout, and plain memory has no i915 residency limit).
//
//   k3cs IDX BIN PORT       IDX lines "<global_id> <gate_type> <down_type> <byte_offset>", BIN = records back to back
//
// Environment (k3x names where the meaning is the same):
//   K3X_MUFU=<dir>          MUFU tables (rcp/ex2p/ex2n/tanh.bin)
//   K3X_MAX_GB=<GB>         stop loading at this many GB of records (default 24); later ids answer NaN
//   K3X_DISK_TIER=1         serve nonresident records from a full K3XS store (default off)
//   K3X_DISK_STORE=<file>   full store path (default BIN); IDX/BIN still select the startup RAM records
//   K3X_DISK_LRU_GB=<GB>    preallocated disk cache (decimal GB, default 1), rounded down to whole slots
//   K3X_DISK_THREADS=<n>    parallel readers (default 4); optional liburing, else O_DIRECT pread
//   K3X_DISK_DEADLINE_MS=n  disk budget per received request (default 20); late/missing rows answer NaN
//   K3X_SETOUT=<file>       write the served "layer expert" list (the engine's LLAMA_EXPERT_STORE_REMOTE_SET)
//   K3X_FULLCHECK=1         every served record: one token through the serving kernels vs the scalar host reference
//                           (k3x_math.h, the same reference k3x's FULLCHECK uses), bit for bit; exit 6 on any difference
//   K3X_CPU_CPUS=<list>     worker cpus (one pinned worker each; required)
//   K3X_CPU_RT=<prio>       SCHED_FIFO workers (spin capped at 2000 us), K3X_CPU_SPIN_US, K3X_CPU_LOOKUP, K3X_CPU_INTERLEAVE
//   K3X_RT=<prio>           request thread SCHED_FIFO; it blocks in poll() after K3X_RX_SPIN_US (default 1000) of idle
//   K3X_MAIN_CPU=<cpu>      pin the request thread there once serving (loading and FULLCHECK use every cpu; do not
//                           start k3cs under taskset, or FULLCHECK's reference threads all share that cpu)
//   K3X_MEMWARM=<MB/s>, K3X_MEMWARM_CPU=<cpu>   as in k3x: stream a 64 MiB buffer while idle (off by default)
//   K3X_TRACE=<entries>, K3X_TRACE_OUT=<file>   per-request timestamps (CLOCK_MONOTONIC_RAW ns) written on SIGUSR1 and
//                           at every connection close; columns as k3x's (subset): idx seq conn k t_wait t_first t_hdr t_rx
//                           t_x t_row0..3 t_sent0..3 t_done c_gu c_dn
// The arena is anonymous memory in 1 GiB chunks with transparent huge pages, locked (mlockall) after loading, so the
// serving path never page-faults. A record whose row comes back all zero (never a real expert row) is sent as NaN.
#include "k3cpu.h"
#include "../expert_disk.h"
#include "../sycl/k3x_act.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL
#include "../sycl/ggml-common.h"

static const int D = 3584, F = 3072;
static inline uint64_t ns_raw() { timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return (uint64_t) t.tv_sec * 1000000000ull + t.tv_nsec; }
static inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
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

// ---- scalar host reference (k3x_math.h; identical to k3cpu_test.cpp's and k3x's host_ref)
static size_t rowb(int t, int k) { return (size_t) (k / 256) * (t == 19 ? 50 : 66); }
static bool supported(const k3disk::Record & r) {
    return (r.tg == 16 || r.tg == 19) && (r.tu == 16 || r.tu == 19) &&
           (r.td == 16 || r.td == 19) && r.gate == rowb(r.tg, D) * F &&
           r.up == rowb(r.tu, D) * F && r.down == rowb(r.td, F) * D;
}
static float ref_row(int t, const uint8_t * row, int nb, const k3_q8_1 * q8) {
    float p[128];
    for (int i = 0; i < 128; i++) {
        const int kb = i / 8, l = i % 8;
        p[i] = kb < nb ? k3_vdot(t, row + kb * (t == 19 ? 50 : 66), q8 + kb * 8, l, iq1s_grid_gpu, iq2xxs_grid) : 0.0f;
    }
    return k3_row_reduce(p);
}
static void ref_expert(const uint8_t * rec, int tg, int tu, int td, const float * x, const k3_mufu_tabs & M, float * dn) {
    std::vector<k3_q8_1> qx(D / 32), qh(F / 32);
    std::vector<float> h(F);
    for (int b = 0; b < D / 32; b++) k3_quantize_block(x + 32 * b, &qx[b], M);
    const uint8_t * gate = rec, * up = gate + rowb(tg, D) * F, * down = up + rowb(tu, D) * F;
    for (int i = 0; i < F; i++) {
        const float g = ref_row(tg, gate + i * rowb(tg, D), D / 256, qx.data());
        const float u = ref_row(tu, up + i * rowb(tu, D), D / 256, qx.data());
        h[i] = k3_act(g, u, M);
    }
    for (int b = 0; b < F / 32; b++) k3_quantize_block(h.data() + 32 * b, &qh[b], M);
    for (int i = 0; i < D; i++) dn[i] = ref_row(td, down + i * rowb(td, F), F / 256, qh.data());
}

// ---- trace
struct Tr { int64_t seq; int conn, k; uint64_t t_wait, t_first, t_hdr, t_rx, t_x, t_row[4], t_sent[4], t_done, c_gu, c_dn; };
static Tr * g_tr = nullptr; static size_t g_tr_n = 0, g_tr_i = 0; static int64_t g_tr_idx = 0;
static volatile sig_atomic_t g_tr_dump = 0;
static void tr_sig(int) { g_tr_dump = 1; }
static void tr_dump() {
    g_tr_dump = 0;
    if (!g_tr) return;
    const char * p = getenv("K3X_TRACE_OUT") ? getenv("K3X_TRACE_OUT") : "/tmp/k3cs-trace.tsv";
    FILE * f = fopen(p, "w"); if (!f) { perror(p); return; }
    fprintf(f, "idx\tseq\tconn\tk\tt_wait\tt_first\tt_hdr\tt_rx\tt_x\tt_row0\tt_row1\tt_row2\tt_row3\tt_sent0\tt_sent1\tt_sent2\tt_sent3\tt_done\tc_gu\tc_dn\n");
    const size_t n = std::min(g_tr_i, g_tr_n);
    for (size_t j = 0; j < n; j++) {
        const size_t i = g_tr_i > g_tr_n ? (g_tr_i + j) % g_tr_n : j;
        const Tr & r = g_tr[i];
        fprintf(f, "%lld\t%lld\t%d\t%d\t%llu\t%llu\t%llu\t%llu\t%llu", (long long) (g_tr_idx - (int64_t) n + (int64_t) j), (long long) r.seq, r.conn, r.k,
                (unsigned long long) r.t_wait, (unsigned long long) r.t_first, (unsigned long long) r.t_hdr, (unsigned long long) r.t_rx, (unsigned long long) r.t_x);
        for (int q = 0; q < 4; q++) fprintf(f, "\t%llu", (unsigned long long) r.t_row[q]);
        for (int q = 0; q < 4; q++) fprintf(f, "\t%llu", (unsigned long long) r.t_sent[q]);
        fprintf(f, "\t%llu\t%llu\t%llu\n", (unsigned long long) r.t_done, (unsigned long long) r.c_gu, (unsigned long long) r.c_dn);
    }
    fclose(f);
    fprintf(stderr, "K3X_TRACE: %zu requests written to %s\n", n, p);
}

// ---- receive: spin, then block (bounded spin when the thread is SCHED_FIFO)
static long g_rx_spin_ns = -1;
static bool rd(int fd, void * p, size_t n, uint64_t * first = nullptr) {
    char * c = (char *) p; uint64_t t_idle = 0;
    while (n) {
        ssize_t r = recv(fd, c, n, MSG_DONTWAIT);
        if (r == 0) return false;
        if (r < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) return false;
            if (g_tr_dump && first && !*first) tr_dump();
            if (g_rx_spin_ns >= 0) {
                const uint64_t now = ns_raw();
                if (!t_idle) t_idle = now;
                else if (now - t_idle > (uint64_t) g_rx_spin_ns) {
                    pollfd pf{fd, POLLIN, 0};
                    if (poll(&pf, 1, 100) < 0 && errno != EINTR) return false;
                    t_idle = 0;
                }
            }
            continue;
        }
        if (first && !*first) *first = ns_raw();
        c += r; n -= r;
    }
    return true;
}
static bool wr(int fd, const void * p, size_t n) { const char * c = (const char *) p; while (n) { ssize_t r = send(fd, c, n, MSG_NOSIGNAL); if (r <= 0) return false; c += r; n -= r; } return true; }

// ---- memwarm (as k3x)
static std::atomic<int> g_warm_on{0};
static void memwarm_start(double mbps) {
    std::thread([mbps]() {
        { sched_param sp{}; pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp); }
        if (const char * mc = getenv("K3X_MEMWARM_CPU")) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(atoi(mc), &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs); }
        const size_t N = (size_t) 64 << 20; uint8_t * b = (uint8_t *) aligned_alloc(4096, N); memset(b, 1, N);
        const size_t step = 64 << 10; size_t off = 0; volatile uint64_t sink = 0;
        const double burst_us = step / (mbps * 1e6) * 1e6;
        for (;;) {
            if (!g_warm_on.load(std::memory_order_relaxed)) { usleep(50); continue; }
            const uint64_t t0 = ns_raw(); uint64_t acc = 0;
            for (size_t i = 0; i < step; i += 64) acc += *(const uint64_t *) (b + off + i);
            sink = acc; off = (off + step) % N;
            while ((ns_raw() - t0) / 1e3 < burst_us && g_warm_on.load(std::memory_order_relaxed)) {}
        }
    }).detach();
    fprintf(stderr, "K3X_MEMWARM: %.0f MB/s while idle\n", mbps);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: k3cs IDX BIN PORT (see the head of k3cs.cpp)\n"); return 1; }
    const char * idxp = argv[1], * binp = argv[2]; const int port = atoi(argv[3]);
    struct R { long gid; int tg, td; size_t off; };
    std::vector<R> idx; { FILE * f = fopen(idxp, "r"); if (!f) { perror(idxp); return 1; } R r; while (fscanf(f, "%ld %d %d %zu", &r.gid, &r.tg, &r.td, &r.off) == 4) idx.push_back(r); fclose(f); }
    const bool disk_on = getenv("K3X_DISK_TIER") && atoi(getenv("K3X_DISK_TIER")) != 0;
    std::unique_ptr<k3disk::Store> store;
    std::unique_ptr<k3disk::Cache> disk;
    double disk_deadline_ms = 0;
    if (disk_on) {
        const char * path = getenv("K3X_DISK_STORE") ? getenv("K3X_DISK_STORE") : binp;
        const double gb = getenv("K3X_DISK_LRU_GB") ? atof(getenv("K3X_DISK_LRU_GB")) : 1.0;
        disk_deadline_ms = getenv("K3X_DISK_DEADLINE_MS") ? atof(getenv("K3X_DISK_DEADLINE_MS")) : 20.0;
        const int threads = getenv("K3X_DISK_THREADS") ? atoi(getenv("K3X_DISK_THREADS")) : 4;
        if (!std::isfinite(gb) || gb <= 0 || gb * 1e9 >= double(SIZE_MAX) ||
            !std::isfinite(disk_deadline_ms) || disk_deadline_ms < 0 || disk_deadline_ms > 3600000 ||
            threads < 1 || threads > 64) {
            fprintf(stderr, "k3cs: invalid disk tier configuration\n"); return 1;
        }
        try {
            store.reset(new k3disk::Store(path));
            size_t largest = 0;
            for (const auto & r : store->records) largest = std::max(largest, size_t(r.bytes));
            disk.reset(new k3disk::Cache(path, size_t(gb * 1e9), largest, threads));
        } catch (const std::exception & e) {
            fprintf(stderr, "k3cs: disk tier: %s\n", e.what()); return 1;
        }
        fprintf(stderr, "K3X_DISK_TIER: %zu records, %u experts/layer, %.3f GB LRU, %d readers, %.3f ms deadline\n",
                store->records.size(), store->experts, gb, threads, disk_deadline_ms);
    }
    const char * cs = getenv("K3X_CPU_CPUS");
    if (!cs || !*cs) { fprintf(stderr, "k3cs: K3X_CPU_CPUS is required\n"); return 1; }
    const double max_gb = getenv("K3X_MAX_GB") ? atof(getenv("K3X_MAX_GB")) : 24.0;
    // load: GGUF records verbatim into 1 GiB THP chunks
    const size_t CH = (size_t) 1 << 30; uint8_t * chunk = nullptr; size_t used = CH, tot = 0;
    struct Rec { const uint8_t * p; int tg, tu, td; };
    std::unordered_map<long, Rec> map;
    std::vector<const R *> served;
    const uint64_t t0 = ns_raw();
    int fd = open(binp, O_RDONLY); if (fd < 0) { perror(binp); return 1; }
    for (auto & r : idx) {
        const auto * meta = store ? store->find(r.gid) : nullptr;
        if (store && (!meta || !supported(*meta))) continue;
        const int tg = meta ? meta->tg : r.tg, tu = meta ? meta->tu : r.tg, td = meta ? meta->td : r.td;
        const size_t rec = k3cpu_rec_bytes(tg, tu, td);
        if ((tot + rec) / 1e9 > max_gb) { fprintf(stderr, "K3X_MAX_GB %.1f reached: serving %zu of %zu records\n", max_gb, map.size(), idx.size()); break; }
        if (used + rec > CH) {
            chunk = (uint8_t *) mmap(nullptr, CH, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (chunk == MAP_FAILED) { fprintf(stderr, "arena mmap failed after %.1f GB\n", tot / 1e9); return 1; }
            madvise(chunk, CH, MADV_HUGEPAGE);
            used = 0;
        }
        size_t got = 0;
        while (got < rec) { ssize_t n = pread(fd, chunk + used + got, rec - got, (off_t) (r.off + got)); if (n <= 0) { fprintf(stderr, "short read gid %ld\n", r.gid); return 1; } got += n; }
        posix_fadvise(fd, (off_t) r.off, (off_t) rec, POSIX_FADV_DONTNEED);
        map[r.gid] = {chunk + used, tg, tu, td}; served.push_back(&r);
        used += rec; tot += rec;
    }
    close(fd);
    fprintf(stderr, "loaded %zu of %zu records (%.1f GB, GGUF layout, THP arena) in %.1f s\n", map.size(), idx.size(), tot / 1e9, (ns_raw() - t0) / 1e9);
    // the k3cpu pool
    k3_mufu_tabs mt;
    const char * mdir = getenv("K3X_MUFU") ? getenv("K3X_MUFU") : "../ref/mufu";
    if (!k3cpu_load_mufu(mdir, &mt)) { fprintf(stderr, "k3cs: MUFU tables from %s failed\n", mdir); return 1; }
    static std::vector<int> cpus = parse_cpus(cs);
    k3cpu_opts o; o.cpus = cpus.data(); o.ncpu = (int) cpus.size();
    o.spin_us = getenv("K3X_CPU_SPIN_US") ? atoi(getenv("K3X_CPU_SPIN_US")) : 20000;
    o.lookup = getenv("K3X_CPU_LOOKUP") ? atoi(getenv("K3X_CPU_LOOKUP")) : 0;
    o.interleave = getenv("K3X_CPU_INTERLEAVE") && atoi(getenv("K3X_CPU_INTERLEAVE"));
    o.rt_prio = getenv("K3X_CPU_RT") ? atoi(getenv("K3X_CPU_RT")) : 0;
    if (o.rt_prio > 0 && (o.spin_us < 0 || o.spin_us > 2000)) o.spin_us = 2000;
    k3cpu_pool * P = k3cpu_create(&mt, o);
    if (!P) { fprintf(stderr, "k3cs: k3cpu_create failed (F16C check)\n"); return 1; }
    fprintf(stderr, "K3X_CPU: %d workers on cpus %s, spin %d us, SCHED_FIFO %d, lookup %d, interleave %d (GGUF records)\n", o.ncpu, cs, o.spin_us, o.rt_prio, o.lookup, (int) o.interleave);
    float * rows = (float *) aligned_alloc(4096, (size_t) 16 * D * 4);
    if (const char * so = getenv("K3X_SETOUT")) {
        FILE * f = fopen(so, "w"); if (!f) { perror(so); return 1; }
        if (store) {
            for (const auto & r : store->records) if (supported(r))
                fprintf(f, "%llu %llu\n", (unsigned long long) (r.gid / store->experts),
                        (unsigned long long) (r.gid % store->experts));
        } else {
            for (auto * r : served) fprintf(f, "%ld %ld\n", r->gid / 896, r->gid % 896);
        }
        fclose(f);
    }
    std::vector<float> xin(D); for (int i = 0; i < D; i++) xin[i] = 0.03f * (float) ((i * 53) % 67 - 33);
    if (getenv("K3X_FULLCHECK")) {   // every served record through the serving kernels vs the scalar host reference
        const uint64_t tc = ns_raw();
        const size_t ns = served.size();
        std::vector<float> got(ns * D);
        for (size_t s = 0; s < ns; s++) {
            const Rec & r = map[served[s]->gid];
            k3cpu_expert e{r.p, r.tg, r.tu, r.td, &got[s * D], nullptr, nullptr, nullptr, K3CPU_GGUF};
            k3cpu_run(P, xin.data(), &e, 1);
        }
        const double t_srv = (ns_raw() - tc) / 1e9;
        std::atomic<size_t> next{0}, bad{0}, zero{0}; std::atomic<long> badval{0};
        auto worker = [&]() {
            std::vector<float> ref(D);
            for (size_t s; (s = next++) < ns;) {
                const Rec & r = map.at(served[s]->gid);
                ref_expert(r.p, r.tg, r.tu, r.td, xin.data(), mt, ref.data());
                long bv = 0, nz = 0; for (int i = 0; i < D; i++) { bv += fbits(ref[i]) != fbits(got[s * D + i]); nz += got[s * D + i] != 0.0f; }
                if (bv) { bad++; badval += bv; }
                if (!nz) zero++;
            }
        };
        // the pool's workers sleep after their spin; the reference threads run on every cpu
        std::vector<std::thread> th; for (unsigned i = 0; i < std::max(1u, std::thread::hardware_concurrency()); i++) th.emplace_back([&, i]() {
            cpu_set_t all; CPU_ZERO(&all); for (unsigned c = 0; c < std::thread::hardware_concurrency(); c++) CPU_SET(c, &all);
            pthread_setaffinity_np(pthread_self(), sizeof all, &all);   // whatever the caller's affinity was
            worker();
        });
        for (auto & t : th) t.join();
        fprintf(stderr, "FULLCHECK: serving pass %.1f s; %zu of %zu records bit-identical to the host reference (%ld values differ, %zu all-zero rows), %.1f s\n",
                t_srv, ns - bad.load(), ns, badval.load(), zero.load(), (ns_raw() - tc) / 1e9);
        if (bad || zero) { fprintf(stderr, "FULLCHECK FAILED\n"); return 6; }
    }
    if (mlockall(MCL_CURRENT | MCL_FUTURE)) perror("mlockall (continuing unlocked)");
    else fprintf(stderr, "memory locked (mlockall)\n");
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
    fprintf(stderr, "serving on %d (busy-poll)\n", port);
    if (const char * t = getenv("K3X_TRACE")) {
        g_tr_n = atol(t) > 1 ? (size_t) atol(t) : 32768; g_tr = (Tr *) calloc(g_tr_n, sizeof(Tr));
        signal(SIGUSR1, tr_sig);
        fprintf(stderr, "K3X_TRACE: %zu entries, SIGUSR1 dumps to %s\n", g_tr_n, getenv("K3X_TRACE_OUT") ? getenv("K3X_TRACE_OUT") : "/tmp/k3cs-trace.tsv");
    }
    if (getenv("K3X_MEMWARM") && atof(getenv("K3X_MEMWARM")) > 0) memwarm_start(atof(getenv("K3X_MEMWARM")));
    if (const char * mc = getenv("K3X_MAIN_CPU")) {
        cpu_set_t c1; CPU_ZERO(&c1); CPU_SET(atoi(mc), &c1);
        if (pthread_setaffinity_np(pthread_self(), sizeof c1, &c1)) perror("K3X_MAIN_CPU"); else fprintf(stderr, "K3X_MAIN_CPU: request thread on cpu %d\n", atoi(mc));
    }
    if (const char * rt = getenv("K3X_RT")) if (atoi(rt) > 0) {
        sched_param sp{}; sp.sched_priority = atoi(rt);
        if (sched_setscheduler(0, SCHED_FIFO | SCHED_RESET_ON_FORK, &sp)) perror("K3X_RT sched_setscheduler");
        else fprintf(stderr, "K3X_RT: request thread SCHED_FIFO %d\n", sp.sched_priority);
        g_rx_spin_ns = 1000 * 1000L;
    }
    if (const char * rs = getenv("K3X_RX_SPIN_US")) g_rx_spin_ns = atol(rs) < 0 ? -1 : atol(rs) * 1000L;
    if (g_rx_spin_ns >= 0) fprintf(stderr, "K3X_RX_SPIN_US: %ld (then poll)\n", g_rx_spin_ns / 1000);
    std::vector<float> nanrows(16 * D, NAN);
    int32_t hdr[17]; std::vector<float> x(D);
    long zero_rows = 0; int nconn = 0;
    for (;;) {
        int c = accept(ls, nullptr, nullptr);
        if (c < 0) { if (errno == EINTR) { if (g_tr_dump) tr_dump(); continue; } perror("accept"); return 1; }
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        nconn++;
        for (;;) {
            Tr * tr = nullptr;
            if (g_tr) { tr = &g_tr[g_tr_i % g_tr_n]; memset(tr, 0, sizeof *tr); tr->t_wait = ns_raw(); }
            uint64_t t_first = 0;
            g_warm_on.store(1, std::memory_order_relaxed);
            const bool got = rd(c, hdr, sizeof hdr, &t_first);
            g_warm_on.store(0, std::memory_order_relaxed);
            if (!got) break;
            if (tr) { tr->t_first = t_first; tr->t_hdr = ns_raw(); }
            if (!rd(c, x.data(), (size_t) D * 4)) break;
            const int k = hdr[0];
            if (tr) { tr->t_rx = ns_raw(); tr->k = k; tr->conn = nconn; tr->seq = k >= 1 && k < 16 ? hdr[16] : -1; }
            if (k < 1 || k > 16) break;
            k3cpu_expert e[16]; bool ok = true;
            int handles[16], computed[16], count = 0;
            std::fill(handles, handles + 16, -1);
            const auto deadline = k3disk::Clock::now() +
                std::chrono::microseconds((int64_t) (disk_deadline_ms * 1000));
            if (disk) {
                // Queue all misses before waiting, so one request uses every reader.
                for (int i = 0; i < k; ++i) if (map.find(hdr[1 + i]) == map.end()) {
                    const auto * r = store->find(uint32_t(hdr[1 + i]));
                    // Storage is type-agnostic; never reinterpret an unsupported model as IQ2_XXS.
                    if (r && supported(*r))
                        handles[i] = disk->acquire(*r);
                }
            }
            for (int i = 0; i < k; i++) {
                auto it = map.find(hdr[1 + i]);
                if (!disk) {
                    if (it == map.end()) { ok = false; break; }
                    e[i] = k3cpu_expert{it->second.p, it->second.tg, it->second.tu, it->second.td, rows + (size_t) i * D, nullptr, nullptr, nullptr, K3CPU_GGUF};
                } else {
                    computed[i] = -1;
                    const uint8_t * p = nullptr; int tg = 0, tu = 0, td = 0;
                    if (it != map.end()) {
                        p = it->second.p; tg = it->second.tg; tu = it->second.tu; td = it->second.td;
                    } else if (handles[i] >= 0) {
                        p = disk->wait(handles[i], deadline);
                        const auto * r = store->find(uint32_t(hdr[1 + i]));
                        tg = r->tg; tu = r->tu; td = r->td;
                    } else if (const auto * r = store->find(uint32_t(hdr[1 + i])); r && supported(*r)) {
                        disk->wait(-1, deadline);
                    }
                    if (p) {
                        computed[i] = count;
                        e[count++] = k3cpu_expert{p, tg, tu, td, rows + (size_t) i * D, nullptr, nullptr, nullptr, K3CPU_GGUF};
                    }
                }
            }
            bool okw = true;
            if (disk) {
                if (tr) tr->t_x = ns_raw();
                if (count) k3cpu_submit(P, x.data(), e, count);
                for (int j = 0; j < k && okw; ++j) {
                    const float * r = nanrows.data();
                    if (computed[j] >= 0) {
                        k3cpu_wait(P, computed[j]);
                        r = rows + (size_t) j * D;
                        bool nz = false;
                        for (int i = 0; i < D && !nz; ++i) nz = r[i] != 0.0f;
                        if (!nz) { ++zero_rows; r = nanrows.data(); }
                    }
                    if (tr && j < 4) tr->t_row[j] = ns_raw();
                    okw = wr(c, r, (size_t) D * 4);
                    if (tr && j < 4) tr->t_sent[j] = ns_raw();
                }
                // Even a disconnected client must not allow eviction of weights still in use.
                if (count) k3cpu_wait_all(P);
                for (int i = 0; i < k; ++i) disk->release(handles[i]);
            }
            else if (!ok) okw = wr(c, nanrows.data(), (size_t) k * D * 4);
            else {
                if (tr) tr->t_x = ns_raw();
                k3cpu_submit(P, x.data(), e, k);
                for (int j = 0; j < k && okw; j++) {
                    k3cpu_wait(P, j);
                    if (tr && j < 4) tr->t_row[j] = ns_raw();
                    const float * r = rows + (size_t) j * D; bool nz = false;
                    for (int i = 0; i < D && !nz; i++) nz = r[i] != 0.0f;
                    if (!nz) { zero_rows++; r = nanrows.data(); }
                    okw = wr(c, r, (size_t) D * 4);
                    if (tr && j < 4) tr->t_sent[j] = ns_raw();
                }
                if (tr) { k3cpu_wait_all(P); k3cpu_times ct; k3cpu_last_times(P, &ct); tr->c_gu = ct.gu_done[k - 1]; tr->c_dn = ct.dn_done[k - 1]; }
            }
            if (tr) { tr->t_done = ns_raw(); g_tr_i++; g_tr_idx++; }
            if (!okw) break;
        }
        close(c);
        if (zero_rows) fprintf(stderr, "conn %d: %ld all-zero rows sent as NaN so far\n", nconn, zero_rows);
        if (disk) disk->stats(stderr);
        tr_dump();
    }
}
