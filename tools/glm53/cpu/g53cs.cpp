// GLM server adapted from tools/bitexact/cpu/k3cs.cpp. Same disk and wire
// contract.
#include "g53cpu.h"
#ifdef G53_ALLOC_TEST
#include "alloc_test.h"
#endif
#include "../../bitexact/expert_disk.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <memory>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_SYCL
#include "../ref/ggml-common.h"

static const int D = 6144, F = 2048;
static inline uint64_t ns_raw() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static inline uint32_t fbits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static std::vector<int> parse_cpus(const char *s) {
  std::vector<int> v;
  std::string t(s);
  size_t i = 0;
  while (i < t.size()) {
    size_t j = t.find(',', i);
    if (j == std::string::npos)
      j = t.size();
    std::string r = t.substr(i, j - i);
    size_t d = r.find('-');
    if (d == std::string::npos)
      v.push_back(atoi(r.c_str()));
    else
      for (int c = atoi(r.substr(0, d).c_str());
           c <= atoi(r.substr(d + 1).c_str()); c++)
        v.push_back(c);
    i = j + 1;
  }
  return v;
}

// ---- scalar host reference (k3x_math.h; identical to g53cpu_test.cpp's and
// k3x's host_ref)
static size_t rowb(int t, int k) {
  return size_t(k / 256) * g53exact::blockbytes(t);
}
static bool supported(const k3disk::Record &r) {
  return rowb(r.tg, D) && rowb(r.tu, D) && rowb(r.td, F) &&
         r.gate == rowb(r.tg, D) * F && r.up == rowb(r.tu, D) * F &&
         r.down == rowb(r.td, F) * D;
}
static void ref_expert(const uint8_t *rec, int tg, int tu, int td,
                       const float *x, const g53_mufu_tabs &M, float *dn) {
  g53cpu_reference(rec, tg, tu, td, x, M, dn);
}

// ---- trace
struct Tr {
  int64_t seq;
  int conn, k;
  uint64_t t_wait, t_first, t_hdr, t_rx, t_x, t_row[4], t_sent[4], t_done, c_gu,
      c_dn;
};
static Tr *g_tr = nullptr;
static size_t g_tr_n = 0, g_tr_i = 0;
static int64_t g_tr_idx = 0;
static volatile sig_atomic_t g_tr_dump = 0;
static void tr_sig(int) { g_tr_dump = 1; }
static void tr_dump() {
  g_tr_dump = 0;
  if (!g_tr)
    return;
  const char *p = getenv("K3X_TRACE_OUT") ? getenv("K3X_TRACE_OUT")
                                          : "/tmp/g53cs-trace.tsv";
  FILE *f = fopen(p, "w");
  if (!f) {
    perror(p);
    return;
  }
  fprintf(
      f,
      "idx\tseq\tconn\tk\tt_wait\tt_first\tt_hdr\tt_rx\tt_x\tt_row0\tt_row1\tt_"
      "row2\tt_row3\tt_sent0\tt_sent1\tt_sent2\tt_sent3\tt_done\tc_gu\tc_dn\n");
  const size_t n = std::min(g_tr_i, g_tr_n);
  for (size_t j = 0; j < n; j++) {
    const size_t i = g_tr_i > g_tr_n ? (g_tr_i + j) % g_tr_n : j;
    const Tr &r = g_tr[i];
    fprintf(f, "%lld\t%lld\t%d\t%d\t%llu\t%llu\t%llu\t%llu\t%llu",
            (long long)(g_tr_idx - (int64_t)n + (int64_t)j), (long long)r.seq,
            r.conn, r.k, (unsigned long long)r.t_wait,
            (unsigned long long)r.t_first, (unsigned long long)r.t_hdr,
            (unsigned long long)r.t_rx, (unsigned long long)r.t_x);
    for (int q = 0; q < 4; q++)
      fprintf(f, "\t%llu", (unsigned long long)r.t_row[q]);
    for (int q = 0; q < 4; q++)
      fprintf(f, "\t%llu", (unsigned long long)r.t_sent[q]);
    fprintf(f, "\t%llu\t%llu\t%llu\n", (unsigned long long)r.t_done,
            (unsigned long long)r.c_gu, (unsigned long long)r.c_dn);
  }
  fclose(f);
  fprintf(stderr, "K3X_TRACE: %zu requests written to %s\n", n, p);
}

// ---- receive: spin, then block (bounded spin when the thread is SCHED_FIFO)
static long g_rx_spin_ns = -1;
static bool rd(int fd, void *p, size_t n, uint64_t *first = nullptr) {
  char *c = (char *)p;
  uint64_t t_idle = 0;
  while (n) {
#ifdef G53_PIPE_TEST
    ssize_t r = read(fd, c, n);
#else
    ssize_t r = recv(fd, c, n, MSG_DONTWAIT);
#endif
    if (r == 0)
      return false;
    if (r < 0) {
      if (errno != EAGAIN && errno != EWOULDBLOCK)
        return false;
      if (g_tr_dump && first && !*first)
        tr_dump();
      if (g_rx_spin_ns >= 0) {
        const uint64_t now = ns_raw();
        if (!t_idle)
          t_idle = now;
        else if (now - t_idle > (uint64_t)g_rx_spin_ns) {
          pollfd pf{fd, POLLIN, 0};
          if (poll(&pf, 1, 100) < 0 && errno != EINTR)
            return false;
          t_idle = 0;
        }
      }
      continue;
    }
    if (first && !*first)
      *first = ns_raw();
    c += r;
    n -= r;
  }
  return true;
}
static bool wr(int fd, const void *p, size_t n) {
  const char *c = (const char *)p;
  while (n) {
    ssize_t r =
#ifdef G53_PIPE_TEST
        write(STDOUT_FILENO, c, n);
#else
        send(fd, c, n, MSG_NOSIGNAL);
#endif
    if (r <= 0)
      return false;
    c += r;
    n -= r;
  }
  return true;
}

// ---- memwarm (as k3x)
static std::atomic<int> g_warm_on{0};
static void memwarm_start(double mbps) {
  std::thread([mbps]() {
    {
      sched_param sp{};
      pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
    }
    if (const char *mc = getenv("K3X_MEMWARM_CPU")) {
      cpu_set_t cs;
      CPU_ZERO(&cs);
      CPU_SET(atoi(mc), &cs);
      pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    const size_t N = (size_t)64 << 20;
    uint8_t *b = (uint8_t *)aligned_alloc(4096, N);
    memset(b, 1, N);
    const size_t step = 64 << 10;
    size_t off = 0;
    volatile uint64_t sink = 0;
    const double burst_us = step / (mbps * 1e6) * 1e6;
    for (;;) {
      if (!g_warm_on.load(std::memory_order_relaxed)) {
        usleep(50);
        continue;
      }
      const uint64_t t0 = ns_raw();
      uint64_t acc = 0;
      for (size_t i = 0; i < step; i += 64)
        acc += *(const uint64_t *)(b + off + i);
      sink = acc;
      (void)sink;
      off = (off + step) % N;
      while ((ns_raw() - t0) / 1e3 < burst_us &&
             g_warm_on.load(std::memory_order_relaxed)) {
      }
    }
  }).detach();
  fprintf(stderr, "K3X_MEMWARM: %.0f MB/s while idle\n", mbps);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: g53cs IDX BIN PORT (see the head of g53cs.cpp)\n");
    return 1;
  }
  const char *idxp = argv[1], *binp = argv[2];
  const int port = atoi(argv[3]);
  struct R {
    long gid;
    int tg, td;
    size_t off;
  };
  std::vector<R> idx;
  {
    FILE *f = fopen(idxp, "r");
    if (!f) {
      perror(idxp);
      return 1;
    }
    R r;
    while (fscanf(f, "%ld %d %d %zu", &r.gid, &r.tg, &r.td, &r.off) == 4)
      idx.push_back(r);
    fclose(f);
  }
  const bool disk_on =
      getenv("K3X_DISK_TIER") && atoi(getenv("K3X_DISK_TIER")) != 0;
  std::unique_ptr<k3disk::Store> store;
  std::unique_ptr<k3disk::Cache> disk;
  double disk_deadline_ms = 0;
  if (disk_on) {
    const char *path =
        getenv("K3X_DISK_STORE") ? getenv("K3X_DISK_STORE") : binp;
    const double gb =
        getenv("K3X_DISK_LRU_GB") ? atof(getenv("K3X_DISK_LRU_GB")) : 1.0;
    disk_deadline_ms = getenv("K3X_DISK_DEADLINE_MS")
                           ? atof(getenv("K3X_DISK_DEADLINE_MS"))
                           : 20.0;
    const int threads =
        getenv("K3X_DISK_THREADS") ? atoi(getenv("K3X_DISK_THREADS")) : 4;
    if (!std::isfinite(gb) || gb <= 0 || gb * 1e9 >= double(SIZE_MAX) ||
        !std::isfinite(disk_deadline_ms) || disk_deadline_ms < 0 ||
        disk_deadline_ms > 3600000 || threads < 1 || threads > 64) {
      fprintf(stderr, "g53cs: invalid disk tier configuration\n");
      return 1;
    }
    try {
      store.reset(new k3disk::Store(path));
      size_t largest = 0;
      for (const auto &r : store->records)
        largest = std::max(largest, size_t(r.bytes));
      disk.reset(new k3disk::Cache(path, size_t(gb * 1e9), largest, threads));
    } catch (const std::exception &e) {
      fprintf(stderr, "g53cs: disk tier: %s\n", e.what());
      return 1;
    }
    fprintf(stderr,
            "K3X_DISK_TIER: %zu records, %u experts/layer, %.3f GB LRU, %d "
            "readers, %.3f ms deadline\n",
            store->records.size(), store->experts, gb, threads,
            disk_deadline_ms);
  }
  if (!store) {
    int checkfd = open(binp, O_RDONLY);
    char magic[4]{};
    if (checkfd >= 0) {
      ssize_t n = pread(checkfd, magic, 4, 0);
      close(checkfd);
      if (n == 4 && !memcmp(magic, "K3XS", 4)) {
        try {
          store.reset(new k3disk::Store(binp));
        } catch (const std::exception &e) {
          fprintf(stderr, "invalid store: %s\n", e.what());
          return 1;
        }
      }
    }
  }
  const char *cs = getenv("K3X_CPU_CPUS");
  if (!cs || !*cs) {
    fprintf(stderr, "g53cs: K3X_CPU_CPUS is required\n");
    return 1;
  }
  const double max_gb =
      getenv("K3X_MAX_GB") ? atof(getenv("K3X_MAX_GB")) : 24.0;
  // load: GGUF records verbatim into 1 GiB THP chunks
  const size_t CH = (size_t)1 << 30;
  uint8_t *chunk = nullptr;
  size_t used = CH, tot = 0;
  struct Rec {
    const uint8_t *p;
    int tg, tu, td;
  };
  std::unordered_map<long, Rec> map;
  std::vector<const R *> served;
  const uint64_t t0 = ns_raw();
  int fd = open(binp, O_RDONLY);
  if (fd < 0) {
    perror(binp);
    return 1;
  }
  for (auto &r : idx) {
    const auto *meta = store ? store->find(r.gid) : nullptr;
    if (store && (!meta || !supported(*meta) || meta->tg != unsigned(r.tg) ||
                  meta->td != unsigned(r.td)))
      continue;
    const int tg = meta ? meta->tg : r.tg, tu = meta ? meta->tu : r.tg,
              td = meta ? meta->td : r.td;
    const size_t rec = g53cpu_rec_bytes(tg, tu, td);
    if (!rec)
      continue;
    if ((tot + rec) / 1e9 > max_gb) {
      fprintf(stderr, "K3X_MAX_GB %.1f reached: serving %zu of %zu records\n",
              max_gb, map.size(), idx.size());
      break;
    }
    if (used + rec > CH) {
      chunk = (uint8_t *)mmap(nullptr, CH, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (chunk == MAP_FAILED) {
        fprintf(stderr, "arena mmap failed after %.1f GB\n", tot / 1e9);
        return 1;
      }
      madvise(chunk, CH, MADV_HUGEPAGE);
      used = 0;
    }
    size_t got = 0;
    while (got < rec) {
      ssize_t n =
          pread(fd, chunk + used + got, rec - got, (off_t)(r.off + got));
      if (n <= 0) {
        fprintf(stderr, "short read gid %ld\n", r.gid);
        return 1;
      }
      got += n;
    }
    posix_fadvise(fd, (off_t)r.off, (off_t)rec, POSIX_FADV_DONTNEED);
    map[r.gid] = {chunk + used, tg, tu, td};
    served.push_back(&r);
    used += rec;
    tot += rec;
  }
  close(fd);
  fprintf(
      stderr,
      "loaded %zu of %zu records (%.1f GB, GGUF layout, THP arena) in %.1f s\n",
      map.size(), idx.size(), tot / 1e9, (ns_raw() - t0) / 1e9);
  // the g53cpu pool
  g53_mufu_tabs mt;
  const char *mdir = getenv("K3X_MUFU") ? getenv("K3X_MUFU") : "../ref/mufu";
  if (!g53cpu_load_mufu(mdir, &mt)) {
    fprintf(stderr, "g53cs: MUFU tables from %s failed\n", mdir);
    return 1;
  }
  static std::vector<int> cpus = parse_cpus(cs);
  g53cpu_opts o;
  o.cpus = cpus.data();
  o.ncpu = (int)cpus.size();
  o.spin_us =
      getenv("K3X_CPU_SPIN_US") ? atoi(getenv("K3X_CPU_SPIN_US")) : 20000;
  o.lookup = getenv("K3X_CPU_LOOKUP") ? atoi(getenv("K3X_CPU_LOOKUP")) : 0;
  o.interleave =
      getenv("K3X_CPU_INTERLEAVE") && atoi(getenv("K3X_CPU_INTERLEAVE"));
  o.rt_prio = getenv("K3X_CPU_RT") ? atoi(getenv("K3X_CPU_RT")) : 0;
  if (o.rt_prio > 0 && (o.spin_us < 0 || o.spin_us > 2000))
    o.spin_us = 2000;
  g53cpu_pool *P = g53cpu_create(&mt, o);
  if (!P) {
    fprintf(stderr,
            "g53cs: g53cpu_create failed (invalid worker configuration)\n");
    return 1;
  }
  fprintf(stderr,
          "K3X_CPU: %d workers on cpus %s, spin %d us, SCHED_FIFO %d, lookup "
          "%d, interleave %d (GGUF records)\n",
          o.ncpu, cs, o.spin_us, o.rt_prio, o.lookup, (int)o.interleave);
  float *rows = (float *)aligned_alloc(4096, (size_t)16 * D * 4);
  if (!rows) {
    perror("rows");
    return 1;
  }
  // Exercise every worker's stack, math code and per-expert scratch before
  // locking.
  {
    std::vector<uint8_t> warm(g53cpu_rec_bytes(17, 17, 18), 0);
    float zeros[D]{};
    g53cpu_expert es[16];
    for (int i = 0; i < 16; i++)
      es[i] = {warm.data(), 17, 17, 18, rows + size_t(i) * D};
    g53cpu_run(P, zeros, es, 16);
  }
  if (const char *so = getenv("K3X_SETOUT")) {
    FILE *f = fopen(so, "w");
    if (!f) {
      perror(so);
      return 1;
    }
    if (disk) {
      for (const auto &r : store->records)
        if (supported(r))
          fprintf(f, "%llu %llu\n",
                  (unsigned long long)(r.gid / store->experts),
                  (unsigned long long)(r.gid % store->experts));
    } else {
      for (auto *r : served)
        fprintf(f, "%ld %ld\n", r->gid / (store ? store->experts : 256),
                r->gid % (store ? store->experts : 256));
    }
    fclose(f);
  }
  std::vector<float> xin(D);
  for (int i = 0; i < D; i++)
    xin[i] = 0.03f * (float)((i * 53) % 67 - 33);
  if (getenv("K3X_FULLCHECK") &&
      atoi(
          getenv("K3X_FULLCHECK"))) { // every served record through the serving
                                      // kernels vs the scalar host reference
    const uint64_t tc = ns_raw();
    const size_t ns = served.size();
    std::vector<float> got(ns * D);
    for (size_t s = 0; s < ns; s++) {
      const Rec &r = map[served[s]->gid];
      g53cpu_expert e{r.p,     r.tg,    r.tu,    r.td,       &got[s * D],
                      nullptr, nullptr, nullptr, G53CPU_GGUF};
      g53cpu_run(P, xin.data(), &e, 1);
    }
    const double t_srv = (ns_raw() - tc) / 1e9;
    std::atomic<size_t> next{0}, bad{0}, zero{0};
    std::atomic<long> badval{0};
    auto worker = [&]() {
      std::vector<float> ref(D);
      for (size_t s; (s = next++) < ns;) {
        const Rec &r = map.at(served[s]->gid);
        ref_expert(r.p, r.tg, r.tu, r.td, xin.data(), mt, ref.data());
        long bv = 0, nz = 0;
        for (int i = 0; i < D; i++) {
          bv += fbits(ref[i]) != fbits(got[s * D + i]);
          nz += got[s * D + i] != 0.0f;
        }
        if (bv) {
          bad++;
          badval += bv;
        }
        if (!nz)
          zero++;
      }
    };
    // the pool's workers sleep after their spin; the reference threads run on
    // every cpu
    std::vector<std::thread> th;
    for (unsigned i = 0; i < std::max(1u, std::thread::hardware_concurrency());
         i++)
      th.emplace_back([&, i]() {
        cpu_set_t all;
        CPU_ZERO(&all);
        for (unsigned c = 0; c < std::thread::hardware_concurrency(); c++)
          CPU_SET(c, &all);
        pthread_setaffinity_np(pthread_self(), sizeof all,
                               &all); // whatever the caller's affinity was
        worker();
      });
    for (auto &t : th)
      t.join();
    fprintf(
        stderr,
        "FULLCHECK: serving pass %.1f s; %zu of %zu records bit-identical to "
        "the host reference (%ld values differ, %zu all-zero rows), %.1f s\n",
        t_srv, ns - bad.load(), ns, badval.load(), zero.load(),
        (ns_raw() - tc) / 1e9);
    if (bad) {
      fprintf(stderr, "FULLCHECK FAILED\n");
      return 6;
    }
  }

#ifndef G53_PIPE_TEST
  int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a = {};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  a.sin_addr.s_addr = INADDR_ANY;
  if (bind(ls, (sockaddr *)&a, sizeof a) || listen(ls, 4)) {
    perror("bind");
    return 1;
  }
  fprintf(stderr, "serving on %d (busy-poll)\n", port);
#else
  fprintf(stderr, "serving on pipes (test transport)\n");
#endif
  if (const char *t = getenv("K3X_TRACE")) {
    g_tr_n = atol(t) > 1 ? (size_t)atol(t) : 32768;
    g_tr = (Tr *)calloc(g_tr_n, sizeof(Tr));
    signal(SIGUSR1, tr_sig);
    fprintf(stderr, "K3X_TRACE: %zu entries, SIGUSR1 dumps to %s\n", g_tr_n,
            getenv("K3X_TRACE_OUT") ? getenv("K3X_TRACE_OUT")
                                    : "/tmp/g53cs-trace.tsv");
  }
  if (getenv("K3X_MEMWARM") && atof(getenv("K3X_MEMWARM")) > 0)
    memwarm_start(atof(getenv("K3X_MEMWARM")));
  if (const char *mc = getenv("K3X_MAIN_CPU")) {
    cpu_set_t c1;
    CPU_ZERO(&c1);
    CPU_SET(atoi(mc), &c1);
    if (pthread_setaffinity_np(pthread_self(), sizeof c1, &c1))
      perror("K3X_MAIN_CPU");
    else
      fprintf(stderr, "K3X_MAIN_CPU: request thread on cpu %d\n", atoi(mc));
  }
  if (const char *rt = getenv("K3X_RT"))
    if (atoi(rt) > 0) {
      sched_param sp{};
      sp.sched_priority = atoi(rt);
      if (sched_setscheduler(0, SCHED_FIFO | SCHED_RESET_ON_FORK, &sp))
        perror("K3X_RT sched_setscheduler");
      else
        fprintf(stderr, "K3X_RT: request thread SCHED_FIFO %d\n",
                sp.sched_priority);
      g_rx_spin_ns = 1000 * 1000L;
    }
  if (const char *rs = getenv("K3X_RX_SPIN_US"))
    g_rx_spin_ns = atol(rs) < 0 ? -1 : atol(rs) * 1000L;
  if (g_rx_spin_ns >= 0)
    fprintf(stderr, "K3X_RX_SPIN_US: %ld (then poll)\n", g_rx_spin_ns / 1000);
  std::vector<float> nanrows(16 * D, NAN);
  int32_t hdr[17]{};
  std::vector<float> x(D);
  memset(rows, 0, size_t(16) * D * 4);
  if (mlockall(MCL_CURRENT | MCL_FUTURE)) {
    if (!getenv("G53_ALLOW_UNLOCKED")) {
      perror(
          "mlockall: raise memlock limit, or G53_ALLOW_UNLOCKED=1 for tests");
      return 1;
    }
    perror("mlockall: explicitly allowed unlocked test mode");
  } else
    fprintf(stderr, "memory locked (mlockall)\n");
  long zero_rows = 0;
  int nconn = 0;
  for (;;) {
#ifdef G53_PIPE_TEST
    int c = STDIN_FILENO;
#else
    int c = accept(ls, nullptr, nullptr);
    if (c < 0) {
      if (errno == EINTR) {
        if (g_tr_dump)
          tr_dump();
        continue;
      }
      perror("accept");
      return 1;
    }
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#endif
    nconn++;
    for (;;) {
      Tr *tr = nullptr;
      if (g_tr) {
        tr = &g_tr[g_tr_i % g_tr_n];
        memset(tr, 0, sizeof *tr);
        tr->t_wait = ns_raw();
      }
      uint64_t t_first = 0;
      g_warm_on.store(1, std::memory_order_relaxed);
      const bool got = rd(c, hdr, sizeof hdr, &t_first);
      g_warm_on.store(0, std::memory_order_relaxed);
      if (!got)
        break;
      if (tr) {
        tr->t_first = t_first;
        tr->t_hdr = ns_raw();
      }
      if (!rd(c, x.data(), (size_t)D * 4))
        break;
      const int k = hdr[0];
      if (tr) {
        tr->t_rx = ns_raw();
        tr->k = k;
        tr->conn = nconn;
        tr->seq = k >= 1 && k < 16 ? hdr[16] : -1;
      }
      if (k < 1 || k > 16)
        break;
#ifdef G53_ALLOC_TEST
      allocation_begin();
#endif
      g53cpu_expert e[16];
      bool ok = true;
      int handles[16], computed[16], count = 0;
      std::fill(handles, handles + 16, -1);
      const auto deadline =
          k3disk::Clock::now() +
          std::chrono::microseconds((int64_t)(disk_deadline_ms * 1000));
      if (disk) {
        // Queue all misses before waiting, so one request uses every reader.
        for (int i = 0; i < k; ++i)
          if (map.find(hdr[1 + i]) == map.end()) {
            const auto *r = store->find(uint32_t(hdr[1 + i]));
            // Storage is type-agnostic; never reinterpret an unsupported model
            // as IQ2_XXS.
            if (r && supported(*r))
              handles[i] = disk->acquire(*r);
          }
      }
      for (int i = 0; i < k; i++) {
        auto it = map.find(hdr[1 + i]);
        if (!disk) {
          if (it == map.end()) {
            ok = false;
            break;
          }
          e[i] = g53cpu_expert{
              it->second.p,         it->second.tg, it->second.tu, it->second.td,
              rows + (size_t)i * D, nullptr,       nullptr,       nullptr,
              G53CPU_GGUF};
        } else {
          computed[i] = -1;
          const uint8_t *p = nullptr;
          int tg = 0, tu = 0, td = 0;
          if (it != map.end()) {
            p = it->second.p;
            tg = it->second.tg;
            tu = it->second.tu;
            td = it->second.td;
          } else if (handles[i] >= 0) {
            p = disk->wait(handles[i], deadline);
            const auto *r = store->find(uint32_t(hdr[1 + i]));
            tg = r->tg;
            tu = r->tu;
            td = r->td;
          } else if (const auto *r = store->find(uint32_t(hdr[1 + i]));
                     r && supported(*r)) {
            disk->wait(-1, deadline);
          }
          if (p) {
            computed[i] = count;
            e[count++] = g53cpu_expert{
                p,       tg,      tu,      td,         rows + (size_t)i * D,
                nullptr, nullptr, nullptr, G53CPU_GGUF};
          }
        }
      }
      bool okw = true;
      if (disk) {
        if (tr)
          tr->t_x = ns_raw();
        if (count)
          g53cpu_submit(P, x.data(), e, count);
        for (int j = 0; j < k && okw; ++j) {
          const float *r = nanrows.data();
          if (computed[j] >= 0) {
            g53cpu_wait(P, computed[j]);
            r = rows + (size_t)j * D;
            bool nz = false;
            for (int i = 0; i < D && !nz; ++i)
              nz = r[i] != 0.0f;
            if (!nz)
              ++zero_rows;
          }
          if (tr && j < 4)
            tr->t_row[j] = ns_raw();
          okw = wr(c, r, (size_t)D * 4);
          if (tr && j < 4)
            tr->t_sent[j] = ns_raw();
        }
        // Even a disconnected client must not allow eviction of weights still
        // in use.
        if (count)
          g53cpu_wait_all(P);
        for (int i = 0; i < k; ++i)
          disk->release(handles[i]);
      } else if (!ok)
        okw = wr(c, nanrows.data(), (size_t)k * D * 4);
      else {
        if (tr)
          tr->t_x = ns_raw();
        g53cpu_submit(P, x.data(), e, k);
        for (int j = 0; j < k && okw; j++) {
          g53cpu_wait(P, j);
          if (tr && j < 4)
            tr->t_row[j] = ns_raw();
          const float *r = rows + (size_t)j * D;
          bool nz = false;
          for (int i = 0; i < D && !nz; i++)
            nz = r[i] != 0.0f;
          if (!nz)
            zero_rows++;
          okw = wr(c, r, (size_t)D * 4);
          if (tr && j < 4)
            tr->t_sent[j] = ns_raw();
        }
        g53cpu_wait_all(P);
        if (tr) {
          g53cpu_times ct;
          g53cpu_last_times(P, &ct);
          tr->c_gu = ct.gu_done[k - 1];
          tr->c_dn = ct.dn_done[k - 1];
        }
      }
      if (tr) {
        tr->t_done = ns_raw();
        g_tr_i++;
        g_tr_idx++;
      }
#ifdef G53_ALLOC_TEST
      allocation_end();
#endif
      if (!okw)
        break;
    }
    close(c);
    if (zero_rows)
      fprintf(stderr, "conn %d: %ld valid all-zero rows returned so far\n",
              nconn, zero_rows);
    if (disk)
      disk->stats(stderr);
    tr_dump();
#ifdef G53_PIPE_TEST
    return 0;
#endif
  }
}
