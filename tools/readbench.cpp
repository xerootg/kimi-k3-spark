// Read-shape microbenchmark for the expert store: fills of K random records (decode pattern) read with
// O_DIRECT into a preallocated, populated region, comparing a pread thread pool (what the engine does) with
// io_uring (one submitter, fixed buffers, registered file) across piece sizes and page sizes.
//
//   readbench <file> <mode: pool|uring|uring-sqpoll> <piece KiB, 0 = whole record> <threads|qd cap> <huge 0|1> [fills] [k]
#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <liburing.h>
#include <mutex>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

static const size_t REC = 6606848; // ~one K3 expert record, 4 KiB aligned

struct piece { char * dst; size_t len; off_t off; };

struct pool {
    std::vector<std::thread> th;
    std::mutex m; std::condition_variable cv, done;
    std::vector<piece> jobs; std::atomic<size_t> next{0}, remaining{0};
    uint64_t gen = 0; bool stop = false; int fd;
    pool(int n, int fd) : fd(fd) { for (int i = 0; i < n; ++i) th.emplace_back([this] { worker(); }); }
    ~pool() { { std::lock_guard<std::mutex> l(m); stop = true; } cv.notify_all(); for (auto & t : th) t.join(); }
    void work() {
        for (;;) {
            size_t i = next.fetch_add(1); if (i >= jobs.size()) return;
            piece & p = jobs[i];
            if (pread(fd, p.dst, p.len, p.off) != (ssize_t) p.len) { perror("pread"); exit(1); }
            if (remaining.fetch_sub(1) == 1) { std::lock_guard<std::mutex> l(m); done.notify_all(); }
        }
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return stop || gen != seen; }); if (stop) return; seen = gen; }
            work();
        }
    }
    void run(std::vector<piece> & in) {
        { std::lock_guard<std::mutex> l(m); jobs.swap(in); remaining = jobs.size(); next = 0; gen++; }
        cv.notify_all(); work();
        std::unique_lock<std::mutex> l(m); done.wait(l, [&] { return remaining.load() == 0; });
    }
};

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    const char * path = argv[1]; std::string mode = argv[2];
    size_t pkib = atol(argv[3]); int par = atoi(argv[4]); bool huge = atoi(argv[5]);
    int fills = argc > 6 ? atoi(argv[6]) : 400; int k = argc > 7 ? atoi(argv[7]) : 7;
    size_t piece_sz = pkib ? pkib * 1024 : REC;

    int fd = open(path, O_RDONLY | O_DIRECT); if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    const size_t n_rec = st.st_size / REC;

    // destination: 64 slots, populated before timing (no allocation or page faults in the measured path)
    const int n_slots = 64; size_t region = n_slots * REC; region = (region + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    char * buf = (char *) mmap(nullptr, region, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(buf, region, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    madvise(buf, region, MADV_POPULATE_WRITE);

    std::mt19937_64 rng(42);
    std::vector<double> ms;
    size_t total = 0;

    pool * pl = nullptr; io_uring ring;
    if (mode == "pool") pl = new pool(par - 1, fd);
    else {
        io_uring_params p{}; if (mode == "uring-sqpoll") { p.flags |= IORING_SETUP_SQPOLL; p.sq_thread_idle = 2000; }
        p.flags |= IORING_SETUP_SINGLE_ISSUER;
        if (io_uring_queue_init_params(512, &ring, &p) < 0) { perror("uring"); return 1; }
        iovec iov{ buf, region }; // one fixed buffer covering the whole region
        if (mode != "uring-plain") {
            if (io_uring_register_buffers(&ring, &iov, 1) < 0) { perror("regbuf"); return 1; }
            if (io_uring_register_files(&ring, &fd, 1) < 0) { perror("regfile"); return 1; }
        }
    }

    for (int f = -20; f < fills; ++f) { // 20 warm-up fills
        std::vector<piece> jobs;
        for (int r = 0; r < k; ++r) {
            off_t off = (off_t) ((rng() % n_rec) * REC);
            char * dst = buf + ((size_t)(((f + 20) * k + r) % n_slots)) * REC;
            for (size_t o = 0; o < REC; o += piece_sz) jobs.push_back({ dst + o, std::min(piece_sz, REC - o), off + (off_t) o });
        }
        auto t0 = std::chrono::steady_clock::now();
        if (pl) pl->run(jobs);
        else {
            size_t sub = 0, comp = 0, n = jobs.size();
            while (comp < n) {
                while (sub < n && sub - comp < (size_t) par) {
                    io_uring_sqe * s = io_uring_get_sqe(&ring); if (!s) break;
                    if (mode == "uring-plain") io_uring_prep_read(s, fd, jobs[sub].dst, jobs[sub].len, jobs[sub].off);
                    else { io_uring_prep_read_fixed(s, 0, jobs[sub].dst, jobs[sub].len, jobs[sub].off, 0); s->flags |= IOSQE_FIXED_FILE; }
                    ++sub;
                }
                io_uring_submit(&ring);
                io_uring_cqe * c; if (io_uring_wait_cqe(&ring, &c) < 0) { perror("wait"); return 1; }
                unsigned head, cnt = 0;
                io_uring_for_each_cqe(&ring, head, c) { if (c->res < 0) { fprintf(stderr, "read %s\n", strerror(-c->res)); return 1; } ++cnt; }
                io_uring_cq_advance(&ring, cnt); comp += cnt;
            }
        }
        double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (f >= 0) { ms.push_back(d); total += (size_t) k * REC; }
    }
    std::sort(ms.begin(), ms.end());
    double sum = 0; for (double d : ms) sum += d;
    printf("%-13s piece %5zuK par %3d huge %d k %d: fill mean %.3f ms p50 %.3f p99 %.3f  %.2f GB/s\n", mode.c_str(),
        piece_sz / 1024, par, huge, k, sum / ms.size(), ms[ms.size() / 2], ms[ms.size() * 99 / 100], total / (sum / 1e3) / 1e9);
    delete pl;
    return 0;
}
