// Shared, compute-independent K3XS metadata and bounded disk residency.
// Store format: tools/repack-experts.py. v1: a layer's experts back to back (stride = record_size, 4096 alignment);
// v2: the index entry's last u64 is the stride (record_size rounded up to the header's power-of-two alignment).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef K3X_HAVE_URING
#include <liburing.h>
#endif

namespace k3disk {
using Clock = std::chrono::steady_clock;
struct Record {
    uint64_t gid, offset, bytes, gate, up, down;
    uint32_t tg, tu, td;
    uint64_t stride = 0;   // bytes from one expert's record to the next (v1: = bytes)
};

class Store {
    static uint32_t u32(const uint8_t * p) {
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    }
    static uint64_t u64(const uint8_t * p) { return u32(p) | uint64_t(u32(p + 4)) << 32; }
public:
    uint32_t experts = 0, version = 0, align = 0;
    std::vector<Record> records;
    explicit Store(const char * path) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open K3XS store");
        try {
            struct stat st{};
            uint8_t h[64];
            if (fstat(fd, &st) || pread(fd, h, sizeof h, 0) != sizeof h ||
                memcmp(h, "K3XS", 4) || ((version = u32(h + 4)) != 1 && version != 2) || !(experts = u32(h + 12)) ||
                (align = u32(h + 16)) < 4096 || (align & (align - 1)) || (version == 1 && align != 4096) ||
                u32(h + 20) != 64)
                throw std::runtime_error("invalid K3XS header");
            const uint64_t end = st.st_size, index = u64(h + 24), data = u64(h + 32);
            const uint32_t layers = u32(h + 8);
            if (!layers || index < 64 || index > end || uint64_t(layers) * 64 > end - index ||
                data < index + uint64_t(layers) * 64 || data > end)
                throw std::runtime_error("invalid K3XS index bounds");
            uint32_t previous = 0;
            std::vector<std::pair<uint64_t, uint64_t>> ranges;
            for (uint32_t i = 0; i < layers; ++i) {
                uint8_t e[64];
                if (pread(fd, e, sizeof e, index + uint64_t(i) * 64) != sizeof e)
                    throw std::runtime_error("short K3XS index");
                const uint32_t layer = u32(e);
                Record r{uint64_t(layer) * experts, u64(e + 48), u64(e + 40),
                         u64(e + 16), u64(e + 24), u64(e + 32),
                         u32(e + 4), u32(e + 8), u32(e + 12)};
                r.stride = version == 2 ? u64(e + 56) : r.bytes;
                if ((i && layer <= previous) || !r.bytes || r.bytes % 4096 || r.offset % align ||
                    r.stride < r.bytes || r.stride % align ||
                    r.offset < data || r.offset > end || r.stride > (end - r.offset) / experts ||
                    !r.gate || !r.up || !r.down || r.gate > r.bytes ||
                    r.up > r.bytes - r.gate || r.down > r.bytes - r.gate - r.up)
                    throw std::runtime_error("invalid K3XS layer record");
                previous = layer;
                ranges.emplace_back(r.offset, r.offset + r.stride * experts);
                for (uint32_t j = 0; j < experts; ++j) {
                    records.push_back(r); ++r.gid; r.offset += r.stride;
                }
            }
            std::sort(ranges.begin(), ranges.end());
            for (size_t i = 1; i < ranges.size(); ++i)
                if (ranges[i].first < ranges[i - 1].second)
                    throw std::runtime_error("overlapping K3XS layer records");
        } catch (...) { close(fd); throw; }
        close(fd);
    }
    const Record * find(uint64_t gid) const {
        auto it = std::lower_bound(records.begin(), records.end(), gid,
                                  [](const Record & r, uint64_t g) { return r.gid < g; });
        return it != records.end() && it->gid == gid ? &*it : nullptr;
    }
};

// Bounded LRU of whole records read with O_DIRECT by a fixed pool of reader threads (liburing or pread).
// A record is read in pieces (Options::piece bytes, 0 = one read per record): every reader takes the next piece of the
// oldest queued record, so one record keeps all readers busy (queue depth = threads) and records complete in FIFO order.
// Options::transform, if set, rewrites a record in place while it lands: transform(ctx, record, buf, lo, hi) is called
// once per piece [lo, hi) after that piece and the next one are complete (the caller's units, e.g. rows, may start in
// [lo, hi) and run into the next piece, never further). The record is ready after its last transform.
// Options::arena: caller-owned memory (slots * largest bytes, 4096-aligned; e.g. GPU-visible host memory) instead of
// an internal posix_memalign arena. Nothing allocates after construction.
class Cache {
public:
    using Transform = void (*)(void * ctx, const Record & r, uint8_t * rec, uint64_t lo, uint64_t hi);
    struct Options {
        uint8_t * arena = nullptr;
        size_t piece = 0;
        Transform transform = nullptr;
        void * ctx = nullptr;
    };
    static size_t slots_for(size_t budget, size_t largest) { return largest ? budget / largest : 0; }
private:
    enum State { empty, queued, reading, ready, failed };
    struct Slot {
        Record record{};
        State state = empty;
        size_t pins = 0;
        uint64_t age = 0;
        Clock::time_point posted{}, done{};
        uint64_t piece = 0, mask = 0;     // piece bytes, completed pieces
        unsigned pieces = 0, issued = 0, inflight = 0, transforming = 0, transformed = 0;
        bool bad = false, in_ring = false;
        std::atomic<int> settled{0};      // ready or failed (lock-free poll)
    };
    int fd_ = -1;
    uint8_t * arena_ = nullptr;
    bool own_arena_ = true;
    size_t stride_, nslots_ = 0, piece_ = 0;
    Transform transform_ = nullptr;
    void * ctx_ = nullptr;
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<size_t[]> ring_;   // FIFO of slots with pieces left to issue (capacity nslots_)
    size_t ring_head_ = 0, ring_n_ = 0;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable changed_, work_;
    bool stop_ = false;
    unsigned started_ = 0, busy_ = 0;
    Clock::time_point busy_since_{};
    uint64_t age_ = 0, hits_ = 0, lru_ = 0, bytes_ = 0, misses_ = 0, hist_[64]{}, busy_ns_ = 0;

    void ring_pop() { ring_head_ = (ring_head_ + 1) % nslots_; --ring_n_; }
    // under the lock: a record with nothing in flight is ready (all pieces transformed) or failed (a bad piece)
    void settle(Slot & s, Clock::time_point now) {
        if (s.inflight || s.transforming || s.in_ring || (!s.bad && s.transformed < s.pieces)) return;
        s.done = now;
        s.state = s.bad ? failed : ready;
        if (s.state == ready) {
            ++hits_;
            uint64_t us = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::microseconds>(now - s.posted).count());
            unsigned b = 0; while (us > 1 && b < 63) { us = (us + 1) / 2; ++b; }
            ++hist_[b];
        }
        s.settled.store(1, std::memory_order_release);
        changed_.notify_all();
    }
    void worker() {
#ifdef K3X_HAVE_URING
        io_uring ring{};
        const bool uring = io_uring_queue_init(1, &ring, 0) == 0;
#endif
        std::unique_lock<std::mutex> lock(mutex_);
        ++started_;
        changed_.notify_all();
        for (;;) {
            work_.wait(lock, [&] { return stop_ || ring_n_; });
            if (stop_) break;
            const size_t i = ring_[ring_head_];
            Slot & s = slots_[i];
            if (s.bad) { ring_pop(); s.in_ring = false; settle(s, Clock::now()); continue; }
            const unsigned p = s.issued++;
            if (s.issued == s.pieces) { ring_pop(); s.in_ring = false; }
            s.state = reading;
            ++s.inflight;
            if (!busy_++) busy_since_ = Clock::now();
            const uint64_t lo = p * s.piece, len = std::min<uint64_t>(s.piece, s.record.bytes - lo);
            uint8_t * buf = arena_ + i * stride_;
            const uint64_t off = s.record.offset + lo;
            lock.unlock();
            ssize_t n;
#ifdef K3X_HAVE_URING
            if (uring) {
                io_uring_sqe * sqe = io_uring_get_sqe(&ring);
                io_uring_prep_read(sqe, fd_, buf + lo, len, off);
                int rc = io_uring_submit(&ring);
                io_uring_cqe * cqe = nullptr;
                if (rc == 1) {
                    do { rc = io_uring_wait_cqe(&ring, &cqe); } while (rc == -EINTR);
                } else rc = rc < 0 ? rc : -EIO;
                n = rc < 0 ? rc : cqe->res;
                if (cqe) io_uring_cqe_seen(&ring, cqe);
            } else
#endif
            {
                do { n = pread(fd_, buf + lo, len, off); } while (n < 0 && errno == EINTR);
            }
            auto now = Clock::now();
            lock.lock();
            if (!--busy_) busy_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(now - busy_since_).count();
            if (n > 0) bytes_ += n;
            --s.inflight;
            if (n != static_cast<ssize_t>(len)) s.bad = true;
            else if (!s.bad) {
                // piece p completes the transform windows of p - 1 (if p - 1 was done) and p (if p + 1 was done)
                s.mask |= uint64_t(1) << p;
                const bool t_prev = p > 0 && (s.mask >> (p - 1) & 1);
                const bool t_this = p + 1 == s.pieces || (s.mask >> (p + 1) & 1);
                const unsigned nt = unsigned(t_prev) + unsigned(t_this);
                if (nt && transform_) {
                    s.transforming += nt;
                    const Record r = s.record;
                    const uint64_t piece = s.piece;
                    lock.unlock();
                    if (t_prev) transform_(ctx_, r, buf, lo - piece, lo);
                    if (t_this) transform_(ctx_, r, buf, lo, lo + len);
                    now = Clock::now();
                    lock.lock();
                    s.transforming -= nt;
                }
                s.transformed += nt;
            }
            settle(s, now);
        }
#ifdef K3X_HAVE_URING
        if (uring) io_uring_queue_exit(&ring);
#endif
    }
public:
    Cache(const char * path, size_t budget, size_t largest, unsigned threads) : Cache(path, budget, largest, threads, Options()) {}
    Cache(const char * path, size_t budget, size_t largest, unsigned threads, const Options & o)
        : stride_(largest), piece_(o.piece), transform_(o.transform), ctx_(o.ctx) {
        if (!largest || largest % 4096 || largest > unsigned(std::numeric_limits<int>::max()) ||
            budget / largest == 0 || budget / largest > unsigned(std::numeric_limits<int>::max()) ||
            !threads || threads > 64 || o.piece % 4096 || (o.arena && reinterpret_cast<uintptr_t>(o.arena) % 4096))
            throw std::runtime_error("invalid disk LRU size or worker count");
        nslots_ = budget / largest;
        slots_.reset(new Slot[nslots_]);
        ring_.reset(new size_t[nslots_]);
        if (o.arena) { arena_ = o.arena; own_arena_ = false; }
        else if (posix_memalign(reinterpret_cast<void **>(&arena_), 4096, nslots_ * stride_))
            throw std::runtime_error("disk LRU allocation failed");
        // Fault in every page before serving; the arena is also covered by the server's mlockall.
        memset(arena_, 0, nslots_ * stride_);
        fd_ = open(path, O_RDONLY | O_DIRECT);
        if (fd_ < 0) { if (own_arena_) free(arena_); throw std::runtime_error("cannot open O_DIRECT store"); }
        try {
            workers_.reserve(threads);
            for (unsigned i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return started_ == threads; });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; work_.notify_all(); }
            for (auto & t : workers_) t.join();
            close(fd_); if (own_arena_) free(arena_); throw;
        }
    }
    ~Cache() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; work_.notify_all(); }
        for (auto & t : workers_) t.join();
        close(fd_); if (own_arena_) free(arena_);
    }
    size_t slots() const { return nslots_; }
    size_t stride() const { return stride_; }
    // piece size used for a record of `bytes` (at most 64 pieces per record)
    static uint64_t piece_for(uint64_t bytes, size_t piece) {
        if (!piece || piece >= bytes) return bytes;
        const uint64_t min = ((bytes + 63) / 64 + 4095) / 4096 * 4096;
        return std::max<uint64_t>(piece, min);
    }
    // Handles stay pinned through compute. Timed-out reads own their slot until completion;
    // releasing a handle never cancels or reuses an in-flight buffer.
    int acquire(const Record & r) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (r.bytes > stride_) return -1;
        size_t victim = nslots_;
        for (size_t i = 0; i < nslots_; ++i) {
            Slot & s = slots_[i];
            if (s.state != empty && s.state != failed && s.record.gid == r.gid) {
                ++s.pins; s.age = ++age_; if (s.state == ready) ++lru_; return i;
            }
            if (!s.pins && s.state != queued && s.state != reading &&
                (victim == nslots_ || s.age < slots_[victim].age)) victim = i;
        }
        if (victim == nslots_) return -1;
        Slot & s = slots_[victim];
        s.settled.store(0, std::memory_order_relaxed);
        s.record = r; s.state = queued; s.pins = 1; s.age = ++age_; s.posted = Clock::now();
        s.piece = piece_for(r.bytes, piece_);
        s.pieces = unsigned((r.bytes + s.piece - 1) / s.piece);
        s.mask = 0; s.issued = s.inflight = s.transforming = s.transformed = 0; s.bad = false;
        s.in_ring = true;
        ring_[(ring_head_ + ring_n_++) % nslots_] = victim;
        work_.notify_all();
        return victim;
    }
    // lock-free: the handle's read has finished (ready or failed); wait() then returns without blocking
    bool settled(int handle) const {
        return handle < 0 || slots_[handle].settled.load(std::memory_order_acquire);
    }
    const uint8_t * wait(int handle, Clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (handle < 0) { ++misses_; return nullptr; }
        Slot & s = slots_[handle];
        changed_.wait_until(lock, deadline, [&] { return s.state == ready || s.state == failed; });
        if (s.state != ready || s.done > deadline) {
            if (s.state != failed) ++misses_;
            return nullptr;
        }
        return arena_ + size_t(handle) * stride_;
    }
    void release(int handle) {
        if (handle < 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        --slots_[handle].pins;
    }
    // gbps: bytes over the time at least one read was in flight (the drive's achieved rate while busy)
    void stats(FILE * f) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto percentile = [&](uint64_t percent) {
            uint64_t count = 0, target = hits_ / 100 * percent + (hits_ % 100 * percent + 99) / 100;
            if (!target) return uint64_t(0);
            for (unsigned b = 0; b < 64; ++b) if ((count += hist_[b]) >= target) return uint64_t(1) << b;
            return uint64_t(0);
        };
        fprintf(f, "disk: hits=%llu lru_hits=%llu bytes=%llu p50_us=%llu p99_us=%llu deadline_misses=%llu busy_ms=%.1f gbps=%.2f\n",
                (unsigned long long) hits_, (unsigned long long) lru_, (unsigned long long) bytes_,
                (unsigned long long) percentile(50), (unsigned long long) percentile(99), (unsigned long long) misses_,
                busy_ns_ / 1e6, busy_ns_ ? bytes_ / (double) busy_ns_ : 0.0);
    }
};
} // namespace k3disk
