// Shared, compute-independent K3XS metadata and bounded disk residency.
// Store format: tools/repack-experts.py. v1: a layer's experts back to back (stride = record_size, 4096 alignment);
// v2: the index entry's last u64 is the stride (record_size rounded up to the header's power-of-two alignment).
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

class Cache {
    enum State { empty, queued, reading, ready, failed };
    struct Slot {
        Record record{};
        State state = empty;
        size_t pins = 0;
        uint64_t age = 0;
        Clock::time_point posted{}, done{};
    };
    int fd_ = -1;
    uint8_t * arena_ = nullptr;
    size_t stride_;
    std::vector<Slot> slots_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable changed_;
    bool stop_ = false;
    unsigned started_ = 0;
    uint64_t age_ = 0, hits_ = 0, lru_ = 0, bytes_ = 0, misses_ = 0, hist_[64]{};

    void worker() {
#ifdef K3X_HAVE_URING
        io_uring ring{};
        const bool uring = io_uring_queue_init(1, &ring, 0) == 0;
#endif
        std::unique_lock<std::mutex> lock(mutex_);
        ++started_;
        changed_.notify_all();
        for (;;) {
            size_t i = slots_.size();
            changed_.wait(lock, [&] {
                if (stop_) return true;
                for (size_t j = 0; j < slots_.size(); ++j)
                    if (slots_[j].state == queued) { i = j; return true; }
                return false;
            });
            if (stop_) break;
            Slot & s = slots_[i];
            s.state = reading;
            const Record r = s.record;
            lock.unlock();
            ssize_t n;
#ifdef K3X_HAVE_URING
            if (uring) {
                io_uring_sqe * sqe = io_uring_get_sqe(&ring);
                io_uring_prep_read(sqe, fd_, arena_ + i * stride_, r.bytes, r.offset);
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
                do { n = pread(fd_, arena_ + i * stride_, r.bytes, r.offset); } while (n < 0 && errno == EINTR);
            }
            const auto done = Clock::now();
            lock.lock();
            if (n > 0) bytes_ += n;
            s.done = done;
            s.state = n == static_cast<ssize_t>(r.bytes) ? ready : failed;
            if (s.state == ready) {
                ++hits_;
                uint64_t us = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::microseconds>(done - s.posted).count());
                unsigned b = 0; while (us > 1 && b < 63) { us = (us + 1) / 2; ++b; }
                ++hist_[b];
            }
            changed_.notify_all();
        }
#ifdef K3X_HAVE_URING
        if (uring) io_uring_queue_exit(&ring);
#endif
    }
public:
    Cache(const char * path, size_t budget, size_t largest, unsigned threads) : stride_(largest) {
        if (!largest || largest % 4096 || largest > unsigned(std::numeric_limits<int>::max()) ||
            budget / largest == 0 || budget / largest > unsigned(std::numeric_limits<int>::max()) ||
            !threads || threads > 64)
            throw std::runtime_error("invalid disk LRU size or worker count");
        slots_.resize(budget / largest);
        if (posix_memalign(reinterpret_cast<void **>(&arena_), 4096, slots_.size() * stride_))
            throw std::runtime_error("disk LRU allocation failed");
        // Fault in every page before serving; the arena is also covered by the server's mlockall.
        memset(arena_, 0, slots_.size() * stride_);
        fd_ = open(path, O_RDONLY | O_DIRECT);
        if (fd_ < 0) { free(arena_); throw std::runtime_error("cannot open O_DIRECT store"); }
        try {
            for (unsigned i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return started_ == threads; });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; changed_.notify_all(); }
            for (auto & t : workers_) t.join();
            close(fd_); free(arena_); throw;
        }
    }
    ~Cache() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; changed_.notify_all(); }
        for (auto & t : workers_) t.join();
        close(fd_); free(arena_);
    }
    // Handles stay pinned through compute. Timed-out reads own their slot until completion;
    // releasing a handle never cancels or reuses an in-flight buffer.
    int acquire(const Record & r) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (r.bytes > stride_) return -1;
        size_t victim = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            Slot & s = slots_[i];
            if (s.state != empty && s.state != failed && s.record.gid == r.gid) {
                ++s.pins; s.age = ++age_; if (s.state == ready) ++lru_; return i;
            }
            if (!s.pins && s.state != queued && s.state != reading &&
                (victim == slots_.size() || s.age < slots_[victim].age)) victim = i;
        }
        if (victim == slots_.size()) return -1;
        Slot & s = slots_[victim];
        s.record = r; s.state = queued; s.pins = 1; s.age = ++age_; s.posted = Clock::now();
        changed_.notify_all();
        return victim;
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
    void stats(FILE * f) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto percentile = [&](uint64_t percent) {
            uint64_t count = 0, target = hits_ / 100 * percent + (hits_ % 100 * percent + 99) / 100;
            if (!target) return uint64_t(0);
            for (unsigned b = 0; b < 64; ++b) if ((count += hist_[b]) >= target) return uint64_t(1) << b;
            return uint64_t(0);
        };
        fprintf(f, "disk: hits=%llu lru_hits=%llu bytes=%llu p50_us=%llu p99_us=%llu deadline_misses=%llu\n",
                (unsigned long long) hits_, (unsigned long long) lru_, (unsigned long long) bytes_,
                (unsigned long long) percentile(50), (unsigned long long) percentile(99), (unsigned long long) misses_);
    }
};
} // namespace k3disk
