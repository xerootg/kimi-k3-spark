// Run through test_disk_tier.py; stdlib/compiler only. No model or GPU required.
#include "../expert_disk.h"
#include <atomic>
#include <new>
#include <string>
#include <sys/syscall.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static thread_local bool track_allocations = false;
static thread_local size_t allocations = 0;
extern "C" void * __real_malloc(size_t);
extern "C" void * __real_calloc(size_t, size_t);
extern "C" void * __real_realloc(void *, size_t);
extern "C" int __real_posix_memalign(void **, size_t, size_t);
extern "C" void * __real_aligned_alloc(size_t, size_t);
extern "C" void * __wrap_malloc(size_t n) {
    if (track_allocations) ++allocations;
    return __real_malloc(n);
}
extern "C" void * __wrap_calloc(size_t n, size_t size) {
    if (track_allocations) ++allocations;
    return __real_calloc(n, size);
}
extern "C" void * __wrap_realloc(void * p, size_t n) {
    if (track_allocations) ++allocations;
    return __real_realloc(p, n);
}
extern "C" int __wrap_posix_memalign(void ** p, size_t align, size_t n) {
    if (track_allocations) ++allocations;
    return __real_posix_memalign(p, align, n);
}
extern "C" void * __wrap_aligned_alloc(size_t align, size_t n) {
    if (track_allocations) ++allocations;
    return __real_aligned_alloc(align, n);
}
void * operator new(size_t n) {
    if (track_allocations) ++allocations;
    if (void * p = malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { free(p); }
void operator delete[](void * p) noexcept { free(p); }
void operator delete(void * p, size_t) noexcept { free(p); }
void operator delete[](void * p, size_t) noexcept { free(p); }

static std::atomic<bool> delay_read{false}, read_entered{false}, finish_read{false};
static std::atomic<uint64_t> delay_offset{0}, delay_offset2{UINT64_MAX};
static std::atomic<unsigned> reads_blocked{0};
extern "C" ssize_t __wrap_pread(int fd, void * p, size_t n, off_t off) {
    if (delay_read.load() && (uint64_t(off) == delay_offset.load() ||
                             uint64_t(off) == delay_offset2.load())) {
        read_entered = true;
        ++reads_blocked;
        while (!finish_read.load()) std::this_thread::yield();
    }
    return syscall(SYS_pread64, fd, p, n, off);
}

static void u32(uint8_t * p, uint32_t n) {
    for (unsigned i = 0; i < 4; ++i) p[i] = n >> (8 * i);
}
static void u64(uint8_t * p, uint64_t n) {
    for (unsigned i = 0; i < 8; ++i) p[i] = n >> (8 * i);
}
static void write_file(const std::string & path, const std::vector<uint8_t> & b) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0);
    CHECK(write(fd, b.data(), b.size()) == ssize_t(b.size()));
    CHECK(close(fd) == 0);
}
static std::vector<uint8_t> fixture(unsigned experts, unsigned layers = 2) {
    // K3XS v1: LE 64-byte header, 64-byte layer descriptors; data and
    // per-expert stride 4096 aligned. Sparse layer IDs 2,7,...; arbitrary counts.
    std::vector<uint8_t> b(4096 + layers * experts * 4096);
    memcpy(b.data(), "K3XS", 4);
    u32(&b[4], 1); u32(&b[8], layers); u32(&b[12], experts);
    u32(&b[16], 4096); u32(&b[20], 64);
    u64(&b[24], 64); u64(&b[32], 4096);
    for (unsigned l = 0; l < layers; ++l) {
        auto * e = b.data() + 64 + l * 64;
        u32(e, 2 + l * 5);
        // IQ2_XS=17 / IQ3_XXS=18, intentionally independent of CPU compute.
        u32(e + 4, l % 2 ? 18 : 17); u32(e + 8, 17); u32(e + 12, 18);
        u64(e + 16, 256); u64(e + 24, 512); u64(e + 32, 1024);
        u64(e + 40, 4096); u64(e + 48, 4096 + l * experts * 4096);
        for (unsigned x = 0; x < experts; ++x)
            memset(b.data() + 4096 + (l * experts + x) * 4096, 1 + l * experts + x, 4096);
    }
    return b;
}
static auto deadline() { return k3disk::Clock::now() + std::chrono::seconds(5); }
static void bytes(const uint8_t * p, unsigned value) {
    CHECK(p);
    for (unsigned i = 0; i < 4096; ++i) CHECK(p[i] == value);
}
template<class F> static void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    CHECK(rejected);
}
struct Metrics {
    unsigned long long hits, lru, bytes, p50, p99, misses;
};
static Metrics metrics(k3disk::Cache & cache, const std::string & path) {
    const std::string name = path + ".stats";
    FILE * f = fopen(name.c_str(), "w+");
    CHECK(f);
    cache.stats(f);
    rewind(f);
    Metrics m{};
    CHECK(fscanf(f, "disk: hits=%llu lru_hits=%llu bytes=%llu p50_us=%llu p99_us=%llu deadline_misses=%llu",
                 &m.hits, &m.lru, &m.bytes, &m.p50, &m.p99, &m.misses) == 6);
    CHECK(fclose(f) == 0);
    CHECK(unlink(name.c_str()) == 0);
    return m;
}
int main(int argc, char ** argv) {
    CHECK(argc == 2);
    const std::string path = argv[1], bad = path + ".bad";
    for (unsigned layers : {1u, 2u, 5u}) for (unsigned experts : {1u, 3u, 7u}) {
        write_file(path, fixture(experts, layers));
        k3disk::Store s(path.c_str());
        CHECK(s.experts == experts && s.records.size() == size_t(experts) * layers);
        CHECK(s.find(2 * experts)->tg == 17);
        if (layers > 1) CHECK(s.find(7 * experts)->tg == 18);
        CHECK(!s.find(0) && !s.find(3 * experts));
        CHECK(s.find((3 + (layers - 1) * 5) * experts - 1)->offset ==
              4096 + (layers * experts - 1) * 4096);
    }
    auto b = fixture(3);
    for (int kind = 0; kind < 24; ++kind) {
        auto invalid = b;
        switch (kind) {
        case 0: invalid.resize(63); break;
        case 1: invalid[0] = 'X'; break;
        case 2: u32(&invalid[4], 2); break;
        case 3: u32(&invalid[8], UINT32_MAX); break;
        case 4: u32(&invalid[12], 0); break;
        case 5: u32(&invalid[20], 32); break;
        case 6: u64(&invalid[24], UINT64_MAX); break;
        case 7: u64(&invalid[32], 64); break;
        case 8: u64(&invalid[64 + 40], 4095); break;
        case 9: u64(&invalid[64 + 48], 4097); break;
        case 10: u64(&invalid[64 + 16], 4097); break;
        case 11: u32(&invalid[128], 2); break; // duplicate layer/GIDs
        case 12: invalid.resize(invalid.size() - 1); break;
        case 13: u64(&invalid[64 + 48], UINT64_MAX - 4095); break;
        case 14: u32(&invalid[8], 0); break;
        case 15: u32(&invalid[16], 2048); break;
        case 16: u64(&invalid[24], 32); break;
        case 17: u64(&invalid[32], invalid.size() + 1); break;
        case 18: u64(&invalid[64 + 40], 0); break;
        case 19: u64(&invalid[64 + 16], 0); break;
        case 20: u64(&invalid[64 + 24], UINT64_MAX); break;
        case 21: u64(&invalid[64 + 32], 4096); break;
        case 22: u64(&invalid[128 + 48], 4096); break; // identical layer ranges
        case 23: u64(&invalid[128 + 48], 8192); break; // partial layer overlap
        }
        write_file(bad, invalid);
        rejects([&] { k3disk::Store s(bad.c_str()); });
    }
    // Layer IDs must be sorted, but disjoint physical ranges need not be.
    auto reversed = b;
    u64(&reversed[64 + 48], 4096 + 3 * 4096);
    u64(&reversed[128 + 48], 4096);
    write_file(bad, reversed);
    {
        k3disk::Store disjoint(bad.c_str());
        CHECK(disjoint.find(6)->offset == 4096 + 3 * 4096);
        CHECK(disjoint.find(21)->offset == 4096);
    }
    write_file(path, b);
    k3disk::Store s(path.c_str());
    const auto & a = *s.find(6), & c = *s.find(7), & d = *s.find(8);
    rejects([&] { k3disk::Cache x(path.c_str(), 4095, 4096, 1); });
    rejects([&] { k3disk::Cache x(path.c_str(), 8192, 4095, 1); });
    rejects([&] { k3disk::Cache x(path.c_str(), 8192, 4096, 0); });
    rejects([&] { k3disk::Cache x(path.c_str(), 8192, 4096, 65); });
    rejects([&] { k3disk::Cache x(path.c_str(), SIZE_MAX, 4096, 1); });
    {
        k3disk::Cache cache(path.c_str(), 8192, 4096, 4);
        int h = cache.acquire(a), dup = cache.acquire(a), other = cache.acquire(c);
        CHECK(h >= 0 && h == dup && other >= 0 && other != h);
        CHECK(cache.acquire(d) == -1); // all slots pinned
        bytes(cache.wait(h, deadline()), 1);
        bytes(cache.wait(other, deadline()), 2);
        cache.release(h);
        CHECK(cache.acquire(d) == -1); // duplicate still pins A
        cache.release(dup); cache.release(other);
        // Touch A, making C the least recently used.
        h = cache.acquire(a); bytes(cache.wait(h, deadline()), 1); cache.release(h);
        int replaced = cache.acquire(d);
        CHECK(replaced == other);
        bytes(cache.wait(replaced, deadline()), 3); cache.release(replaced);
        // No allocation in caller's cache paths, including misses and waits.
        track_allocations = true;
        h = cache.acquire(a); bytes(cache.wait(h, deadline()), 1); cache.release(h);
        other = cache.acquire(c); bytes(cache.wait(other, deadline()), 2); cache.release(other);
        CHECK(!cache.wait(-1, deadline()));
        track_allocations = false;
        CHECK(allocations == 0);
        // Concurrent handles/readers, with duplicate GIDs and repeated eviction.
        std::atomic<bool> failed{false};
        std::vector<std::thread> readers;
        for (int t = 0; t < 4; ++t) readers.emplace_back([&, t] {
            for (int n = 0; n < 30; ++n) {
                const auto & r = s.records[(n + t) % s.records.size()];
                int slot;
                do { slot = cache.acquire(r); if (slot < 0) std::this_thread::yield(); } while (slot < 0);
                const auto * p = cache.wait(slot, deadline());
                if (!p || p[0] != 1 + (n + t) % s.records.size()) failed = true;
                cache.release(slot);
            }
        });
        for (auto & t : readers) t.join();
        CHECK(!failed);
        cache.stats(stdout);
    }
#ifndef K3X_HAVE_URING
    {
        k3disk::Cache cache(path.c_str(), 8192, 4096, 2);
        delay_offset = a.offset; delay_offset2 = c.offset;
        finish_read = false; reads_blocked = 0; delay_read = true;
        int first = cache.acquire(a), second = cache.acquire(c);
        const auto limit = deadline();
        while (reads_blocked < 2 && k3disk::Clock::now() < limit) std::this_thread::yield();
        CHECK(reads_blocked == 2); // both reader threads are concurrently inside pread
        CHECK(!cache.wait(first, k3disk::Clock::now()));
        CHECK(!cache.wait(second, k3disk::Clock::now()));
        finish_read = true;
        bytes(cache.wait(first, deadline()), 1); bytes(cache.wait(second, deadline()), 2);
        cache.release(first); cache.release(second);
        delay_read = false; delay_offset2 = UINT64_MAX;
    }
    {
        k3disk::Cache cache(path.c_str(), 4096, 4096, 1);
        delay_offset = a.offset; finish_read = false; read_entered = false; delay_read = true;
        int h = cache.acquire(a);
        const auto limit = deadline();
        while (!read_entered && k3disk::Clock::now() < limit) std::this_thread::yield();
        CHECK(read_entered);
        CHECK(!cache.wait(h, k3disk::Clock::now())); // deterministic zero deadline
        CHECK(!cache.wait(h, k3disk::Clock::now() + std::chrono::microseconds(100)));
        const auto timed_out = metrics(cache, path);
        CHECK(!timed_out.hits && !timed_out.bytes && timed_out.misses == 2);
        cache.release(h);
        CHECK(cache.acquire(c) == -1); // timed-out read still owns its buffer
        int duplicate = cache.acquire(a);
        CHECK(duplicate == h);
        finish_read = true;
        bytes(cache.wait(duplicate, deadline()), 1); cache.release(duplicate);
        delay_read = false;
        int next = cache.acquire(c);
        CHECK(next == h); bytes(cache.wait(next, deadline()), 2); cache.release(next);
        // Rechecking A proves no late completion overwrote the new occupant.
        next = cache.acquire(a); bytes(cache.wait(next, deadline()), 1); cache.release(next);
    }
#endif
    // Counters distinguish successful reads, LRU hits, deadlines and read errors.
    {
        k3disk::Cache cache(path.c_str(), 4096, 4096, 1);
        auto m = metrics(cache, path);
        CHECK(!m.hits && !m.lru && !m.bytes && !m.p50 && !m.p99 && !m.misses);
        int h = cache.acquire(a); bytes(cache.wait(h, deadline()), 1); cache.release(h);
        m = metrics(cache, path);
        CHECK(m.hits == 1 && m.bytes == 4096 && !m.lru && !m.misses);
        CHECK(m.p50 >= 1 && m.p99 >= m.p50);
        h = cache.acquire(a); bytes(cache.wait(h, deadline()), 1); cache.release(h);
        CHECK(!cache.wait(-1, k3disk::Clock::now()));
        m = metrics(cache, path);
        CHECK(m.hits == 1 && m.lru == 1 && m.bytes == 4096 && m.misses == 1);
    }
    // EOF and O_DIRECT EINVAL are failures, not deadline misses; slots recover.
    // tmpfs accepts misaligned O_DIRECT reads, so only test EINVAL where the filesystem enforces it.
    bool enforces_align;
    {
        alignas(4096) static char probe[8192];
        int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        CHECK(fd >= 0);
        enforces_align = pread(fd, probe + 1, 4096, 1) < 0;
        close(fd);
    }
    for (uint64_t offset : {uint64_t(b.size() + 4096), uint64_t(1)}) {
        if (offset == 1 && !enforces_align) continue;
        k3disk::Cache cache(path.c_str(), 4096, 4096, 1);
        auto invalid = a; invalid.offset = offset;
        int h = cache.acquire(invalid); CHECK(!cache.wait(h, deadline())); cache.release(h);
        auto m = metrics(cache, path);
        CHECK(!m.hits && !m.bytes && !m.misses);
        h = cache.acquire(c); bytes(cache.wait(h, deadline()), 2); cache.release(h);
        m = metrics(cache, path);
        CHECK(m.hits == 1 && m.bytes == 4096 && !m.misses);
    }
    // A positive but short read records bytes, never a successful disk hit.
    {
        k3disk::Cache cache(path.c_str(), 8192, 8192, 1);
        auto short_read = a; short_read.offset = b.size() - 4096; short_read.bytes = 8192;
        int h = cache.acquire(short_read); CHECK(!cache.wait(h, deadline())); cache.release(h);
        auto m = metrics(cache, path);
        CHECK(!m.hits && m.bytes == 4096 && !m.misses);
        h = cache.acquire(c); bytes(cache.wait(h, deadline()), 2); cache.release(h);
        m = metrics(cache, path);
        CHECK(m.hits == 1 && m.bytes == 8192 && !m.misses);
    }
    unlink(bad.c_str());
    puts("PASS shared disk metadata/cache (counts/types, bounds, pins, LRU, no allocations, concurrency, deadlines)");
}
