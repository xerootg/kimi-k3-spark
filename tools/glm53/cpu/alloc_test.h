// Test builds only: counts allocations across caller and all compute workers.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <sys/resource.h>
inline rusage request_faults{};
inline std::atomic<bool> alloc_gate{false};
inline std::atomic<long> alloc_count{0};
extern "C" void *__real_malloc(size_t);
extern "C" void *__real_calloc(size_t, size_t);
extern "C" void *__real_realloc(void *, size_t);
extern "C" void *__real_aligned_alloc(size_t, size_t);
extern "C" int __real_posix_memalign(void **, size_t, size_t);
extern "C" void *__wrap_malloc(size_t n) {
  if (alloc_gate)
    alloc_count++;
  return __real_malloc(n);
}
extern "C" void *__wrap_calloc(size_t a, size_t b) {
  if (alloc_gate)
    alloc_count++;
  return __real_calloc(a, b);
}
extern "C" void *__wrap_realloc(void *p, size_t n) {
  if (alloc_gate)
    alloc_count++;
  return __real_realloc(p, n);
}
extern "C" void *__wrap_aligned_alloc(size_t a, size_t n) {
  if (alloc_gate)
    alloc_count++;
  return __real_aligned_alloc(a, n);
}
extern "C" int __wrap_posix_memalign(void **p, size_t a, size_t n) {
  if (alloc_gate)
    alloc_count++;
  return __real_posix_memalign(p, a, n);
}
void *operator new(size_t n) {
  if (alloc_gate)
    alloc_count++;
  auto *p = __real_malloc(n ? n : 1);
  if (!p)
    throw std::bad_alloc();
  return p;
}
void *operator new[](size_t n) { return ::operator new(n); }
void operator delete(void *p) noexcept { free(p); }
void operator delete[](void *p) noexcept { free(p); }
void operator delete(void *p, size_t) noexcept { free(p); }
void operator delete[](void *p, size_t) noexcept { free(p); }
inline void allocation_begin() {
  getrusage(RUSAGE_SELF, &request_faults);
  alloc_count = 0;
  alloc_gate = true;
}
inline void allocation_end() {
  alloc_gate = false;
  if (alloc_count) {
    fprintf(stderr, "FAIL request allocations=%ld\n", alloc_count.load());
    abort();
  }
  rusage after{};
  getrusage(RUSAGE_SELF, &after);
  long minor = after.ru_minflt - request_faults.ru_minflt,
       major = after.ru_majflt - request_faults.ru_majflt;
  if ((minor || major) && !getenv("G53_ALLOW_UNLOCKED")) {
    fprintf(stderr, "FAIL locked request page faults=%ld/%ld\n", minor, major);
    abort();
  }
  fprintf(stderr, "PASS request allocations=0 minor/major faults=%ld/%ld\n",
          minor, major);
}
