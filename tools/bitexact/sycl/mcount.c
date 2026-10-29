// LD_PRELOAD shim (S1): counts malloc/calloc/realloc/aligned allocations; SIGUSR2 prints the totals to stderr. Build:
//   gcc -O2 -shared -fPIC -o mcount.so mcount.c ; run k3x with LD_PRELOAD=$PWD/mcount.so and signal it before/after a run.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
static atomic_long n_malloc, n_calloc, n_realloc, n_memalign, n_free;
extern void *__libc_malloc(size_t), *__libc_calloc(size_t, size_t), *__libc_realloc(void *, size_t), *__libc_memalign(size_t, size_t);
extern void __libc_free(void *);
void *malloc(size_t n) { atomic_fetch_add_explicit(&n_malloc, 1, memory_order_relaxed); return __libc_malloc(n); }
void *calloc(size_t a, size_t b) { atomic_fetch_add_explicit(&n_calloc, 1, memory_order_relaxed); return __libc_calloc(a, b); }
void *realloc(void *p, size_t n) { atomic_fetch_add_explicit(&n_realloc, 1, memory_order_relaxed); return __libc_realloc(p, n); }
void free(void *p) { if (p) atomic_fetch_add_explicit(&n_free, 1, memory_order_relaxed); __libc_free(p); }
int posix_memalign(void **r, size_t a, size_t n) { atomic_fetch_add_explicit(&n_memalign, 1, memory_order_relaxed); *r = __libc_memalign(a, n); return *r ? 0 : 12; }
void *aligned_alloc(size_t a, size_t n) { atomic_fetch_add_explicit(&n_memalign, 1, memory_order_relaxed); return __libc_memalign(a, n); }
static void dump(int s) { (void) s; char b[200]; int l = snprintf(b, sizeof b, "MCOUNT malloc %ld calloc %ld realloc %ld memalign %ld free %ld\n",
    atomic_load(&n_malloc), atomic_load(&n_calloc), atomic_load(&n_realloc), atomic_load(&n_memalign), atomic_load(&n_free)); write(2, b, l); }
__attribute__((constructor)) static void init(void) { signal(SIGUSR2, dump); }
