// allocgate.so: LD_PRELOAD allocation counter for the serveidx allocation gate (disk_test.py gate).
//   icx -O2 -shared -fPIC -o allocgate.so allocgate.c -ldl
// Counts, process-wide and only while the window is open: malloc, calloc, realloc, posix_memalign / aligned_alloc /
// memalign / valloc / pvalloc, operator new (all variants), free / operator delete, mmap / munmap / mremap, and the
// Level Zero allocations zeMemAllocHost / zeMemAllocDevice / zeMemAllocShared / zeMemFree plus zeEventCreate,
// zeEventPoolCreate, zeCommandListCreate(Immediate) (diagnostics). The Level Zero counters also run outside the window
// ("total"), so the report shows whether the runtime's calls are reachable by interposition at all.
// SIGUSR2 toggles the window: the first opens it (counters reset), the next closes it and writes one line
//   ALLOCGATE window malloc=.. calloc=.. ... ze_host=.. (total ze_host=..)
// to stderr (write(2) only). ALLOCGATE_BT=<n>: the first n allocations in the window also print a backtrace
// (backtrace_symbols_fd, no malloc) to find the caller. ALLOCGATE_FAULTS=<n>: while the window is open, sample every
// user-mode page fault of every thread (perf_event_open, software page-faults, period 1; needs perf_event_paranoid
// <= 2) and at the close print the first n as "ALLOCGATE fault tid= ip= addr=" plus the /proc/self/maps line of the
// address.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

extern void * __libc_malloc(size_t);
extern void * __libc_calloc(size_t, size_t);
extern void * __libc_realloc(void *, size_t);
extern void * __libc_memalign(size_t, size_t);
extern void __libc_free(void *);

enum { C_MALLOC, C_CALLOC, C_REALLOC, C_MEMALIGN, C_NEW, C_FREE, C_MMAP, C_MUNMAP, C_ZE_HOST, C_ZE_DEV, C_ZE_SHARED,
       C_ZE_FREE, C_ZE_EVENT, C_ZE_POOL, C_ZE_CL, C_N };
static const char * NAMES[C_N] = {"malloc", "calloc", "realloc", "memalign", "new", "free", "mmap", "munmap", "ze_host",
                                  "ze_device", "ze_shared", "ze_free", "ze_event", "ze_eventpool", "ze_cmdlist"};
static atomic_long win[C_N], tot[C_N];
static atomic_int on = 0;
static atomic_long bt_left = 0;
static long bt_max = 0;
static __thread int in_bt = 0;

static void hit(int c) {
    atomic_fetch_add_explicit(&tot[c], 1, memory_order_relaxed);
    if (!atomic_load_explicit(&on, memory_order_relaxed)) return;
    atomic_fetch_add_explicit(&win[c], 1, memory_order_relaxed);
    if (c == C_FREE || c == C_MUNMAP || c == C_ZE_FREE) return;
    if (bt_max && !in_bt && atomic_fetch_sub(&bt_left, 1) > 0) {
        in_bt = 1;
        void * f[32]; const int n = backtrace(f, 32);
        char h[64]; const int l = snprintf(h, sizeof h, "ALLOCGATE bt %s:\n", NAMES[c]);
        if (write(2, h, l) < 0) {}
        backtrace_symbols_fd(f, n, 2);
        in_bt = 0;
    }
}

void * malloc(size_t n) { hit(C_MALLOC); return __libc_malloc(n); }
void * calloc(size_t a, size_t b) { hit(C_CALLOC); return __libc_calloc(a, b); }
void * realloc(void * p, size_t n) { hit(C_REALLOC); return __libc_realloc(p, n); }
void free(void * p) { if (p) hit(C_FREE); __libc_free(p); }
int posix_memalign(void ** r, size_t a, size_t n) { hit(C_MEMALIGN); *r = __libc_memalign(a, n); return *r ? 0 : 12; }
void * aligned_alloc(size_t a, size_t n) { hit(C_MEMALIGN); return __libc_memalign(a, n); }
void * memalign(size_t a, size_t n) { hit(C_MEMALIGN); return __libc_memalign(a, n); }
void * valloc(size_t n) { hit(C_MEMALIGN); return __libc_memalign(4096, n); }
void * pvalloc(size_t n) { hit(C_MEMALIGN); return __libc_memalign(4096, (n + 4095) & ~(size_t) 4095); }

// operator new / delete (Itanium mangled names), counted as "new" / "free"
static void * new_(size_t n) { hit(C_NEW); void * p = __libc_malloc(n ? n : 1); if (!p) abort(); return p; }
static void * new_al(size_t n, size_t a) { hit(C_NEW); void * p = __libc_memalign(a, n ? n : 1); if (!p) abort(); return p; }
void * _Znwm(size_t n) { return new_(n); }
void * _Znam(size_t n) { return new_(n); }
void * _ZnwmRKSt9nothrow_t(size_t n, const void * t) { (void) t; hit(C_NEW); return __libc_malloc(n ? n : 1); }
void * _ZnamRKSt9nothrow_t(size_t n, const void * t) { (void) t; hit(C_NEW); return __libc_malloc(n ? n : 1); }
void * _ZnwmSt11align_val_t(size_t n, size_t a) { return new_al(n, a); }
void * _ZnamSt11align_val_t(size_t n, size_t a) { return new_al(n, a); }
void * _ZnwmSt11align_val_tRKSt9nothrow_t(size_t n, size_t a, const void * t) { (void) t; hit(C_NEW); return __libc_memalign(a, n ? n : 1); }
void * _ZnamSt11align_val_tRKSt9nothrow_t(size_t n, size_t a, const void * t) { (void) t; hit(C_NEW); return __libc_memalign(a, n ? n : 1); }
static void del_(void * p) { if (p) hit(C_FREE); __libc_free(p); }
void _ZdlPv(void * p) { del_(p); }
void _ZdaPv(void * p) { del_(p); }
void _ZdlPvm(void * p, size_t n) { (void) n; del_(p); }
void _ZdaPvm(void * p, size_t n) { (void) n; del_(p); }
void _ZdlPvSt11align_val_t(void * p, size_t a) { (void) a; del_(p); }
void _ZdaPvSt11align_val_t(void * p, size_t a) { (void) a; del_(p); }
void _ZdlPvmSt11align_val_t(void * p, size_t n, size_t a) { (void) n; (void) a; del_(p); }
void _ZdaPvmSt11align_val_t(void * p, size_t n, size_t a) { (void) n; (void) a; del_(p); }
void _ZdlPvRKSt9nothrow_t(void * p, const void * t) { (void) t; del_(p); }
void _ZdaPvRKSt9nothrow_t(void * p, const void * t) { (void) t; del_(p); }

// mmap family (the runtime's own pools, glibc's large mallocs)
typedef void * (*mmap_t)(void *, size_t, int, int, int, off_t);
typedef int (*munmap_t)(void *, size_t);
typedef void * (*mremap_t)(void *, size_t, size_t, int, ...);
static mmap_t r_mmap; static munmap_t r_munmap; static mremap_t r_mremap;
void * mmap(void * a, size_t n, int p, int f, int fd, off_t o) {
    if (!r_mmap) r_mmap = (mmap_t) dlsym(RTLD_NEXT, "mmap");
    hit(C_MMAP); return r_mmap(a, n, p, f, fd, o);
}
void * mmap64(void * a, size_t n, int p, int f, int fd, off_t o) { return mmap(a, n, p, f, fd, o); }
int munmap(void * a, size_t n) {
    if (!r_munmap) r_munmap = (munmap_t) dlsym(RTLD_NEXT, "munmap");
    hit(C_MUNMAP); return r_munmap(a, n);
}
void * mremap(void * a, size_t o, size_t n, int f, ...) {
    if (!r_mremap) r_mremap = (mremap_t) dlsym(RTLD_NEXT, "mremap");
    va_list ap; va_start(ap, f); void * na = (f & MREMAP_FIXED) ? va_arg(ap, void *) : NULL; va_end(ap);
    hit(C_MMAP); return r_mremap(a, o, n, f, na);
}

// Level Zero (ze_result_t is an int-sized enum; handles and descriptors are pointers)
#define ZE_FWD(name, cnt, params, args)                                                     \
    typedef int (*name##_t) params;                                                         \
    static name##_t r_##name;                                                               \
    int name params {                                                                       \
        if (!r_##name) r_##name = (name##_t) dlsym(RTLD_NEXT, #name);                       \
        hit(cnt);                                                                           \
        return r_##name ? r_##name args : 0x78000001; /* ZE_RESULT_ERROR_UNINITIALIZED */   \
    }
ZE_FWD(zeMemAllocHost, C_ZE_HOST, (void * c, const void * d, size_t n, size_t a, void ** p), (c, d, n, a, p))
ZE_FWD(zeMemAllocDevice, C_ZE_DEV, (void * c, const void * d, size_t n, size_t a, void * dev, void ** p), (c, d, n, a, dev, p))
ZE_FWD(zeMemAllocShared, C_ZE_SHARED, (void * c, const void * dd, const void * hd, size_t n, size_t a, void * dev, void ** p),
       (c, dd, hd, n, a, dev, p))
ZE_FWD(zeMemFree, C_ZE_FREE, (void * c, void * p), (c, p))
ZE_FWD(zeEventCreate, C_ZE_EVENT, (void * pool, const void * d, void ** e), (pool, d, e))
ZE_FWD(zeEventPoolCreate, C_ZE_POOL, (void * c, const void * d, uint32_t n, void ** devs, void ** pool), (c, d, n, devs, pool))
ZE_FWD(zeCommandListCreate, C_ZE_CL, (void * c, void * dev, const void * d, void ** cl), (c, dev, d, cl))
ZE_FWD(zeCommandListCreateImmediate, C_ZE_CL, (void * c, void * dev, const void * d, void ** cl), (c, dev, d, cl))

// The UR Level Zero adapter does not link libze_loader: it dlopen()s it and takes the entry points with dlsym, either
// one by one or as DDI tables (zeGet*ProcAddrTable). So dlsym is interposed too: it hands out the wrappers above and
// patches the tables (field order from level_zero/ze_ddi.h: mem {AllocShared, AllocDevice, AllocHost, Free, ...},
// event {Create, ...}, event pool {Create, ...}, command list {Create, CreateImmediate, ...}).
typedef void * (*dlsym_t)(void *, const char *);
static dlsym_t r_dlsym;
typedef int (*ddi_t)(int, void **);
static ddi_t r_mem_tab, r_ev_tab, r_pool_tab, r_cl_tab;
static int mem_tab(int v, void ** t) {
    const int r = r_mem_tab(v, t);
    if (!r) {
        if (t[0] && t[0] != (void *) zeMemAllocShared) { r_zeMemAllocShared = (zeMemAllocShared_t) t[0]; t[0] = (void *) zeMemAllocShared; }
        if (t[1] && t[1] != (void *) zeMemAllocDevice) { r_zeMemAllocDevice = (zeMemAllocDevice_t) t[1]; t[1] = (void *) zeMemAllocDevice; }
        if (t[2] && t[2] != (void *) zeMemAllocHost) { r_zeMemAllocHost = (zeMemAllocHost_t) t[2]; t[2] = (void *) zeMemAllocHost; }
        if (t[3] && t[3] != (void *) zeMemFree) { r_zeMemFree = (zeMemFree_t) t[3]; t[3] = (void *) zeMemFree; }
    }
    return r;
}
static int ev_tab(int v, void ** t) {
    const int r = r_ev_tab(v, t);
    if (!r && t[0] && t[0] != (void *) zeEventCreate) { r_zeEventCreate = (zeEventCreate_t) t[0]; t[0] = (void *) zeEventCreate; }
    return r;
}
static int pool_tab(int v, void ** t) {
    const int r = r_pool_tab(v, t);
    if (!r && t[0] && t[0] != (void *) zeEventPoolCreate) { r_zeEventPoolCreate = (zeEventPoolCreate_t) t[0]; t[0] = (void *) zeEventPoolCreate; }
    return r;
}
static int cl_tab(int v, void ** t) {
    const int r = r_cl_tab(v, t);
    if (!r) {
        if (t[0] && t[0] != (void *) zeCommandListCreate) { r_zeCommandListCreate = (zeCommandListCreate_t) t[0]; t[0] = (void *) zeCommandListCreate; }
        if (t[1] && t[1] != (void *) zeCommandListCreateImmediate) {
            r_zeCommandListCreateImmediate = (zeCommandListCreateImmediate_t) t[1]; t[1] = (void *) zeCommandListCreateImmediate;
        }
    }
    return r;
}
void * dlsym(void * h, const char * name) {
    if (!r_dlsym) {
        r_dlsym = (dlsym_t) dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (!r_dlsym) r_dlsym = (dlsym_t) dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    // a tail call keeps the caller's return address, which RTLD_NEXT (and the caller's namespace) depend on
    if (h == RTLD_NEXT || h == RTLD_DEFAULT || name[0] != 'z' || name[1] != 'e') return r_dlsym(h, name);
    void * p = r_dlsym(h, name);
    if (!p) return p;
    {   // only the loader's entry points (the loader itself dlsym()s the driver's table getters: leave those alone)
        Dl_info di;
        if (!dladdr(p, &di) || !di.dli_fname || !strstr(di.dli_fname, "libze_loader")) return p;
    }
#define SWAP(fn) if (!strcmp(name, #fn)) { r_##fn = (fn##_t) p; return (void *) fn; }
    SWAP(zeMemAllocHost) SWAP(zeMemAllocDevice) SWAP(zeMemAllocShared) SWAP(zeMemFree) SWAP(zeEventCreate)
    SWAP(zeEventPoolCreate) SWAP(zeCommandListCreate) SWAP(zeCommandListCreateImmediate)
#undef SWAP
    if (!strcmp(name, "zeGetMemProcAddrTable")) { r_mem_tab = (ddi_t) p; return (void *) mem_tab; }
    if (!strcmp(name, "zeGetEventProcAddrTable")) { r_ev_tab = (ddi_t) p; return (void *) ev_tab; }
    if (!strcmp(name, "zeGetEventPoolProcAddrTable")) { r_pool_tab = (ddi_t) p; return (void *) pool_tab; }
    if (!strcmp(name, "zeGetCommandListProcAddrTable")) { r_cl_tab = (ddi_t) p; return (void *) cl_tab; }
    return p;
}

// ---- page-fault sampler (ALLOCGATE_FAULTS)
enum { PF_MAX_THREADS = 64, PF_PAGES = 64 };
static long pf_max = 0;
static int pf_fd[PF_MAX_THREADS], pf_n = 0;
static void * pf_ring[PF_MAX_THREADS];
static void pf_open(void) {
    pf_n = 0;
    DIR * d = opendir("/proc/self/task");
    if (!d) return;
    for (struct dirent * e; (e = readdir(d)) && pf_n < PF_MAX_THREADS;) {
        if (e->d_name[0] == '.') continue;
        struct perf_event_attr a; memset(&a, 0, sizeof a);
        a.size = sizeof a; a.type = PERF_TYPE_SOFTWARE; a.config = PERF_COUNT_SW_PAGE_FAULTS; a.sample_period = 1;
        a.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_ADDR; a.exclude_kernel = 1; a.exclude_hv = 1;
        const int fd = (int) syscall(SYS_perf_event_open, &a, atoi(e->d_name), -1, -1, 0);
        if (fd < 0) continue;
        void * r = r_mmap(NULL, (1 + PF_PAGES) * 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (r == MAP_FAILED) { close(fd); continue; }
        pf_fd[pf_n] = fd; pf_ring[pf_n++] = r;
    }
    closedir(d);
}
static char pf_maps[1 << 20];
static void pf_report(void) {
    int ml = 0;
    {
        const int fd = open("/proc/self/maps", O_RDONLY);
        for (int n; fd >= 0 && ml < (int) sizeof pf_maps - 1 && (n = read(fd, pf_maps + ml, sizeof pf_maps - 1 - ml)) > 0;) ml += n;
        if (fd >= 0) close(fd);
        pf_maps[ml] = 0;
    }
    long shown = 0, total = 0;
    for (int t = 0; t < pf_n; t++) {
        struct perf_event_mmap_page * m = pf_ring[t];
        const uint64_t head = __atomic_load_n(&m->data_head, __ATOMIC_ACQUIRE);
        const char * data = (const char *) pf_ring[t] + 4096;
        for (uint64_t pos = m->data_tail; pos < head;) {
            const struct perf_event_header * h = (const void *) (data + pos % (PF_PAGES * 4096));
            if (h->type == PERF_RECORD_SAMPLE) {
                const uint64_t * v = (const uint64_t *) (h + 1);   // ip, pid/tid, addr
                total++;
                if (shown < pf_max) {
                    shown++;
                    char b[512]; int l = snprintf(b, sizeof b, "ALLOCGATE fault tid=%u ip=0x%lx addr=0x%lx", (unsigned) (v[1] >> 32),
                                                  (unsigned long) v[0], (unsigned long) v[2]);
                    for (char * line = pf_maps; *line;) {   // "start-end perms offset dev inode path"
                        char * nl = strchr(line, '\n'); if (!nl) nl = line + strlen(line);
                        char * dash; const unsigned long lo = strtoul(line, &dash, 16), hi = strtoul(dash + 1, NULL, 16);
                        if (v[2] >= lo && v[2] < hi) { l += snprintf(b + l, sizeof b - l, "  %.*s", (int) (nl - line), line); break; }
                        line = *nl ? nl + 1 : nl;
                    }
                    if (l > (int) sizeof b - 2) l = sizeof b - 2;
                    b[l++] = '\n';
                    if (write(2, b, l) < 0) {}
                }
            }
            pos += h->size;
        }
        r_munmap(pf_ring[t], (1 + PF_PAGES) * 4096); close(pf_fd[t]);
    }
    char b[128]; const int l = snprintf(b, sizeof b, "ALLOCGATE faults sampled=%ld in %d threads\n", total, pf_n);
    if (write(2, b, l) < 0) {}
    pf_n = 0;
}

static void report(void) {
    char b[1024]; int l = snprintf(b, sizeof b, "ALLOCGATE window");
    for (int i = 0; i < C_N; i++) l += snprintf(b + l, sizeof b - l, " %s=%ld", NAMES[i], atomic_load(&win[i]));
    l += snprintf(b + l, sizeof b - l, " (total ze_host=%ld ze_device=%ld ze_shared=%ld ze_event=%ld ze_cmdlist=%ld)\n",
                  atomic_load(&tot[C_ZE_HOST]), atomic_load(&tot[C_ZE_DEV]), atomic_load(&tot[C_ZE_SHARED]),
                  atomic_load(&tot[C_ZE_EVENT]), atomic_load(&tot[C_ZE_CL]));
    if (write(2, b, l) < 0) {}
}
static void toggle(int s) {
    (void) s;
    if (!atomic_load(&on)) {
        for (int i = 0; i < C_N; i++) atomic_store(&win[i], 0);
        atomic_store(&bt_left, bt_max);
        if (pf_max) pf_open();
        atomic_store(&on, 1);
        static const char m[] = "ALLOCGATE on\n";
        if (write(2, m, sizeof m - 1) < 0) {}
    } else {
        atomic_store(&on, 0);
        report();
        if (pf_max) pf_report();
    }
}
__attribute__((constructor)) static void init(void) {
    r_mmap = (mmap_t) dlsym(RTLD_NEXT, "mmap");
    r_munmap = (munmap_t) dlsym(RTLD_NEXT, "munmap");
    r_mremap = (mremap_t) dlsym(RTLD_NEXT, "mremap");
    if (getenv("ALLOCGATE_FAULTS")) pf_max = atol(getenv("ALLOCGATE_FAULTS"));
    if (getenv("ALLOCGATE_BT")) { bt_max = atol(getenv("ALLOCGATE_BT")); void * f[2]; backtrace(f, 2); }   // loads libgcc now
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = toggle; sa.sa_flags = SA_RESTART;
    sigaction(SIGUSR2, &sa, NULL);
}
