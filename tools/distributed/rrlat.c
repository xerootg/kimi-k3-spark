// rrlat: TCP request/response latency with the host-side latency levers as flags, to attribute and cut the
// Spark <-> NUC round trip of the remote-expert tier (14 KB activation out, k x 14 KB rows back).
//
//   rrlat serve PORT [opts]                    echo server: reads REQ bytes, busy-waits -w us, replies RESP bytes
//   rrlat client HOST PORT REQ RESP N [opts]   prints p50/p90/p99/min/max in microseconds
//
// opts (both sides):
//   -L US   hold /dev/cpu_dma_latency at US for the life of the process (root; 0 = no C-states beyond polling)
//   -B US   SO_BUSY_POLL US on the data socket (root to raise it), plus SO_PREFER_BUSY_POLL
//   -S      spin: non-blocking recv loop instead of sleeping in recv (user-space busy polling)
//   -C CPU  pin the process to CPU
//   -I DEV  SO_BINDTODEVICE (e.g. force the LAN NIC when a host route points over Thunderbolt; both ends need it,
//           or the reply takes the other path)
//   -w US   server: busy-wait US before replying (stand-in for expert compute)
//   -g US   client: idle gap between requests (sleep, like the engine between layers; default 0)
//   -G      client: busy-wait the gap instead of sleeping
// The client sends REQ bytes; the first 4 are RESP so one server serves any size. The server reports, in the first
// 4 bytes of each reply, the time from the first to the last byte of that request (us*100) -- packet spread seen
// by the receiver (interrupt coalescing / GRO / pacing on the way in).
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifndef SO_PREFER_BUSY_POLL
#define SO_PREFER_BUSY_POLL 69
#endif

static int spin, busy_us = -1, lat_us = -1, cpu = -1, wait_us, gap_us, gap_spin;
static const char *dev;

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

static int rd(int fd, void *p, size_t n, double *first) {
    char *c = p;
    while (n) {
        ssize_t r = recv(fd, c, n, spin ? MSG_DONTWAIT : 0);
        if (r < 0 && (errno == EAGAIN || errno == EINTR)) continue;
        if (r <= 0) return 0;
        if (first && *first == 0) *first = now_us();
        c += r; n -= r;
    }
    return 1;
}
static int wr(int fd, const void *p, size_t n) {
    const char *c = p;
    while (n) {
        ssize_t r = send(fd, c, n, MSG_NOSIGNAL);
        if (r < 0 && (errno == EAGAIN || errno == EINTR)) continue;
        if (r <= 0) return 0;
        c += r; n -= r;
    }
    return 1;
}

static void tune_sock(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (busy_us >= 0) {
        if (setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &busy_us, sizeof busy_us)) perror("SO_BUSY_POLL");
        if (setsockopt(fd, SOL_SOCKET, SO_PREFER_BUSY_POLL, &one, sizeof one)) perror("SO_PREFER_BUSY_POLL");
    }
}

static void tune_proc(void) {
    if (cpu >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); if (sched_setaffinity(0, sizeof s, &s)) perror("affinity"); }
    if (lat_us >= 0) {
        int fd = open("/dev/cpu_dma_latency", O_WRONLY);  // kept open on purpose: the request lasts while it is open
        int32_t v = lat_us;
        if (fd < 0 || write(fd, &v, sizeof v) != sizeof v) perror("cpu_dma_latency");
    }
}

static int cmp(const void *a, const void *b) { double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }

static int opts(int argc, char **argv, int i) {
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "-S")) spin = 1;
        else if (!strcmp(argv[i], "-G")) gap_spin = 1;
        else if (i + 1 < argc && !strcmp(argv[i], "-L")) lat_us = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "-B")) busy_us = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "-C")) cpu = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "-I")) dev = argv[++i];
        else if (i + 1 < argc && !strcmp(argv[i], "-w")) wait_us = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "-g")) gap_us = atoi(argv[++i]);
        else { fprintf(stderr, "bad option %s\n", argv[i]); return 0; }
    }
    return 1;
}

static int serve(int port) {
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (dev && setsockopt(ls, SOL_SOCKET, SO_BINDTODEVICE, dev, strlen(dev))) perror("SO_BINDTODEVICE");
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port);
    if (bind(ls, (struct sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
    static char buf[1 << 20];
    for (;;) {
        int c = accept(ls, NULL, NULL);
        if (c < 0) continue;
        tune_sock(c);
        uint32_t hdr[2];
        while (1) {
            double first = 0;
            if (!rd(c, hdr, 8, &first)) break;
            uint32_t req = hdr[0], resp = hdr[1];
            if (req < 8 || req > sizeof buf || resp < 4 || resp > sizeof buf) break;
            if (!rd(c, buf, req - 8, NULL)) break;
            double t = now_us();
            *(uint32_t *) buf = (uint32_t) ((t - first) * 100);
            if (wait_us) while (now_us() < t + wait_us) { }
            if (!wr(c, buf, resp)) break;
        }
        close(c);
    }
}

static int client(const char *host, int port, int req, int resp, int n) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (dev && setsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, dev, strlen(dev))) perror("SO_BINDTODEVICE");
    tune_sock(fd);
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, host, &a.sin_addr);
    if (connect(fd, (struct sockaddr *) &a, sizeof a)) { perror("connect"); return 1; }
    static char out[1 << 20], in[1 << 20];
    memset(out, 1, sizeof out);
    double *t = malloc(sizeof(double) * n), *sp = malloc(sizeof(double) * n);
    for (int it = -20; it < n; it++) {
        if (gap_us) {
            if (gap_spin) { double u = now_us() + gap_us; while (now_us() < u) { } }
            else { struct timespec s = {0, gap_us * 1000L}; nanosleep(&s, NULL); }
        }
        ((uint32_t *) out)[0] = req; ((uint32_t *) out)[1] = resp;
        double t0 = now_us();
        if (!wr(fd, out, req) || !rd(fd, in, resp, NULL)) { fprintf(stderr, "io error\n"); return 1; }
        double dt = now_us() - t0;
        if (it >= 0) { t[it] = dt; sp[it] = *(uint32_t *) in / 100.0; }
    }
    qsort(t, n, sizeof(double), cmp); qsort(sp, n, sizeof(double), cmp);
    printf("%s req=%d resp=%d n=%d gap=%d%s: p50 %.0f  p90 %.0f  p99 %.0f  min %.0f  max %.0f us | req spread at server p50 %.1f p90 %.1f us\n",
           host, req, resp, n, gap_us, gap_spin ? "spin" : "", t[n / 2], t[n * 9 / 10], t[n * 99 / 100], t[0], t[n - 1], sp[n / 2], sp[n * 9 / 10]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 3 && !strcmp(argv[1], "serve")) { if (!opts(argc, argv, 3)) return 1; tune_proc(); return serve(atoi(argv[2])); }
    if (argc >= 7 && !strcmp(argv[1], "client")) {
        if (!opts(argc, argv, 7)) return 1;
        tune_proc();
        return client(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    }
    fprintf(stderr, "usage: rrlat serve PORT [opts] | rrlat client HOST PORT REQ RESP N [opts]\n");
    return 1;
}
