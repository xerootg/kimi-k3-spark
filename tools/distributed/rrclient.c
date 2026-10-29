// rrclient HOST PORT ITERS k1,k2,... | dist  -- remote expert round trip from the Spark: 17 int32 header (k, ids)
// + 3584 f32 activation out, k x 3584 f32 rows back. "dist" draws k from the replay's per-layer remote counts.
// RR_IDX=<server index file> sends real global ids (k distinct experts of one random layer the server holds, as the
// engine does) instead of random ones, and counts replies with NaN rows (ids the server refused).
// Pacing and tracing (to compare with the engine, which sends one request per layer every few ms):
//   RR_GAP_US=<us>  wait this long between a reply and the next request (default 0 = back to back)
//   RR_SPIN=1       busy-wait the gap instead of sleeping (keeps this side's cpu awake)
//   RR_SEQ=1        put the request's number in header word 16 (k < 16) for a k3x K3X_TRACE dump
//   RR_OUT=<file>   one line per request: seq k t_send t_sent t_first t_done (CLOCK_MONOTONIC_RAW ns)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#define D 3584
static int wr(int fd, const void *p, size_t n) { const char *c = p; while (n) { ssize_t r = send(fd, c, n, 0); if (r <= 0) return 0; c += r; n -= r; } return 1; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static unsigned long long ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec * 1000000000ull + t.tv_nsec; }
static unsigned long long t_first;
static int rd_first(int fd, void *p, size_t n) { char *c = p; t_first = 0; while (n) { ssize_t r = recv(fd, c, n, 0); if (r <= 0) return 0; if (!t_first) t_first = ns(); c += r; n -= r; } return 1; }
static int cmp(const void *a, const void *b) { double x = *(double *) a, y = *(double *) b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(atoi(argv[2])); inet_pton(AF_INET, argv[1], &a.sin_addr);
    if (connect(fd, (struct sockaddr *) &a, sizeof a)) { perror("connect"); return 1; }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    int iters = atoi(argv[3]);
    static int gid[1 << 16]; int ng = 0;
    if (getenv("RR_IDX")) { FILE *f = fopen(getenv("RR_IDX"), "r"); long g, o; int a, b; while (f && ng < (1 << 16) && fscanf(f, "%ld %d %d %ld", &g, &a, &b, &o) == 4) gid[ng++] = (int) g; if (f) fclose(f); fprintf(stderr, "%d ids from %s\n", ng, getenv("RR_IDX")); }
    long nan_replies = 0;
    const long gap_us = getenv("RR_GAP_US") ? atol(getenv("RR_GAP_US")) : 0; const int spin = getenv("RR_SPIN") != NULL, useq = getenv("RR_SEQ") != NULL;
    FILE *out = getenv("RR_OUT") ? fopen(getenv("RR_OUT"), "w") : NULL; long seq = 0;
    if (out) fprintf(out, "seq\tk\tt_send\tt_sent\tt_first\tt_done\n");
    static float x[D], rows[16 * D];
    for (int i = 0; i < D; i++) x[i] = (float) (rand() % 2001 - 1000) / 1000.f;
    // per-layer remote count distribution from the replay (7000-record set, prose-trained, code eval): filled below
    int dist = !strcmp(argv[4], "dist");
    int ks[32], nk = 0;
    if (!dist) for (char *s = strtok(argv[4], ","); s; s = strtok(NULL, ",")) ks[nk++] = atoi(s);
    else { ks[0] = 0; nk = 1; }
    for (int ki = 0; ki < nk; ki++) {
        double *t = malloc(sizeof(double) * iters); int n = 0; double sumk = 0;
        for (int it = 0; it < iters + 10; it++) {
            int k = ks[ki];
            if (dist) { double u = (double) rand() / RAND_MAX; // P(k) for k=0..8+ from the replay histogram (argv[5])
                double c = 0; k = 0; char buf[256]; strcpy(buf, argv[5]); for (char *s = strtok(buf, ","); s; s = strtok(NULL, ","), k++) { c += atof(s); if (u <= c) break; } if (k > 16) k = 16; }
            if (k == 0) continue;
            int hdr[17] = {0}; hdr[0] = k; for (int i = 0; i < k; i++) hdr[1 + i] = rand();
            if (ng > 0) { // k distinct experts of one layer (the index is sorted by gid, 896 per layer)
                int s0 = rand() % ng, L = gid[s0] / 896, lo = s0, hi = s0;
                while (lo > 0 && gid[lo - 1] / 896 == L) lo--;
                while (hi + 1 < ng && gid[hi + 1] / 896 == L) hi++;
                if (k > hi - lo + 1) k = hi - lo + 1;
                hdr[0] = k; for (int i = 0; i < k; i++) hdr[1 + i] = gid[lo + (s0 - lo + i) % (hi - lo + 1)];
            }
            if (gap_us > 0) { if (spin) { unsigned long long tg = ns() + gap_us * 1000ull; while (ns() < tg) {} } else { struct timespec g = { gap_us / 1000000, (gap_us % 1000000) * 1000 }; nanosleep(&g, NULL); } }
            if (useq && hdr[0] < 16) hdr[16] = (int) seq;
            double t0 = now(); unsigned long long a0 = ns(), a1;
            if (!wr(fd, hdr, sizeof hdr) || !wr(fd, x, sizeof x)) { fprintf(stderr, "io error\n"); return 1; }
            a1 = ns();
            if (!rd_first(fd, rows, (size_t) hdr[0] * D * 4)) { fprintf(stderr, "io error\n"); return 1; }
            double dt = now() - t0;
            if (out) fprintf(out, "%ld\t%d\t%llu\t%llu\t%llu\t%llu\n", seq, hdr[0], a0, a1, t_first, ns());
            seq++;
            if (it >= 10) { t[n++] = dt; sumk += k; }
            if (rows[0] != rows[0]) nan_replies++;
        }
        qsort(t, n, sizeof(double), cmp);
        printf("%s k=%s n=%d mean_k=%.2f: p50 %.3f ms  p90 %.3f ms  p99 %.3f ms  min %.3f ms\n", argv[1], dist ? "dist" : (char[8]){0}, n, sumk / n,
               t[n / 2], t[(int) (n * .9)], t[(int) (n * .99)], t[0]);
        if (!dist) printf("   (k=%d)\n", ks[ki]);
        if (ng > 0) printf("   replies with NaN rows: %ld\n", nan_replies);
        free(t);
    }
    if (out) fclose(out);
    return 0;
}
