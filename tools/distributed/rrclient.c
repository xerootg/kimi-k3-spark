// rrclient HOST PORT ITERS k1,k2,... | dist  -- remote expert round trip from the Spark: 17 int32 header (k, ids)
// + 3584 f32 activation out, k x 3584 f32 rows back. "dist" draws k from the replay's per-layer remote counts.
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
static int rd(int fd, void *p, size_t n) { char *c = p; while (n) { ssize_t r = recv(fd, c, n, 0); if (r <= 0) return 0; c += r; n -= r; } return 1; }
static int wr(int fd, const void *p, size_t n) { const char *c = p; while (n) { ssize_t r = send(fd, c, n, 0); if (r <= 0) return 0; c += r; n -= r; } return 1; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp(const void *a, const void *b) { double x = *(double *) a, y = *(double *) b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(atoi(argv[2])); inet_pton(AF_INET, argv[1], &a.sin_addr);
    if (connect(fd, (struct sockaddr *) &a, sizeof a)) { perror("connect"); return 1; }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    int iters = atoi(argv[3]);
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
            int hdr[17]; hdr[0] = k; for (int i = 0; i < k; i++) hdr[1 + i] = rand();
            double t0 = now();
            if (!wr(fd, hdr, sizeof hdr) || !wr(fd, x, sizeof x) || !rd(fd, rows, (size_t) k * D * 4)) { fprintf(stderr, "io error\n"); return 1; }
            double dt = now() - t0;
            if (it >= 10) { t[n++] = dt; sumk += k; }
        }
        qsort(t, n, sizeof(double), cmp);
        printf("%s k=%s n=%d mean_k=%.2f: p50 %.3f ms  p90 %.3f ms  p99 %.3f ms  min %.3f ms\n", argv[1], dist ? "dist" : (char[8]){0}, n, sumk / n,
               t[n / 2], t[(int) (n * .9)], t[(int) (n * .99)], t[0]);
        if (!dist) printf("   (k=%d)\n", ks[ki]);
        free(t);
    }
    return 0;
}
