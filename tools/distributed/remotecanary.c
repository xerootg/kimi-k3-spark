// remotecanary HOST PORT DUMPDIR NCASES LAYER FIRST STRIDE [K]: end-to-end bit-exactness check of a remote expert
// server through its serving path, against agent A's production dumps (tools/bitexact/FORMAT.md). The server must
// hold the dump's records under their global ids (layer * 896 + FIRST + STRIDE * slot), e.g. rec_L1.bin appended to
// its expert file with index lines for gids 896 + 3 + 28 * i. For each case it sends the dump's x with the case's
// first K slots (default 16) and compares the K returned rows with dn.bin bit for bit. Exit status 0 only if every
// row is identical, or NaN where the dump's row is all zero (k3x answers an all-zero row with NaN, its guard against
// dropped GPU submissions, and the engine then reads that expert locally). Run it before the engine connects (a
// server serves one connection at a time).
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#define D 3584
static int rd(int fd, void *p, size_t n) { char *c = p; while (n) { ssize_t r = recv(fd, c, n, 0); if (r <= 0) return 0; c += r; n -= r; } return 1; }
static int wr(int fd, const void *p, size_t n) { const char *c = p; while (n) { ssize_t r = send(fd, c, n, MSG_NOSIGNAL); if (r <= 0) return 0; c += r; n -= r; } return 1; }
static void *slurp(const char *dir, const char *name, size_t *n) {
    char p[4096]; snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); *n = (size_t) ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*n); if (fread(b, 1, *n, f) != *n) { perror(p); exit(2); } fclose(f); return b;
}
int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: remotecanary HOST PORT DUMPDIR NCASES LAYER FIRST STRIDE [K]\n"); return 2; }
    int ncase = atoi(argv[4]), layer = atoi(argv[5]), first = atoi(argv[6]), stride = atoi(argv[7]);
    int K = argc > 8 ? atoi(argv[8]) : 16;
    size_t nx, ni, nd;
    float *x = slurp(argv[3], "x.bin", &nx); int32_t *ids = slurp(argv[3], "ids.bin", &ni); float *dn = slurp(argv[3], "dn.bin", &nd);
    int avail = (int) (nx / (D * 4)); if (ncase > avail) ncase = avail;
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(atoi(argv[2])); inet_pton(AF_INET, argv[1], &a.sin_addr);
    if (connect(fd, (struct sockaddr *) &a, sizeof a)) { perror("connect"); return 2; }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    static float rows[16 * D];
    long same = 0, tot = 0, rows_ok = 0, rows_tot = 0, nan_rows = 0, zero_nan = 0;
    for (int c = 0; c < ncase; c++) {
        int32_t hdr[17] = {0}; hdr[0] = K;
        for (int j = 0; j < K; j++) hdr[1 + j] = layer * 896 + first + stride * ids[c * 16 + j];
        if (!wr(fd, hdr, sizeof hdr) || !wr(fd, x + (size_t) c * D, D * 4) || !rd(fd, rows, (size_t) K * D * 4)) { fprintf(stderr, "io error at case %d\n", c); return 2; }
        for (int j = 0; j < K; j++) {
            const uint32_t *r = (const uint32_t *) (rows + (size_t) j * D), *e = (const uint32_t *) (dn + ((size_t) c * 16 + j) * D);
            int ok = 1, ezero = 1;
            if (rows[(size_t) j * D] != rows[(size_t) j * D]) nan_rows++;
            for (int i = 0; i < D; i++) { same += r[i] == e[i]; ok &= r[i] == e[i]; ezero &= (e[i] << 1) == 0; }
            if (!ok && ezero && rows[(size_t) j * D] != rows[(size_t) j * D]) { ok = 1; zero_nan++; }
            tot += D; rows_ok += ok; rows_tot++;
        }
    }
    printf("canary %s:%s layer %d: %d cases, rows ok %ld/%ld (bit-identical %ld, NaN for an all-zero row %ld; values %ld/%ld), NaN rows %ld -> %s\n",
           argv[1], argv[2], layer, ncase, rows_ok, rows_tot, rows_ok - zero_nan, zero_nan, same, tot, nan_rows, rows_ok == rows_tot ? "PASS" : "FAIL");
    close(fd);
    return rows_ok == rows_tot ? 0 : 1;
}
