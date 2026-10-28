// remote-standin PORT [BASE_US PER_K_US]: a stand-in for the remote expert server, for plumbing and timing of the
// engine's remote path (LLAMA_EXPERT_STORE_REMOTE) before a bit-exact server exists. Same wire protocol as
// moebench serve / rrclient: int32 [17] {k, ids} + 3584 f32 in, k x 3584 f32 rows out, in order. It computes
// nothing: it busy-waits BASE_US + k * PER_K_US (default 300 + 450, about nuc2's CPU time for the k experts) and
// replies zero rows. Output text with it on is NOT the gold one (the remote experts contribute 0): plumbing only.
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
static int wr(int fd, const void *p, size_t n) { const char *c = p; while (n) { ssize_t r = send(fd, c, n, MSG_NOSIGNAL); if (r <= 0) return 0; c += r; n -= r; } return 1; }
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: remote-standin PORT [BASE_US PER_K_US]\n"); return 1; }
    int port = atoi(argv[1]);
    double base = argc > 2 ? atof(argv[2]) : 300, perk = argc > 3 ? atof(argv[3]) : 450;
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (struct sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
    fprintf(stderr, "remote-standin on %d, %.0f + k * %.0f us per request, zero rows\n", port, base, perk);
    static float x[D], rows[16 * D];
    for (;;) {
        int c = accept(ls, NULL, NULL);
        if (c < 0) continue;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        long n = 0; double ksum = 0;
        int hdr[17];
        while (rd(c, hdr, sizeof hdr) && rd(c, x, sizeof x)) {
            int k = hdr[0];
            if (k < 1 || k > 16) break;
            double t0 = now_us(), until = t0 + base + k * perk;
            while (now_us() < until) { }
            if (!wr(c, rows, (size_t) k * D * 4)) break;
            n++; ksum += k;
            if (n % 10000 == 0) fprintf(stderr, "%ld requests, mean k %.2f\n", n, ksum / n);
        }
        fprintf(stderr, "connection closed after %ld requests\n", n);
        close(c);
    }
}
