// Row layout (repacked at load, same bytes): each super-block field is gathered per row into its own run,
// fields in the order listed (e.g. IQ2_XS: all qs, then all scales, then all d). Every run starts 16-byte aligned for
// the row lengths used here, so a lane's unit is one aligned vector load and the 16 lanes of a sub-group read one
// contiguous span. FS = field sizes in repacked order, FO = the field offsets inside ggml's block.
template <int T> struct Ty;
#define FIELD(i) (row + NSB * pre[i])
template <> struct Ty<T_IQ2_XS> {
    static constexpr int per_sb = 8, ny = 8, nf = 3, FS[3] = {64, 8, 2}, FO[3] = {2, 66, 0}, pre[3] = {0, 64, 72};
    static inline void ly(int u, const int * xq, int * y) { for (int i = 0; i < 8; i++) y[i] = xq[u * 8 + i]; }
    template <int NSB> static inline float dot(const uint8_t * row, int u, const int * y, const float * xd, const float *, const Tabs & t) {
        const sycl::uint2 qv = *(const sycl::uint2 *) (FIELD(0) + 8 * u);
        const uint32_t sc = FIELD(1)[u];
        const uint32_t qq[4] = {qv.x() & 0xffff, qv.x() >> 16, qv.y() & 0xffff, qv.y() >> 16};
        int s1 = 0, s2 = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t q = qq[l];
            const uint64_t g = t.g2[q & 511], m = sign_mask(q >> 9, t);
            int & s = l < 2 ? s1 : s2;
            s = dp4a(apply_sign((uint32_t) g, (uint32_t) m), y[2 * l], s);
            s = dp4a(apply_sign((uint32_t) (g >> 32), (uint32_t) (m >> 32)), y[2 * l + 1], s);
        }
        return h2f(FIELD(2) + 2 * (u >> 3)) * xd[u] * 0.25f * ((0.5f + (sc & 15)) * s1 + (0.5f + (sc >> 4)) * s2);
    }
};
template <> struct Ty<T_IQ3_XXS> {
    static constexpr int per_sb = 8, ny = 8, nf = 3, FS[3] = {64, 32, 2}, FO[3] = {2, 66, 0}, pre[3] = {0, 64, 96};
    static inline void ly(int u, const int * xq, int * y) { for (int i = 0; i < 8; i++) y[i] = xq[u * 8 + i]; }
    template <int NSB> static inline float dot(const uint8_t * row, int u, const int * y, const float * xd, const float *, const Tabs & t) {
        const sycl::uint2 qv = *(const sycl::uint2 *) (FIELD(0) + 8 * u);
        const uint32_t aux = *(const uint32_t *) (FIELD(1) + 4 * u);
        int s = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t w = l < 2 ? qv.x() : qv.y(), sh = 16 * (l & 1);
            const uint64_t m = sign_mask(aux >> (7 * l), t);
            s = dp4a(apply_sign(t.g3[(w >> sh) & 255], (uint32_t) m), y[2 * l], s);
            s = dp4a(apply_sign(t.g3[(w >> (sh + 8)) & 255], (uint32_t) (m >> 32)), y[2 * l + 1], s);
        }
        return h2f(FIELD(2) + 2 * (u >> 3)) * (0.5f + (aux >> 28)) * 0.5f * xd[u] * s;
    }
};
template <> struct Ty<T_IQ4_XS> {
    static constexpr int per_sb = 8, ny = 8, nf = 4, FS[4] = {128, 4, 2, 2}, FO[4] = {8, 4, 2, 0}, pre[4] = {0, 128, 132, 134};
    static inline void ly(int u, const int * xq, int * y) { for (int i = 0; i < 8; i++) y[i] = xq[u * 8 + i]; }
    template <int NSB> static inline float dot(const uint8_t * row, int u, const int * y, const float * xd, const float *, const Tabs & t) {
        const int sb = u >> 3, ib = u & 7;
        const sycl::uint4 qv = *(const sycl::uint4 *) (FIELD(0) + 16 * u);
        const int ls = ((FIELD(1)[4 * sb + ib / 2] >> 4 * (ib & 1)) & 15) | (((ld16(FIELD(2) + 2 * sb) >> 2 * ib) & 3) << 4);
        int s = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t q = qv[l];
            const uint32_t p0 = t.k4[q & 255], p1 = t.k4[(q >> 8) & 255], p2 = t.k4[(q >> 16) & 255], p3 = t.k4[q >> 24];
            const int lo = (int) ((p0 & 255) | ((p1 & 255) << 8) | ((p2 & 255) << 16) | ((p3 & 255) << 24));
            const int hi = (int) ((p0 >> 8) | (p1 & 0xff00) | ((p2 & 0xff00) << 8) | ((p3 & 0xff00) << 16));
            s = dp4a(lo, y[l], s);
            s = dp4a(hi, y[4 + l], s);
        }
        return h2f(FIELD(3) + 2 * sb) * (float) (ls - 32) * xd[u] * s;
    }
};
static inline void k4_scale_min(const uint8_t * sc, int j, int & s, int & m) {
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
}
template <int T5> struct TyK {   // Q4_K (T5 = 0) and Q5_K (T5 = 1): unit = block pair p of a super-block
    static constexpr int per_sb = 4, ny = 16, nf = T5 ? 4 : 3;
    static constexpr int FS[4] = {128, T5 ? 32 : 12, T5 ? 12 : 4, T5 ? 4 : 0}, FO[4] = {T5 ? 48 : 16, T5 ? 16 : 4, T5 ? 4 : 0, 0};
    static constexpr int pre[4] = {0, 128, T5 ? 160 : 140, T5 ? 172 : 0};
    static inline void ly(int u, const int * xq, int * y) { const int b0 = (u >> 2) * 8 + 2 * (u & 3); for (int i = 0; i < 16; i++) y[i] = xq[b0 * 8 + i]; }
    template <int NSB> static inline float dot(const uint8_t * row, int u, const int * y, const float * xd, const float * xs, const Tabs &) {
        const int sb = u >> 2, j0 = 2 * (u & 3), blk0 = sb * 8 + j0;
        const sycl::uint4 q0 = *(const sycl::uint4 *) (FIELD(0) + 32 * u), q1 = *(const sycl::uint4 *) (FIELD(0) + 32 * u + 16);
        sycl::uint4 h0, h1;
        if (T5) { h0 = *(const sycl::uint4 *) (FIELD(1) + 32 * sb); h1 = *(const sycl::uint4 *) (FIELD(1) + 32 * sb + 16); }
        const uint8_t * sc = FIELD(T5 ? 2 : 1) + 12 * sb, * dm = FIELD(T5 ? 3 : 2) + 4 * sb;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int l = 0; l < 8; l++) {
            const int q = (int) (l < 4 ? q0[l] : q1[l - 4]);
            int lo = q & 0x0F0F0F0F, hi = (q >> 4) & 0x0F0F0F0F;
            if (T5) {
                const int h = (int) (l < 4 ? h0[l] : h1[l - 4]);
                lo |= ((h >> j0) & 0x01010101) << 4;
                hi |= ((h >> (j0 + 1)) & 0x01010101) << 4;
            }
            s0 = dp4a(lo, y[l], s0);
            s1 = dp4a(hi, y[8 + l], s1);
        }
        int sc0, m0, sc1, m1;
        k4_scale_min(sc, j0, sc0, m0); k4_scale_min(sc, j0 + 1, sc1, m1);
        return h2f(dm) * (sc0 * xd[blk0] * s0 + sc1 * xd[blk0 + 1] * s1) - h2f(dm + 2) * (m0 * xs[blk0] + m1 * xs[blk0 + 1]);
    }
};
template <> struct Ty<T_Q4_K> : TyK<0> {};
template <> struct Ty<T_Q5_K> : TyK<1> {};
template <> struct Ty<T_Q6_K> {   // unit = 32-block u = (super-block, half n, t): 32 ql nibbles + 32 qh 2-bit fields, 2 scales
    static constexpr int per_sb = 8, ny = 8, nf = 4, FS[4] = {128, 64, 16, 2}, FO[4] = {0, 128, 192, 208}, pre[4] = {0, 128, 192, 208};
    static inline void ly(int u, const int * xq, int * y) { for (int i = 0; i < 8; i++) y[i] = xq[u * 8 + i]; }
    template <int NSB> static inline float dot(const uint8_t * row, int u, const int * y, const float * xd, const float *, const Tabs &) {
        const int sb = u >> 3, n = (u >> 2) & 1, t = u & 3;
        const uint8_t * ql = FIELD(0) + 128 * sb + 64 * n + 32 * (t & 1), * qh = FIELD(1) + 64 * sb + 32 * n;
        const sycl::uint4 l0 = *(const sycl::uint4 *) ql, l1 = *(const sycl::uint4 *) (ql + 16);
        const sycl::uint4 h0 = *(const sycl::uint4 *) qh, h1 = *(const sycl::uint4 *) (qh + 16);
        const int8_t * sc = (const int8_t *) (FIELD(2) + 16 * sb + 8 * n + 2 * t);
        int sa = 0, sb2 = 0;
#pragma unroll
        for (int w = 0; w < 8; w++) {
            const uint32_t l4 = w < 4 ? l0[w] : l1[w - 4], h4 = w < 4 ? h0[w] : h1[w - 4];
            const uint32_t q = ((l4 >> 4 * (t >> 1)) & 0x0F0F0F0F) | (((h4 >> 2 * t) & 0x03030303) << 4);
            int & s = w < 4 ? sa : sb2;
            s = dp4a((int) ((q + 0x60606060u) ^ 0x80808080u), y[w], s);
        }
        return h2f(FIELD(3) + 2 * sb) * xd[u] * (sc[0] * sa + sc[1] * sb2);
    }
};
#undef FIELD

// host: ggml blocks -> the per-row field runs above (rows of n values)
template <int T> static void repack_rows(const uint8_t * src, uint8_t * dst, size_t nrows, int n) {
    using Y = Ty<T>;
    const int nsb = n / 256; int bsz = 0;
    for (int f = 0; f < Y::nf; f++) bsz += Y::FS[f];
    for (size_t r = 0; r < nrows; r++) {
        const uint8_t * s = src + r * nsb * bsz; uint8_t * d = dst + r * nsb * bsz;
        for (int f = 0; f < Y::nf; f++)
            for (int sb = 0; sb < nsb; sb++) memcpy(d + nsb * Y::pre[f] + sb * Y::FS[f], s + sb * bsz + Y::FO[f], Y::FS[f]);
    }
}
static void repack(int t, const uint8_t * src, uint8_t * dst, size_t nrows, int n) {
    switch (t) {
        case T_IQ2_XS: return repack_rows<T_IQ2_XS>(src, dst, nrows, n); case T_IQ3_XXS: return repack_rows<T_IQ3_XXS>(src, dst, nrows, n);
        case T_IQ4_XS: return repack_rows<T_IQ4_XS>(src, dst, nrows, n); case T_Q4_K: return repack_rows<T_Q4_K>(src, dst, nrows, n);
        case T_Q5_K: return repack_rows<T_Q5_K>(src, dst, nrows, n); case T_Q6_K: return repack_rows<T_Q6_K>(src, dst, nrows, n);
    }
}
