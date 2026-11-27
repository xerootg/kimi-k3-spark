#!/usr/bin/env python3
"""Per-token GPU breakdown of an nsys cuda_gpu_trace CSV of a 64-token Flash decode (kw legs).
Tokens are delimited by the Q4_K output-head GEMV ((ggml_type)12). Prints wall and busy per token (median over the
steady tokens), the kernel families, the mailbox waits after k_bank_request, and one mid layer in full with --layer."""
import csv, re, sys, collections, statistics, argparse
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--layer", type=int, default=-1); ap.add_argument("--skip", type=int, default=8)
a = ap.parse_args()
rows = sorted((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"]) for r in csv.DictReader(open(a.csv)))
outs = [i for i, (s, d, n) in enumerate(rows) if "(ggml_type)12," in n]
def short(n):
    m = re.match(r"(?:void )?([\w:\[\] -]+?)(<[^(]*>)?(\(|$)", n)
    if not m: return n[:60]
    t = m.group(2) or ""; t = re.sub(r"\(ggml_type\)(\d+)", r"T\1", t); t = re.sub(r"\((?:int|bool|unsigned long|long)\)", "", t); t = re.sub(r"\s", "", t)
    return m.group(1).strip() + t
def fam(n):
    if "mul_mat_vec_q" in n:
        t = n.split("(ggml_type)")[1].split(",")[0]
        return {"13":"spine Q5_K","14":"spine Q6_K","8":"spine Q8_0","12":"output Q4_K","17":"experts IQ2_XS","18":"experts IQ3_XXS","23":"experts IQ4_XS"}.get(t, "mmvq T"+t) + (" (moe)" if "_moe" in n else "")
    if "mul_mat_vec_f" in n: return "router F32"
    if "quantize_q8_1" in n: return "quantize"
    if "memcpy" in n or "memset" in n: return "memcpy"
    return "small kernels"
walls, busies, fams, waits, kern_n, covered = [], [], collections.Counter(), [], [], []
for t in range(a.skip, len(outs) - 1):
    a0, b0 = outs[t], outs[t + 1]; tok = rows[a0 + 1:b0 + 1]
    walls.append((rows[b0][0] + rows[b0][1] - rows[a0 + 1][0]) / 1e3); busies.append(sum(d for s, d, n in tok) / 1e3); kern_n.append(len(tok))
    # union of the kernel intervals: kernels on several streams overlap, so the plain sum over-counts; idle = wall - union
    cov = 0; cs, ce = tok[0][0], tok[0][0] + tok[0][1]
    for s, d, n in tok[1:]:
        if s > ce: cov += ce - cs; cs, ce = s, s + d
        else: ce = max(ce, s + d)
    cov += ce - cs; covered.append(cov / 1e3)
    f = collections.Counter()
    for s, d, n in tok: f[fam(n)] += d / 1e3
    fams.update(f)
    for i in range(len(tok) - 1):
        if tok[i][2].startswith("k_bank_request"): waits.append((tok[i + 1][0] - (tok[i][0] + tok[i][1])) / 1e3)
nt = len(walls)
print(f"{a.csv}: {len(outs)} tokens traced, {nt} measured; per token: wall median {statistics.median(walls):.1f} us, kernel sum {statistics.median(busies):.1f} us (overlapping streams double-count), GPU covered {statistics.median(covered):.1f} us, idle {statistics.median(walls)-statistics.median(covered):.1f} us, kernels {statistics.median(kern_n):.0f}")
for k, v in fams.most_common(): print(f"  {k:18s} {v/nt:8.1f} us/token")
if waits:
    w = sorted(waits); print(f"  mailbox waits after k_bank_request: {len(waits)/nt:.0f}/token, p50 {w[len(w)//2]:.1f} us, p90 {w[int(len(w)*.9)]:.1f} us, sum {sum(waits)/nt:.0f} us/token")
if a.layer >= 0:
    a0, b0 = outs[a.skip + 20], outs[a.skip + 21]; tok = rows[a0 + 1:b0 + 1]
    segs, cur = [], []
    for s, d, n in tok:
        if n.startswith("k_bank_request") and cur: segs.append(cur); cur = []
        cur.append((s, d, n))
    segs.append(cur)
    for si in (a.layer, a.layer + 1):
        seg = segs[si]; t0 = seg[0][0]; print(f"--- segment {si}: {len(seg)} kernels, span {(seg[-1][0]+seg[-1][1]-t0)/1e3:.1f} us")
        for s, d, n in seg: print(f"  +{(s-t0)/1e3:8.1f} {d/1e3:7.1f}  {short(n)[:90]}")
