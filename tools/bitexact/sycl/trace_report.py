#!/usr/bin/env python3
"""Per-stage breakdown of the remote expert round trip from the trace dumps.

  trace_report.py [--server K3X_TRACE.tsv] [--engine LLAMA_EXPERT_STORE_REMOTE_TRACE.tsv.0] [--client RR_OUT.tsv]
                  [--k K] [--conn N] [--last N]

--server: a k3x serveidx dump (K3X_TRACE=1, optionally K3X_TRACE_PROF=1 / K3X_TRACE_FREQ=1).
--engine: the engine's dump for one box (LLAMA_EXPERT_STORE_REMOTE_TRACE=<path> writes <path>.<box>).
--client: rrclient RR_OUT. Engine or client rows are joined with the server's by the sequence number in header word 16
(each side's clock is only compared with itself: the network time is the client's span minus the server's span).
All times in microseconds: p50 / p90 / mean over the requests that have both stamps.
"""
import argparse
import csv
import sys


def load(path):
    with open(path) as f:
        return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f, delimiter="\t")]


def stats(v):
    v = sorted(v)
    if not v:
        return None
    return v[len(v) // 2], v[min(len(v) - 1, int(len(v) * 0.9))], sum(v) / len(v), len(v)


def table(title, rows):
    print(f"\n{title}")
    print(f"  {'stage':<46} {'n':>6} {'p50':>9} {'p90':>9} {'mean':>9}")
    for name, vals in rows:
        s = stats(vals)
        if s:
            print(f"  {name:<46} {s[3]:>6} {s[0]:>9.1f} {s[1]:>9.1f} {s[2]:>9.1f}")


def d(r, a, b):
    """r[b] - r[a] in us, or None when either stamp is missing"""
    if r.get(a, 0) and r.get(b, 0):
        return (r[b] - r[a]) / 1e3
    return None


def col(rs, f):
    out = []
    for r in rs:
        v = f(r)
        if v is not None:
            out.append(v)
    return out


def server_rows(S):
    rows = [
        ("idle before the request (t_first - t_wait)", col(S, lambda r: d(r, "t_wait", "t_first"))),
        ("header+x received after the first byte", col(S, lambda r: d(r, "t_first", "t_rx"))),
        ("ids looked up, x into host USM", col(S, lambda r: d(r, "t_rx", "t_x"))),
        ("submit: x copy", col(S, lambda r: d(r, "t_x", "t_sub0"))),
        ("submit: q8_1(x)", col(S, lambda r: d(r, "t_sub0", "t_sub1"))),
        ("submit: gate/up", col(S, lambda r: d(r, "t_sub1", "t_sub2"))),
        ("submit: SiTU + q8_1(h)", col(S, lambda r: d(r, "t_sub2", "t_sub3"))),
        ("submit: down 0 + row 0 copy", col(S, lambda r: d(r, "t_sub3", "t_sub5"))),
        ("submit: down 1 + row 1 copy", col(S, lambda r: d(r, "t_sub5", "t_sub7") if r["k"] >= 2 else None)),
        ("all submissions (t_x -> last submit)", col(S, lambda r: d(r, "t_x", "t_sub%d" % (3 + 2 * min(r["k"], 4))))),
        ("x in USM -> row 0 in host memory", col(S, lambda r: d(r, "t_x", "t_row0"))),
        ("row 0 send()", col(S, lambda r: d(r, "t_row0", "t_sent0"))),
        ("row 0 sent -> row 1 in host memory", col(S, lambda r: d(r, "t_sent0", "t_row1") if r["k"] >= 2 else None)),
        ("row 1 send()", col(S, lambda r: d(r, "t_row1", "t_sent1") if r["k"] >= 2 else None)),
        ("queue drained after the last send", col(S, lambda r: d(r, "t_sent%d" % (min(r["k"], 4) - 1), "t_end"))),
        ("SERVER: first byte -> last row sent", col(S, lambda r: d(r, "t_first", "t_sent%d" % (min(r["k"], 4) - 1)))),
        ("SERVER: x received -> last row sent", col(S, lambda r: d(r, "t_rx", "t_sent%d" % (min(r["k"], 4) - 1)))),
    ]
    if any(r.get("d_start1", 0) for r in S):
        dv = []
        names = ["x copy", "q8_1(x)", "gate/up", "SiTU+q8_1(h)", "down 0", "row 0 copy", "down 1", "row 1 copy"]
        for i, nm in enumerate(names):
            # d_sub (command_submit) is on another time base than start/end with Level Zero on the Arc iGPU: not used
            dv.append((f"GPU {nm}: run", col(S, lambda r, i=i: d(r, f"d_start{i}", f"d_end{i}"))))
            if i:
                dv.append((f"GPU gap before {nm}", col(S, lambda r, i=i: d(r, f"d_end{i-1}", f"d_start{i}"))))
        dv.append(("GPU: x copy start -> last row copy end", col(S, lambda r: d(r, "d_start0", "d_end%d" % (3 + 2 * min(r["k"], 4))))))
        rows += dv
    if any(r.get("f_before", 0) or r.get("f_after", 0) for r in S):
        rows.append(("GT MHz before the GPU work (not us)", [r["f_before"] for r in S if r.get("t_x")]))
        rows.append(("GT MHz after (not us)", [r["f_after"] for r in S if r.get("t_x")]))
    return rows


def engine_rows(E):
    return [
        ("early lookup hook -> rpost hook (GPU)", col(E, lambda r: d(r, "t_early", "t_hook"))),
        ("rpost hook -> cache lock", col(E, lambda r: d(r, "t_hook", "t_lock"))),
        ("lock -> posted (x copy)", col(E, lambda r: d(r, "t_lock", "t_post"))),
        ("posted -> network thread picks it up", col(E, lambda r: d(r, "t_post", "t_pick"))),
        ("send() header + x", col(E, lambda r: d(r, "t_pick", "t_sent"))),
        ("sent -> first reply byte", col(E, lambda r: d(r, "t_sent", "t_first"))),
        ("first byte -> row 0 complete", col(E, lambda r: d(r, "t_first", "t_row0"))),
        ("row 0 -> last byte", col(E, lambda r: d(r, "t_row0", "t_done") if r["k"] >= 2 else None)),
        ("ENGINE RTT (posted -> last byte, the logged one)", col(E, lambda r: d(r, "t_post", "t_done"))),
        ("network thread: sent -> last byte", col(E, lambda r: d(r, "t_sent", "t_done"))),
        ("WINDOW: posted -> real lookup hook", col(E, lambda r: d(r, "t_post", "t_late_hook"))),
        ("real lookup hook -> local reads done (resolve)", col(E, lambda r: d(r, "t_late_hook", "t_resolve"))),
        ("WINDOW+reads: posted -> resolve", col(E, lambda r: d(r, "t_post", "t_resolve"))),
        ("resolve: waited for the reply", col(E, lambda r: d(r, "t_resolve", "t_wait_end"))),
        ("resolve: rows placed", col(E, lambda r: d(r, "t_wait_end", "t_placed"))),
        ("resolve: total incl. local fallback reads", col(E, lambda r: d(r, "t_resolve", "t_resolve_end"))),
        ("slack: last byte -> resolve (negative = late)", col(E, lambda r: d(r, "t_done", "t_resolve"))),
        ("network thread idle naps before pick-up (count)", [r["sleeps"] for r in E]),
        ("network thread involuntary switches per request", [r["ivcsw_done"] - r["ivcsw_pick"] for r in E if r["ivcsw_done"] > 0]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server")
    ap.add_argument("--engine")
    ap.add_argument("--client")
    ap.add_argument("--k", type=int, default=0, help="only requests with this k")
    ap.add_argument("--conn", type=int, default=0, help="server: only this connection")
    ap.add_argument("--last", type=int, default=0, help="only the last N requests of each dump")
    a = ap.parse_args()
    S = load(a.server) if a.server else []
    E = load(a.engine) if a.engine else []
    C = load(a.client) if a.client else []
    if a.conn:
        S = [r for r in S if r["conn"] == a.conn]
    if a.k:
        S = [r for r in S if r["k"] == a.k]
        E = [r for r in E if r["k"] == a.k]
        C = [r for r in C if r["k"] == a.k]
    if a.last:
        S, E, C = S[-a.last:], E[-a.last:], C[-a.last:]
    if S:
        ks = {}
        for r in S:
            ks[r["k"]] = ks.get(r["k"], 0) + 1
        print(f"server: {len(S)} requests, k histogram {dict(sorted(ks.items()))}")
        table("server (k3x), per request", server_rows(S))
    if E:
        ks = {}
        for r in E:
            ks[r["k"]] = ks.get(r["k"], 0) + 1
        print(f"\nengine: {len(E)} requests, k histogram {dict(sorted(ks.items()))}, "
              f"done by the deadline {sum(1 for r in E if r['flags'] & 1)}, abandoned {sum(1 for r in E if r['flags'] & 2)}")
        table("engine (Spark), per request", engine_rows(E))
    sidx = {}
    for r in S:
        if r["seq"] >= 0:
            sidx.setdefault(r["seq"], r)   # first connection wins on a repeated number
    if E and S:
        J = [(e, sidx[e["seq"]]) for e in E if e["seq"] in sidx and sidx[e["seq"]]["k"] == e["k"]]
        last = lambda s: "t_sent%d" % (min(s["k"], 4) - 1)
        table(f"joined engine x server ({len(J)} requests)", [
            ("engine sent -> last byte", [d(e, "t_sent", "t_done") for e, s in J if d(e, "t_sent", "t_done") is not None]),
            ("  server first byte -> last row sent", [d(s, "t_first", last(s)) for e, s in J if d(s, "t_first", last(s)) is not None]),
            ("  = wire + stacks (both directions)", [d(e, "t_sent", "t_done") - d(s, "t_first", last(s)) for e, s in J
                                                     if d(e, "t_sent", "t_done") is not None and d(s, "t_first", last(s)) is not None]),
            ("  server idle before it (us)", [d(s, "t_wait", "t_first") for e, s in J if d(s, "t_wait", "t_first") is not None]),
        ])
    if C:
        table("client (rrclient), per request", [
            ("send() header + x", col(C, lambda r: (r["t_sent"] - r["t_send"]) / 1e3)),
            ("sent -> first reply byte", col(C, lambda r: (r["t_first"] - r["t_sent"]) / 1e3)),
            ("first byte -> last byte", col(C, lambda r: (r["t_done"] - r["t_first"]) / 1e3)),
            ("CLIENT RTT", col(C, lambda r: (r["t_done"] - r["t_send"]) / 1e3)),
        ])
        if S:
            J = [(c, sidx[c["seq"]]) for c in C if c["seq"] in sidx and sidx[c["seq"]]["k"] == c["k"]]
            last = lambda s: "t_sent%d" % (min(s["k"], 4) - 1)
            table(f"joined client x server ({len(J)} requests)", [
                ("client RTT", [(c["t_done"] - c["t_send"]) / 1e3 for c, s in J]),
                ("  server first byte -> last row sent", [d(s, "t_first", last(s)) for c, s in J if d(s, "t_first", last(s)) is not None]),
                ("  = wire + stacks", [(c["t_done"] - c["t_send"]) / 1e3 - d(s, "t_first", last(s)) for c, s in J
                                       if d(s, "t_first", last(s)) is not None]),
            ])
    if not (S or E or C):
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
