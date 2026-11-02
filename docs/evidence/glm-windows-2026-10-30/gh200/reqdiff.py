#!/usr/bin/env python3
"""Compare tensor digests between requests inside one digest TSV.

A request starts at a graph whose column count is > 1 (the prompt prefill,
recorded with LLAMA_DEBUG_TENSOR_PREFILL=1). Graphs are aligned by their
index within the request and nodes by their order within the graph.
Prints, for each request after the first, the first differing node against
request 1 with three preceding matching nodes, and per-graph diff counts.
"""
import sys
from collections import Counter


def load(path):
    reqs = []  # list of requests; request = list of graphs; graph = list of rows
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("# graph"):
            cols = int(line.split("columns=")[1])
            if cols > 1 or not reqs:
                reqs.append([])
            reqs[-1].append([])
            continue
        if line.startswith("#") or not line:
            continue
        f = line.split("\t")
        if reqs and reqs[-1]:
            reqs[-1][-1].append(f)
    return reqs


def main():
    reqs = load(sys.argv[1])
    print("requests:", len(reqs), "graphs per request:", [len(r) for r in reqs])
    # drop groups that are not full requests (e.g. a lone 2-column setup graph)
    full = max(len(r) for r in reqs)
    reqs = [r for r in reqs if len(r) == full]
    import os
    bi = int(os.environ.get("BASE", "1")) - 1
    skip = os.environ.get("SKIP_DIM")  # e.g. 8192: skip whole-cache tensors
    if skip:
        reqs = [[[row for row in g if skip not in row[5].split(",")] for g in r] for r in reqs]
    base = reqs[bi]
    for k, r in enumerate(reqs, start=1):
        if k - 1 == bi:
            continue
        first = None
        per_graph = []
        ops = Counter()
        for g, (ga, gb) in enumerate(zip(base, r)):
            nd = 0
            for i, (a, b) in enumerate(zip(ga, gb)):
                if a[2:8] != b[2:8]:
                    print(f"req{k} graph {g} node {i}: STRUCTURE differs: {a[2:8]} vs {b[2:8]}")
                    break
                if a[8] != b[8]:
                    nd += 1
                    ops[a[3]] += 1
                    if first is None:
                        first = (g, i, ga[max(0, i - 3):i], a, b)
            if len(ga) != len(gb):
                print(f"req{k} graph {g}: node count {len(ga)} vs {len(gb)}")
            per_graph.append(nd)
        print(f"req{k}: diff nodes per graph {per_graph}")
        if first:
            g, i, ctx, a, b = first
            print(f"req{k}: FIRST DIFF graph {g} node {i}")
            for c in ctx:
                print("   same ", "\t".join(c[2:9]))
            print(f"   req{bi+1} ", "\t".join(a[2:10]))
            print(f"   req{k} ", "\t".join(b[2:10]))
            print(f"req{k}: differing ops (first-diff onward) {ops.most_common(8)}")
        else:
            print(f"req{k}: IDENTICAL to base")


if __name__ == "__main__":
    main()
