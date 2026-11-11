"""Which pair of pi0.5 runs is the one the manuscript reports, and what are its two totals?

Reads ~/pi05/runs/out_<tag>/eval_info.json, groups tags by stripping the _t<digit> suffix,
and prints every run's suite total, per-task counts, and the pairwise episode agreement.

Usage:  ~/pi05/env/bin/python runs_matrix.py [base ...]
"""
import glob
import json
import os
import sys
from itertools import combinations

R = os.path.expanduser("~/pi05/runs")


def load_run(base):
    flags = {}
    for t in range(10):
        p = os.path.join(R, "out_%s_t%d" % (base, t), "eval_info.json")
        if not os.path.exists(p):
            return None
        if base == "t":
            p = os.path.join(R, "out_t%d" % t, "eval_info.json")
        with open(p) as f:
            d = json.load(f)
        task = d["per_task"][0]
        sr = task["metrics"]["sum_rewards"]
        for e, v in enumerate(sr):
            flags[(t, e)] = 1 if float(v) >= 0.5 else 0
    return flags


def find_bases():
    seen = []
    for d in sorted(glob.glob(os.path.join(R, "out_*"))):
        tag = os.path.basename(d)[4:]
        if tag == "t" or (tag.startswith("t") and tag[1:].isdigit()):
            b = "t"
        elif "_t" in tag and tag.rsplit("_t", 1)[1].isdigit():
            b = tag.rsplit("_t", 1)[0]
        else:
            continue
        if b not in seen:
            seen.append(b)
    return seen


bases = sys.argv[1:] or [b for b in find_bases() if b == "t" or b.startswith("s") and b[1:].isdigit()]

runs = {}
for b in bases:
    f = load_run(b)
    if f is None:
        print("skip %s (incomplete)" % b)
        continue
    runs[b] = f

print("%-14s %s" % ("run", "suite  " + "  ".join("t%d" % t for t in range(10))))
print("-" * 70)
for b, f in runs.items():
    per = [sum(f[(t, e)] for e in range(30)) for t in range(10)]
    print("%-14s %3d/300  %s" % (b, sum(per), "  ".join("%2d" % x for x in per)))

print("\n=== pairwise episode agreement (of 300) ===")
print("%-10s %-10s %8s %8s %10s" % ("run A", "run B", "succ A", "succ B", "agree"))
for a, b in combinations(runs, 2):
    fa, fb = runs[a], runs[b]
    keys = set(fa) & set(fb)
    agree = sum(1 for k in keys if fa[k] == fb[k])
    print("%-10s %-10s %8d %8d %10d" % (a, b, sum(fa.values()), sum(fb.values()), agree))

print("\n=== episodes that flip, for the closest pair to 30 flips ===")
best = None
for a, b in combinations(runs, 2):
    fa, fb = runs[a], runs[b]
    flips = sum(1 for k in fa if k in fb and fa[k] != fb[k])
    if best is None or abs(flips - 30) < abs(best[2] - 30):
        best = (a, b, flips)
if best:
    a, b, fl = best
    fa, fb = runs[a], runs[b]
    up = sum(1 for k in fa if k in fb and fa[k] == 1 and fb[k] == 0)
    down = fl - up
    print("  %s vs %s: %d flips (%d ->0, %d ->1)" % (a, b, fl, up, down))