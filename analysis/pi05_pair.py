"""The pi0.5 pair that matches the OpenVLA control: two launches, same settings.

out_aaq0r0 and out_aaq0r1 are two unquantised full-suite runs with the same seed offset,
which is exactly the construct the OpenVLA comparison uses (eval_openvla_*_q1, quant 1).

Prints per-task success for both runs, the episode-level agreement, and the same numbers
for the pair chosen to have the median flip count, so the table can be filled either way.

Usage:  ~/pi05/env/bin/python pi05_pair.py [arm_prefix]
"""
import json
import os
import sys
from itertools import combinations

R = os.path.expanduser("~/pi05/runs")
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "aaq0"


def load(k):
    flags = {}
    for t in range(10):
        p = os.path.join(R, "out_%sr%d_s0_t%d" % (PREFIX, k, t), "eval_info.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            d = json.load(f)
        for e, v in enumerate(d["per_task"][0]["metrics"]["sum_rewards"]):
            flags[(t, e)] = 1 if float(v) >= 0.5 else 0
    return flags


runs = {k: load(k) for k in range(10)}
runs = {k: v for k, v in runs.items() if v is not None}

flips = {ab: sum(1 for i in runs[ab[0]] if runs[ab[0]][i] != runs[ab[1]][i])
         for ab in combinations(sorted(runs), 2)}
order = sorted(flips, key=lambda ab: flips[ab])
print("pairwise flips over %d pairs: min %d, median %d, max %d"
      % (len(order), flips[order[0]],
         flips[order[len(order) // 2]], flips[order[-1]]))

pairs = [order[0], order[len(order) // 2], order[-1]]
print("\nusing: min-flip %s, median-flip %s, max-flip %s" %
      (pairs[0], pairs[1], pairs[2]))

for a, b in pairs:
    fa, fb = runs[a], runs[b]
    print("\n=== r%d vs r%d : %d flips ===" % (a, b, flips[(a, b)]))
    print("task  succA  succB  agree(30)")
    ta = tb = agree = 0
    for t in range(10):
        sa = sum(fa[(t, e)] for e in range(30))
        sb = sum(fb[(t, e)] for e in range(30))
        ag = sum(1 for e in range(30) if fa[(t, e)] == fb[(t, e)])
        ta += sa
        tb += sb
        agree += ag
        print("%4d   %2d/30  %2d/30    %2d/30" % (t, sa, sb, ag))
    print("suite  %d/300 %d/300   %d/300   (%.2f%% vs %.2f%%)"
          % (ta, tb, agree, 100.0 * ta / 300, 100.0 * tb / 300))