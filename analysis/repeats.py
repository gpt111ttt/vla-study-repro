"""Do the ten-repeat arms reproduce Table 7, and what does "one episode in ten" refer to?

Reads out_aaq0r{0..9}_s0_t{0..9} (unquantised) and out_aaq8r{0..9}_s0_t{0..9} (step 8),
each a full 300-episode suite, and reports for each arm

  unstable     episodes whose flag is not identical across all ten repeats
  pairwise     mean, over the 45 repeat pairs, of the fraction of episodes that flip
  per pair     min / median / max flips over those 45 pairs
  five-pool    mean, over the 252 five-repeat subsets, of the fraction unstable

Usage:  ~/pi05/env/bin/python repeats.py
"""
import itertools
import json
import os
import statistics
import sys

R = os.path.expanduser("~/pi05/runs")


def load_arm(prefix, k):
    flags = {}
    for t in range(10):
        p = os.path.join(R, "out_%sr%d_s0_t%d" % (prefix, k, t), "eval_info.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            d = json.load(f)
        for e, v in enumerate(d["per_task"][0]["metrics"]["sum_rewards"]):
            flags[(t, e)] = 1 if float(v) >= 0.5 else 0
    return flags


for prefix, label in (("aaq0", "unquantised"), ("aaq8", "step 8")):
    runs = [load_arm(prefix, k) for k in range(10)]
    if any(r is None for r in runs):
        missing = [k for k, r in enumerate(runs) if r is None]
        print("%s: missing repeats %s, skipped" % (label, missing))
        continue
    keys = sorted(runs[0])
    n = len(keys)

    unstable = sum(1 for k in keys if len({r[k] for r in runs}) > 1)

    flips = []
    for a, b in itertools.combinations(range(10), 2):
        flips.append(sum(1 for k in keys if runs[a][k] != runs[b][k]))
    pairwise = statistics.mean(flips) / n

    five = []
    for sub in itertools.combinations(range(10), 5):
        u = sum(1 for k in keys if len({runs[i][k] for i in sub}) > 1)
        five.append(u / n)

    succ = statistics.mean(sum(r.values()) for r in runs) / n

    print("\n=== %s ===" % label)
    print("  episodes                          : %d" % n)
    print("  unstable across all ten repeats   : %d/%d = %.1f%%" % (unstable, n, 100.0 * unstable / n))
    print("  pairwise flips (mean of 45 pairs) : %.2f%%   [%d/%d episodes]" % (100 * pairwise, round(pairwise * n), n))
    print("  per-pair flips                    : min %d, median %d, max %d"
          % (min(flips), int(statistics.median(flips)), max(flips)))
    print("  of which a single pair flips      : %.1f%% to %.1f%%"
          % (100.0 * min(flips) / n, 100.0 * max(flips) / n))
    print("  five repeats pooled, unstable     : %.1f%%  (range %.1f--%.1f%%)"
          % (100 * statistics.mean(five), 100 * min(five), 100 * max(five)))
    print("  success                           : %.2f%%" % (100 * succ))