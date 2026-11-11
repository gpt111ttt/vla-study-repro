"""Paired (McNemar) analysis of the stability comparison.

An episode is "unstable" in a setting if the repeats of that setting do not all return the
same success flag. The two settings then give each of the 300 episodes a paired 2x2 label,
so the comparison is McNemar's exact test, not a two-proportion z.

  unquantised repeats : out_aaq0r{0..9}_s*_t*
  quantised (step 8)  : out_aaq8r{0..9}_s*_t*
  bf16 / fp32         : out_e5bf16r{0..4}_*, out_e5fp32r{0..4}_*
"""
import glob
import json
import math
import os
import re

R = os.path.expanduser("~/pi05/runs")


def collect(pattern, rmax):
    """-> {(task, state): [success flags, one per repeat]} plus per-repeat suite means."""
    lab, means = {}, {}
    for r in range(rmax):
        n_r = 0
        for f in glob.glob("%s/out_%s_s*_t*/eval_info.json" % (R, pattern % r)):
            tid = int(re.search(r"_t(\d+)/eval_info\.json$", f).group(1))
            for t in json.load(open(f))["per_task"]:
                for i, v in enumerate(t["metrics"]["successes"]):
                    lab.setdefault((tid, i), []).append(float(v))
                    n_r += 1
        means[r] = (100.0 * sum(1.0 for k, v in lab.items() if k[0] == 9 and v[-1]) / max(n_r, 1)
                    if False else None)
    return lab


def unstable(lab, n_rep):
    """Episodes whose n_rep repeats are not unanimous."""
    return {k: int(len(v) != n_rep or len(set(v)) > 1)
            for k, v in lab.items() if len(v) == n_rep}


def suite_mean(lab):
    vals = [sum(v) / len(v) for v in lab.values()]
    return 100.0 * sum(vals) / len(vals), len(vals)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def report(name, A, B, nA, nB):
    keys = sorted(set(A) & set(B))
    a = sum(1 for k in keys if A[k] and B[k])
    b = sum(1 for k in keys if A[k] and not B[k])
    c = sum(1 for k in keys if B[k] and not A[k])
    d = sum(1 for k in keys if not A[k] and not B[k])
    n = len(keys)
    p = mcnemar_exact(b, c)
    print("=== %s ===" % name)
    print("   paired episodes: %d" % n)
    print("   unstable in both      : %d" % a)
    print("   unstable only %-6s  : %d" % (nA, b))
    print("   unstable only %-6s  : %d" % (nB, c))
    print("   stable in both        : %d" % d)
    print("   marginal rates: %.1f%% and %.1f%%" % (100.0 * (a + b) / n, 100.0 * (a + c) / n))
    print("   exact McNemar p = %.6f  (discordant pairs %d)" % (p, b + c))
    print("   range of p over all feasible pairings: %.6f .. %.6f"
          % (mcnemar_exact(max(a + c, a + b), min(a + c, a + b)), p))
    print()


q0 = collect("aaq0r%d", 10)
q8 = collect("aaq8r%d", 10)
m0, n0 = suite_mean(q0)
m8, n8 = suite_mean(q8)
print("aq0 (unquantised): %d episodes, mean %.2f%%" % (n0, m0))
print("aq8 (step 8)     : %d episodes, mean %.2f%%\n" % (n8, m8))

U0 = unstable(q0, 10)
U8 = unstable(q8, 10)
report("unquantised vs step 8 (10 repeats, 300 episodes)", U0, U8, "aq0", "aq8")

f16 = collect("e5bf16r%d", 5)
f32 = collect("e5fp32r%d", 5)
if f16 and f32:
    U16 = unstable(f16, 5)
    U32 = unstable(f32, 5)
    report("bf16 vs fp32 (5 repeats)", U16, U32, "bf16", "fp32")

# per-repeat suite means, to confirm these are the ten-repeat sets of Table 5
print("per-repeat suite means (aq0):")
for r in range(10):
    lab = {}
    for f in glob.glob("%s/out_aaq0r%d_s*_t*/eval_info.json" % (R, r)):
        tid = int(re.search(r"_t(\d+)/eval_info\.json$", f).group(1))
        for t in json.load(open(f))["per_task"]:
            for i, v in enumerate(t["metrics"]["successes"]):
                lab[(tid, i)] = float(v)
    v = list(lab.values())
    print("   r%d  %d episodes  %.2f%%" % (r, len(v), 100.0 * sum(v) / len(v)))