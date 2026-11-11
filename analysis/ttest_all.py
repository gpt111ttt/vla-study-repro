"""Every statistic the paper reports, on the initial state as the unit.

Paired contrasts: five seeds averaged within a state, paired t, df = n-1.
Independent contrasts (two different checkpoints): Welch t on the state rates.
Self-contained t tail (regularised incomplete beta); no SciPy needed.
"""
import glob
import json
import math
import os
import re

R = os.path.expanduser("~/pi05/runs")


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lb + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def t_p(t, df):
    return betai(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df, alpha=0.05):
    lo, hi = 0.0, 80.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if t_p(mid, df) > alpha else (lo, mid)
    return (lo + hi) / 2.0


def load(tag):
    raw = {}
    for f in glob.glob("%s/out_%s_seed*_s*_t*/eval_info.json" % (R, tag)):
        tid = int(re.search(r"_t(\d+)/eval_info\.json$", f).group(1))
        for t in json.load(open(f))["per_task"]:
            for i, v in enumerate(t["metrics"]["successes"]):
                raw.setdefault((tid, i), []).append(float(v))
    return {k: sum(v) / len(v) for k, v in raw.items()}


def paired(A, B, keys=None):
    keys = sorted(keys if keys is not None else (set(A) & set(B)))
    d = [A[k] - B[k] for k in keys]
    n = len(d)
    m, sd = sum(d) / n, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se else 0.0
    c = t_crit(n - 1)
    return n, m, t, t_p(t, n - 1), m - c * se, m + c * se


def indep(A, B):
    def st(c):
        v = list(c.values())
        n = len(v)
        m = sum(v) / n
        return n, m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) / math.sqrt(n)
    na, ma, sa = st(A)
    nb, mb, sb = st(B)
    se = math.sqrt(sa * sa + sb * sb)
    df = se ** 4 / ((sa ** 4) / (na - 1) + (sb ** 4) / (nb - 1))
    t = (ma - mb) / se if se else 0.0
    c = t_crit(df)
    return ma - mb, t, t_p(t, df), (ma - mb) - c * se, (ma - mb) + c * se, df


def pct(x):
    return "%+6.2f pp  t=%6.2f df=%3.0f p=%.5f  CI [%+.2f, %+.2f]"


TAGS = ("ms_none", "ms_cont", "ms_place", "ms_none50k", "ms_cont50k", "ms_place50k",
        "ms_noneb3", "ms_noneb4", "ms_ref", "ms_none100k")
C = {t: load(t) for t in TAGS}

print("=== cell means with cluster intervals (300 states, t) ===")
for t in TAGS:
    v = list(C[t].values())
    n = len(v)
    m = sum(v) / n
    se = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) / math.sqrt(n)
    c = t_crit(n - 1)
    print("   %-12s %5.1f%%  CI [%.1f, %.1f]" % (t, 100 * m, 100 * (m - c * se),
                                                  100 * (m + c * se)))

print("\n=== Table 15 contrasts (paired, 300 states) ===")
for bud, a, b, c2 in (("30k", "ms_none", "ms_cont", "ms_place"),
                      ("50k", "ms_none50k", "ms_cont50k", "ms_place50k")):
    for x, y in ((b, a), (c2, a), (b, c2)):
        print("   %s %-22s %s" % (bud, x[3:] + " - " + y[3:], pct(paired(C[x], C[y])[:2] + paired(C[x], C[y])[2:])))

print("\n=== Table 13: reference vs batch 4 ===")
print("   SUITE %s" % (pct(paired(C["ms_ref"], C["ms_noneb4"]))))
for tid in range(10):
    keys = [k for k in C["ms_ref"] if k[0] == tid]
    r = paired(C["ms_ref"], C["ms_noneb4"], keys)
    print("   task %d  %+6.2f  t=%5.2f df=%d p=%.4f" % (tid, 100 * r[1], r[2], r[0] - 1, r[3]))

print("\n=== Table 17: two task effects (paired, 30 states) ===")
for lab, b30, o30, b50, o50, tid in (
        ("task 4 place-none", "ms_none", "ms_place", "ms_none50k", "ms_place50k", 4),
        ("task 1 cont-none", "ms_none", "ms_cont", "ms_none50k", "ms_cont50k", 1)):
    for bud, A, B in (("30k", b30, o30), ("50k", b50, o50)):
        keys = [k for k in C[A] if k[0] == tid]
        r = paired(C[A], C[B], keys)
        print("   %s %s %+6.2f  t=%5.2f df=%d p=%.5f" % (lab, bud, 100 * r[1], r[2], r[0] - 1, r[3]))

print("\n=== independent contrasts (different checkpoints, Welch) ===")
for lab, x, y in (("30k->50k none", "ms_none50k", "ms_none"),
                  ("30k->50k cont", "ms_cont50k", "ms_cont"),
                  ("30k->50k place", "ms_place50k", "ms_place"),
                  ("b2 vs b3", "ms_noneb3", "ms_none50k"),
                  ("b3 vs b4", "ms_noneb4", "ms_noneb3"),
                  ("b2 vs b4", "ms_noneb4", "ms_none50k"),
                  ("ref vs b4", "ms_ref", "ms_noneb4"),
                  ("100k vs 50k", "ms_none100k", "ms_none50k")):
    m, t, p, lo, hi, df = indep(C[x], C[y])
    print("   %-16s %+6.2f pp  t=%6.2f df=%3.0f p=%.5f  CI [%+.2f, %+.2f]"
          % (lab, 100 * m, t, df, p, 100 * lo, 100 * hi))