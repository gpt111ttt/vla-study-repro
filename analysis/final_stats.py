"""All contrasts, paired on the shared initial states, with a Student t tail."""
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


def tp(t, df):
    return betai(df / 2.0, 0.5, df / (df + t * t))


def tc(df, alpha=0.05):
    lo, hi = 0.0, 80.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if tp(mid, df) > alpha else (lo, mid)
    return (lo + hi) / 2.0


def load(tag):
    raw = {}
    for f in glob.glob("%s/out_%s_seed*_s*_t*/eval_info.json" % (R, tag)):
        tid = int(re.search(r"_t(\d+)/eval_info\.json$", f).group(1))
        for t in json.load(open(f))["per_task"]:
            for i, v in enumerate(t["metrics"]["successes"]):
                raw.setdefault((tid, i), []).append(float(v))
    return {k: sum(v) / len(v) for k, v in raw.items()}


def show(label, A, B, keys=None):
    keys = sorted(keys if keys is not None else (set(A) & set(B)))
    d = [A[k] - B[k] for k in keys]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se else 0.0
    c = tc(n - 1)
    print("   %-24s %+6.2f pp  t=%6.2f df=%3d p=%.5f  CI [%+.2f, %+.2f]"
          % (label, 100 * m, t, n - 1, tp(t, n - 1), 100 * (m - c * se), 100 * (m + c * se)))


TAGS = ("ms_none", "ms_cont", "ms_place", "ms_none50k", "ms_cont50k", "ms_place50k",
        "ms_noneb3", "ms_noneb4", "ms_ref", "ms_none100k")
C = {t: load(t) for t in TAGS}

print("=== Table 15: state-pathway contrasts ===")
show("30k cont - none", C["ms_cont"], C["ms_none"])
show("30k place - none", C["ms_place"], C["ms_none"])
show("30k cont - place", C["ms_cont"], C["ms_place"])
show("50k cont - none", C["ms_cont50k"], C["ms_none50k"])
show("50k place - none", C["ms_place50k"], C["ms_none50k"])
show("50k cont - place", C["ms_cont50k"], C["ms_place50k"])

print("\n=== Table 13 suite row ===")
show("ref - batch4", C["ms_ref"], C["ms_noneb4"])

print("\n=== budget effects (same states) ===")
show("none 50k - 30k", C["ms_none50k"], C["ms_none"])
show("cont 50k - 30k", C["ms_cont50k"], C["ms_cont"])
show("place 50k - 30k", C["ms_place50k"], C["ms_place"])
show("none 100k - 50k", C["ms_none100k"], C["ms_none50k"])

print("\n=== batch ladder (same states) ===")
show("b3 - b2", C["ms_noneb3"], C["ms_none50k"])
show("b4 - b3", C["ms_noneb4"], C["ms_noneb3"])
show("b4 - b2", C["ms_noneb4"], C["ms_none50k"])
show("ref - b4", C["ms_ref"], C["ms_noneb4"])