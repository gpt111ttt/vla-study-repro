#!/usr/bin/env python3
"""C1 -- close the pixel-to-outcome link on the borderline episodes.

The paper currently shows: two natural runs of one checkpoint disagree on some
episodes, and the wrist image differs at 2-3 pixels of one grey level. That is a
correlation. This script turns it into an intervention on the episodes that
actually flipped:

    same pinned initial state, three arms, identical except for one pixel
      off   no injection                     -> the baseline verdict
      noop  +0 at the same pixel             -> proves the patch is inert
      edge  +1 at the same pixel             -> the intervention

Everything else (weights, seed, quantisation, episode list) is fixed, and each
episode runs to termination. The question is how often `edge` flips the verdict
that `off` produced, against how often `noop` does.

    closure.py list                     show the runs available to pair
    closure.py pick <runA> <runB>       find the flipped episodes, write the list
    closure.py submit [--dry]           queue 3 arms per flipped episode
    closure.py report                   paired table + McNemar

Run dirs are ~/pi05/runs/out_<tag>_s<seedoffset>_t<task>/eval_info.json and the
episode list written by `pick` is ~/pi05/closure_episodes.tsv.
"""
import glob
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.expanduser("~/pi05/runs")
PIX_EDGE = os.path.expanduser("~/pi05/e4_pixels.json")
PIX_NOOP = os.path.expanduser("~/pi05/e4_pixels_noop.json")
LIST = os.path.expanduser("~/pi05/closure_episodes.tsv")
CKPT = os.path.expanduser("~/pi05/models/_ckpt_expert_best50k")
SWEEP = os.path.expanduser("~/pi05/eval_sweep2.sh")

NAME = re.compile(r"out_(?P<tag>.+)_s(?P<off>\d+)_t(?P<task>\d+)$")


def load_run(tag):
    """{(task, episode_index): bool} for every task dir of one tag."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "out_%s_s*_t*" % tag))):
        m = NAME.search(os.path.basename(d))
        p = os.path.join(d, "eval_info.json")
        if not m or not os.path.exists(p):
            continue
        task = int(m.group("task"))
        try:
            j = json.load(open(p))
        except Exception as e:
            print("  skip %s (%s)" % (p, e))
            continue
        for t in j.get("per_task", []):
            if int(t.get("task_id", task)) != task:
                continue
            for i, s in enumerate(t["metrics"]["successes"]):
                out[(task, i)] = bool(s)
    return out


def cmd_list():
    rows = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "out_*_s*_t*"))):
        m = NAME.search(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "eval_info.json")):
            continue
        rows.setdefault(m.group("tag"), []).append(int(m.group("task")))
    if not rows:
        print("no eval_info.json under", ROOT)
        return 1
    print("%-34s %-14s %s" % ("tag", "tasks", "episodes/task"))
    for tag in sorted(rows):
        tasks = sorted(rows[tag])
        n = None
        for task in tasks:
            p = glob.glob(os.path.join(ROOT, "out_%s_s*_t%d" % (tag, task), "eval_info.json"))
            if not p:
                continue
            try:
                j = json.load(open(p[0]))
                for t in j.get("per_task", []):
                    n = len(t["metrics"]["successes"])
            except Exception:
                pass
            break
        print("%-34s %-14s %s" % (tag, ",".join(str(t) for t in tasks), n if n else "?"))
    return 0


def cmd_pick(a_tag, b_tag):
    a = load_run(a_tag)
    b = load_run(b_tag)
    keys = sorted(set(a) & set(b))
    if not keys:
        print("the two runs share no (task, episode) pair -- check the tags")
        return 2
    tasks = sorted({t for t, _ in keys})
    print("%-6s %-8s %-8s %-9s" % ("task", "A", "B", "flipped"))
    rows = []
    for task in tasks:
        ks = [k for k in keys if k[0] == task]
        na = sum(a[k] for k in ks)
        nb = sum(b[k] for k in ks)
        fl = [k for k in ks if a[k] != b[k]]
        print("%-6d %-8s %-8s %d/%d" % (task, "%d/%d" % (na, len(ks)),
                                        "%d/%d" % (nb, len(ks)),
                                        len(fl), len(ks)))
        for k in fl:
            rows.append((task, k[1], int(a[k]), int(b[k])))
    if not rows:
        print("\nno flipped episodes -- nothing to close.")
        return 3
    with open(LIST, "w") as f:
        f.write("# task\tepisode\tA\tB\n")
        for r in rows:
            f.write("%d\t%d\t%d\t%d\n" % r)
    print("\n%d flipped episodes -> %s" % (len(rows), LIST))
    print("A=1,B=0: %d   A=0,B=1: %d" % (sum(1 for r in rows if r[2] and not r[3]),
                                         sum(1 for r in rows if r[3] and not r[2])))
    return 0


def _noop_file():
    """Same pixels as the edge arm, delta zeroed."""
    j = json.load(open(PIX_EDGE))
    json.dump({"pixels": [[int(y), int(x), 0] for y, x, _ in j["pixels"]]},
              open(PIX_NOOP, "w"), indent=1)
    return PIX_NOOP


def cmd_submit(args):
    """submit [--dry] [--quant 1|8] [--seed-offset N] [--ckpt PATH]"""
    dry = "--dry" in args
    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default
    quant = opt("--quant", "1")
    off = opt("--seed-offset", "0")
    ckpt = opt("--ckpt", CKPT)
    if not os.path.exists(LIST):
        print("no episode list -- run `closure.py pick` first")
        return 2
    if not os.path.exists(PIX_EDGE):
        print("missing %s -- run e4_locate.py on two dump directories first" % PIX_EDGE)
        return 2
    noop = _noop_file()
    rows = [l.split("\t") for l in open(LIST) if l.strip() and not l.startswith("#")]
    only = opt("--tasks", "")
    if only:
        keep = set(only.split(","))
        rows = [r for r in rows if r[0] in keep]
    lim = int(opt("--limit", "0") or 0)
    if lim:
        rows = rows[:lim]
    if not rows:
        print("no episodes selected -- check --tasks against %s" % LIST)
        return 2
    jobs = []
    for task, ep, _, _ in rows:
        for arm, mode, pix in (("off", "none", ""), ("noop", "target", noop),
                               ("edge", "target", PIX_EDGE)):
            tag = "cl_%s_t%s_e%s" % (arm, task, ep)
            env = ["ALL", "E4_MODE=%s" % mode, "E4_AT_STEP=0", "E4_SEED=7",
                   "P0_QUANT=%s" % quant, "P0_SEED_OFFSET=%s" % off,
                   "P0_PIN_INIT_STATES=1", "P0_SEED=1"]
            if pix:
                env.append("E4_PIXELS=%s" % pix)
            jobs.append((arm, task, ep, tag, env))
    print("%d episodes x 3 arms = %d jobs   (quant=%s seed_offset=%s)"
          % (len(rows), len(jobs), quant, off))
    for arm, task, ep, tag, env in jobs:
        cmd = ("sbatch --job-name=%s --gres=gpu:1 -c 4 -t 00:40:00 "
               "--export=%s --wrap=\"bash %s %s %s 1 %s %s\""
               % (tag[:32], ",".join(env), SWEEP, task, ep, tag, ckpt))
        if dry:
            print(cmd)
        else:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print("%-6s t%s e%-3s -> %s" % (arm, task, ep,
                                            (r.stdout or r.stderr).strip()[:60]))
    if not dry:
        print("\nwatch: squeue -u $USER    results land in %s/out_cl_*" % ROOT)
    return 0


def cmd_report():
    rec = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "out_cl_*_s*_t*"))):
        m = NAME.search(os.path.basename(d))
        p = os.path.join(d, "eval_info.json")
        if not m or not os.path.exists(p):
            continue
        mm = re.match(r"cl_(off|noop|edge)_t(\d+)_e(\d+)$", m.group("tag"))
        if not mm:
            continue
        arm, task, ep = mm.group(1), int(mm.group(2)), int(mm.group(3))
        try:
            j = json.load(open(p))
        except Exception:
            continue
        for t in j.get("per_task", []):
            s = t["metrics"]["successes"]
            if s:
                rec[(task, ep, arm)] = bool(s[0])
    eps = sorted({(t, e) for (t, e, _) in rec})
    if not eps:
        print("no results yet under %s/out_cl_*" % ROOT)
        return 1
    print("%-6s %-8s %-6s %-6s %-6s" % ("task", "episode", "off", "noop", "edge"))
    for t, e in eps:
        print("%-6d %-8d %-6s %-6s %-6s" % (t, e,
                                            rec.get((t, e, "off"), "?"),
                                            rec.get((t, e, "noop"), "?"),
                                            rec.get((t, e, "edge"), "?")))
    have = [(t, e) for t, e in eps
            if all((t, e, a) in rec for a in ("off", "noop", "edge"))]
    if not have:
        print("\nnothing complete yet")
        return 0

    def mcnemar(x, y):
        b = sum(1 for k in have if x[k] and not y[k])
        c = sum(1 for k in have if y[k] and not x[k])
        n = b + c
        if n == 0:
            return b, c, 1.0
        k = min(b, c)
        p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
        return b, c, min(1.0, p)

    off = {k: rec[k + ("off",)] for k in have}
    noop = {k: rec[k + ("noop",)] for k in have}
    edge = {k: rec[k + ("edge",)] for k in have}
    print("\ncomplete episodes: %d" % len(have))
    b, c, p = mcnemar(noop, off)
    print("noop vs off : %d->%d  %d->%d  p=%.4f   (must be null)" % (b, c, c, b, p))
    b, c, p = mcnemar(edge, off)
    print("edge vs off : %d->%d  %d->%d  p=%.4f" % (b, c, c, b, p))
    same_e = sum(1 for k in have if edge[k] == off[k])
    same_n = sum(1 for k in have if noop[k] == off[k])
    print("\nverdicts unchanged by injection: noop %d/%d, edge %d/%d"
          % (same_n, len(have), same_e, len(have)))
    return 0


def cmd_churn(prefix):
    """churn <prefix>  -- pairwise flip rate and unstable-episode share across a family of runs."""
    tags = sorted({NAME.search(os.path.basename(d)).group("tag")
                   for d in glob.glob(os.path.join(ROOT, "out_%s*_s*_t*" % prefix))
                   if NAME.search(os.path.basename(d))})
    if len(tags) < 2:
        print("need at least two runs matching %s*, found %d" % (prefix, len(tags)))
        return 2
    runs = {t: load_run(t) for t in tags}
    tags = [t for t in tags if runs[t]]
    keys = sorted(set.intersection(*[set(runs[t]) for t in tags]))
    if not keys:
        print("runs share no episodes")
        return 2
    print("%d runs, %d shared episodes\n" % (len(tags), len(keys)))

    tot = totf = 0
    worst = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = runs[tags[i]], runs[tags[j]]
            f = sum(1 for k in keys if a[k] != b[k])
            tot += len(keys)
            totf += f
            worst.append((f, tags[i], tags[j]))
    print("pairwise: %d pairs, %d/%d episode pairs differ = %.1f%%"
          % (len(worst), totf, tot, 100.0 * totf / tot))
    worst.sort(reverse=True)
    print("  highest: " + ", ".join("%s/%s %d" % (x[1], x[2], x[0]) for x in worst[:3]))
    print("  lowest : " + ", ".join("%s/%s %d" % (x[1], x[2], x[0]) for x in worst[-3:]))

    un = {k: len({runs[t][k] for t in tags}) for k in keys}
    unstable = [k for k in keys if un[k] > 1]
    print("\nunanimity: %d/%d episodes unstable = %.1f%%"
          % (len(unstable), len(keys), 100.0 * len(unstable) / len(keys)))
    hist = {}
    for k in keys:
        hist[un[k]] = hist.get(un[k], 0) + 1
    print("  distinct verdicts per episode: " +
          ", ".join("%d:%d" % (n, hist[n]) for n in sorted(hist)))
    by_task = {}
    for t, _ in unstable:
        by_task[t] = by_task.get(t, 0) + 1
    print("  unstable per task: " +
          ", ".join("t%d:%d" % (t, by_task[t]) for t in sorted(by_task)))
    json.dump({"unstable": [[t, e] for t, e in unstable]},
              open(os.path.expanduser("~/pi05/unstable_episodes.json"), "w"))
    print("  wrote ~/pi05/unstable_episodes.json")
    return 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if a[0] == "list":
        return cmd_list()
    if a[0] == "pick" and len(a) >= 3:
        return cmd_pick(a[1], a[2])
    if a[0] == "submit":
        return cmd_submit(a[1:])
    if a[0] == "report":
        return cmd_report()
    if a[0] == "churn" and len(a) >= 2:
        return cmd_churn(a[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())