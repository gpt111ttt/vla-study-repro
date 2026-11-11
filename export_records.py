"""Export the per-episode success flags behind every number in the paper.

Walks a tree of evaluation outputs, where each run is a directory containing an
``eval_info.json`` as written by lerobot-eval, and emits one row per episode:

    run,task,episode,success

The run name is taken from the directory name with the standard ``out_`` prefix
removed, e.g. ``out_aaq8r3_s0_t7`` becomes ``aaq8r3_s0`` in the *run* column and
``7`` in the *task* column, so a full suite shares one run label across ten
directories.

    python export_records.py /path/to/pi05/runs -o records.csv

Also writes ``runs.csv`` beside it, with one row per (run, task).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

TAG = re.compile(r"^(?:out_)?(?P<run>.+?)_t(?P<task>\d+)$")
# out_t0 .. out_t9 are one suite, one task per directory, so this has to be tried before
# the generic pattern, which would otherwise read them as run="out".
BARE = re.compile(r"^(?:out_)?t(?P<task>\d+)$")


def parse_tag(dirname):
    m = BARE.match(dirname)
    if m:
        return "t", int(m.group("task"))
    m = TAG.match(dirname)
    if not m:
        return None
    return m.group("run"), int(m.group("task"))


def episodes(path):
    """Yield (task_id, episode_index, success) for one eval_info.json."""
    with open(path) as f:
        d = json.load(f)
    for block in d.get("per_task", []):
        tid = block.get("task_id")
        rewards = block.get("metrics", {}).get("sum_rewards", [])
        for i, r in enumerate(rewards):
            yield tid, i, 1 if float(r) >= 0.5 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory holding out_*/eval_info.json")
    ap.add_argument("-o", "--out", default="records.csv")
    args = ap.parse_args()

    rows = []
    skipped = []
    for path in sorted(glob.glob(os.path.join(args.root, "*", "eval_info.json"))):
        dirname = os.path.basename(os.path.dirname(path))
        tag = parse_tag(dirname)
        if tag is None:
            skipped.append(dirname)
            continue
        run, task_from_name = tag
        try:
            found = list(episodes(path))
        except (ValueError, KeyError) as exc:
            skipped.append("%s (%s)" % (dirname, exc))
            continue
        for tid, i, s in found:
            task = task_from_name if tid is None else tid
            rows.append((run, task, i, s))

    if not rows:
        print("no eval_info.json found under %s" % args.root, file=sys.stderr)
        return 1

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "task", "episode", "success"])
        w.writerows(rows)

    per_run = {}
    for run, task, _, s in rows:
        per_run.setdefault(run, {}).setdefault(task, [0, 0])
        per_run[run][task][0] += s
        per_run[run][task][1] += 1

    runs_path = os.path.join(os.path.dirname(args.out) or ".", "runs.csv")
    with open(runs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "task", "successes", "episodes"])
        for run in sorted(per_run):
            for task in sorted(per_run[run]):
                s, n = per_run[run][task]
                w.writerow([run, task, s, n])

    print("wrote %d rows from %d runs to %s" % (len(rows), len(per_run), args.out))
    print("wrote per-run counts to %s" % runs_path)
    if skipped:
        print("skipped %d directories:" % len(skipped))
        for s in skipped[:10]:
            print("  " + s)
    return 0


if __name__ == "__main__":
    sys.exit(main())