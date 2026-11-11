"""Compare two OpenVLA suite runs episode by episode.

    python3 cmp_ovla.py n1_r0 n1_r1_r1
"""
import json
import os
import sys

ROOT = os.path.expanduser("~/openvla_probe/runs")
A = sys.argv[1] if len(sys.argv) > 1 else "n1_r0"
B = sys.argv[2] if len(sys.argv) > 2 else "n1_r1_r1"


def load(tag, t):
    p = os.path.join(ROOT, tag, "eval_openvla_t%d_q1.json" % t)
    if not os.path.isfile(p):
        return None
    return json.load(open(p))


def episode_list(d):
    """Find the per-episode records, whatever the file calls them."""
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                if "success" in v[0] or "steps" in v[0]:
                    return v
        for k, v in d.items():
            if isinstance(v, dict):
                r = episode_list(v)
                if r:
                    return r
    return None


first = load(A, 0)
if first is None:
    print("no file for", A)
    sys.exit(2)
print("schema of", A, "task 0:")
if isinstance(first, dict):
    for k, v in first.items():
        print("  %-14s %s %s" % (k, type(v).__name__, len(v) if hasattr(v, "__len__") else v))
else:
    print("  list of", len(first))

tot = 0
diff_s = 0
diff_n = 0
for t in range(10):
    da, db = load(A, t), load(B, t)
    if da is None or db is None:
        print("task %d: missing side" % t)
        continue
    ea, eb = episode_list(da), episode_list(db)
    if ea is None or eb is None:
        print("task %d: no episode records" % t)
        continue
    same_s = sum(1 for x, y in zip(ea, eb) if x.get("success") == y.get("success"))
    steps_a = [x.get("steps") for x in ea]
    steps_b = [y.get("steps") for y in eb]
    same_n = sum(1 for x, y in zip(steps_a, steps_b) if x is not None and x == y)
    if steps_a and None not in steps_a:
        same_n = sum(1 for x, y in zip(steps_a, steps_b) if x == y)
    n = min(len(ea), len(eb))
    tot += n
    diff_s += n - same_s
    diff_n += n - same_n
    print("task %d: n=%d success-identical %d/%d  steps-identical %d/%d" % (t, n, same_s, n, same_n, n))

print()
print("TOTAL episodes %d   success differences %d   step-count differences %d" % (tot, diff_s, diff_n))