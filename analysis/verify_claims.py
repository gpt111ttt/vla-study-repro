"""Check the remaining prose counts against the ten-repeat arms.

Claims under test:
  "54 of the 300 episodes do change their success rate, and the changes cancel"
  "the paired difference in success is +0.87 points"
  "quantisation reduces instability by 58%"

Usage:  ~/pi05/env/bin/python verify_claims.py
"""
import json
import os
import statistics

R = os.path.expanduser("~/pi05/runs")


def load_arm(prefix):
    rates = {}
    for k in range(10):
        for t in range(10):
            p = os.path.join(R, "out_%sr%d_s0_t%d" % (prefix, k, t), "eval_info.json")
            if not os.path.exists(p):
                return None
            with open(p) as f:
                d = json.load(f)
            for e, v in enumerate(d["per_task"][0]["metrics"]["sum_rewards"]):
                rates.setdefault((t, e), []).append(1 if float(v) >= 0.5 else 0)
    return {k: sum(v) / len(v) for k, v in rates.items()}


a = load_arm("aaq0")
b = load_arm("aaq8")
if a is None or b is None:
    print("arm incomplete")
    raise SystemExit(1)

keys = sorted(a)
n = len(keys)
changed = [k for k in keys if abs(a[k] - b[k]) > 1e-9]
up = [k for k in changed if b[k] > a[k]]
down = [k for k in changed if b[k] < a[k]]

print("episodes                                  : %d" % n)
print("episodes whose success rate changes       : %d" % len(changed))
print("  better at step 8                        : %d" % len(up))
print("  worse at step 8                         : %d" % len(down))
print("mean success unquantised                  : %.2f%%" % (100 * statistics.mean(a.values())))
print("mean success at step 8                    : %.2f%%" % (100 * statistics.mean(b.values())))
print("paired difference                         : %+.2f points"
      % (100 * (statistics.mean(b.values()) - statistics.mean(a.values()))))

un_a = sum(1 for k in keys if 0 < a[k] < 1)
un_b = sum(1 for k in keys if 0 < b[k] < 1)
print("\nunstable episodes (rate strictly between 0 and 1)")
print("  unquantised                             : %d/300 = %.1f%%" % (un_a, 100.0 * un_a / n))
print("  step 8                                  : %d/300 = %.1f%%" % (un_b, 100.0 * un_b / n))
if un_a:
    print("  reduction                               : %.1f%%" % (100.0 * (un_a - un_b) / un_a))

only_a = [k for k in keys if 0 < a[k] < 1 and (b[k] == 0 or b[k] == 1)]
only_b = [k for k in keys if 0 < b[k] < 1 and (a[k] == 0 or a[k] == 1)]
both = [k for k in keys if 0 < a[k] < 1 and 0 < b[k] < 1]
print("\ncross-tab of instability")
print("  unstable only unquantised               : %d" % len(only_a))
print("  unstable only at step 8                 : %d" % len(only_b))
print("  unstable under both                     : %d" % len(both))