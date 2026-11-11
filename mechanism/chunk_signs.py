"""Signed difference between two action chunks: does the perturbation accumulate or cancel?

Fig. 3(b) plots |delta|, so it cannot say whether the per-step deviations point the same way.
This reads the two chunk dumps directly and answers:

  * is the deviation a constant offset across the chunk (a global shift), or does it change
    sign from step to step (jitter)?
  * over the 50 steps executed open-loop, do the signed deviations add up or cancel?
  * does that differ between the 7 real action dimensions and the 25 padding dimensions?

Usage:  ~/pi05/env/bin/python chunk_signs.py [dump_dir_a] [dump_dir_b]
"""
import os
import sys

import numpy as np

D = os.path.expanduser("~/pi05/dumps")
A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "a")
B = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "b")
NAME = "ep001_chunk0.npy"


def load(d):
    p = os.path.join(d, NAME)
    v = np.load(p)
    v = np.squeeze(v)
    if v.ndim != 2:
        v = v.reshape(-1, v.shape[-1])
    return v.astype(np.float64)


a, b = load(A), load(B)
if a.shape != b.shape:
    print("shape mismatch %s vs %s" % (a.shape, b.shape))
    sys.exit(1)
T, K = a.shape
d = a - b
print("chunk %d steps x %d dims   from %s vs %s" % (T, K, os.path.basename(A), os.path.basename(B)))

print("\n=== per step ===")
print(" step   max|d|   signed d at that dim   dim   sum over dims")
for t in range(T):
    k = int(np.argmax(np.abs(d[t])))
    print("  %2d   %.4f      %+.4f                %2d    %+.4f"
          % (t, abs(d[t, k]), d[t, k], k, d[t].sum()))

# ---- is it a constant offset or jitter?
mean_t = d.mean(axis=0)          # average deviation per dimension
sd_t = d.std(axis=0)             # how much it changes across steps
shift = np.abs(mean_t) > sd_t
print("\n=== shift or jitter (per dimension) ===")
print("  dimensions where |mean over steps| > sd over steps : %d of %d"
      % (int(shift.sum()), K))
print("  real dims 0-6 : %s" % "".join("S" if s else "j" for s in shift[:7]))
print("  padding 7-31  : %s" % "".join("S" if s else "j" for s in shift[7:]))
print("  (S = a roughly constant offset across the chunk, j = sign changes along it)")

# ---- accumulate or cancel
tot_signed = d.sum(axis=0)
tot_abs = np.abs(d).sum(axis=0)
ratio = np.where(np.abs(tot_signed) > 1e-12, tot_abs / np.abs(tot_signed), np.inf)
print("\n=== accumulate or cancel over the %d open-loop steps ===" % T)
print("  dim   sum(signed d)   sum|d|   cancellation ratio")
for k in range(K):
    tag = "" if k < 7 else "  (padding)"
    print("   %2d    %+.4f        %.4f    %s%s"
          % (k, tot_signed[k], tot_abs[k],
             "inf" if np.isinf(ratio[k]) else "%.2f" % ratio[k], tag))

print("\n  real dims (0-6)  : sum|d| %.4f, |sum signed| %.4f, ratio %.2f"
      % (tot_abs[:7].sum(), abs(tot_signed[:7].sum()),
         tot_abs[:7].sum() / max(abs(tot_signed[:7].sum()), 1e-12)))
print("  padding (7-31)   : sum|d| %.4f, |sum signed| %.4f, ratio %.2f"
      % (tot_abs[7:].sum(), abs(tot_signed[7:].sum()),
         tot_abs[7:].sum() / max(abs(tot_signed[7:].sum()), 1e-12)))
print("  (ratio 1.0 = the deviations all point the same way and add up;")
print("   large ratio = they cancel along the chunk)")

# ---- sign consistency per dimension
print("\n=== sign consistency across the 50 steps ===")
same = 0
for k in range(K):
    s = np.sign(d[:, k])
    pos = int((s > 0).sum())
    maj = max(pos, T - pos)
    if maj >= 0.8 * T:
        same += 1
print("  dimensions whose deviation keeps one sign for >=80%% of the steps: %d of %d"
      % (same, K))

# ---- does the first step predict the rest?
print("\n=== does the step-0 deviation predict the chunk ===")
c = np.corrcoef(d[0], d.reshape(T, K).mean(axis=0))[0, 1]
print("  correlation between d[step 0] and the mean signed deviation: %.2f" % c)
print("  max|d| at step 0 %.4f vs mean max|d| over all steps %.4f"
      % (np.abs(d[0]).max(), np.abs(d).max(axis=1).mean()))