"""Fifty launches of one pinned episode: where do the images actually differ?

Reads ~/pi05/dumps50/r*/ (one directory per launch, same config every time) and answers
three questions the two-pair comparison in the manuscript cannot:

  1. Does the agent view ever differ, or is it stable in a way the wrist camera is not?
  2. Are the differing pixels always one grey level apart, and always on an edge?
  3. Does the first action chunk follow the image, one-to-one?

Usage:  ~/pi05/env/bin/python dumps50_analysis.py [dumps-root]
"""
import glob
import hashlib
import json
import os
import sys
from itertools import combinations

import numpy as np
from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/pi05/dumps50")
AGENT = "raw_agentview_image.png"
WRIST = "raw_robot0_eye_in_hand_image.png"
CHUNK = "ep001_chunk0.npy"


def run_key(p):
    tail = os.path.basename(p).lstrip("r")
    return int(tail) if tail.isdigit() else 10 ** 9


def load(name):
    runs = sorted([d for d in glob.glob(os.path.join(BASE, "*"))
                   if os.path.isdir(d) and os.path.exists(os.path.join(d, name))], key=run_key)
    if not runs:
        return None, []
    arr = np.stack([np.array(Image.open(os.path.join(d, name)).convert("RGB"), dtype=np.uint8)
                    for d in runs])
    return arr, [os.path.basename(d) for d in runs]


def unstable_pixels(A):
    """(n,H,W,3) -> boolean (H,W): any channel is not unanimous across launches."""
    return (A.max(axis=0) != A.min(axis=0)).any(axis=2)


def gradient(U):
    g = U.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(g, axis=1, append=g[:, -1:]))
    gy = np.abs(np.diff(g, axis=0, append=g[-1:, :]))
    return np.hypot(gx, gy)


def report(name, A, runs):
    n = A.shape[0]
    print("\n=== %s === %d launches, %dx%d" % (name, n, A.shape[2], A.shape[1]))
    variants = len({hashlib.sha256(A[i].tobytes()).hexdigest() for i in range(n)})
    print("  distinct images: %d of %d" % (variants, n))
    U = unstable_pixels(A)
    k = int(U.sum())
    print("  pixels that are not unanimous: %d of %d" % (k, U.size))
    if k == 0:
        return A, U, None
    vals = A[:, U, :].reshape(n, k * 3)
    rng = (vals.max(axis=0) - vals.min(axis=0))
    print("  spread at those pixels: max %d level(s), median %d"
          % (int(rng.max()), int(np.median(rng))))
    grad = gradient(A.mean(axis=0))
    pct = [(grad[U] > g).mean() * 100 for g in (grad[U])]
    print("  edge position: the unstable pixels sit at the %.0f-%.0f percentile of"
          " local gradient" % (np.percentile(pct, 5), np.percentile(pct, 95)))
    return A, U, grad


A, arun = load(AGENT)
W, wrun = load(WRIST)
if A is None or W is None:
    print("need both images in every run directory; found agent=%s wrist=%s"
          % (len(arun), len(wrun)))
    sys.exit(1)
if arun != wrun:
    print("warning: run lists differ, comparing the %d common runs"
          % len(set(arun) & set(wrun)))
    common = [r for r in arun if r in set(wrun)]
    A = A[[arun.index(r) for r in common]]
    W = W[[wrun.index(r) for r in common]]
    arun = common

report("agent view", A, arun)
report("wrist camera", W, arun)

# ---- pairwise: how often does a random pair differ, per camera
n = A.shape[0]
ca = cw = both = neither = 0
for i, j in combinations(range(n), 2):
    da = not np.array_equal(A[i], A[j])
    dw = not np.array_equal(W[i], W[j])
    ca += da
    cw += dw
    both += da and dw
    neither += not da and not dw
tot = ca * 0 + n * (n - 1) // 2
print("\n=== all %d pairs of launches ===" % tot)
print("  agent view differs      : %4d  (%.1f%%)" % (ca, 100.0 * ca / tot))
print("  wrist camera differs    : %4d  (%.1f%%)" % (cw, 100.0 * cw / tot))
print("  both differ             : %4d" % both)
print("  neither differs         : %4d  (%.1f%%)" % (neither, 100.0 * neither / tot))

# ---- does the chunk follow the image?
chunks = {}
for d in [os.path.join(BASE, r) for r in arun]:
    p = os.path.join(d, CHUNK)
    if os.path.exists(p):
        chunks[r] = np.load(p)
if chunks:
    keys = {}
    for r, v in chunks.items():
        keys[r] = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    print("\n=== first action chunk ===")
    print("  distinct chunks: %d of %d launches" % (len(set(keys.values())), len(keys)))
    groups = {}
    for r, k in keys.items():
        groups.setdefault(k, []).append(r)
    imgs = {r: hashlib.sha256(W[arun.index(r)].tobytes()).hexdigest() for r in chunks}
    print("  image-hash -> chunk-hash mapping: %s"
          % ("one-to-one" if len(set(zip(imgs.values(), keys.values()))) == len(set(imgs.values()))
             else "NOT one-to-one"))
    for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:6]:
        print("    chunk %s  %2d launches, %2d distinct wrist images"
              % (k[:8], len(g), len({imgs[r] for r in g})))

# ---- what was pinned
info = os.path.join(BASE, arun[0], "info.json")
if os.path.exists(info):
    d = json.load(open(info))
    keep = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
    print("\n=== info.json of %s ===" % arun[0])
    for k, v in sorted(keep.items())[:20]:
        print("  %-24s %s" % (k, v))