"""Locate the dump pair behind Figure 3 / Table 10 by pixel-value fingerprint.

Table 10 says the differing pixels hold 45 vs 46 and 84 vs 85. We print, for every candidate
pair, each differing pixel as (row,col) and (x,y) with its two values, so the fingerprint can
be matched exactly rather than guessed.
"""
import os

import numpy as np
from PIL import Image

BASE = os.path.expanduser("~/pi05")
CANDIDATES = [
    ("dumps a/b", "dumps/a", "dumps/b"),
    ("dumps same_a/same_b", "dumps/same_a", "dumps/same_b"),
    ("dumps bf16c/bf16d", "dumps/bf16c", "dumps/bf16d"),
]
for t in range(10):
    CANDIDATES.append(("dumps_q t%d" % t, "dumps_q/t%d_a" % t, "dumps_q/t%d_b" % t))

for name, da, db in CANDIDATES:
    pa = os.path.join(BASE, da, "raw_robot0_eye_in_hand_image.png")
    pb = os.path.join(BASE, db, "raw_robot0_eye_in_hand_image.png")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    A = np.array(Image.open(pa).convert("RGB"), dtype=np.int16)
    B = np.array(Image.open(pb).convert("RGB"), dtype=np.int16)
    m = np.abs(A - B).max(axis=2)
    ys, xs = np.nonzero(m)
    print("== %s   %d differing px" % (name, len(ys)))
    for y, x in zip(ys.tolist(), xs.tolist()):
        va = tuple(int(c) for c in A[y, x])
        vb = tuple(int(c) for c in B[y, x])
        print("   (row,col)=(%d,%d)  (x,y)=(%d,%d)   %s vs %s" % (y, x, x, y, va, vb))