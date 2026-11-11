"""Is the agent view really identical across every dump pair, not just the two in Table 8?

Compares raw_agentview_image.png and, for cross-reference, fmt_image.png / fmt_image2.png
in every dump pair, and reports differing pixels and the largest level difference.

If the agent view is identical in all thirteen pairs, Table 8's "65536/65536 identical"
holds at n=13 rather than n=2, which is the same claim at six times the sample.

Usage:  ~/pi05/env/bin/python agentview_pairs.py [root ...]
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

ROOTS = sys.argv[1:] or [os.path.expanduser("~/pi05/dumps"),
                         os.path.expanduser("~/pi05/dumps_q")]


def pairs():
    out = []
    for root in ROOTS:
        for a in sorted(glob.glob(os.path.join(root, "*_a"))):
            b = a[:-2] + "_b"
            if os.path.isdir(b):
                out.append((os.path.basename(a)[:-2], a, b))
        for a, b in (("a", "b"), ("same_a", "same_b"), ("bf16c", "bf16d")):
            pa, pb = os.path.join(root, a), os.path.join(root, b)
            if os.path.isdir(pa) and os.path.isdir(pb):
                out.append(("%s %s" % (os.path.basename(root), a), pa, pb))
    seen, uniq = set(), []
    for nm, pa, pb in out:
        if (pa, pb) not in seen:
            seen.add((pa, pb))
            uniq.append((nm, pa, pb))
    return uniq


def load(d, f):
    p = os.path.join(d, f)
    return np.array(Image.open(p).convert("RGB"), dtype=np.int16) if os.path.exists(p) else None


FILES = [("raw_agentview_image.png", "agent"),
         ("fmt_image.png", "fmt1"),
         ("fmt_image2.png", "fmt2")]

print("%-18s %14s %14s %14s" % ("pair", "raw agent view", "fmt_image", "fmt_image2"))
print("-" * 66)
tot = 0
clean = 0
for nm, pa, pb in pairs():
    cells = []
    for f, _ in FILES:
        A, B = load(pa, f), load(pb, f)
        if A is None or B is None or A.shape != B.shape:
            cells.append("n/a")
            continue
        k = int((A != B).any(axis=2).sum())
        cells.append("identical" if k == 0 else "%d px, max|d| %d" % (k, int(np.abs(A - B).max())))
    tot += 1
    if cells[0] == "identical":
        clean += 1
    print("%-18s %14s %14s %14s" % (nm, cells[0], cells[1], cells[2]))

print("\nagent view identical in %d of %d pairs" % (clean, tot))
if clean == tot:
    print("=> the agent-view column of the camera table can be stated at n=%d, not n=2." % tot)
else:
    print("=> the agent view is NOT always identical; the exceptions above need explaining.")