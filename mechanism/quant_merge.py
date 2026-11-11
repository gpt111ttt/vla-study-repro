"""Does quantisation explain why some differing wrist images leave the chunk untouched?

For every dump pair, compare the wrist images three ways:
  raw          the renderer's output
  q_round      round(x/8)*8      (one plausible quantiser)
  q_floor      (x//8)*8          (the other)
and check whether the *quantised* images are identical. The manuscript predicts:

  quantised images identical -> the policy sees the same input -> chunk bit-identical
  quantised images differ    -> chunk differs

Fmt images dumped by the harness are compared too, as an independent witness.

Usage:  ~/pi05/env/bin/python quant_merge.py [root ...]
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ROOTS = sys.argv[1:] or [os.path.expanduser("~/pi05/dumps"),
                         os.path.expanduser("~/pi05/dumps_q")]
WRIST = "raw_robot0_eye_in_hand_image.png"


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


def load_img(d):
    p = os.path.join(d, WRIST)
    return np.array(Image.open(p).convert("RGB"), dtype=np.int64) if os.path.exists(p) else None


def load_chunk(d):
    p = os.path.join(d, "ep001_chunk0.npy")
    return np.load(p) if os.path.exists(p) else None


def fmt_state(d):
    """Which dumped fmt files are byte-identical is reported separately."""
    out = {}
    for f in ("fmt_image.png", "fmt_image2.png"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            out[f] = np.array(Image.open(p).convert("RGB"), dtype=np.int64)
    return out


rows = []
for nm, pa, pb in pairs():
    A, B = load_img(pa), load_img(pb)
    ca, cb = load_chunk(pa), load_chunk(pb)
    if A is None or B is None or ca is None or cb is None or A.shape != B.shape:
        continue
    qa_r, qb_r = np.round(A / 8.0) * 8, np.round(B / 8.0) * 8
    qa_f, qb_f = (A // 8) * 8, (B // 8) * 8
    raw = int((A != B).any(axis=2).sum())
    qr = int((qa_r != qb_r).any(axis=2).sum())
    qf = int((qa_f != qb_f).any(axis=2).sum())
    ch_same = bool(np.array_equal(ca, cb))
    fa, fb = fmt_state(pa), fmt_state(pb)
    fmt = {}
    for k in fa:
        if k in fb:
            fmt[k] = bool(np.array_equal(fa[k], fb[k]))
    q = ""
    for d, key in ((pa, "P0_QUANT"), (pb, "P0_QUANT")):
        p = os.path.join(d, "info.json")
        if os.path.exists(p):
            try:
                j = json.load(open(p))
                if key in j:
                    q = str(j[key])
                    break
            except Exception:
                pass
    rows.append((nm, raw, qr, qf, ch_same, fmt, q))

print("%-18s %5s %8s %8s  %-10s  %-22s %s"
      % ("pair", "raw", "q_round", "q_floor", "chunk", "fmt identical?", "QUANT"))
print("-" * 92)
for nm, raw, qr, qf, ch_same, fmt, q in rows:
    fs = ",".join("%s:%s" % (k.replace("fmt_image", "f"), "same" if v else "diff")
                  for k, v in fmt.items()) or "-"
    print("%-18s %5d %8d %8d  %-10s  %-22s %s"
          % (nm, raw, qr, qf, "same" if ch_same else "differs", fs, q or "?"))

print("\n=== the prediction: quantised-identical <-> chunk-identical ===")
for label, idx in (("round(x/8)*8", 2), ("(x//8)*8", 3)):
    agree = sum(1 for r in rows if (r[idx] == 0) == r[4])
    print("  %-14s agrees in %d of %d pairs" % (label, agree, len(rows)))

print("\n=== the cases that matter: raw images differ ===")
print("%-18s %5s %8s %8s  %-10s" % ("pair", "raw", "q_round", "q_floor", "chunk"))
for nm, raw, qr, qf, ch_same, fmt, q in rows:
    if raw > 0:
        print("%-18s %5d %8d %8d  %-10s" % (nm, raw, qr, qf, "same" if ch_same else "differs"))
n = sum(1 for r in rows if r[1] > 0)
print("\n  of %d pairs with differing raw wrists, %d keep an identical quantised image,"
      " and of those %d leave the chunk untouched"
      % (n, sum(1 for r in rows if r[1] > 0 and min(r[2], r[3]) == 0),
         sum(1 for r in rows if r[1] > 0 and min(r[2], r[3]) == 0 and r[4])))