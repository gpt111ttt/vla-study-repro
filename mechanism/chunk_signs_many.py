"""Signed chunk differences across every dump pair available: is "real dims accumulate,
padding cancels" a one-off or a pattern?

For each pair of dump directories it reports

  px        differing wrist pixels (the renderer difference)
  step0     max |delta| at action step 0
  Sreal     sum |delta| over the 7 real action dims
  net       |sum of the signed delta| over those dims
  r_real    Sreal / net      (1 = deviations all point one way and add up)
  r_pad     the same ratio for the 25 padding dims, which carry no information

and then checks the two invariants the manuscript relies on:
  images identical -> chunk identical, and images differ -> chunk differs.

Usage:  ~/pi05/env/bin/python chunk_signs_many.py [root ...]
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

ROOTS = sys.argv[1:] or [os.path.expanduser("~/pi05/dumps"),
                         os.path.expanduser("~/pi05/dumps_q")]
WRIST = "raw_robot0_eye_in_hand_image.png"
CHUNK = "ep001_chunk0.npy"


def find_pairs():
    out = []
    for root in ROOTS:
        for a in sorted(glob.glob(os.path.join(root, "*_a"))):
            b = a[:-2] + "_b"
            if os.path.isdir(b):
                out.append((os.path.basename(a)[:-2], a, b))
        for a, b in (("a", "b"), ("same_a", "same_b"), ("bf16c", "bf16d")):
            pa, pb = os.path.join(root, a), os.path.join(root, b)
            if os.path.isdir(pa) and os.path.isdir(pb):
                out.append(("%s %s/%s" % (os.path.basename(root), a, b), pa, pb))
    seen, uniq = set(), []
    for nm, pa, pb in out:
        if (pa, pb) not in seen:
            seen.add((pa, pb))
            uniq.append((nm, pa, pb))
    return uniq


def chunk(d):
    p = os.path.join(d, CHUNK)
    if not os.path.exists(p):
        return None
    v = np.squeeze(np.load(p))
    return (v if v.ndim == 2 else v.reshape(-1, v.shape[-1])).astype(np.float64)


def img(d):
    p = os.path.join(d, WRIST)
    return np.array(Image.open(p).convert("RGB"), dtype=np.int16) if os.path.exists(p) else None


rows = []
for name, pa, pb in find_pairs():
    ca, cb = chunk(pa), chunk(pb)
    if ca is None or cb is None or ca.shape != cb.shape:
        rows.append((name, None))
        continue
    ia, ib = img(pa), img(pb)
    npx = int((np.abs(ia - ib).max(axis=2) > 0).sum()) if ia is not None and ib is not None else -1
    d = ca - cb
    T, K = d.shape
    nreal = min(7, K)
    real, pad = d[:, :nreal], d[:, nreal:]
    s_real, s_pad = np.abs(real).sum(), np.abs(pad).sum()
    n_real, n_pad = abs(real.sum()), abs(pad.sum())
    r_real = s_real / max(n_real, 1e-12) if n_real > 0 else np.inf
    r_pad = s_pad / max(n_pad, 1e-12) if n_pad > 0 else np.inf
    net = d.sum(axis=0)
    top = sorted(range(K), key=lambda k: -abs(net[k]))[:2]
    same = sum(1 for k in range(K) if max((d[:, k] > 0).sum(), (d[:, k] < 0).sum()) >= 0.8 * T)
    rows.append((name, dict(px=npx, T=T, K=K, step0=np.abs(d[0]).max(), dmax=np.abs(d).max(),
                            s_real=s_real, n_real=n_real, r_real=r_real,
                            s_pad=s_pad, n_pad=n_pad, r_pad=r_pad,
                            top=top, net=net, same=same)))

print("%-22s %5s %7s %7s %7s %6s %7s %8s %6s %-8s %4s"
      % ("pair", "px", "step0", "Sreal", "net", "r_real", "Spad", "net_pad", "r_pad", "top", "same"))
print("-" * 104)
good = []
for name, m in rows:
    if m is None:
        print("%-22s (no comparable chunk)" % name)
        continue
    print("%-22s %5d %7.4f %7.3f %7.3f %6.2f %7.3f %8.3f %6.2f %-8s %4d"
          % (name, m["px"], m["step0"], m["s_real"], m["n_real"],
             m["r_real"], m["s_pad"], m["n_pad"], m["r_pad"],
             ",".join(str(k) for k in m["top"]), m["same"]))
    good.append((name, m))

print("\n=== is the pattern consistent? ===")
live = [(n, m) for n, m in good if m["dmax"] > 0]
dead = [(n, m) for n, m in good if m["dmax"] == 0]
print("  pairs with identical chunks : %d of %d" % (len(dead), len(good)))
for n, m in dead:
    print("      %-20s wrist pixels differing: %d" % (n, m["px"]))
print("  pairs with differing chunks : %d" % len(live))
if live:
    rr = np.array([m["r_real"] for _, m in live])
    rp = np.array([m["r_pad"] for _, m in live])
    print("  r_real  median %.2f  range %.2f-%.2f" % (np.median(rr), rr.min(), rr.max()))
    print("  r_pad   median %.2f  range %.2f-%.2f" % (np.median(rp), rp.min(), rp.max()))
    more = sum(1 for _, m in live if m["r_real"] < m["r_pad"])
    print("  pairs where the real dims cancel LESS than the padding dims: %d of %d"
          % (more, len(live)))
    ratio_net = np.array([m["n_real"] / max(m["step0"] * m["K"], 1e-12) for _, m in live])
    print("  net real deviation as a multiple of step-0 amplitude: median %.1fx"
          % np.median([m["n_real"] / max(m["step0"], 1e-12) for _, m in live]))
    topdims = {}
    for _, m in live:
        for k in m["top"]:
            topdims[k] = topdims.get(k, 0) + 1
    print("  dimensions most often carrying the net deviation: %s"
          % sorted(topdims.items(), key=lambda kv: -kv[1]))

print("\n=== images <-> chunks ===")
ok = True
for name, m in good:
    img_same = (m["px"] == 0)
    chunk_same = (m["dmax"] == 0)
    if img_same != chunk_same and m["px"] >= 0:
        ok = False
        print("  EXCEPTION %-20s wrist px %d but chunk %s"
              % (name, m["px"], "identical" if chunk_same else "differs"))
print("  %s" % ("images identical <-> chunk identical holds in every pair"
                if ok else "the correspondence is broken in the pairs listed above"))