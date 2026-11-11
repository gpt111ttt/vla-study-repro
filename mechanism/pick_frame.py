"""Find the dump pair whose wrist images differ at (45,64) and (47,65), and export that frame.

Prints, for every task pair, how many wrist pixels differ and where; then emits the base64 of
one frame from the pair that matches the coordinates Figure 3 marks.
"""
import base64
import glob
import io
import os

import numpy as np
from PIL import Image

D = os.path.expanduser("~/pi05/dumps_q")
WANT = {(45, 64), (47, 65)}
hits = []

for t in range(10):
    fa = os.path.join(D, "t%d_a" % t, "raw_robot0_eye_in_hand_image.png")
    fb = os.path.join(D, "t%d_b" % t, "raw_robot0_eye_in_hand_image.png")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print("t%d: missing" % t)
        continue
    A = np.array(Image.open(fa).convert("RGB"), dtype=np.int16)
    B = np.array(Image.open(fb).convert("RGB"), dtype=np.int16)
    if A.shape != B.shape:
        print("t%d: shape mismatch %s %s" % (t, A.shape, B.shape))
        continue
    m = np.abs(A - B).max(axis=2)
    ys, xs = np.nonzero(m)
    coords = sorted(set(zip(ys.tolist(), xs.tolist())))
    lvl = [int(m[y, x]) for (y, x) in coords]
    print("t%d: %dx%d  %d differing px  %s  levels %s"
          % (t, A.shape[1], A.shape[0], len(coords), coords[:6], lvl[:6]))
    if WANT.issubset(set(coords)):
        hits.append((t, fa))

print()
if not hits:
    print("NO PAIR MATCHES (45,64)&(47,65)")
    # fall back: the pair with the fewest differing pixels
    raise SystemExit(0)

t, path = hits[0]
im = Image.open(path).convert("RGB")
print("PICK t%d_a  %s  size %s" % (t, path, im.size))
buf = io.BytesIO()
im.save(buf, "JPEG", quality=90, optimize=True)
b64 = base64.b64encode(buf.getvalue()).decode()
print("B64 %d" % len(b64))
print(b64)