"""Render the four figures as PNGs. Every figure is included at \\linewidth = 175 mm.

Design rules enforced here and checked at the end of the run:
  * no title, subtitle or summary sentence inside the image -- those belong in the caption
  * every text element is at least 8 pt once the figure is scaled to 175 mm
  * multi-panel figures carry (a), (b), (c) letters
  * no text box leaves its panel

Data provenance:
  repeats   -- aaq0 / aaq8 sweeps, 10 repeats x 10 tasks x 30 episodes
  per task  -- ms_report / ms_task over 5 training runs x 30 episodes = 150 per cell
"""
import os

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)


def _src(name):
    """Find a source image: the output directory first, then next to this script."""
    for d in (OUT, os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit("missing source image: " + name)
S = 6
MM = 175.0  # printed width of every figure

INK = (26, 26, 26)
NONE_FILL = (176, 176, 176)   # luminance 176
PLACE_FILL = (120, 32, 24)     # luminance 52, BLUE is 84
GREY = (110, 110, 110)
LGREY = (222, 222, 222)
BLUE = (47, 93, 138)
RED = (163, 58, 44)

FONTS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
FONTS_R = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

AAQ0 = [168, 171, 168, 168, 168, 168, 165, 166, 170, 164]
AAQ8 = [165, 167, 162, 168, 163, 163, 164, 167, 166, 165]

K30 = {
    "none": [56, 106, 57, 33, 92, 38, 33, 116, 54, 30],
    "cont": [71, 86, 63, 28, 101, 41, 29, 114, 60, 26],
    "place": [72, 87, 59, 28, 62, 34, 21, 121, 53, 32],
}
K50 = {
    "none": [78, 105, 54, 38, 107, 41, 38, 129, 74, 38],
    "cont": [75, 90, 64, 38, 98, 41, 34, 119, 77, 33],
    "place": [80, 105, 66, 36, 89, 40, 35, 128, 61, 26],
}


class Canvas:
    """Image plus a registry of every text box, so the layout can be checked."""

    def __init__(self, w, h, tag):
        self.W, self.H, self.tag = w, h, tag
        self.img = Image.new("RGB", (w * S, h * S), "white")
        self.d = ImageDraw.Draw(self.img)
        self.boxes = []
        self.minpt = 99.0

    def font(self, size, bold=False):
        for p in (FONTS if bold else FONTS_R):
            if os.path.exists(p):
                return ImageFont.truetype(p, int(size * S))
        return ImageFont.load_default()

    def txt(self, x, y, s, size, anchor="mm", fill=INK, bold=False):
        f = self.font(size, bold)
        pt = size * MM / self.W * 2.835
        self.minpt = min(self.minpt, pt)
        self.d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)
        l, t, r, b = self.d.textbbox((x * S, y * S), s, font=f, anchor=anchor)
        self.boxes.append((l / S, t / S, r / S, b / S, s))

    def vtxt(self, x, y, s, size, fill=GREY):
        f = self.font(size)
        pt = size * MM / self.W * 2.835
        self.minpt = min(self.minpt, pt)
        probe = Image.new("RGBA", (10, 10))
        w = int(ImageDraw.Draw(probe).textlength(s, font=f)) + 4
        tmp = Image.new("RGBA", (w, int(size * S) + 12), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((2, 2), s, font=f, fill=fill + (255,))
        tmp = tmp.crop(tmp.getbbox()).rotate(90, expand=True)
        self.img.paste(tmp, (int(x * S - tmp.width / 2), int(y * S - tmp.height / 2)), tmp)
        self.boxes.append((x - tmp.width / S / 2, y - tmp.height / S / 2,
                           x + tmp.width / S / 2, y + tmp.height / S / 2, s))

    def line(self, *a, **k):
        self.d.line(*a, **k)

    def rect(self, *a, **k):
        self.d.rectangle(*a, **k)

    def fig(self, name):
        bad = [b for b in self.boxes
               if b[0] < -1 or b[1] < -1 or b[2] > self.W + 1 or b[3] > self.H + 1]
        clash = []
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                a, b = self.boxes[i], self.boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    clash.append((a[4], b[4]))
        print("%-28s %4dx%-4d  text %2d  min %.1f pt  %s%s"
              % (name, self.W, self.H, len(self.boxes), self.minpt,
                 "OK" if not bad else "OVERFLOW: %s" % [b[4] for b in bad][:3],
                 "" if not clash else "  TEXT-CLASH: %s" % clash[:4]))
        from PIL import ImageChops
        bg = Image.new("RGB", self.img.size, "white")
        mask = ImageChops.difference(self.img, bg).convert("L").point(lambda v: 255 if v > 6 else 0)
        bb = mask.getbbox()
        if bb:
            pad = 7 * S
            bb = (max(0, bb[0] - pad), max(0, bb[1] - pad),
                  min(self.img.width, bb[2] + pad), min(self.img.height, bb[3] + pad))
            self.img = self.img.crop(bb)
        self.img.save(os.path.join(OUT, name), dpi=(300, 300))


# --------------------------------------------------------------------- fig 1
def fig_spread():
    W, H = 700, 400
    L, R, T, B = 92, 34, 70, 74
    pw, ph = W - L - R, H - T - B
    c = Canvas(W, H, "fig1")
    lo, hi = 158, 174

    def X(i):
        return L + pw * i / 9.0

    def Y(v):
        return T + ph - ph * (v - lo) / (hi - lo)

    for v in range(lo, hi + 1, 2):
        c.line([(L * S, Y(v) * S), ((L + pw) * S, Y(v) * S)], fill=LGREY, width=S)
        c.txt(L - 12, Y(v), "%.1f" % (100.0 * v / 300), 15, anchor="rm", fill=GREY)
    c.line([(L * S, T * S), (L * S, (T + ph) * S)], fill=INK, width=S)
    c.line([(L * S, (T + ph) * S), ((L + pw) * S, (T + ph) * S)], fill=INK, width=S)
    c.vtxt(26, T + ph / 2, "success rate (%)", 13)

    for series, colr, lab, sq in ((AAQ0, RED, "unquantised", False),
                                  (AAQ8, BLUE, "step 8", True)):
        m = sum(series) / len(series)
        y = Y(m)
        x = L
        while x < L + pw:
            dash, spacing = (3, 7) if sq else (8, 13)
            c.line([(x * S, y * S), (min(x + dash, L + pw) * S, y * S)], fill=colr, width=S)
            x += spacing
        for i, v in enumerate(series):
            cx, cy, r = X(i) * S, Y(v) * S, 5 * S
            if sq:
                c.rect([cx - r, cy - r, cx + r, cy + r], fill=colr, outline="white", width=S)
            else:
                c.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colr,
                            outline="white", width=S)

    ly = 27
    for k, (colr, lab, sq) in enumerate(((RED, "unquantised", False),
                                         (BLUE, "step 8", True))):
        yy = ly
        legend_x = L + k * 205
        if sq:
            c.rect([(legend_x + 5) * S - 5 * S, yy * S - 5 * S, (legend_x + 5) * S + 5 * S, yy * S + 5 * S],
                   fill=colr)
        else:
            c.d.ellipse([(legend_x + 5) * S - 5 * S, yy * S - 5 * S, (legend_x + 5) * S + 5 * S,
                         yy * S + 5 * S], fill=colr)
        c.txt(legend_x + 18, yy, lab, 15, anchor="lm", fill=colr)

    for i in range(10):
        c.txt(X(i), T + ph + 20, str(i + 1), 15, fill=GREY)
    c.txt(L + pw / 2, T + ph + 46, "repeat index", 15, fill=GREY)
    c.fig("fig1-repeat-spread.png")


# --------------------------------------------------------------------- fig 2
def fig_task_bars():
    W, H = 760, 440
    L0, gap = 74, 46
    pw = (W - L0 - 22 - gap) / 2.0
    T, B = 40, 96
    ph = H - T - B
    c = Canvas(W, H, "fig2")
    panels = [(K30, "(a) 30k steps"), (K50, "(b) 50k steps")]
    colours = {"none": NONE_FILL, "cont": BLUE, "place": PLACE_FILL}

    for pi, (data, lab) in enumerate(panels):
        L = L0 + pi * (pw + gap)
        c.txt(L, T - 14, lab, 15, anchor="lm", bold=True)
        for v in (0, 25, 50, 75, 100):
            y = T + ph - ph * v / 100.0
            c.line([(L * S, y * S), ((L + pw) * S, y * S)], fill=LGREY, width=S)
            c.txt(L - 10, y, str(v), 15, anchor="rm", fill=GREY)
        c.line([(L * S, T * S), (L * S, (T + ph) * S)], fill=INK, width=S)
        c.line([(L * S, (T + ph) * S), ((L + pw) * S, (T + ph) * S)], fill=INK, width=S)
        if pi == 0:
            c.vtxt(26, T + ph / 2, "success (%)", 15)
        gw = pw / 10.0
        bw = gw / 4.2
        for t in range(10):
            for k, arm in enumerate(("none", "cont", "place")):
                h = ph * (data[arm][t] / 150.0)
                x = L + gw * t + gw * 0.10 + k * bw
                c.rect([x * S, (T + ph - h) * S, (x + bw * 0.84) * S, (T + ph) * S],
                       fill=colours[arm])
            c.txt(L + gw * t + gw / 2, T + ph + 18, str(t), 15, fill=GREY)
        c.txt(L + pw / 2, T + ph + 44, "task id", 15, fill=GREY)

    lx, ly = W / 2 - 190, H - 18
    for k, arm in enumerate(("none", "cont", "place")):
        x = lx + k * 150
        c.rect([x * S, (ly - 7) * S, (x + 15) * S, (ly + 7) * S], fill=colours[arm])
        c.txt(x + 24, ly, arm, 15, anchor="lm")
    c.fig("fig4-task-bars.png")


# --------------------------------------------------------------------- fig 3
def fig_pixel_chain():
    W, H = 900, 470
    c = Canvas(W, H, "fig3")

    # ---- (a) the rendered wrist frame itself, with the two differing pixels marked
    side = 200
    ax, ay = 40, 74

    def PA(x, y):
        return ax + side * x / 256.0, ay + side * y / 256.0

    c.txt(ax, ay - 16, "(a) wrist camera", 16, anchor="lm", bold=True)
    # Lossless source. The earlier wrist-frame.jpg differed from this frame in 31328 of
    # 65536 pixels (max 15 levels) and read 44 where the data holds 45, which is the exact
    # magnitude the figure is about.
    frame = Image.open(_src(os.environ.get("WRIST_SRC", "wrist-a.png"))).convert("RGB")
    c.img.paste(frame.resize((side * S, side * S), Image.LANCZOS), (ax * S, ay * S))
    c.rect([ax * S, ay * S, (ax + side) * S, (ay + side) * S], outline=INK, width=S)
    mx, my = PA(46.5, 65.0)
    c.d.ellipse([(mx - 8) * S, (my - 8) * S, (mx + 8) * S, (my + 8) * S],
                outline=RED, width=2 * S)
    lx = ax + side / 2
    c.txt(lx, ay + side + 26, "256 x 256, rendered frame", 16, fill=GREY)
    c.txt(lx, ay + side + 55, "2 px of 65536", 16, fill=RED, bold=True)
    c.txt(lx, ay + side + 80, "differ by 1 level", 16, fill=RED)

    # ---- (b) divergence along the chunk
    perstep = [0.0158, 0.0127, 0.0107, 0.013, 0.0068, 0.0059, 0.0093, 0.0039, 0.0068,
               0.0078, 0.0059, 0.0041, 0.0107, 0.0117, 0.0054, 0.0107, 0.0107, 0.0107,
               0.0093, 0.0065, 0.0098, 0.0078, 0.0078, 0.0112, 0.0068, 0.0039, 0.0059,
               0.0059, 0.0045, 0.0033, 0.0033, 0.0042, 0.0042, 0.0071, 0.0078, 0.0049,
               0.0088, 0.0031, 0.0063, 0.0059, 0.0044, 0.0088, 0.0078, 0.0037, 0.0067,
               0.0061, 0.0042, 0.0072, 0.0059, 0.0039]
    bw, bh = 240, 250
    bx, by = 350, 74

    def PBX(i):
        return bx + bw * i / 49.0

    def PBY(v):
        return by + bh - bh * v / 0.017

    c.txt(bx, by - 16, "(b) first action chunk", 16, anchor="lm", bold=True)
    c.vtxt(bx - 65, by + bh / 2, "max |delta|", 16)
    for v in (0, 0.005, 0.010, 0.015):
        c.line([(bx * S, PBY(v) * S), ((bx + bw) * S, PBY(v) * S)], fill=LGREY, width=S)
        c.txt(bx - 8, PBY(v), "%.3f" % v, 16, anchor="rm", fill=GREY)
    c.line([(bx * S, by * S), (bx * S, (by + bh) * S)], fill=INK, width=S)
    c.line([(bx * S, (by + bh) * S), ((bx + bw) * S, (by + bh) * S)], fill=INK, width=S)
    pts = [(PBX(i) * S, PBY(v) * S) for i, v in enumerate(perstep)]
    c.line(pts, fill=BLUE, width=int(1.8 * S), joint="curve")
    c.txt(PBX(0) + 10, PBY(perstep[0]) - 16, "%.4f" % perstep[0], 16, anchor="lm",
          fill=BLUE, bold=True)
    for i in (0, 25, 49):
        c.txt(PBX(i), by + bh + 18, str(i), 16, fill=GREY)
    c.txt(bx + bw / 2, by + bh + 42, "action step", 16, fill=GREY)

    # ---- (c) which dimensions move
    perdim = [0.0059, 0.0029, 0.0039, 0.0112, 0.0088, 0.0158, 0.0072, 0.0057, 0.0039,
              0.0051, 0.0047, 0.0042, 0.0042, 0.0041, 0.0028, 0.003, 0.0032, 0.0039,
              0.0044, 0.0034, 0.0035, 0.0034, 0.0032, 0.0048, 0.0031, 0.0039, 0.0048,
              0.006, 0.0029, 0.004, 0.0033, 0.0024]
    cw = 200
    cx, cy = 675, 74

    def CCY(v):
        return cy + bh - bh * v / 0.017

    c.txt(cx, cy - 16, "(c) 32 dims", 16, anchor="lm", bold=True)
    for v in (0, 0.005, 0.010, 0.015):
        c.line([(cx * S, CCY(v) * S), ((cx + cw) * S, CCY(v) * S)], fill=LGREY, width=S)
        c.txt(cx - 8, CCY(v), "%.3f" % v, 16, anchor="rm", fill=GREY)
    c.line([(cx * S, cy * S), (cx * S, (cy + bh) * S)], fill=INK, width=S)
    c.line([(cx * S, (cy + bh) * S), ((cx + cw) * S, (cy + bh) * S)], fill=INK, width=S)
    gw = cw / 32.0
    for i, v in enumerate(perdim):
        h = bh * v / 0.017
        c.rect([(cx + gw * i + 0.5) * S, CCY(v) * S, (cx + gw * (i + 1) - 0.5) * S,
                (cy + bh) * S], fill=BLUE if i < 7 else GREY)
    c.line([(cx * S, cy * S), ((cx + 7 * gw) * S, cy * S)], fill=BLUE, width=S)
    c.txt(cx + 3.5 * gw, cy + bh + 18, "real", 16, fill=BLUE)
    c.txt(cx + 19 * gw, cy + bh + 18, "padding", 16, fill=GREY)
    c.txt(cx + 3.5 * gw, cy + bh + 40, "0-6", 16, fill=BLUE)
    c.txt(cx + 19 * gw, cy + bh + 40, "7-31", 16, fill=GREY)
    c.txt(cx + cw / 2, cy + bh + 65, "dimension", 16, fill=GREY)
    c.fig("fig2-pixel-chain.png")


# --------------------------------------------------------------------- fig 4
def fig_sweep():
    W, H = 880, 470
    c = Canvas(W, H, "fig4")
    steps = [1, 2, 4, 8, 16, 32]
    totals = [59, 32, 18, 9, 0, 0]
    model = [59.0 / s for s in steps]
    pairs = [("t0", [8, 2, 0, 3, 0, 0]), ("t1", [9, 9, 9, 0, 0, 0]),
             ("t4", [12, 6, 0, 0, 0, 0]), ("t5", [6, 3, 0, 3, 0, 0]),
             ("t7", [12, 3, 3, 3, 0, 0]), ("t8", [12, 9, 6, 0, 0, 0])]

    L, T, pw, ph = 78, 84, 300, 240

    def X(i):
        return L + pw * (i + 0.5) / 6.0

    def Y(v):
        return T + ph - ph * v / 70.0

    c.txt(L - 8, T - 20, "(a) pooled", 15, anchor="lm", bold=True)
    for v in range(0, 61, 20):
        c.line([(L * S, Y(v) * S), ((L + pw) * S, Y(v) * S)], fill=LGREY, width=S)
        c.txt(L - 10, Y(v), str(v), 15, anchor="rm", fill=GREY)
    c.line([(L * S, T * S), (L * S, (T + ph) * S)], fill=INK, width=S)
    c.line([(L * S, (T + ph) * S), ((L + pw) * S, (T + ph) * S)], fill=INK, width=S)
    c.vtxt(24, T + ph / 2, "differing values", 15)

    bw = pw / 6.0 * 0.5
    for i, v in enumerate(totals):
        x0 = X(i) - bw / 2
        c.rect([x0 * S, Y(v) * S, (x0 + bw) * S, (T + ph) * S], fill=BLUE)
        c.txt(X(i), Y(v) - (29 if v == 0 else 20), str(v), 15, fill=BLUE, bold=True)

    pts = [(X(i), Y(model[i])) for i in range(6)]
    for i in range(5):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        for k in range(8):
            if k % 2 == 0:
                xa = x0 + (x1 - x0) * k / 8.0
                ya = y0 + (y1 - y0) * k / 8.0
                xb = x0 + (x1 - x0) * (k + 1) / 8.0
                yb = y0 + (y1 - y0) * (k + 1) / 8.0
                c.line([(xa * S, ya * S), (xb * S, yb * S)], fill=RED, width=2 * S)

    for i in range(6):
        c.txt(X(i), T + ph + 18, str(steps[i]), 15, fill=GREY)
    c.txt(L + pw / 2, T + ph + 42, "step", 15, fill=GREY)
    c.txt(L + 138, T + 16, "measured", 15, fill=BLUE, anchor="lm", bold=True)
    for offset in (0, 14, 28):
        c.line([(L + 138 + offset) * S, (T + 38) * S,
                (L + 146 + offset) * S, (T + 38) * S], fill=RED, width=2 * S)
    c.txt(L + 182, T + 38, "59/step", 15, fill=RED, anchor="lm")

    TX, TY, cww = 448, 140, 58
    c.txt(TX, T - 20, "(b) per task pair", 15, anchor="lm", bold=True)
    for j, s in enumerate(steps):
        c.txt(TX + 62 + cww * j + cww / 2, TY - 26, str(s), 15, fill=GREY)
    for i, (nm, vals) in enumerate(pairs):
        y = TY + 38 * i
        c.txt(TX, y, nm, 15, anchor="lm", bold=True)
        for j, v in enumerate(vals):
            c.txt(TX + 62 + cww * j + cww / 2, y, str(v), 15, anchor="mm",
                  fill=INK if v else (140, 140, 140))
    c.txt(TX + 62 + cww * 3, TY - 54, "step", 15, fill=GREY)
    c.fig("fig3-quantiser-sweep.png")


if __name__ == "__main__":
    fig_spread()
    fig_pixel_chain()
    fig_sweep()
    fig_task_bars()
