"""Pull every quantitative claim out of main.tex so each can be traced to data."""
import re

PATH = r"D:\桌面文件\vla\reports\main.tex"
OUT = r"D:\桌面文件\vla\reports\number-audit.txt"

lines = open(PATH, encoding="utf-8").read().split("\n")

PAT = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:\\%|percent)"
    r"|\d+\s+of\s+\d+"
    r"|\d+/\d+"
    r"|one in [a-z]+"
    r"|\b(?:about|roughly|nearly|almost|around)\b[^.]{0,40}\d)")

rows = []
for i, ln in enumerate(lines, 1):
    if ln.lstrip().startswith("%"):
        continue
    for m in PAT.finditer(ln):
        s = max(0, m.start() - 55)
        e = min(len(ln), m.end() + 55)
        rows.append((i, m.group(0).strip(), ln[s:e].strip()))

# group by section for readability
sec = {}
cur = "front"
for i, tok, ctx in rows:
    sec.setdefault(cur, [])
    sec[cur].append((i, tok, ctx))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("quantitative claims in main.tex: %d\n" % len(rows))
    f.write("lines scanned: %d\n\n" % len(lines))
    for i, tok, ctx in rows:
        f.write("%4d  [%s]\n      %s\n" % (i, tok, ctx))

kinds = {}
for _, tok, _ in rows:
    k = "percent" if "%" in tok or "percent" in tok else ("count" if "/" in tok or " of " in tok else "other")
    kinds[k] = kinds.get(k, 0) + 1
print("wrote %d claims to %s" % (len(rows), OUT))
print(kinds)