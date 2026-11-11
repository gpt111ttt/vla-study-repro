#!/bin/bash
# One repeat of one arm: 10 LIBERO-Goal tasks x 30 episodes, resumable per task.
#   bash run_suite.sh <repeat> [quant] [tag]
set -u
export MUJOCO_GL=egl
export PYTHONPATH=$HOME/caopeng_workspace/libero_resources_v1/LIBERO
cd "$HOME/openvla_probe"

REPEAT=${1:-0}
QUANT=${2:-1}
TAG=${3:-n1}
OUT=$HOME/openvla_probe/runs/${TAG}_r${REPEAT}
mkdir -p "$OUT"

for T in 0 1 2 3 4 5 6 7 8 9; do
  J="$OUT/eval_openvla_t${T}_q${QUANT}.json"
  if [ -f "$J" ]; then echo "skip task $T"; continue; fi
  echo "=== task $T q$QUANT repeat $REPEAT ==="
  ./env/bin/python -u eval_openvla_libero.py \
    --ckpt "$HOME/openvla_probe/ckpt" \
    --task_id "$T" --n_episodes 30 --quant "$QUANT" --out "$OUT" 2>&1 | grep -e "\[eval\]" -e "Traceback" -e "Error"
done

echo "=== summary $TAG repeat $REPEAT q$QUANT ==="
python3 - "$OUT" <<'PYEOF'
import glob, json, os, sys
d = sys.argv[1]
tot = ok = 0
rows = []
for p in sorted(glob.glob(os.path.join(d, "eval_openvla_t*_q*.json"))):
    r = json.load(open(p))
    tot += r["n"]
    ok += r["success"]
    rows.append((r["task_id"], r["success"], r["n"]))
print("SUITE %d/%d = %.4f" % (ok, tot, (ok / tot) if tot else 0.0))
for t, s, n in rows:
    print("  task %2d  %2d/%2d  %.3f" % (t, s, n, s / n))
open(os.path.join(d, "summary.txt"), "w").write(
    "SUITE %d/%d = %.4f\n" % (ok, tot, (ok / tot) if tot else 0.0)
    + "\n".join("task %2d %2d/%2d" % (t, s, n) for t, s, n in rows) + "\n")
PYEOF