#!/bin/bash
# run50.sh -- fifty launches of the SAME pinned episode, same GPU, same flags.
#
#   sbatch --gres=gpu:1 -c 4 -t 03:00:00 -o ~/pi05/logs/run50.log \
#          --wrap="bash ~/pi05/run50.sh 9"
#
# Why separate launches: the manuscript compares two *runs*, and dump_same_gpu.sh shows
# that even with the device pinned the wrist image still differs. Within one process the
# kernels are deterministic, so the variation we are measuring only appears between
# processes. Each launch therefore gets its own evaluation.
#
# Output: ~/pi05/dumps50/r01 .. r50, each with raw_agentview_image.png,
#         raw_robot0_eye_in_hand_image.png, ep001_chunk0.npy, info.json
#
# Then:   ~/pi05/env/bin/python ~/pi05/dumps50_analysis.py
set -u
ROOT=$HOME/pi05
TASK=${1:-9}
CKPT=${2:-$ROOT/models/_ckpt_expert_best50k}
N=${3:-50}
BASE=$ROOT/dumps50

if [ ! -f "$CKPT/model.safetensors" ]; then
  echo "FATAL: no model.safetensors under $CKPT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
mkdir -p "$BASE" "$ROOT/logs"
echo "=== run50 task=$TASK n=$N host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES ==="
echo "=== ckpt=$(basename "$CKPT") base=$BASE ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for k in $(seq -w 1 "$N"); do
  D="$BASE/r$k"
  if [ -f "$D/info.json" ] && [ -f "$D/raw_agentview_image.png" ]; then
    echo "[run50] r$k already present, skip"
    continue
  fi
  rm -rf "$D"
  mkdir -p "$D"
  t0=$(date +%s)
  P0_DUMP_DIR="$D" P0_QUANT=8 \
    bash "$ROOT/eval_sweep2.sh" "$TASK" 0 1 "d50_$k" "$CKPT" > "$ROOT/logs/d50_$k.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  files=$(find "$D" -type f 2>/dev/null | wc -l)
  echo "[run50] r$k rc=$rc ${dt}s files=$files"
  if [ "$files" -eq 0 ]; then
    echo "[run50] WARN: r$k produced no dump; see ~/pi05/logs/d50_$k.log" >&2
  fi
done

ok=$(ls -d "$BASE"/r* 2>/dev/null | wc -l)
echo "=== run50 done: $ok / $N dump dirs present ==="
echo "=== next: ~/pi05/env/bin/python ~/pi05/dumps50_analysis.py ==="