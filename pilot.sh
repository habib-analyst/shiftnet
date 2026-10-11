#!/bin/bash
# KILL-GATE PILOT: oracle-shift vs learned-shift vs feature-only routing.
#
# Three variants:
#   oracle: router gets GROUND-TRUTH family one-hot (upper bound).
#           Tests: CAN shift information help routing at all?
#   shift:  router gets the LEARNED shift signature (the thesis).
#   feat:   router gets features only (baseline).
#
# Kill-gate logic:
#   oracle <= feat  -> KILL (shift info doesn't help routing; thesis dead).
#   oracle > feat but shift <= feat -> estimator is the bottleneck (fix it).
#   shift > feat    -> thesis holds; proceed to GPU phase.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/workspace/venvs/shiftnet/bin/python"
export PYTHONPATH="$HERE"

for MODE in oracle shift feat; do
  OUT="$HERE/runs/pilot_$MODE"
  mkdir -p "$OUT"
  # idempotent: --resume continues if a daemon restart kills us
  # (this script itself runs detached via setsid+nohup)
  $PY "$HERE/train.py" \
    --out "$OUT" --epochs 8 --batch-size 128 --max-n 3000 \
    --width 0.75 --eval-every 2 --seed 0 --router-mode "$MODE" \
    --w-fam 1.0 --resume \
    > "$OUT/train_stdout.log" 2>&1
  echo "TRAIN DONE: $MODE" >&2
  $PY "$HERE/eval_matrix.py" --run-dir "$OUT" --width 0.75 \
    --router-mode "$MODE" --seed 0 \
    > "$OUT/matrix_stdout.log" 2>&1
  echo "MATRIX DONE: $MODE" >&2
done
echo "PILOT DONE" >&2
