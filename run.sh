#!/usr/bin/env bash
# =====================================================================
# One entry point for the whole project. Edit the variables, then:
#     bash run.sh
# (For long runs, launch it inside the 'simulations' tmux session.)
# =====================================================================
set -e
cd "$(dirname "$0")"

PY=../.venv/bin/python      # venv interpreter (torch, pennylane, ...)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8   # avoid thread oversubscription
ENCODER=cnn                 # cnn | gru
SMOKE=1                     # 1 = quick smoke tests first, 0 = skip
RUN_SWEEP=1                 # run_experiment.py      : hybrid vs classical across N
RUN_QUANT=1                 # quantize_eval.py quant : rate-distortion vs bits
RUN_NOISE=1                 # quantize_eval.py noise : robustness vs channel noise
RUN_PARAMEFF=0             # quantize_eval.py parameff (amplitude decoder underfits)

run () { echo; echo ">>> $*"; "$@"; }

if [ "$SMOKE" = 1 ]; then
  run $PY run_experiment.py --encoder $ENCODER --quick
  run $PY quantize_eval.py quant --encoder $ENCODER --quick
  run $PY quantize_eval.py noise --encoder $ENCODER --quick
fi

[ "$RUN_SWEEP"    = 1 ] && run $PY run_experiment.py            --encoder $ENCODER
[ "$RUN_QUANT"    = 1 ] && run $PY quantize_eval.py quant       --encoder $ENCODER
[ "$RUN_NOISE"    = 1 ] && run $PY quantize_eval.py noise       --encoder $ENCODER
[ "$RUN_PARAMEFF" = 1 ] && run $PY quantize_eval.py parameff    --encoder $ENCODER

echo; echo "ALL DONE."
