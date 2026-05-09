#!/bin/bash
# Watchdog for nwd_sal_linear experiment.
# First run: tunes from checkpoint (resets optimizer).
# Subsequent restarts: resumes from last.pth.
# Usage: bash experiments/nwd_sal_linear/watchdog.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="experiments/nwd_sal_linear/config.yml"
RESUME_CKPT="output/nwd_sal_linear/last.pth"
LOG="output/nwd_sal_linear/watchdog.log"
RESTART_DELAY=15

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p output/nwd_sal_linear
echo "=== Watchdog started at $(date) ===" | tee -a "$LOG"

while true; do
    if [ -f "$RESUME_CKPT" ]; then
        echo "--- Resuming from $RESUME_CKPT at $(date) ---" | tee -a "$LOG"
        set +e
        python train.py -c "$CONFIG" --resume "$RESUME_CKPT" --device cuda:0 2>&1 | tee -a "$LOG"
        EXIT=$?
        set -e
    else
        echo "--- First run: tuning from checkpoint at $(date) ---" | tee -a "$LOG"
        set +e
        python train.py -c "$CONFIG" --device cuda:0 2>&1 | tee -a "$LOG"
        EXIT=$?
        set -e
    fi

    if [ $EXIT -eq 0 ]; then
        echo "=== Training complete at $(date) ===" | tee -a "$LOG"
        break
    fi

    echo "--- Exit code $EXIT. Restarting in ${RESTART_DELAY}s at $(date) ---" | tee -a "$LOG"
    sleep $RESTART_DELAY
done
