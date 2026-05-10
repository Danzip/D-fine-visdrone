#!/bin/bash
# Watchdog for ar_aware experiment.
# First run: tunes from checkpoint (resets optimizer).
# Subsequent restarts: resumes from last.pth.
# Usage: bash experiments/ar_aware/watchdog.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="experiments/ar_aware/config.yml"
RESUME_CKPT="output/ar_aware/last.pth"
LOG="output/ar_aware/watchdog.log"
RESTART_DELAY=15


mkdir -p output/ar_aware
echo "=== Watchdog started at $(date) ===" | tee -a "$LOG"

while true; do
    if [ -f "$RESUME_CKPT" ]; then
        echo "--- Resuming from $RESUME_CKPT at $(date) ---" | tee -a "$LOG"
        set +e
        python train.py -c "$CONFIG" --resume "$RESUME_CKPT" -u "tuning=~" --device cuda:0 2>&1 | tee -a "$LOG"
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
