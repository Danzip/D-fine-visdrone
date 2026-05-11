#!/bin/bash
# Watchdog for ar_aware phase 2 — flat 1e-7 extension for 200 epochs.
# Always resumes from last.pth (no tuning phase).
# Usage: nohup bash experiments/ar_aware/watchdog_p2.sh &

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="experiments/ar_aware/config_p2.yml"
RESUME_CKPT="output/ar_aware_p2/last.pth"
INITIAL_CKPT="output/ar_aware/last.pth"   # replace with pod's epoch-110 checkpoint
LOG="output/ar_aware_p2/watchdog.log"
RESTART_DELAY=15

mkdir -p output/ar_aware_p2
echo "=== Watchdog started at $(date) ===" | tee -a "$LOG"

while true; do
    if [ -f "$RESUME_CKPT" ]; then
        CKPT="$RESUME_CKPT"
    else
        CKPT="$INITIAL_CKPT"
    fi

    echo "--- Resuming from $CKPT at $(date) ---" | tee -a "$LOG"
    set +e
    python train.py -c "$CONFIG" --resume "$CKPT" --device cuda:0 2>&1 | tee -a "$LOG"
    EXIT=$?
    set -e

    if [ $EXIT -eq 0 ]; then
        echo "=== Training complete at $(date) ===" | tee -a "$LOG"
        break
    fi

    echo "--- Exit code $EXIT. Restarting in ${RESTART_DELAY}s at $(date) ---" | tee -a "$LOG"
    sleep $RESTART_DELAY
done
