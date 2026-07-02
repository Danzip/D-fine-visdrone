#!/bin/bash
# Watchdog for msfd_640 experiment.
# First run: tunes from checkpoint (resets optimizer).
# Subsequent restarts: resumes from last.pth.
# Usage: bash experiments/msfd_640/watchdog.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONUNBUFFERED=1

CONFIG="experiments/msfd_640/config.yml"
TUNING_CKPT="output/dfine_hgnetv2_s_visdrone_nwd/best_stg1_dfine_s_visdrone_nwd_sqrt_v2.pth"
RESUME_CKPT="output/msfd_640/last.pth"
LOG="output/msfd_640/watchdog.log"
RESTART_DELAY=15

mkdir -p output/msfd_640
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
        python train.py -c "$CONFIG" -t "$TUNING_CKPT" --device cuda:0 2>&1 | tee -a "$LOG"
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
