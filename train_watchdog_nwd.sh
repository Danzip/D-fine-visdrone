#!/bin/bash
# Watchdog for NWD + size-adaptive loss training run.
# First run: --tuning best_stg2.pth (loads weights, resets optimizer for new loss landscape)
# Subsequent restarts: --resume last.pth
# Usage: bash train_watchdog_nwd.sh   (run in tmux)

set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

CONFIG="configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml"
TUNING_CKPT="output/dfine_hgnetv2_s_visdrone_mosaic_resume/best_stg2.pth"
RESUME_CKPT="output/dfine_hgnetv2_s_visdrone_nwd/last.pth"
LOG="output/nwd_watchdog.log"
RESTART_DELAY=15

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== NWD Watchdog started at $(date) ===" | tee -a "$LOG"

# First launch: tune from best_stg2 (reset optimizer for new loss)
if [ ! -f "$RESUME_CKPT" ]; then
    echo "--- First run: tuning from $TUNING_CKPT ---" | tee -a "$LOG"
    echo "--- Clearing stale GPU processes ---" | tee -a "$LOG"
    fuser -k /dev/nvidia* 2>/dev/null || true
    sleep 3

    set +e
    python train.py \
        -c "$CONFIG" \
        --tuning "$TUNING_CKPT" \
        "$@" \
        2>&1 | tee -a "$LOG"
    EXIT=$?
    set -e

    if [ $EXIT -eq 0 ]; then
        echo "=== Phase 1 complete at $(date). Comparing eval@1024 vs eval@1280 ===" | tee -a "$LOG"
        BEST_CKPT="output/dfine_hgnetv2_s_visdrone_nwd/best_stg2.pth"

        AP_1024=$(python train.py -c "$CONFIG" --resume "$BEST_CKPT" --test-only \
            -u "eval_spatial_size=[1024,1024]" 2>&1 \
            | grep "Average Precision.*all.*maxDets" | tail -1 | awk '{print $NF}')
        echo "  AP@1024: $AP_1024" | tee -a "$LOG"

        AP_1280=$(python train.py -c "$CONFIG" --resume "$BEST_CKPT" --test-only \
            -u "eval_spatial_size=[1280,1280]" 2>&1 \
            | grep "Average Precision.*all.*maxDets" | tail -1 | awk '{print $NF}')
        echo "  AP@1280: $AP_1280" | tee -a "$LOG"

        USE_SIZE=$(python3 -c "
ap1024 = float('${AP_1024}') if '${AP_1024}' else 0.0
ap1280 = float('${AP_1280}') if '${AP_1280}' else 0.0
winner = 1280 if ap1280 > ap1024 else 1024
print(winner)
")
        echo "  Winner: ${USE_SIZE}x${USE_SIZE} -- updating config for phase 2" | tee -a "$LOG"
        sed -i "s/eval_spatial_size: \[.*\]/eval_spatial_size: [${USE_SIZE}, ${USE_SIZE}]/" "$CONFIG"

        echo "=== Patching checkpoint for phase 2 (ep161-320) ===" | tee -a "$LOG"
        # Reset scheduler so phase 2 starts a fresh cosine from current LR (eta_min of phase 1)
        # as the new max, decaying to eta_min/10 over 160 epochs. No LR spike.
        python - <<'PYEOF'
import torch
path = "output/dfine_hgnetv2_s_visdrone_nwd/last.pth"
state = torch.load(path, map_location="cpu")
opt_sd = state["optimizer"]["state_dict"]
current_lrs = [pg["lr"] for pg in opt_sd["param_groups"]]
print(f"  LRs at end of phase 1: {[f'{lr:.2e}' for lr in current_lrs]}")
sched = state["lr_scheduler"]
sched["last_epoch"] = 0          # fresh cosine cycle
sched["base_lrs"] = current_lrs  # new max = where phase 1 ended
sched["_last_lr"] = current_lrs
sched["eta_min"] = 1e-7          # floor for phase 2 (10x lower)
torch.save(state, path)
print(f"  Checkpoint patched: new cosine max={current_lrs}, eta_min=1e-7")
PYEOF
        echo "--- Cleared stale GPU processes ---" | tee -a "$LOG"
    else
        echo "--- Crashed (exit $EXIT) at $(date). Restarting in ${RESTART_DELAY}s ---" | tee -a "$LOG"
        sleep $RESTART_DELAY
    fi
fi

# Subsequent launches: resume from last.pth
while true; do
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "Checkpoint not found: $RESUME_CKPT" | tee -a "$LOG"
        exit 1
    fi

    echo "--- Clearing stale GPU processes at $(date) ---" | tee -a "$LOG"
    fuser -k /dev/nvidia* 2>/dev/null || true
    sleep 3

    echo "--- Resuming at $(date) from $RESUME_CKPT ---" | tee -a "$LOG"

    set +e
    python train.py \
        -c "$CONFIG" \
        --resume "$RESUME_CKPT" \
        "$@" \
        2>&1 | tee -a "$LOG"
    EXIT=$?
    set -e

    if [ $EXIT -eq 0 ]; then
        echo "=== Training COMPLETE at $(date) ===" | tee -a "$LOG"
        break
    fi

    echo "--- Crashed (exit $EXIT) at $(date). Restarting in ${RESTART_DELAY}s ---" | tee -a "$LOG"
    sleep $RESTART_DELAY
done
