"""
SWA-style checkpoint averaging for D-FINE.

Averages the model weights from N checkpoints (the EMA module from each)
and saves a new checkpoint with the averaged weights.

The intuition: checkpoints near the end of training orbit a loss basin.
Averaging them finds the centre of that basin, which generalises better
than any single point.

Usage:
  python tools/inference/swa_avg.py \
    --checkpoints output/dfine_hgnetv2_s_visdrone_ms1280_cont/checkpoint0119.pth \
                  output/dfine_hgnetv2_s_visdrone_ms1280_cont/checkpoint0131.pth \
    --output output/dfine_hgnetv2_s_visdrone_ms1280_cont/swa_avg.pth

Then eval:
  python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_cont.yml \
    --test-only --resume output/dfine_hgnetv2_s_visdrone_ms1280_cont/swa_avg.pth \
    --device cuda:0
"""

import argparse
import copy
import torch


def average_checkpoints(paths):
    print(f"Averaging {len(paths)} checkpoints...")
    avg_state = None
    for i, path in enumerate(paths):
        print(f"  Loading {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Use EMA weights (better than raw model)
        if "ema" in ckpt:
            state = ckpt["ema"]["module"]
        else:
            state = ckpt["model"]

        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    n = len(paths)
    for k in avg_state:
        avg_state[k] /= n
        # Cast back to original dtype (usually float16 for some, float32 for most)
        avg_state[k] = avg_state[k].half() if avg_state[k].dtype == torch.float16 else avg_state[k]

    return avg_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    avg_state = average_checkpoints(args.checkpoints)

    # Build a minimal checkpoint: model = averaged weights, ema wraps the same
    out_ckpt = {
        "model": avg_state,
        "ema": {"module": avg_state, "updates": 0},
        "last_epoch": -1,
    }
    torch.save(out_ckpt, args.output)
    print(f"Saved averaged checkpoint to {args.output}")
