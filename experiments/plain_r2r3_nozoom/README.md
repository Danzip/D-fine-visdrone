# Experiment: R2+R3 only (no R1 crop-zoom) — isolation control for `plain_r1r2r3`

## Status: DONE — did not beat standing best; isolates R1's effect as ~neutral

## What this tests

Identical to `plain_r1r2r3` minus R1 (crop-zoom reverted to standard
training) — run in parallel specifically to isolate whether R1 (train-time
SAHI-style tiling) helps or hurts independently of R2 (NWD regression loss)
+ R3 (rare-class CopyPaste 2.0).

## Starting checkpoint

Same lineage as `plain_r1r2r3` (plain, non-P2 architecture).

## Results

| Run | AP50:95 | vs standing best (0.3226) | Notes |
|-----|---------|---------------------------|-------|
| `plain_r2r3_nozoom` (R2+R3, no R1) | 0.3188 | -0.0038 | `output/runpod_results/plain_r2r3_nozoom_last.pth` |

**R1 isolation:** `nozoom` (0.3188) vs `r1r2r3` (0.3183) = **R1's standalone
effect ≈ -0.0005 AP** — essentially neutral, not the hoped-for +1-2 AP.
R1's premise (crop-zoom gives objects more effective pixels during training)
wasn't validated by this run. Neither bundle beat the standing best, so R2+R3
together are the larger drag here, not R1.

An idle-billing incident happened during this run window — see BUG-046.
