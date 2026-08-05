# Experiment: msfd_1024 polish2 — ultra-low-LR polish, augs off (continued from `msfd_1024_polish`)

## Status: DONE — was the standing best before `e6_1280` superseded it

## What this tests

Continuation of `msfd_1024_polish`: 50 epochs, augmentations off, to let the
P2ConvHead/P2FusionLite/NWD architecture fully converge at 1024px before
trying anything else on top of it. Also found+fixed BUG-045 (destructive
`stop_epoch=0` reload) during this run.

## Starting checkpoint

`output/runpod_results/msfd_1024_best_ep109.pth` (AP=0.3219), via
`msfd_1024_polish`.

## Results

| Epoch | AP50:95 | AP-small | Notes |
|-------|---------|----------|-------|
| 44 (peak) | 0.3226 | 0.2344 | `output/runpod_results/polish2_last.pth` — flat plateau ep22-49, genuine convergence, not a lucky spike |

Was the standing best (2026-07-04 → 2026-07-06) until `e6_1280` (1280px
resolution unlock) reached 0.344. Still the reference point for "best at
1024px, single-scale-converged" if resolution is ever held fixed for a
future comparison.
