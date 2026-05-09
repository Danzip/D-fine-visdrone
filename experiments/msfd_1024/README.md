# Experiment: MSFD P2 fusion @ 1024×1024 + NWD matcher + SAL linear (1/area)

## Status: PLANNED

## What this tests
Same as msfd_p2_640 but at 1024×1024 — only practical on RunPod (24GB VRAM).
Starting from our best checkpoint (AP=0.321) which is already NWD-adapted.
Goal: push past 0.321 with the P2 small-object branch.

## Architecture change: MSFD-style P2 fusion
Identical to msfd_p2_640. Depthwise separable convs + SE on P2, fused into P3.
Transformer sees [P3_enhanced, P4, P5]. No extra transformer levels.

## Loss
- **Matcher:** NWD cost (`cost_nwd: 2`, `nwd_constant: 0.5`)
- **Loss:** `size_adaptive: True`, `1/area` (linear).

## Starting checkpoint
`output/dfine_hgnetv2_s_visdrone_nwd/best_dfine_s_visdrone_nwd_sqrt_v2.pth` — AP=0.321
Already NWD-adapted. New MSFD P2 layers init randomly; all other weights loaded.

## LR schedule: DualConvergeLR (two param groups converge over phase 1)
```
Phase 1 (epochs 0–50):
  Old weights (backbone + encoder + decoder):  1e-8  → 2e-5  (linear warmup, rising)
  New MSFD P2 weights:                         1e-4  → 2e-5  (cosine decay, falling)
  Both land at 2e-5 at epoch 50.

Phase 2 (epochs 50–110):
  All weights together:                        2e-5  → 1e-7  (cosine decay)
```

## Files
- `config.yml` — TBD
- `watchdog.sh` — TBD

## Results
| Epoch | AP50:95 | AP-small | Notes |
|-------|---------|----------|-------|
| — | — | — | not started |
