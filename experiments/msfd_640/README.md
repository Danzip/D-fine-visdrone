# Experiment: MSFD P2 fusion @ 640×640 + NWD matcher + SAL linear (1/area)

## Status: ABANDONED — broken on launch, never fixed/relaunched

## What this tests
Lightweight P2 feature fusion (MSFD-style) at 640×640, combined with our best
loss setup (NWD matcher + 1/area SAL). Both the architecture and the loss are
new relative to the mosaic baseline.

## Architecture change: MSFD-style P2 fusion
Our original P2 branch added P2 as a full 4th transformer level → 4× VRAM explosion
(P2 at 1024px = 256×256 = 65,536 positions vs P3+P4+P5 = 21,504 combined).

MSFD-style avoids this: P2 features are processed with lightweight depthwise
separable convs + SE attention, then fused DOWN into P3. Transformer still sees
[P3_enhanced, P4, P5] — same 3 levels, negligible VRAM overhead (~200K params).

## Loss
- **Matcher:** NWD cost (`cost_nwd: 2`, `nwd_constant: 0.5`)
- **Loss:** `size_adaptive: True`, `1/area` (linear). Same reasoning as nwd_sal_linear.

## Starting checkpoint
`output/dfine_hgnetv2_s_visdrone_nwd/best_dfine_s_visdrone_nwd_sqrt_v2.pth` — AP=0.321
Best weights available, already NWD-adapted. New MSFD module inits randomly;
all other weights loaded and protected by very low starting LR (1e-8).

## LR schedule: DualConvergeLR (two param groups converge over phase 1)
```
Phase 1 (epochs 0–50):
  Old weights (backbone + encoder + decoder):  1e-8  → 2e-5  (linear warmup, rising)
  New MSFD P2 weights:                         1e-4  → 2e-5  (cosine decay, falling)
  Both land at 2e-5 at epoch 50.

Phase 2 (epochs 50–110):
  All weights together:                        2e-5  → 1e-7  (cosine decay)
```
Old weights start near-frozen (1e-8) to protect loaded knowledge while new P2
layers (random init) train at high LR (1e-4). They converge and then decay together.

## Files
- `config.yml` — TBD
- `watchdog.sh` — TBD

## Results
| Epoch | AP50:95 | AP-small | Notes |
|-------|---------|----------|-------|
| 1-2 | 0.0 | - | dead on arrival — tuning from a 0.32 checkpoint should give ~0.31 at epoch 0. 3 live bugs found (OOM needing ~14GB, CUDA allocator crash, P2 fusion issue); see `PROJECT_NOTES/11_ablation_study_runpod.md` §W2. Never fixed/relaunched — `msfd_1024` (this dir's 1024px sibling) became the flagship P2 run instead |
