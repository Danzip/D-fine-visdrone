# Experiment: AR-Aware Rectangular Training

## Status: DONE (RunPod) — did not beat baseline

## What this tests
Our original contribution. VisDrone images are 4:3 (46.7%) and 16:9 (53.3%).
Training at square resolution squashes aspect ratio and wastes pixels.
ARBucketSampler groups images by AR into buckets; each batch uses a rectangular
canvas matching the bucket's native shape. No distortion, more preserved pixels.

No SOTA paper (DroneScan-YOLO, Drone-DETR, DAU-YOLO) does this.

## Loss
- **Matcher:** NWD cost (`cost_nwd: 2`, `nwd_constant: 0.5`)
- **Loss:** `size_adaptive: True`, `1/area` (linear) — same as nwd_sal_linear.

## Previous failure — do not repeat
Started from ms1280_cont checkpoint (ep132+, multi-scale mosaic 1024px).
AP recovered only to 0.05 after 6 epochs. Root cause: checkpoint was too
specialized to mosaic+square; AR rectangular batching was too large a shift.

## Starting checkpoint
`output/dfine_hgnetv2_s_visdrone_nwd/best_dfine_s_visdrone_nwd_sqrt_v2.pth` — AP=0.321
Best checkpoint available. Already NWD-adapted. Low starting LR (1e-8) protects
loaded weights from the AR rectangular batching distribution shift.

If AP regresses past epoch 20 → abort and retry from COCO pretrained (`weight/dfine_s_coco.pth`)
as a clean slate (slower, ~160 epochs to converge, but no conflicting priors).

## LR schedule: WarmupCosineHoldLR (single param group — no new layers)
```
Epoch   0:  1e-8   (start — protect loaded weights from distribution shift)
Epoch  50:  2e-5   (peak after linear warmup)
Epoch 110:  1e-7   (cosine decay complete)
Epoch 110+: 1e-7   (hold)
```
Backbone LR = 0.5× global LR throughout.

Note on best_stg1.pth (AP=0.231): kept for reference only, do not use as
starting point. Too low AP to be competitive; the 640px simple training history
is not better than stg2's stronger VisDrone adaptation for this experiment.

## Reference code
- `src/data/` — ARBucketSampler (already implemented)
- `configs/dfine/dfine_hgnetv2_s_visdrone_ar.yml` — previous config (needs update)

## Files
- `config.yml` — TBD
- `watchdog.sh` — TBD

## Results
| Epoch | AP50:95 | AP-small | Notes |
|-------|---------|----------|-------|
| 0 | 0.2625 | - | canvas change (1024² multi-scale → 736×1280) cost ~6 AP instantly |
| 10 | 0.307 | - | recovering |
| 61 | 0.3158 | - | |
| 110 (final) | 0.318 | +0.003 vs baseline | never beat the 0.321 starting checkpoint — 0.3 AP below baseline after 110 epochs; see `PROJECT_NOTES/11_ablation_study_runpod.md` §W-AR |
