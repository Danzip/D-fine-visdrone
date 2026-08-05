# Experiment: SAHI-style tiled training (train + eval both on 640x640, 50% overlap tiles)

## Status: PLANNED

## What this tests

Give SAHI a real chance: an earlier ablation ran SAHI-sliced *inference only*
against a model that was never trained on tiles, and it hurt accuracy
(28.0-28.3% vs 29.7% standard eval; see the root README's Inference-Time
Ablation table). The leading theory was that this is a train/inference
mismatch, not evidence tiling can't help. This experiment removes that
mismatch: train on the same overlapping 640x640 tiles the eval will use.

- **Train:** offline-tiled VisDrone train set (`tools/tiling/build_tiled_visdrone.py`,
  640x640 tiles, 50% overlap / 320px stride, boxes below 20% visible area dropped).
- **Eval:** `tools/tiling/tiled_eval.py` slices each *original* val image with
  the identical tile geometry, runs each tile through the model, and merges
  overlapping detections with class-wise NMS before scoring — the same
  procedure the model would see in a real tiled-inference deployment.

## Starting checkpoint

`weight/dfine_s_coco.pth` — fresh COCO-pretrained, not an already
VisDrone-adapted checkpoint. Deliberate choice: isolates whether tiling
itself is the lever, rather than measuring tiling on top of biases the
model already learned from full-image training. Means this run has to earn
its own VisDrone adaptation from scratch — see the config's header comment
for why warmup is long.

## Architecture

Same P2ConvHead + P2FusionLite + NWD-matcher architecture as `msfd_1024`
(this config `__include__`s it directly) — reuses the strongest available
recipe rather than a stripped-down baseline, since the goal is a real,
comparable result, not just a clean ablation.

## Results

| Run | AP50:95 | AP50 | AP-small | AP-medium | AP-large | Notes |
|-----|---------|------|----------|-----------|----------|-------|
| `e6_1280` std. whole-image eval | 0.344 | 0.549 | 0.257 | 0.453 | 0.626 | today's best, no tiling anywhere |
| `e6_1280` + tiled eval, **no tile training** | 0.146 | 0.363 | 0.108 | 0.215 | 0.242 | fresh control — same tile=640/overlap=0.5 settings this experiment will use. Confirms the train/inference mismatch is severe for this tiling config: -0.198 AP50:95 (-58% relative), AP-large hit hardest (fragmentation/duplication across tile boundaries) |
| this experiment, training not yet run | — | — | — | — | — | not started |

Full run log: `PROJECT_NOTES/13_sahi_tiled_training.md`.
