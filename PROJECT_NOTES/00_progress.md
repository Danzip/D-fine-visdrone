# Project Progress Log

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Environment Setup + Repo Structure | ✅ COMPLETE |
| 2 | Baseline COCO Inference | ✅ COMPLETE |
| 3 | VisDrone Dataset Preparation | ✅ COMPLETE |
| 4 | Fine-tuning on VisDrone (local) | ✅ COMPLETE — 72 epochs, best AP50:95=0.231 (epoch 66) |
| 5 | WSL2 Setup + W&B fix | ✅ COMPLETE |
| 6 | AWS + EC2 Spot training | ⏳ BLOCKED — GPU quota increase pending AWS approval |
| 6b | SAHI inference experiment | ✅ COMPLETE — no improvement over baseline (see results below) |
| 7 | Structured Pruning (FFN neurons, group lasso) | ✅ COMPLETE — epoch 41 best, +recovery → AP=0.2320 (see 07_pruning.md) |
| 7b | Higher-resolution training (attempts 1–4) | ❌ ABANDONED — all failed; root cause: resolution collapse from single-scale priors |
| 10 | Multi-scale training at 1024px (ms1280) | ✅ COMPLETE — AP=0.255 after 80 epochs; +0.024 over baseline (see 09_multiscale_training.md) |
| 11 | Multi-scale continuation (ms1280_cont) | ✅ COMPLETE — AP=0.2966 at epoch 131; crashed ep132 (BUG-017, now fixed); see 09_multiscale_training.md |
| 8 | ONNX Export + Qualcomm AI Hub deployment | ✅ COMPLETE — 47ms/21FPS on Snapdragon 8 Gen 2, 100% NPU (see below) |
| 9 | Flutter app + inference server — model selector, box overlay | ✅ COMPLETE — working web app, both models selectable, boxes aligned |
| 12 | Copy-paste small object augmentation | ✅ COMPLETE — already in ms1280_cont config; active since training started |
| 14 | No-retraining ablations: eval@1280, SAHI, TTA, SWA | ✅ COMPLETE — all neutral/negative; baseline 0.2966 is still best (see 09_multiscale_training.md) |
| 15 | Combined retraining: mosaic + class-balance + multi-res | ✅ COMPLETE — **AP=0.3027 at epoch 15** (W&B l7dygiqx); crashed ep16 (BUG-018); best_stg1.pth kept |
| 16 | AR-aware rectangular training (Experiment C) | ❌ ABANDONED — pipeline built + tested; full training AP recovered only to 0.05 after 6 epochs from cont checkpoint (epoch 132+); too slow vs mosaic path; code left in repo but unused. See 12_ar_pipeline.md |
| 17 | Clean mosaic resume (batch_size=4, AMP=True) | ✅ COMPLETE — **best AP=0.316** (best_stg2.pth, epoch 92, maxDets=500); post-peak plateau confirmed; stopped 2026-05-02 |
| 18 | NWD + size-adaptive (sqrt) loss run | 🔄 IN PROGRESS — peak AP=0.3209 (ep109); 3 critical bugs fixed (BUG-036, BUG-037); restarted 2026-05-04 from best_stg2.pth |
| 19 | Cloud training setup (RunPod) | ⏳ PENDING — AWS quota denied × 2, switching to RunPod RTX A5000 ($0.27/hr, 24GB) |
| 20 | MSFD-style P2 fusion @ 640px + SAL(1/area) + NWD | ⏳ PLANNED — start from best_stg1.pth (AP=0.231); dual-convergence LR |
| 21 | MSFD-style P2 fusion @ 1024px + SAL(1/area) + NWD | ⏳ PLANNED — start from best NWD-sqrt ckpt (AP=0.321); dual-convergence LR |
| 13 | README + GitHub | ⏳ PENDING |

### Possible ablation (low priority, do after step 8–9 if time permits)
Re-run 640×640 fine-tuning with `use_amp: True` and `batch_size=8` to test whether
the larger effective batch improves max AP beyond the current 0.231.
`use_amp: True` is now set in `dfine_hgnetv2_s_visdrone.yml` — just change batch size.
Hypothesis: larger batch → more stable gradients → slightly higher AP ceiling.
Not high priority; the resolution gain from step 7b is likely more impactful.

---

## Step 1 — COMPLETE

**Date:** 2026-03-22

**What was done:**
- Confirmed GPU: NVIDIA RTX 4060 Laptop GPU, 8GB VRAM
- Driver: 555.97, CUDA 12.5 max support
- Created Python 3.12 venv at `D-FINE/venv/`
- Installed PyTorch 2.5.1+cu124 — CUDA confirmed working
- Installed all D-FINE requirements + wandb
- Cloned D-FINE repo, explored all source files
- Wrote `PROJECT_NOTES/01_repo_structure.md`

**Key facts discovered:**
- D-FINE uses `reg_max=32` bins per edge (128 bins total per box)
- GO-LSD is implemented as `loss_ddf` (Decoupled Distillation Focal loss) in `dfine_criterion.py`
- FDR is implemented via `weighting_function` + `distance2bbox` in `dfine_utils.py`
- Model is a 3-piece pipeline: HGNetV2 backbone → HybridEncoder → DFINETransformer
- Training entry point is `train.py` with `-c config -t checkpoint` for fine-tuning

**Next step:** Step 2 — download D-FINE-S COCO pretrained weights and run baseline inference.

---

## Step 4 — COMPLETE (2026-03-25/26)

**Full fine-tuning run: 72 epochs, batch_size=4, cosine LR**

### What changed vs previous runs
- Switched from `MultiStepLR` (milestone=500, never fired in 72 epochs) to `CosineAnnealingLR`
  (T_max=72, eta_min=1e-6) — LR now decays properly over the full run
- Used `--tuning` (weights only) not `--resume` (weights + stale optimizer state)
- batch_size=4 to avoid OOM on RTX 4060 8GB
- W&B fully enabled with per-step loss logging (every 10 steps) + images every 500 steps

### Results (best checkpoint: epoch 66)

| Metric | Score |
|--------|-------|
| **AP50:95** | **0.231** |
| AP50 | 0.389 |
| AP75 | 0.229 |
| AP-small | 0.142 |
| AP-medium | 0.339 |
| AP-large | 0.545 |

Previous best (epoch 9, flat LR): **0.170** → now **0.231** — **+36% improvement**

### SOTA context (standard eval, no slicing)
- DroneScan-YOLO (2025): AP50:95=0.356 — current SOTA, 10M params, 1280×1280 input
- Drone-DETR (2024): AP50:95=0.339 — 28.7M params
- DAU-YOLO (2025): AP50:95=~0.328 — 28.9M params
- UAV-DETR-R50 (2025): AP50:95=0.315 — ~50M params
- RT-DETR-R50 (2023): AP50:95=0.284 — direct predecessor architecture
- Our D-FINE-S at 640px: 0.231 — 65% of SOTA with a 10M param model
- Our D-FINE-S at 1280px (mosaic): **0.314 — 88% of SOTA, same 10M param budget as DroneScan-YOLO**

Gap vs DroneScan-YOLO: ~4.2 AP points. Their advantages: RPA-Block dynamic filtering, MSFD P2 branch, SAL-NWD hybrid loss, square 1280×1280.
More uncomfortable: Drone-DETR gets 0.339 at **640×640** (28.7M params) — architecture/loss differences, not resolution.

### Checkpoint location
`output/dfine_hgnetv2_s_visdrone/best_stg1.pth` (epoch 66)
W&B run: wandb.ai/danziv/D-FINE

---

## Step 6b — SAHI Inference Experiment (2026-03-26)

**SAHI** (Slicing Aided Hyper Inference) slices each image into overlapping 640×640 patches,
runs D-FINE on each patch, then merges with NMS. Designed to help with small objects by
keeping them large relative to the patch.

### Script
`tools/inference/sahi_inf.py` — supports single image and full val set AP eval.

### Results (slice_size=640, overlap=0.2)

| Metric | Baseline | SAHI | Delta |
|--------|----------|------|-------|
| **AP50:95** | **0.231** | **0.225** | -0.006 |
| AP50 | 0.389 | 0.404 | +0.015 |
| AP75 | 0.229 | 0.217 | -0.012 |
| AP-small | 0.142 | 0.153 | **+0.011** |
| AP-medium | 0.339 | 0.317 | -0.022 |
| AP-large | 0.545 | 0.471 | -0.074 |

### Why SAHI didn't help overall
- Small objects improved (+0.011) as expected — slicing keeps them large relative to patch
- Medium and large objects hurt significantly — slicing cuts them across patch boundaries,
  fragmenting detections and reducing localization precision
- AP75 dropped — stitched boxes from adjacent patches are less geometrically precise
- Net effect: SAHI at default settings is a wash (slightly negative overall)

### Why the better approach is higher training resolution
Training at 1280px on AWS (T4 has 16GB VRAM) will help ALL object sizes without the
patch-boundary fragmentation problem. Expected gain: +5–8 AP points → ~0.29–0.31.

---

## Step 7 — Structured Pruning — COMPLETE (2026-04-01)

See `07_pruning.md` for full results table.

**Decision: epoch 41** — Pareto-optimal checkpoint.
- FFN dims: [598, 780, 423] (41.4% of original 3072 neurons removed)
- AP after pruning: 0.2292 (baseline: 0.2308 — within noise)
- AP after 10 recovery epochs: **0.2320** (beat pre-pruning baseline!)
- `output/pruning_recovery/best_recovery.pth`

---

## Step 7b — Higher-Resolution Training — FAILED × 2, now 960×576 progressive — IN PROGRESS (2026-04-03)

**Motivation:**
VisDrone images are 4:3 (46.7%) and 16:9 (53.3%) — no square images exist.
Training at 640×640 distorts aspect ratio AND downscales to ~32% average pixel
preservation. Higher resolution preserves more pixels and no distortion.

**Resolution analysis (pixels preserved vs native):**
| Native res | % | 640×640 | 960×720 | 1280×736 |
|---|---|---|---|---|
| 1400×1050 (4:3) | 35.6% | 28% | 47% | 47% |
| 1400×788 (16:9) | 18.5% | 37% | 47% | 84% |
| 1360×765 (16:9) | 16.4% | 40% | 50% | 89% |
| 2000×1500 (4:3) | 11.0% | 14% | 23% | 23% |
| 1916×1078 (16:9) | 7.7% | 20% | 25% | 45% |
| **Weighted avg** | 100% | **32%** | **45%** | **60%** |

Why 1280×736 not 1280×720: 720/32=22.5 (not integer), causing FPN upsample mismatch.
736/32=23 ✓, 1280/32=40 ✓ — both backbone strides divide evenly.

**Bugs fixed to make non-square training work (all still apply):**
- BUG-008: `PadToSize` API broken in torchvision 0.20 (get_spatial_size removed)
- BUG-009: `profiler_utils.stats()` used square input for non-square model
- BUG-010: `stop_epoch=0` triggered D-FINE stage1→stage2 reload at epoch 0
- BUG-011: WandbViz used hardcoded 640×640 input size

---

### Attempt 1 — 1280×736 from best_stg1.pth — FAILED (2026-04-02/03)

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_1280.yml`
- Start: `output/dfine_hgnetv2_s_visdrone/best_stg1.pth` (AP=0.231)
- LR: global=1e-4, backbone=5e-5, cosine T_max=50, AMP=True, batch=4

**Result:** 43 epochs, AP flatlined at 0.112 → 0.125. Killed.

**Root cause:** The 640×640 checkpoint had FDR distributions specialized for 8,400 anchor
positions. Jumping to 19,320 positions (2.3×) forced the decoder to relearn the spatial
grid while fighting entrenched 640×640 priors. Loss decreased but AP never followed.

---

### Attempt 2 — 1280×736 from COCO pretrained — FAILED (2026-04-03)

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_1280_from_coco.yml`
- Start: `weight/dfine_s_coco.pth`
- LR: global=1e-4, backbone=5e-5, cosine T_max=50, AMP=True, batch=4

**Result:** 27 epochs, AP 0.026 → 0.117 then decelerating hard (+0.002/epoch). Killed.

**Root cause:** 6,471 training images is insufficient to learn 19,320 anchor positions
from COCO scratch in 50 epochs. Each anchor position sees too few training examples
per epoch — the grid is simply too large for this dataset size.

---

### Attempt 3 — Progressive resolution: 960×576 from best_stg1.pth — FAILED (2026-04-03)

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_960.yml`
- Start: best_stg1.pth (AP=0.231), LR: global=1e-4, backbone=5e-5, cosine, AMP, batch=4
- **Result: KILLED at epoch 3** — AP 0.0138 → 0.0143 → 0.0144 (flat). Same wall.

---

### Architecture Analysis — why resolution change is so hard for D-FINE (2026-04-03)

D-FINE uses **encoder-generated proposals** (not static learned queries). The query pipeline:
1. `HybridEncoder` processes backbone features into multi-scale memory
2. `DFINETransformer.enc_output` (Linear+LN) projects encoder memory → proposal features
3. `DFINETransformer.enc_score_head` (Linear) scores each position → top-k selected
4. Top-k positions become decoder queries + reference points

When resolution changes (640×640 → 960×576):
- Backbone: largely **resolution-invariant** (CNN local features) ✅
- Encoder CCFF: largely invariant (depthwise convolutions) ✅
- Encoder AIFI: attention over stride-32 tokens (400→540 tokens) — needs slight re-learning
- `enc_output` + `enc_score_head`: position-scoring learned for 640×640 statistics
- `cross_attn.sampling_offsets`: WHERE each decoder layer samples — resolution-specific
- FDR heads: stride-pixel units unchanged — mostly OK

Root cause of stagnation: when ALL params update simultaneously, the gradient signal
from wrong (confident) 640×640 proposals corrupts the backbone/encoder update, which
corrupts the proposal quality, creating a self-reinforcing destructive cycle.

---

### Attempt 4 — 960×576, decoder-only (freeze backbone+encoder) — FAILED (2026-04-03)

**Hypothesis:** Freeze backbone+encoder (resolution-invariant), train only decoder (4.4M params).
Decoder re-adapts proposal scoring and sampling offsets with stable frozen feature inputs.

**Code change:** `src/solver/_solver.py` — `freeze_except_decoder` flag sets
`requires_grad=False` for all `backbone.*` + `encoder.*` params after checkpoint load.

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_960_queries_only.yml`
- Start: best_stg1.pth, `freeze_except_decoder: True`, LR: 1e-3 decoder only, AMP, batch=4

**Result: KILLED at epoch 0** — AP below 0.014, worse than full training.

**Why it failed:** The encoder's AIFI self-attention was trained for 400 stride-32 tokens
(20×20 at 640×640). At 960×576 it's forced to process 540 tokens (30×18) with frozen
weights from a 400-token context — producing corrupted features for the 140 new positions.
The frozen encoder gave the decoder *worse* inputs than the unfrozen encoder, so AP dropped.

---

### Final verdict — higher-resolution training (2026-04-03)

**All 4 attempts failed. AP=0.231 at 640×640 is the ceiling for now.**

| Attempt | Config | Start | Result |
|---------|--------|-------|--------|
| 1 | 1280×736, all layers | best_stg1.pth | 0.112→0.125 (43 ep) |
| 2 | 1280×736, all layers | COCO | 0.026→0.117 (27 ep, decelerating) |
| 3 | 960×576, all layers | best_stg1.pth | 0.014→0.014 (3 ep, flat) |
| 4 | 960×576, decoder only | best_stg1.pth | <0.014 (0 ep, worse) |

**What to try in the future (if revisiting):**

1. **Two-stage warm-up:** Train encoder-only for 5 epochs (let AIFI adapt to new token count),
   then unfreeze decoder. The encoder adapts first with a stable loss signal from the frozen
   but wrong decoder; then once encoder features are calibrated, decoder can adapt.

2. **Curriculum resolution:** Start training at 640×640 (same as baseline) but gradually
   increase resolution over epochs using multi-scale collate — let the model continuously
   adapt rather than jumping discretely.

3. **Position embedding interpolation at init:** Before starting, explicitly interpolate the
   encoder's AIFI positional encodings from 400→540 tokens and the decoder's cross-attention
   sampling offsets to match the new anchor grid. Then train normally. Reduces the cold-start
   mismatch without requiring a warm-up phase.

4. **Larger dataset:** The core constraint is 6,471 training images. With 20k+ images,
   COCO→960×576 would likely converge in 50 epochs (the COCO→1280×736 run was improving,
   just too slow). VisDrone has a larger unlabeled set that could be pseudo-labeled.

**Relevant code:** `src/solver/_solver.py` has `freeze_except_decoder` implemented and ready.

---

## Step 8 — ONNX Export + Qualcomm AI Hub Deployment — COMPLETE (2026-04-03)

### Export

**Script:** `tools/deployment/export_onnx_pruned.py`
- Loads `ffn_dims=[598, 780, 423]` from checkpoint, resizes FFN layers, then exports
- Output: `output/pruning_recovery/best_recovery.onnx` (38 MB FP32)
- ONNX opset 16, simplified, checked

### Qualcomm AI Hub

**Script:** `tools/deployment/submit_aihub.py`
- Uploads ONNX, compiles FP32→INT8 targeting Snapdragon 8 Gen 2 (Samsung Galaxy S23)
- Compile flags: `--quantize_full_type int8 --quantize_io --truncate_64bit_io`
  (truncate needed because `orig_target_sizes` is int64, Qualcomm only supports int32)
- Compiled model ID: `mmr56re0n` (TFLite format, optimized for Hexagon v73 NPU)
- Profile job ID: `j563vzv65`

### Results on Samsung Galaxy S23 (Snapdragon 8 Gen 2)

| Metric | Value |
|--------|-------|
| Inference latency (median) | **47 ms** |
| Throughput | **~21 FPS** |
| NPU utilization | **100%** (1316/1317 ops on Hexagon NPU) |
| Inference peak memory | 7.1 MB |
| Warm load memory | ~352 MB |
| First load time (one-time) | ~99 ms |
| Model size (INT8) | ~10 MB |

**21 FPS real-time on Snapdragon 8 Gen 2 with full NPU acceleration.**

---

## Step 9 — Flutter App + Inference Server — COMPLETE (2026-04-04)

### Final architecture

**Server-side inference** (instead of on-device TFLite) — simpler, works on both web and Android.

- **Backend:** `dfine_app_server/server_v1.py` — FastAPI, port 8000
  - Loads both models at startup: ONNX Runtime (D-FINE) + Ultralytics (YOLOv8)
  - `POST /detect` — multipart form, fields: `file` (image), `model` (dfine|yolov8)
  - `GET /models` — returns available models
  - CORS enabled for all origins
- **Frontend:** `dfine_app/lib/main.dart` — Flutter web, served on port 8080
  - Model selector dropdown (YOLOv8-X vs D-FINE-S)
  - Gallery + Camera picker
  - Bounding box overlay with per-class colours, labels, scores
  - Settings gear → change server URL (for Android use)

### Model comparison

| Model | Params | AP50 | AP50:95 | Notes |
|-------|--------|------|---------|-------|
| D-FINE-S (ours, pruned) | 10M | 0.389 | 0.232 | INT8 on Snapdragon: 47ms/21FPS |
| YOLOv8-X (mshamrai HuggingFace) | 68M | 0.470 | N/A | 7× larger, stronger on diverse images |

On VisDrone-style aerial images, both models find all objects with some artifacts.
Gap is expected — model size difference. Next step is to close this gap.

### Files

| File | Purpose |
|------|---------|
| `dfine_app_server/server_v1.py` | **Active server** — both models, clean pipeline |
| `dfine_app_server/server_yolo_original.py` | YOLOv8-only backup |
| `dfine_app_server/models/best.pt` | YOLOv8-X VisDrone weights (mshamrai HuggingFace) |
| `dfine_app/lib/main.dart` | Flutter web UI |
| `dfine_app_server/sota_compare.py` | CLI script — runs both models, saves `_yolo.jpg` + `_dfine.jpg` |

### Bugs encountered and fixed (2026-04-04)

**BUG-012: Server preprocessing — naive squish instead of letterbox**
- Original `preprocess()` did `image.resize((640,640))` — squished aspect ratio
- Model was trained with letterbox (scale longest side, pad short side)
- Fix: replaced with letterbox resize + centre-pad, then undo padding in box coords

**BUG-013: YOLOv8 BGR/PIL pipeline mismatch**
- When PIL image was passed to YOLO in the server, adding BGR conversion broke detection
- Attempted fixes (cv2.cvtColor, temp file path) all made things worse
- Root cause: the working pipeline (`sota_compare.py`) passes a file **path** directly to YOLO;
  YOLO reads it with its own pipeline internally
- Fix: reverted to passing PIL image directly — YOLO handles PIL correctly without conversion

**BUG-014: Flutter `imageQuality` / `Image.memory` on web**
- Removing `imageQuality: 90` caused image picker to return raw PNG bytes
- `Image.memory` on Flutter web release build fails silently with raw PNG → black screen
- Fix: restored `imageQuality: 100` — browser re-encodes to JPEG via canvas, `Image.memory` works

**BUG-015: Bounding box coordinate translation**
- Boxes were drawn relative to the full widget size, but `BoxFit.contain` letterboxes the image
- Result: boxes were offset/stretched away from the actual objects
- Fix: painter now computes `scale = min(widgetW/imgW, widgetH/imgH)` and adds
  `offsetX = (widgetW - imgW*scale)/2`, `offsetY = (widgetH - imgH*scale)/2`
  to match BoxFit.contain exactly

**BUG-016: Flutter service worker aggressive caching**
- Normal browser refresh loads old cached build — appears as if changes have no effect
- Fix: always use `Ctrl+Shift+R` (hard refresh) after every `flutter build web`

---

## Step 18 — NWD + Size-Adaptive Loss Run — IN PROGRESS (2026-05-02)

**Goal:** Push AP above 0.316 baseline by making matching and loss more sensitive to small objects.

### Changes vs Step 17

**1. NWD in Hungarian matching cost (`matcher.py`)**
GIoU cost in the original matcher is not size-relative: a 4px displacement on a 4px box
costs the same as on a 40px box. NWD models boxes as 2D Gaussians and measures distance
in (cx, cy, w/2, h/2) space — cost is automatically proportional to box size.
`cost_nwd: 2` added alongside existing cost terms.

**2. Size-adaptive loss weighting — sqrt(1/area) (`dfine_criterion.py`)**
After Hungarian matching, each matched pair's L1, GIoU, and FGL losses are multiplied by
`1 / sqrt(w*h + ε)`, normalized to mean=1. 77% of VisDrone boxes are <16px.
Using `1/area` (linear) was tried first but caused AP regression (0.316 → 0.308 over 20 epochs):
the linear weighting amplified label noise on tiny boxes by 625× vs large boxes.
Switched to `sqrt(1/area)` — gentler scaling, keeps relative emphasis without blowing up noise.

**3. maxDets=500 in COCO eval (`det_engine.py`)**
VisDrone images can have 100-800 objects; default maxDets=100 silently caps scoring.
AP difference is small for D-FINE (no NMS to recover from) but semantically correct.
- maxDets=100: AP=0.3160, AP-small=0.2252
- maxDets=500: AP=0.3162, AP-small=0.2262  ← small but real improvement

**4. num_top_queries=500 in config**

### First attempt (1/area) — ABANDONED after 20 epochs
- Epoch 0 (loaded weights): AP=0.3176 (≈ same as baseline — good sign)
- Epoch 20: AP=0.308, AP-small ≈ 0.217 — REGRESSED
- Root cause: 1/area too aggressive; amplified label-noise gradients on tiny objects

### Current run: sqrt(1/area) from best_stg2.pth
- Started: 2026-05-02 — fresh --tuning from best_stg2.pth (epoch 92, AP=0.316)
- Config: `configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml`
- Watchdog: `train_watchdog_nwd.sh`
- Monitoring: W&B project `dfine-visdrone`, experiment `dfine-nwd-sqrt`
- **Peak:** AP=0.3209 at ep109 (best_stg2.pth, LR~9.5e-6)

### Bugs fixed and restart (2026-05-04)

Three V-shaped dips in the W&B curve traced to two D-FINE bugs:

**BUG-036 — wrong rollback target:**
D-FINE's non-improvement handler loaded `best_stg1.pth` (pre-ep80 weights, AP~0.314)
any time stage-2 eval didn't beat the best. For our 160-ep stage 2 this caused 30+
epochs of regression. Fixed: load `best_stg2.pth` instead; load EMA weights (not live
model weights) into both model copies; do NOT reset optimizer or scheduler.

**BUG-037 — last.pth not saved after stop_epoch:**
`last.pth` was only saved for `epoch < stop_epoch=80`. Every post-ep80 crash caused a
full restart from ep79, redoing all of stage 2 from scratch. The stop_epoch=80 EMA
reset fired 4 separate times as a result. Fixed: `last.pth` now saved every epoch.

**Restarted 2026-05-04** from best_stg2.pth (AP=0.3209, sched last_epoch=95, LR~9.5e-6).
Training continues cosine descent toward eta_min=1e-7. Phase 2 (ep161-320) auto-triggers
via watchdog after phase 1 completes, with eval@1024 vs eval@1280 comparison at boundary.
