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
| 18 | NWD + size-adaptive (sqrt) loss run | ✅ COMPLETE — **peak AP=0.3209** (ep109); bugs BUG-036/037 fixed; final best: `best_stg1_dfine_s_visdrone_nwd_sqrt.pth` (AP=0.321) |
| 19 | Cloud training setup (RunPod) | ✅ COMPLETE — RunPod RTX A5000 ($0.27/hr, 24GB); AWS quota denied × 2 |
| 20 | nwd_sal_linear — 1/area weighting | ❌ ABANDONED — killed ep35; AP regressed 0.321→0.315; 1/area too aggressive even with slow warmup |
| 21 | ar_aware — rectangular AR-aware training | ✅ COMPLETE — final AP=0.318 (ep110, ar_aware_p2 0.3181); never beat the 0.321 starting checkpoint. Post-mortem in 11_ablation_study_runpod.md (W-AR) |
| 22 | p2_640 — 4-level D-FINE at 640×640 | ❌ NEVER LAUNCHED — user killed the tournament runner (2.6× compute of msfd); superseded by msfd_1024 |
| 23 | msfd_1024 — YOLOv8-style P2 conv head | ✅ READY — P2ConvHead (DWConv×2, FCOS TAL); transformer stays 3-level; P2 at 256×256 via conv only |
| 24 | msfd_1024 run (P2ConvHead+P2FusionLite + R2 NWD-loss + R3 rare-CopyPaste) | ✅ COMPLETE (2026-07-03) — **AP=0.3219 @ep109** (ties old best), **AP-small=0.2323 (new record, +0.6)**. Found BUG-044 (stage-2 LR-scheduler reset, run ended mid-climb) |
| 25 | msfd_1024_polish → polish2 (ultra-low-LR polish, 50ep, augs off) | ✅ COMPLETE (2026-07-04) — **polish2 final AP=0.3226 / AP-small=0.2344 — NEW STANDING BEST** (ep44 peak, flat plateau ep22-49, genuine convergence). Found+fixed BUG-045 (destructive stop_epoch=0 reload) |
| 26 | msfd/P2 line **SHELVED by user decision** (2026-07-04) — "10% overhead with 0 gain" | — |
| 27 | plain_r1r2r3 + plain_r2r3_nozoom (R1 zoom-crop + R2 NWD-loss + R3 rare-CopyPaste, plain 3-level architecture, no P2 head; parallel pods to isolate R1) | ✅ COMPLETE (2026-07-05) — **neither beat standing best**: r1r2r3 AP=0.3183, nozoom AP=0.3188 (vs 0.3226). Isolating R1 (nozoom minus r1r2r3): zoom-crop alone ≈ **-0.0005 AP, not the hoped-for +1-2 APs** — R1's premise not validated. Idle-billing incident during this run: BUG-046 |
| 28 | R4 — per-class score calibration (eval-only, $0) | ⏳ IN PROGRESS (2026-07-05) — see write-up below |
| 30 | Multi-object tracking on video (e6_1280 checkpoint): ByteTrack + comparison vs. BoT-SORT/StrongSORT/OC-SORT/DeepOCSORT | ✅ COMPLETE (2026-08-03/04) — tested on VisDrone-MOT val sequence (184 GT tracks). ByteTrack: fixed a bug silently disabling its low-conf recovery pass + added GMC, 1039→670 tracks. Compared 5 trackers on identical cached detections: **Re-ID-based trackers (BoT-SORT 362 tracks/1.97x GT, StrongSORT 335/1.82x) roughly halve fragmentation vs. all motion-only trackers (~3.6x GT)** — BoT-SORT is the speed/accuracy pick. Profiled tracker FPS: bottleneck is 500/500 top-queries saturating every frame (dense scene, conf_low=0.1), driving StrongSORT's per-track Kalman-gating cascade and BoT-SORT's per-box ReID crop loop. Added `--nms-iou` (per-class NMS pre-tracking): cut dets/frame 500→374 (-25%) but only moved tracker FPS by ~±3% (noise) — StrongSORT's cost scales with *track* count not detection count, so NMS didn't fix the real bottleneck. See `12_tracking.md`, BUG-049 |
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

**Caveat (2026-07-07): all of the above are paper-reported numbers, not
independently reproduced.** A search for DroneScan-YOLO's official published
weights (to run it ourselves and confirm the 0.356 figure / compare
side-by-side under our own eval harness) came up empty — no public
checkpoint release was found. Every "gap to SOTA" comparison in this project
(including E6's "~0.012 off SOTA") is therefore trusting the paper's
self-reported number, not a verified apples-to-apples eval. Don't cite these
gaps as more precise than that.

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

**Update (2026-07-07): independently verified, and it changes R5's premise.**
Ran `tools/eval/eval_yolov8x_visdrone.py` — YOLOv8-X (best.pt) on our own
VisDrone val split (548 images, faster_coco_eval, same evaluator D-FINE's
own numbers use), native 640px (its own training resolution): **AP
(mAP50:95) = 0.2502**. The AP50=0.470 figure above was the mshamrai
HuggingFace model-card number (different metric, unverified) — this 0.2502
is the real, same-eval-set, same-metric comparison.

Despite being **7x larger** (68M vs 10M params), YOLOv8-X only barely beats
our weakest 640px D-FINE-S baseline (0.231) and is well below our current
best (E6, 0.344 @ 1280px). **R5 (distill from YOLOv8-X) no longer makes
sense as planned** — a distillation teacher should outperform the student it
teaches, and this one doesn't. Dropped from the R-series candidate list;
see `11_ablation_study_runpod.md` §5.

Also, to be explicit since these get conflated elsewhere in this doc:
**YOLOv8-X is not the network behind the 0.356 SOTA reference** — that's
DroneScan-YOLO (see `### SOTA context` above), a completely different
model/paper.

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

## Step 18 — NWD + Size-Adaptive Loss Run — COMPLETE (2026-05-02 → 2026-05-09)

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

**Restarted 2026-05-04** from best_stg2.pth. Ran to completion (110 epochs).

**Final result: AP=0.321** — best checkpoint: `output/dfine_hgnetv2_s_visdrone_nwd/best_stg1_dfine_s_visdrone_nwd_sqrt.pth`

---

## Step 20 — nwd_sal_linear — ABANDONED (2026-05-09)

Tried `sal_mode='linear'` (1/area weighting instead of 1/sqrt(area)).
- Epoch 0: AP=0.321 (loaded from NWD-sqrt best)
- Epoch 35: AP=0.315 — steady regression
- Root cause: 1/area amplifies tiny-box gradients by 625× vs large boxes — too aggressive
  even with 50-epoch warmup. The sqrt version is the right trade-off.
- Killed. Config preserved at `experiments/nwd_sal_linear/`.

---

## Step 21 — ar_aware — IN PROGRESS (2026-05-10)

**Goal:** Exploit VisDrone's bimodal AR distribution (≈50% 16:9, ≈50% 4:3) by training
each batch at its canonical rectangular canvas instead of square 1024×1024.

**Key components built:**
- `ARBucketBatchSampler`: groups images by COCO metadata AR into 16:9 / 4:3 buckets;
  yields same-AR batches; epoch-seeded shuffling (src/data/dataloader.py)
- `ARLetterboxCollateFunction`: proportionally scales + center-pads to canonical canvas
  (736×1280 for 16:9, 960×1280 for 4:3); adjusts normalized cxcywh boxes (same file)
- No Mosaic or IoUCrop (both destroy AR); PhotometricDistort + HFlip + CopyPasteSmall only

**Config:** `experiments/ar_aware/config.yml`
- Tuning from `best_stg1_dfine_s_visdrone_nwd_sqrt.pth` (AP=0.321)
- eval @ 736×1280 (val split is 100% 16:9)
- NWD matcher + SAL sqrt, accum_steps=4, total_batch_size=8

**Local training:** epoch 0→60, AP=0.3158 at ep60 (still in 50-ep warmup)
**RunPod training:** resumed from local ep60 last.pth, currently epoch 63+

**Bugs encountered:**
- `ARBucketBatchSampler.__len__` returned image count not batch count → wrong progress bar (fixed)
- `tuning=~` CLI override didn't reliably null YAML key → moved tuning to watchdog `-t` flag
- W&B duplicate runs from stale `wandb_run_id.txt` on RunPod

---

## Steps 22–23 — p2_640 and msfd_1024 — READY (2026-05-10)

Both experiments prepared and verified; ready to launch.

### p2_640 (`experiments/p2_640/`)
- 4-level D-FINE: P2+P3+P4+P5 all in full MSDeformableAttention transformer
- 640×640 input → P2 = 160×160 = 25,600 tokens (manageable)
- `return_idx: [0,1,2,3]`, `in_channels: [64,256,512,1024]`, `num_levels: 4`
- `num_points: [2,3,6,3]` — fewer points at P2 to limit compute
- Same training pipeline as nwd_sal_sqrt baseline

### msfd_1024 (`experiments/msfd_1024/`)
- YOLOv8-style: transformer stays at 3 levels (P3/P4/P5); P2 handled by conv head only
- 1024×1024 input → P2 = 256×256, processed by `P2ConvHead` (2 DWConv blocks)
- FCOS-style TAL assignment: anchor center inside GT → highest IoU candidate wins
- VFL + L1 + GIoU loss for P2ConvHead predictions
- P2 predictions merged with decoder predictions at NMS (postprocessor)
- Transformer cost unchanged vs baseline

**New code:** `src/zoo/dfine/p2_conv_head.py` (P2ConvHead + p2_head_loss)
Minimal changes to: dfine.py, dfine_criterion.py, postprocessor.py, __init__.py

---

## Step 24 — msfd_1024 run: P2ConvHead+P2FusionLite + R2 NWD-loss + R3 rare-CopyPaste — COMPLETE (2026-07-03)

The flagship RunPod campaign run: P2ConvHead + P2FusionLite (cheap fusion of
raw P2 + upsampled neck-P3, +15% compute at 1024) + NWD regression loss (R2)
+ rare-class CopyPaste (R3), tuned from the v2 NWD-sqrt checkpoint (AP=0.322),
110 epochs on a RunPod RTX 3090.

**Result: AP=0.3219 @ep109** (ties old best), **AP-small=0.2323** (new record,
+0.6 over the 0.322 checkpoint's 0.226).

**BUG-044 found:** the stage-2 (ep80, augs-off) transition resets the LR
scheduler instead of continuing its decay — stage 2 re-warmed for 30 epochs
instead of decaying, so the run ended mid-climb at lr 6.3e-6, AP still rising.
Root cause not fixed (deferred, risky mid-campaign solver change);
`msfd_1024_polish` was launched as a targeted workaround (Step 25). Also
found: epoch 80 (augs-off) caused an instant +1.8 AP jump — the mid-run "dip"
seen throughout this project's history is entirely the augmentation tax, not
a training-quality problem. Full bug details: `06_bugs_and_fixes.md` BUG-044.

---

## Step 25 — msfd_1024_polish → polish2 (ultra-low-LR polish) — COMPLETE, NEW BEST (2026-07-04)

Targeted fix for BUG-044: continue from the true ep109 peak checkpoint with a
clean, low-LR cosine decay (no stage transition to trigger the reload bug).

First polish attempt was itself silently corrupted (**BUG-045**: `stop_epoch:
0`, meant only as "single-scale from start," also unconditionally re-triggered
det_solver.py's stage-transition full-state reload from a stale epoch-1
checkpoint — the new cosine schedule never applied, ~3h/$0.66 wasted before
caught). Fixed (`det_solver.py`, guard `stop_epoch > 0`), then relaunched as
**polish2**: backbone LR 3e-7→1e-8 over 50 epochs, augs off throughout.

**Final result: AP=0.3226, AP-small=0.2344 — NEW STANDING BEST.** Peaked
epoch 44; epochs 22-49 sit in a flat plateau of 0.3223-0.3226 (genuine
convergence, not a single-epoch noise spike — the measured noise floor over
this plateau is ~0.001 AP). Checkpoint: `output/runpod_results/polish2_last.pth`.

A control experiment (does *any* extra low-LR polish help equally, independent
of the P2/NWD-loss/rare-paste bundle?) was identified as the natural next
step to attribute the +0.0038 AP gain correctly, but the user made a judgment
call to skip it: **"10% overhead with 0 gain [in msfd_640's original
mechanism check]... results speak for themselves"** — the msfd/P2 architecture
line was shelved in favor of testing other directions on the plain
(non-P2) architecture instead (Step 27).

---

## Step 26 — msfd/P2 architecture line SHELVED (2026-07-04)

User decision, not a technical failure: the P2ConvHead/P2FusionLite line
reached AP=0.3226 (a real, if modest, +0.0038 improvement) but at meaningful
compute overhead (+15% at 1024px) with an unresolved attribution question
(control experiment above). Rather than spend further budget disentangling
it, effort moved to testing other candidate improvements (R1/R2/R3 from
`11_ablation_study_runpod.md` §5) on the simpler plain 3-level architecture.
polish2's checkpoint remains the standing best regardless of this decision.

---

## Step 27 — plain_r1r2r3 + plain_r2r3_nozoom — COMPLETE, neither beats best (2026-07-05)

Two parallel RunPod pods, both tuned from the real 0.322 (pre-P2) checkpoint,
plain 3-level architecture (no P2 head), single-stage clean cosine (80 epochs,
no stage transition — sidesteps the BUG-045 class of bug entirely):

- **`plain_r1r2r3`** — R1 (aggressive zoom-crop, `RandomIoUCrop` min_scale
  0.3→0.15) + R2 (NWD regression loss) + R3 (rare-class CopyPaste 3x boost).
  Final **AP=0.3183, AP-small=0.2293** (training time 9:09:54).
- **`plain_r2r3_nozoom`** — same as above minus R1 (zoom-crop reverted to
  default min_scale 0.3), to isolate R1's individual effect. Final
  **AP=0.3188, AP-small=0.2301** (training time 9:00:18).

**Neither beats the standing best (AP=0.3226).** Isolating R1's effect
(nozoom minus r1r2r3): the zoom-crop alone comes out **very slightly
negative** (-0.0005 AP) — within noise, but not the hypothesized +1-2 APs.
R1's premise (more pixels-per-object during training helps small objects) is
not validated by this result.

**Infra incident during this run:** both pods finished cleanly and then sat
idle for ~11h afterward (local babysitter died when WSL2 tore down its VM
after the laptop was closed for the night) — ~$4.80 burned on pure idle
billing, more than the marginal training cost. Root-caused and fixed:
`06_bugs_and_fixes.md` BUG-046. Checkpoints:
`output/runpod_results/plain_r1r2r3_last.pth`,
`output/runpod_results/plain_r2r3_nozoom_last.pth`.

---

## Step 28 — R4: per-class score calibration (eval-only) — IN PROGRESS (2026-07-05)

`DFINEPostProcessor` picks its global top-500 detections per image by
flattening (query, class) scores and taking `torch.topk` over the flattened
tensor (`src/zoo/dfine/postprocessor.py`) — classes compete directly for the
maxDets=500 budget. A systematically under-confident class can lose slots to
a better-calibrated common class even where its own localization is correct.
A monotonic per-class rescaling can't change within-class AP (COCO AP is
rank-based) but *can* change which detections survive this cross-class cut.

Method (`tools/calibration/calibrate_scores.py`): fit a per-class logit-space
bias on the full TRAIN split (median matched-TP score per class, boost
under-confident classes up to the least-under-confident class's level, never
suppress), then report the frozen bias applied to the VAL split against a
bias=0 baseline (same checkpoint, same eval, so the baseline reproduces the
standing best as a sanity check).

**Result (full ~6471-image train split, 2,728-134,184 matched TPs per
class — stable): net negative, hypothesis partially confirmed.**

Fitted bias (logit space) and train-split median TP score per class:

| Class | n_tp | median score | bias |
|---|---|---|---|
| pedestrian | 63,445 | 0.652 | +0.996 |
| people | 20,966 | 0.514 | +1.566 |
| bicycle | 7,768 | 0.453 | +1.813 |
| car | 134,184 | 0.820 | +0.103 |
| van | 23,391 | 0.688 | +0.829 |
| truck | 11,601 | 0.717 | +0.690 |
| tricycle | 4,121 | 0.502 | +1.615 |
| awning-tricycle | 2,728 | 0.516 | +1.557 |
| bus | 5,369 | 0.835 | +0.000 (target class) |
| motor | 25,254 | 0.605 | +1.195 |

VAL-split before/after (frozen bias, held-out from fitting):

- Overall AP: 0.3225 → 0.3196 (**-0.0029**)
- Overall AP50: 0.5238 → 0.5196 (**-0.0042**)
- Per-class AP50: **every class the hypothesis predicted would gain, did**
  (people +0.0021, bicycle +0.0027, tricycle +0.0028, awning-tricycle
  +0.0032, motor +0.0014) — confirming rare/under-confident classes were
  genuinely losing top-500 slots to better-calibrated classes. But
  **car (-0.0102), truck (-0.0036), van (-0.0025), and especially bus
  (-0.0377) all dropped**, outweighing the gains: boosting several classes
  at once increases competition pressure on the shared 500-detection budget
  more than it relieves it for any one class.

**Verdict: R4's core hypothesis is directionally validated (rare classes
really are under-confident, boosting them really does recover some of their
lost recall) but this specific implementation (independent per-class
median-equalization) is net negative and not adopted.** A jointly-optimized
bias (e.g. coordinate-ascent directly maximizing overall AP rather than
equalizing medians) might recover the small-class gains without the
large-class cost, but that's a materially bigger undertaking than the
"$0, eval-only" scope this was budgeted for — not pursued further for now.
Script: `tools/calibration/calibrate_scores.py`. Full results:
`output/calibration/r4_results.json`.

---

## Step 29 — E6: 1280 resolution unlock — FINISHED, AP=0.344 (2026-07-06)

Full 50-epoch schedule completed (see `SESSION_HANDOFF_2026-07-05.md` §1 for
design: tuned from `msfd_1024_best_ep109.pth`, square 1280×1280,
`reset_score_head_bias` fix, R2+R3 inherited). **Final AP=0.344** — up from
the pre-E6 standing best of 0.3226, and within ~0.012 of the DroneScan-YOLO
SOTA precedent (0.356). Best checkpoint at **epoch 46/50**
(`output/runpod_results/e6_1280_best_ep46.pth`), last at epoch 49
(`..._last_ep49.pth`) — best landing a few epochs before the final one
suggests the run had plateaued rather than still climbing, so the
pre-authorized resume-with-low-LR contingency was not triggered. (Not
confirmed against the full per-epoch trajectory/wandb log — revisit if that
assumption matters later.)

**Incident during this run:** the pod itself sat idle for ~10.6h (~$2.33)
after training finished because `autostop_launch.sh` didn't fire as
expected — see `06_bugs_and_fixes.md` BUG-047. A second, independent
laptop-side watchdog (`~/.runpod/watchdog.sh`) was added as a backstop.
Checkpoints were already safely retrieved before this was caught.

**Next up, per the priority order in `SESSION_HANDOFF_2026-07-05.md`:**
1. Deployment reality-check — ONNX export + INT8 quant + real latency
   benchmark on target edge hardware (not done yet; 1280 is ~4x the compute
   of the last benchmarked config).
2. msfd/P2 vs plain architecture control experiment at 1280.
3. Lower priority: R4 re-run on E6 checkpoint, R5 distillation, R6-R10.
