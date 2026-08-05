# D-FINE VisDrone: Aerial Object Detection on the Edge

Fine-tuning [D-FINE](https://arxiv.org/abs/2410.13842) (ICLR 2025) on the VisDrone aerial dataset, with structured pruning and INT8 deployment to a Snapdragon mobile NPU. Includes a Flutter web app for live inference.

---

## Results

The model starts from COCO-pretrained D-FINE-S weights (see `PROJECT_NOTES/00_progress.md`);
every row below is measured on VisDrone val.

| Stage | AP50:95 | AP50 | AP-small | Latency | Model Size |
|-------|---------|------|----------|---------|------------|
| VisDrone fine-tuned (640px) | 23.1% | 38.9% | 14.2% | - | 38 MB FP32 |
| + Structured pruning + recovery | 23.2% | - | - | - | ~28 MB FP32 |
| Multi-scale training 1024px (80 ep) | 25.5% | 42.4% | 17.8% | - | 38 MB FP32 |
| + Extended training (131 ep) | 29.7% | 47.9% | 20.8% | - | 38 MB FP32 |
| + Mosaic + multi-scale retraining (160 ep) | 31.6% | 50.7% | 22.5% | - | 38 MB FP32 |
| + NWD matching + size-adaptive loss (ep109) | 32.1% | 50.4% | 23.0% | - | 38 MB FP32 |
| + P2 conv head + P2FusionLite + NWD-loss + rare CopyPaste, ultra-low-LR polish (RunPod, 2026-07-04) | 32.26% | - | 23.44% | - | 38 MB FP32 |
| + 1280px resolution unlock via score-head bias reset (E6, RunPod, 2026-07-06) — current best | **34.4%** | **54.9%** | **25.7%** | - | 38 MB FP32 |
| INT8 on Snapdragon 8 Gen 2 | - | - | - | **47 ms / 21 FPS** | **10 MB INT8** |

SOTA context (VisDrone val, standard eval): DroneScan-YOLO (2026) = 35.6% (10M params, purpose-built for aerial; this figure was never independently reproduced — the paper's weights were never found publicly, treat it as directional, not verified), Drone-DETR (2024) = 33.9%, VRF-DETR (2024) = 32.2%, RT-DETR-R50 (2023) = 28.4%. D-FINE-S reaches **34.4%** with 10M params as a general-purpose detector fine-tuned on VisDrone — within ~1.2 AP of the (unverified) DroneScan-YOLO figure and already ahead of Drone-DETR/VRF-DETR, despite no aerial-specific architecture beyond the added P2 head. 100% NPU utilization on Hexagon v73 (1316/1317 ops offloaded) — note the INT8/latency figures above are from the older 32.1% checkpoint; neither the 32.26% nor the current-best 34.4% checkpoint has been re-deployed yet. A follow-up RunPod campaign (`PROJECT_NOTES/11_ablation_study_runpod.md`) tested the P2 conv-head architecture and several augmentation/loss ideas (crop-zoom, NWD regression loss, rare-class CopyPaste) that did not beat the P2 result; the 1280px resolution unlock that produced the current best (E6, `experiments/e6_1280/`) turned out to be a score-head bias-reset fix, not AIFI positional-embedding interpolation as originally planned — AIFI's sin-cos position embeddings are recomputed fresh per forward pass and never needed interpolation. See that doc and `experiments/e6_1280/config.yml`'s header comment for the full root-cause writeup.

### Per-class AP - epoch-131 checkpoint (baseline for Mosaic+RFS retraining)

| Class | Train freq | AP50:95 | AP50 | AP-small | AP-med | AP-large |
|-------|-----------|---------|------|----------|--------|---------|
| car | 42% | **0.619** | **0.858** | 0.453 | 0.750 | **0.873** |
| bus | 5% | 0.458 | 0.607 | 0.232 | 0.575 | 0.856 |
| van | 8% | 0.373 | 0.516 | 0.203 | 0.519 | 0.586 |
| pedestrian | 22% | 0.286 | 0.562 | 0.247 | 0.523 | **0.843** |
| motor | 11% | 0.294 | 0.591 | 0.265 | 0.416 | 0.561 |
| truck | 4% | 0.284 | 0.395 | 0.124 | 0.322 | 0.604 |
| people | 14% | 0.213 | 0.486 | 0.201 | 0.327 | 0.648 |
| tricycle | 3% | 0.201 | 0.334 | 0.161 | 0.262 | 0.491 |
| bicycle | 4% | 0.129 | 0.253 | 0.097 | 0.275 | 0.316 |
| awning-tricycle | 1% | **0.110** | **0.185** | **0.094** | **0.144** | **0.113** |
| **mean** | | **0.297** | **0.479** | **0.208** | **0.411** | **0.589** |

The two rarest classes (awning-tricycle 1%, tricycle 3%) are the weakest. RFS oversampling and Mosaic augmentation in the current retraining run specifically target this gap.

---

## Inference-Time Ablation (no retraining)

All experiments run on the epoch-131 checkpoint (AP=29.7%, 1024px eval). Multi-scale training already internalizes the benefits that inference-time tricks try to add - none improved on standard eval. Full analysis in `PROJECT_NOTES/00_progress.md` (Step 14) and `PROJECT_NOTES/11_ablation_study_runpod.md`.

| Method | AP50:95 | AP-small | Delta | Verdict |
|--------|---------|----------|-------|---------|
| Standard eval (1024×1024) | **29.7%** | **20.8%** | - | Best |
| Eval at 1280×1280 | 29.6% | 21.2% | −0.1% | No gain |
| SAHI 1024px slices | 28.0% | 19.9% | −1.7% | Hurts |
| SAHI 640px slices (→1024px) | 28.3% | 20.8% | −1.4% | Hurts |
| TTA 1024px + hflip (WBF) | 27.6% | 18.4% | −2.1% | Hurts |
| TTA 3-scale + hflip (WBF) | 28.0% | 19.1% | −1.7% | Hurts |
| SWA ep107+119+131 | 29.5% | - | −0.2% | Negligible |
| SWA ep119+131 | 29.6% | 20.7% | ±0 | Neutral |

**Why everything fails:** Each trick assumes the model hasn't learned something - scale invariance (SAHI, TTA), flip invariance (TTA hflip), or basin convergence (SWA). After 130+ epochs of [768–1280] multi-scale training with `RandomHorizontalFlip` and copy-paste augmentation, all of these are already internalized. The only lever left is retraining with new signal: new augmentation diversity (mosaic, now in progress) or architectural changes (P2 detection head, NWD loss).

---

## Demo

![D-FINE-S detecting vehicles and pedestrians on VisDrone aerial footage](demo.gif)

*D-FINE-S (pruned, INT8-ready) running on VisDrone validation images - 10-class aerial detection at 21 FPS on Snapdragon 8 Gen 2.*

---

## Key Engineering Decisions

- **D-FINE-S over YOLO or larger DETR variants** - sacrifices ~8 AP points vs YOLOv8-X but is 7× smaller; for Snapdragon NPU deployment, parameter efficiency matters more than peak accuracy, and the FDR distribution head compensates for scale
- **CosineAnnealingLR over MultiStepLR** - milestone-based decay never fired in 72 epochs on this dataset; switching to cosine alone lifted AP from 0.170 → 0.231 (+36%), no architecture change
- **Decoder FFNs as pruning target** - decoder FFN layers dominated inference cost; used group lasso regularization to let the model self-select neuron importance rather than applying a fixed compression ratio, achieving 41.4% reduction with no AP regression
- **Multi-scale training from COCO (not from 640px checkpoint)** - 4 direct attempts to train at 960–1280px all failed (AP flatlined at 0.11–0.13); root cause was anchor grid collapse when jumping from 8,400 → 19,320 positions. The fix: start from the COCO checkpoint with multi-scale [768–1280] from epoch 0, so the model never locks into a single-scale prior. This lifted AP from 0.231 → 0.297 (+28%).
- **Adaptive batch size over fixed batch** - training at 1280px with fixed batch=2 would waste capacity at 768px and OOM at 1280px; adaptive batch keeps total pixel budget constant (`n ∝ (base_size/sz)²`), giving effective batch=8 across all scales
- **ONNX + Qualcomm AI Hub over on-device PyTorch** - AI Hub handles Hexagon NPU mapping and INT8 quantization automatically; offloads hardware-specific compiler complexity and gives profiling data (latency, memory, NPU utilization) without owning a device
- **Score-head bias reset unlocks resolution jumps, not AIFI interpolation** - 4 earlier attempts to train at higher resolution from an already-adapted checkpoint collapsed (AP 0.11-0.13); the actual cause was `enc_score_head`/`dec_score_head` biases staying miscalibrated for a denser anchor grid, not AIFI's position embeddings (already resolution-agnostic, recomputed per forward pass). Resetting just those biases before tuning to 1280px lifted AP 32.26% → 34.4%

---

## Limitations

- AP trails the unverified DroneScan-YOLO figure (34.4% vs 35.6%) by ~1.2 points, and is already ahead of Drone-DETR (33.9%) / VRF-DETR (32.2%) / RT-DETR-R50 (28.4%) - closer than earlier checkpoints since the P2 stride-4 head and NWD loss (added in the P2/E6 lineage) already target the aerial-specific gap that used to be the main limitation
- Tiny crowded objects (46–53% of VisDrone instances are < 32px) remain the hardest case; AP-small is 25.7% (up from 14.2% at 640px, but still the weakest area - awning-tricycle and bicycle AP are well below the mean, see per-class table above)
- All inference-time tricks tested at the pre-P2 checkpoint (SAHI, TTA, SWA, higher eval resolution) failed to improve over standard eval; a from-scratch, train-time-tiled SAHI variant is being tested separately (`PROJECT_NOTES/13_sahi_tiled_training.md`) to see if training on tiles (not just evaluating on them) closes that gap
- Current best (34.4%, E6/1280px) has not yet been re-deployed through the pruning/ONNX/INT8 pipeline - the Snapdragon latency numbers above are from the older 32.1% checkpoint
- Next step (lower priority): every SOTA figure above except YOLOv8-X is a paper-reported number, not independently reproduced (DroneScan-YOLO's own weights were never found publicly - see `PROJECT_NOTES/00_progress.md`). Find and evaluate a real, downloadable, comparably strong model with the same protocol used for YOLOv8-X (`tools/eval/eval_yolov8x_visdrone.py`), instead of citing unverified literature numbers

---

## Why VisDrone is Hard

| Property | VisDrone | COCO |
|----------|----------|------|
| Viewpoint | Aerial / top-down | Ground level |
| Objects/image | 53–70 | ~7 |
| Tiny objects (< 32px) | 46–53% | ~15% |
| Classes | 10 (vehicles + pedestrians) | 80 |

Classes: `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`

---

## What This Repo Contains

```
DFine/
├── D-FINE/                        <- main training codebase (D-FINE + VisDrone configs)
│   ├── configs/dfine/             <- YAML configs for COCO, VisDrone, pruning experiments
│   ├── src/zoo/dfine/             <- D-FINE decoder, encoder, FDR + GO-LSD losses
│   ├── src/nn/backbone/           <- HGNetV2 backbone
│   ├── tools/
│   │   ├── visdrone2coco.py       <- VisDrone .txt annotations -> COCO-format JSON
│   │   ├── eval/                  <- eval_yolov8x_visdrone.py (YOLOv8-X baseline eval)
│   │   ├── tracking/              <- track_video.py, compare_trackers.py
│   │   └── calibration/           <- calibrate_scores.py
│   ├── train.py                   <- single entry point for training + eval
│   └── PROJECT_NOTES/             <- lab notebook (all decisions, results, bugs)
├── dfine_app_server/              <- FastAPI inference server (D-FINE + YOLOv8)
│   ├── server_v1.py               <- active server: both models, letterbox preprocessing
│   ├── sota_compare.py            <- side-by-side comparison script
│   └── models/best.pt             <- YOLOv8-X VisDrone weights
└── dfine_app/                     <- Flutter web app
    └── lib/main.dart              <- model selector, camera/gallery picker, box overlay
```

Earlier deployment/pruning/ONNX-export tooling (`tools/inference/`, `tools/deployment/`,
`tools/pruning/`) predates the current `experiments` branch and isn't in this working
tree — it's preserved in git history on `master` (`git show master:<path>`) if needed
again. See `PROJECT_NOTES/00_progress.md` Steps 8-9 for what it produced.

---

## Architecture

**3-stage pipeline:** HGNetV2 backbone → HybridEncoder (neck) → DFINETransformer (decoder)

### D-FINE Innovations

**FDR (Fine-grained Distribution Refinement):** Instead of predicting a single (Δx,Δy,Δw,Δh) offset per box edge, the model predicts a probability distribution over `reg_max=32` non-uniformly spaced bins. The final edge position is the weighted expectation over those bins. This lets the model express localization uncertainty and produces tighter boxes than single-point regression.

**GO-LSD (Global Optimal Localization Self-Distillation):** The final decoder layer's predicted distributions are used as soft targets for earlier layers during training. Zero inference overhead - only active during the forward pass at training time.

---

## Setup

**Requirements:** Python 3.12, PyTorch 2.5.1+cu124, NVIDIA GPU (tested on RTX 4060 Laptop 8GB)

```bash
cd D-FINE
python -m venv venv
source venv/Scripts/activate       # Windows / WSL2
# or: source venv/bin/activate     # Linux
pip install -r requirements.txt
pip install wandb                  # optional, for W&B logging
```

---

## Training

```bash
# Fine-tune from COCO pretrained weights on VisDrone (base NWD recipe)
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml \
    --device cuda:0 --tuning weight/dfine_s_coco.pth

# Override batch size for single-GPU
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml \
    --device cuda:0 --tuning weight/dfine_s_coco.pth \
    -u train_dataloader.total_batch_size=4

# Reproduce the current best checkpoint (P2ConvHead + P2FusionLite + NWD, 1280px)
python train.py -c experiments/e6_1280/config.yml \
    --device cuda:0 --tuning output/runpod_results/msfd_1024_best_ep109.pth

# Eval only
python train.py -c experiments/e6_1280/config.yml \
    --device cuda:0 --test-only --resume output/runpod_results/e6_1280_best_ep46.pth
```

**Key flags:**
- `-t / --tuning` - load weights, reset optimizer (use for domain transfer)
- `-r / --resume` - load weights + optimizer state (use to resume an interrupted run)

---

## Structured Pruning + ONNX Deployment (historical)

Earlier in the project the checkpoint was structurally pruned (group-lasso on decoder
FFN neurons, 41.4% FFN reduction with no AP regression) and exported through ONNX to
Qualcomm AI Hub for INT8 compilation on a Snapdragon 8 Gen 2 NPU (47ms/21FPS, 100% NPU
utilization). Best pruned checkpoint: `output/pruning_recovery/best_recovery.pth`
(ONNX: `output/pruning_recovery/best_recovery.onnx`, FFN dims `[598, 780, 423]` from
`[1024, 1024, 1024]`).

The scripts that produced this (`tools/pruning/`, `tools/deployment/`, and the generic
`tools/inference/{torch_inf,onnx_inf,tta_inf}.py`) predate the current `experiments`
branch and aren't in this working tree — they're preserved in git history on `master`
(`git show master:<path>`) if needed again. See `PROJECT_NOTES/00_progress.md` Steps 8-9
for the full pruning/export log.

---

## Tiled (SAHI-style) Training + Inference

A from-scratch tiled-training experiment (train and eval both on overlapping 640×640
windows, 50% overlap, NMS-merged predictions) is in progress — see
`PROJECT_NOTES/13_sahi_tiled_training.md` and `experiments/sahi_tiled/` for the run log
and final results. This replaces an earlier, inference-only SAHI ablation (see table
above) that hurt accuracy, likely because the model was never trained on tiles.

A fresh control measurement (today's best checkpoint, tiled+NMS eval, but *not* trained
on tiles) confirms the mismatch is severe at this tiling density: AP50:95 0.344 → 0.146
(-58% relative), AP-large hit hardest (0.626 → 0.242) from objects fragmenting/duplicating
across the denser 50%-overlap tile grid. This is the number train-time tiling has to beat.

---

## Multi-Object Tracking

Adds ByteTrack — and, via `boxmot`, BoT-SORT/StrongSORT/OC-SORT/DeepOCSORT — on top of D-FINE detections to track objects across video. Tested on real aerial footage from VisDrone2019-MOT (the DET data used above has no video/sequences of its own).

```bash
# Track with ByteTrack (fixed low-confidence recovery pass + camera motion compensation)
python tools/tracking/track_video.py \
    -c experiments/e6_1280/config.yml \
    -r output/runpod_results/e6_1280_best_ep46.pth \
    --video path/to/input.mp4 --output output/tracked.mp4 --device cuda:0

# Compare all 5 trackers on the same cached D-FINE detections (fair, single-pass comparison)
python tools/tracking/compare_trackers.py \
    -c experiments/e6_1280/config.yml \
    -r output/runpod_results/e6_1280_best_ep46.pth \
    --video path/to/input.mp4 --output-dir output/tracking/compare --device cuda:0
```

**Tracker comparison** (VisDrone-MOT `uav0000137_00458_v`, 233 frames, 184 ground-truth tracks, dense street intersection):

| Tracker | Unique tracks | vs. 184 GT | FPS |
|---|---|---|---|
| ByteTrack (motion-only + GMC) | 670 | 3.64x | 10.8 |
| OC-SORT (motion-only) | 660 | 3.59x | 13.8 |
| **BoT-SORT** (GMC + Re-ID) | **362** | **1.97x** | 7.2 |
| **StrongSORT** (Re-ID) | **335** | **1.82x** | 3.1 |
| DeepOCSORT (motion + Re-ID) | 664 | 3.61x | 4.2 |

**Appearance (Re-ID) matching roughly halves track fragmentation** vs. motion-only tracking on this dense, panning aerial scene — BoT-SORT is the practical pick (best speed/accuracy tradeoff); StrongSORT if accuracy is the only axis that matters. Full methodology, a bug fix (ByteTrack's low-confidence recovery pass was silently disabled by an over-eager pre-filter — BUG-049), and a profiling deep-dive into why tracker FPS doesn't scale with detection count the way you'd expect are in `PROJECT_NOTES/12_tracking.md`.

---

## Flutter App + Inference Server

A web app that runs D-FINE-S and YOLOv8-X side-by-side on any image.

**Start the server:**
```bash
cd dfine_app_server
pip install fastapi uvicorn onnxruntime ultralytics pillow
uvicorn server_v1:app --host 0.0.0.0 --port 8000
```

**Start the Flutter app:**
```bash
cd dfine_app
flutter run -d chrome --release
# or: flutter build web && serve build/web on port 8080
```

The server loads both models at startup. `POST /detect` accepts `file` + `model` (dfine|yolov8) as multipart form fields. The Flutter app has a model selector dropdown, gallery/camera picker, and a bounding box overlay with per-class colours.

**Model comparison** (VisDrone val, same 548-image split, same evaluator, maxDets=500):

| Model | Params | AP50:95 | AP50 | AP-small |
|-------|--------|---------|------|----------|
| D-FINE-S (ours, current best, E6/1280px) | 10M | 0.344 | 0.549 | 0.257 |
| D-FINE-S (ours, pruned INT8)* | 10M | 0.232 | 0.389 | - |
| YOLOv8-X (mshamrai HuggingFace)† | 68M | 0.250 | 0.404 | 0.156 |

\* Predates the P2/E6 architecture (pruned from an earlier, non-P2 lineage) — not a pruned
version of the current best. AP-small isn't filled in because reading this checkpoint
needs the FFN-resize loading logic that was only ever in `export_onnx_pruned.py`
(archived on `master`, out of scope here).
† Re-measured directly with `tools/eval/eval_yolov8x_visdrone.py` against this repo's own
VisDrone val split, with the same `maxDets=500` D-FINE's own eval uses — the model card's
own claimed 0.47 AP50 is a different, unverified measurement.

---

## Lab Notebook

All decisions, experiments, results, and bugs are documented in `D-FINE/PROJECT_NOTES/`:

| File | Contents |
|------|----------|
| `00_progress.md` | Step-by-step log of every experiment, decision, and result — the primary lab notebook |
| `06_bugs_and_fixes.md` | All bugs encountered and fixed (BUG-001 → BUG-049) |
| `10_next_experiments.md` | Ablations: crop-zoom, NWD regression loss, rare-class CopyPaste |
| `11_ablation_study_runpod.md` | RunPod ablation campaign log (P2 conv-head, resolution unlock, calibration) |
| `12_tracking.md` | Multi-object tracker comparison (ByteTrack/BoT-SORT/StrongSORT/OC-SORT/DeepOCSORT) |
| `SESSION_HANDOFF_*.md` | Dated session handoff notes (branch strategy, in-flight run status) |

---

## Reference

- [D-FINE paper](https://arxiv.org/abs/2410.13842) (ICLR 2025)
- [Original D-FINE repo](https://github.com/Peterande/D-FINE)
- [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- W&B training runs: [wandb.ai/danziv/D-FINE](https://wandb.ai/danziv/D-FINE)
