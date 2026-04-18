# D-FINE VisDrone: Aerial Object Detection on the Edge

Fine-tuning [D-FINE](https://arxiv.org/abs/2410.13842) (ICLR 2025) on the VisDrone aerial dataset, with structured pruning and INT8 deployment to a Snapdragon mobile NPU. Includes a Flutter web app for live inference.

---

## Results

**Accuracy** (VisDrone val, standard eval, 640×640 input):

| Stage | Model | AP50:95 | AP50 | Notes |
|-------|-------|---------|------|-------|
| COCO baseline | D-FINE-S | 48.5% | 65.4% | Matched reported paper result |
| VisDrone fine-tuned | D-FINE-S | **23.1%** | **38.9%** | 72 epochs, cosine LR |
| After pruning + recovery | D-FINE-S (pruned) | **23.2%** | — | 41.4% FFN reduction, slightly improves AP50:95 over fine-tuned checkpoint after recovery |

SOTA context: UAV-DETR-R50 (2025) = 31.5%, RT-DETR-R50 (2023) = 28.4%. D-FINE-S reaches 23.1% with a 10M parameter model trained on a single laptop GPU — gap is explained by model size and input resolution (640px vs 1280–1536px in SOTA).

**Edge deployment** (Samsung Galaxy S23, Snapdragon 8 Gen 2, INT8):

| Metric | Value |
|--------|-------|
| Inference latency (median) | **47 ms** |
| Throughput | **~21 FPS** |
| NPU utilization | **100%** (1316/1317 ops on Hexagon v73) |
| Model size | ~10 MB INT8 (vs 38 MB FP32 ONNX) |

---

## Demo

![D-FINE-S detecting vehicles and pedestrians on VisDrone aerial footage](demo.gif)

*D-FINE-S (pruned, INT8-ready) running on VisDrone validation images — 10-class aerial detection at 21 FPS on Snapdragon 8 Gen 2.*

---

## Key Engineering Decisions

- **Chose D-FINE-S over larger models** — 10M params fits comfortably in mobile VRAM; the FDR distribution head recovers localization quality that pure scale would require more parameters to match
- **Used VisDrone to stress-test domain transfer** — COCO pretraining gives strong priors, but aerial viewpoint and dense tiny objects require deliberate fine-tuning; switching from MultiStepLR to CosineAnnealingLR alone lifted AP from 0.170 → 0.231
- **Applied structured pruning to decoder FFNs specifically** — profiling showed decoder FFN layers dominated latency; group lasso regularization let the model self-select which neurons to drop, achieving 41.4% FFN reduction without architectural surgery
- **Exported to ONNX then compiled INT8 via Qualcomm AI Hub** — ONNX gives hardware-neutral export; AI Hub handles INT8 quantization and Hexagon NPU mapping, offloading the quantization complexity entirely
- **Built a side-by-side inference server** — comparing D-FINE-S against YOLOv8-X in the same app makes the accuracy/efficiency tradeoff concrete rather than abstract

---

## Limitations

- AP trails published VisDrone-specific SOTA (23.1% vs 31.5%) — gap is primarily input resolution; attempts to train at 1280px failed due to anchor grid mismatch with the dataset size
- Tiny crowded objects (46–53% of VisDrone instances are < 32px) remain the hardest case; AP-small is 0.142
- SAHI sliced inference improved AP-small (+0.011) but hurt overall AP (-0.006) due to patch-boundary fragmentation of larger objects — no clean win
- Accuracy comparison against YOLOv8-X favors YOLO (AP50 0.470 vs 0.389) — expected given 68M vs 10M parameters; D-FINE-S is the better deployment target, not the better accuracy target

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
│   │   ├── inference/             <- torch_inf.py, onnx_inf.py, sahi_inf.py
│   │   ├── deployment/            <- export_onnx_pruned.py, submit_aihub.py
│   │   └── pruning/               <- prune_dfine.py, recovery_train.py
│   ├── train.py                   <- single entry point for training + eval
│   └── PROJECT_NOTES/             <- lab notebook (all decisions, results, bugs)
├── dfine_app_server/              <- FastAPI inference server (D-FINE + YOLOv8)
│   ├── server_v1.py               <- active server: both models, letterbox preprocessing
│   ├── sota_compare.py            <- side-by-side comparison script
│   └── models/best.pt             <- YOLOv8-X VisDrone weights
└── dfine_app/                     <- Flutter web app
    └── lib/main.dart              <- model selector, camera/gallery picker, box overlay
```

---

## Architecture

**3-stage pipeline:** HGNetV2 backbone → HybridEncoder (neck) → DFINETransformer (decoder)

### D-FINE Innovations

**FDR (Fine-grained Distribution Refinement):** Instead of predicting a single (Δx,Δy,Δw,Δh) offset per box edge, the model predicts a probability distribution over `reg_max=32` non-uniformly spaced bins. The final edge position is the weighted expectation over those bins. This lets the model express localization uncertainty and produces tighter boxes than single-point regression.

**GO-LSD (Global Optimal Localization Self-Distillation):** The final decoder layer's predicted distributions are used as soft targets for earlier layers during training. Zero inference overhead — only active during the forward pass at training time.

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
# Fine-tune from COCO pretrained weights on VisDrone
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --device cuda:0 --tuning weight/dfine_s_coco.pth

# Override batch size for single-GPU (default config assumes 4×GPU)
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --device cuda:0 --tuning weight/dfine_s_coco.pth \
    -u train_dataloader.total_batch_size=4

# Eval only
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --device cuda:0 --test-only --resume output/dfine_hgnetv2_s_visdrone/best_stg1.pth
```

**Key flags:**
- `-t / --tuning` — load weights, reset optimizer (use for domain transfer)
- `-r / --resume` — load weights + optimizer state (use to resume an interrupted run)

---

## Structured Pruning

Removes FFN neurons from the 3 transformer decoder layers using group lasso regularization, then runs a 10-epoch recovery phase. Achieves 41.4% FFN reduction with no AP regression.

```bash
# Run pruning loop (saves checkpoint at each epoch, stops when AP drops below floor)
python tools/pruning/prune_dfine.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --checkpoint output/dfine_hgnetv2_s_visdrone/best_stg1.pth \
    --output-dir output/pruning --device cuda:0

# Recovery training after pruning
python tools/pruning/recovery_train.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --pruned-checkpoint output/pruning/best_pruned.pth \
    --output-dir output/pruning_recovery --device cuda:0
```

Best checkpoint: `output/pruning_recovery/best_recovery.pth`
FFN dims after pruning: `[598, 780, 423]` (from `[1024, 1024, 1024]`)

---

## ONNX Export + Deployment

```bash
# Export pruned model to ONNX (handles non-standard FFN dims automatically)
python tools/deployment/export_onnx_pruned.py \
    --config configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --checkpoint output/pruning_recovery/best_recovery.pth \
    --output output/pruning_recovery/best_recovery.onnx

# Submit to Qualcomm AI Hub for INT8 compilation + profiling
python tools/deployment/submit_aihub.py \
    --onnx output/pruning_recovery/best_recovery.onnx
```

---

## Inference

```bash
# PyTorch inference (single image)
python tools/inference/torch_inf.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    -r output/pruning_recovery/best_recovery.pth \
    --input image.jpg --device cuda:0

# ONNX inference
python tools/inference/onnx_inf.py \
    --onnx output/pruning_recovery/best_recovery.onnx \
    --input image.jpg

# SAHI (sliced) inference — improves AP-small slightly, use with care
python tools/inference/sahi_inf.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    -r output/pruning_recovery/best_recovery.pth \
    --input image.jpg
```

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

**Model comparison:**

| Model | Params | AP50:95 | AP50 |
|-------|--------|---------|------|
| D-FINE-S (ours, pruned) | 10M | 0.232 | 0.389 |
| YOLOv8-X (mshamrai HuggingFace) | 68M | — | 0.470 |

---

## Lab Notebook

All decisions, experiments, results, and bugs are documented in `D-FINE/PROJECT_NOTES/`:

| File | Contents |
|------|----------|
| `00_progress.md` | Step-by-step log, current status |
| `01_repo_structure.md` | Architecture deep-dive, config system |
| `02_coco_baseline.md` | COCO baseline: 48.5 mAP reproduced |
| `03_visdrone_dataset.md` | Dataset stats, class distribution, challenges |
| `04_finetuning_config.md` | Fine-tuning configuration decisions |
| `05_wsl2_aws_kubernetes.md` | WSL2 migration + AWS/K8s plan |
| `06_aws_kubernetes_setup.md` | AWS setup log |
| `06_bugs_and_fixes.md` | All bugs encountered and fixed |
| `07_pruning.md` | Full pruning results table, epoch-by-epoch |

---

## Reference

- [D-FINE paper](https://arxiv.org/abs/2410.13842) (ICLR 2025)
- [Original D-FINE repo](https://github.com/Peterande/D-FINE)
- [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- W&B training runs: [wandb.ai/danziv/D-FINE](https://wandb.ai/danziv/D-FINE)
