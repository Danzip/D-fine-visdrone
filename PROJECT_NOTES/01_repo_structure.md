# D-FINE Repository Structure

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 4060 Laptop GPU |
| VRAM | 8 GB |
| CUDA Driver | 555.97 (supports up to CUDA 12.5) |
| PyTorch | 2.5.1+cu124 |
| Python | 3.12.10 (venv at `D-FINE/venv/`) |

**Training budget implication:** 8GB VRAM is comfortable for D-FINE-S/N.
For D-FINE-L, batch size will need to be reduced to ~4-8. D-FINE-X may require
gradient checkpointing. We will use D-FINE-S for all experiments.

---

## Repository Layout

```
D-FINE/
├── configs/                  <- YAML training configurations (composable, via __include__)
│   ├── dataset/              <- dataset-specific paths and class counts
│   ├── dfine/                <- per-model-size COCO configs + fine-tune templates
│   │   ├── include/          <- shared building blocks: model arch, optimizer, dataloader
│   │   ├── custom/           <- template for any custom dataset (we will use this)
│   │   └── objects365/       <- configs for Objects365 pretraining
│   └── runtime.yml           <- global flags: amp, ema, wandb, checkpoint_freq
│
├── src/                      <- all library code (imported by train.py and tools/)
│   ├── core/                 <- config loading system (YAMLConfig, workspace, registry)
│   ├── data/                 <- dataset classes, dataloaders, transforms
│   ├── misc/                 <- utilities: box ops, distributed, logger, visualizer
│   ├── nn/                   <- generic neural network components (backbone, postprocessor)
│   ├── optim/                <- optimizer, EMA, AMP scaler, warmup scheduler
│   ├── solver/               <- training engine (det_engine.py runs one epoch)
│   └── zoo/dfine/            <- D-FINE-specific code (the core innovation lives here)
│
├── tools/                    <- scripts for non-training tasks
│   ├── inference/            <- torch_inf.py, onnx_inf.py, trt_inf.py
│   ├── deployment/           <- export_onnx.py
│   ├── benchmark/            <- TensorRT benchmarking
│   └── dataset/              <- dataset conversion utilities
│
├── train.py                  <- single entry point for training + eval
├── requirements.txt          <- minimal deps (torch>=2.0.1 + 6 packages)
├── PROJECT_NOTES/            <- our lab notebook (this directory)
└── venv/                     <- Python 3.12 virtual environment (gitignored)
```

---

## Config System (important to understand)

Configs use `__include__` to compose multiple YAML files. For example,
`configs/dfine/dfine_hgnetv2_s_coco.yml` includes:

1. `configs/dataset/coco_detection.yml` — dataset paths and class count
2. `configs/runtime.yml` — global flags (amp, ema, logging)
3. `configs/dfine/include/dataloader.yml` — augmentation pipeline + batch size
4. `configs/dfine/include/optimizer.yml` — AdamW settings, LR warmup, MultiStepLR
5. `configs/dfine/include/dfine_hgnetv2.yml` — full model architecture config

Later keys override earlier ones. The S-size config overrides `num_layers: 3`
and `depth_mult: 0.34` on top of the base config which defaults to 6 layers.

---

## Model Architecture (3-stage pipeline)

### Stage 1: HGNetV2 Backbone (`src/nn/backbone/hgnetv2.py`)

HGNetV2 is a PaddlePaddle-origin CNN backbone. The S-model uses variant `B0`.
- Returns feature maps at strides 8, 16, 32 (channels: 256, 512, 1024 for S)
- `use_lab=True` → uses Light Aggregation Block for efficient feature extraction
- `freeze_at=-1` → nothing frozen by default; during fine-tuning we may freeze early stages

### Stage 2: HybridEncoder (`src/zoo/dfine/hybrid_encoder.py`)

This is the RT-DETR neck, combining two ideas:
1. **AIFI (Attention-based Intra-scale Feature Interaction):** Applies a single
   transformer encoder layer only on the coarsest feature scale (stride 32).
   This is the key RT-DETR efficiency trick — attention is O(N²) in sequence length,
   so applying it only at the coarsest scale (smallest spatial size) is fast.
   Controlled by `use_encoder_idx: [2]` (index 2 = coarsest scale).

2. **CCFF (CNN-based Cross-scale Feature Fusion):** Fuses multi-scale features
   using convolutions (not attention). Fast, parameter-efficient.

Output: three feature maps all projected to `hidden_dim=256`.

### Stage 3: DFINETransformer (`src/zoo/dfine/dfine_decoder.py`)

A 6-layer transformer decoder (S-model uses 3 layers). Key components:

- **MSDeformableAttention:** Multi-scale deformable cross-attention.
  Each query attends to a small set of learned sampling points across all feature scales.
  Much more efficient than full attention.

- **Per-layer outputs:** Each layer outputs `pred_logits` (class scores)
  and `pred_corners` (distribution over edge positions). The last layer
  is the final prediction. Earlier layers produce auxiliary losses during training.

---

## The Two Core Innovations

### Innovation 1: FDR — Fine-grained Distribution Refinement

Located in: `src/zoo/dfine/dfine_utils.py`

**Standard DETR approach:**
- Predict 4 numbers: (Δx, Δy, Δw, Δh) = single-point offset from reference box
- Problem: a single prediction is unstable, especially for ambiguous object boundaries

**D-FINE approach:**
- For each of the 4 box edges (left, top, right, bottom), predict a probability
  distribution over `reg_max=32` discrete bins
- The final edge position is: `weighted_average(W(n) * p(n))` where `W(n)` is a
  learned non-uniform weighting function and `p(n)` is the predicted probability
- The weighting function `W(n)` maps bin indices to actual pixel offsets using
  a non-linear (exponential) spacing — more bins near center (small offsets),
  fewer bins for large offsets

**Why it works:** Instead of committing to a single offset, the model can express
uncertainty by spreading probability mass. During training the distribution sharpens
toward the correct bin. This is analogous to heatmap prediction in pose estimation.

Key functions:
- `weighting_function(reg_max, up, reg_scale)` → generates the non-uniform W(n)
- `distance2bbox(points, distance, reg_scale)` → weighted average → final box
- `bbox2distance(points, bbox, reg_max, reg_scale, up)` → encodes GT box as distribution target

Training loss: `loss_fgl` (Fine-Grained Localization) — unimodal distribution focal loss.
This encourages the predicted distribution to be unimodal (peaked), not spread out.

### Innovation 2: GO-LSD — Global Optimal Localization Self-Distillation

Located in: `src/zoo/dfine/dfine_criterion.py` → `loss_local()` → `loss_ddf`

**Problem being solved:**
In standard multi-layer decoders, each layer is supervised independently.
But layer 6 (final) always has the best box predictions — it's seen the most
context. Layers 1-5 are weaker because they're supervised from GT directly
without the benefit of what layer 6 learned.

**D-FINE solution:**
- During training only: layer 6's predicted distribution (`teacher_corners`) is
  used as a soft target for layers 1-5 (`pred_corners`)
- Loss: KL divergence between student (early layer) and teacher (final layer) distributions
- Temperature T=5 is used to soften the teacher distribution (standard knowledge distillation trick)
- `loss_ddf` = "Decoupled Distillation Focal" — the word "decoupled" means matched and
  unmatched queries are weighted differently (via `num_pos` and `num_neg`)

**Zero inference overhead:** The teacher signal only exists during training.
At inference, all 6 layers run normally and only the final layer is used
(controlled by `eval_idx: -1`).

**Global Optimal part:** The "GO" matching uses a union of Hungarian matches
from all decoder layers (`_get_go_indices`). This gives each layer more
stable training signal than if it were matched independently.

---

## Training Entry Point

```bash
# Single GPU training
python train.py -c configs/dfine/dfine_hgnetv2_s_coco.yml --device cuda:0

# Fine-tuning from checkpoint
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --device cuda:0 --tuning path/to/pretrained.pth

# Evaluation only
python train.py -c configs/dfine/dfine_hgnetv2_s_coco.yml \
    --device cuda:0 --test-only --resume path/to/checkpoint.pth

# Override config params from CLI
python train.py -c config.yml --device cuda:0 \
    -u epochs=50 train_dataloader.total_batch_size=8
```

**`-t / --tuning`** loads weights but starts training fresh (for domain transfer).
**`-r / --resume`** loads weights AND optimizer state (for resuming interrupted training).

---

## Loss Function Summary

| Loss | Variable | Weight | What it measures |
|------|----------|--------|-----------------|
| `loss_vfl` | VFL (Varifocal Loss) | 1.0 | Classification, IoU-weighted |
| `loss_bbox` | L1 | 5.0 | Box center + size regression |
| `loss_giou` | GIoU | 2.0 | Box overlap quality |
| `loss_fgl` | FGL | 0.15 | Distribution over edge positions (FDR) |
| `loss_ddf` | DDF | 1.5 | KL between early and final layer distributions (GO-LSD) |

Each loss is also computed at auxiliary decoder layers and labeled `_aux_0`, `_aux_1`, etc.
This means the actual training log has ~40+ loss terms but they all roll up into the 5 above.

---

## Config Parameters Reference (S-model COCO)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_layers` | 3 | Number of transformer decoder layers (S); L/X use 6 |
| `eval_idx` | -1 | Which layer to use at inference (-1 = last) |
| `reg_max` | 32 | Number of distribution bins per edge |
| `reg_scale` | 4 | Controls W(n) curvature (non-uniform bin spacing) |
| `num_queries` | 300 | Number of object queries |
| `num_denoising` | 100 | Number of denoising queries (contrastive DN training) |
| `hidden_dim` | 256 | Transformer embedding dimension |
| `epochs` | 132 | 120 normal + 12 fine-grained epochs |
| `total_batch_size` | 32 | Designed for 4×GPU setup; on 1 GPU use 4-8 |
| `use_amp` | True | Mixed precision (FP16 forward, FP32 optimizer) |
| `use_ema` | True | Exponential moving average of weights |
| `use_wandb` | False | Set to True to enable W&B logging |
