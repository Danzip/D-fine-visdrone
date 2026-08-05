# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

D-FINE + VisDrone: fine-tuning the D-FINE object detector (ICLR 2025) on the VisDrone aerial dataset,
then exporting to ONNX + INT8 quantization for edge deployment.

## Environment

- Python 3.12 venv at `D-FINE/venv/` — always activate before running anything
- PyTorch 2.5.1+cu124, GPU: RTX 4060 Laptop 8GB VRAM
- All commands run from inside `D-FINE/`

```bash
# Activate venv (Windows bash)
source D-FINE/venv/Scripts/activate

# Fine-tune from COCO pretrained weights (base NWD recipe)
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml \
    --device cuda:0 --tuning weight/dfine_s_coco.pth

# Reproduce the current best checkpoint (P2ConvHead + P2FusionLite + NWD, 1280px)
python train.py -c experiments/e6_1280/config.yml \
    --device cuda:0 --tuning output/runpod_results/msfd_1024_best_ep109.pth

# Eval only
python train.py -c experiments/e6_1280/config.yml \
    --device cuda:0 --test-only --resume output/runpod_results/e6_1280_best_ep46.pth

# Override any config value from CLI
python train.py -c config.yml -u epochs=50 train_dataloader.total_batch_size=8
```

ONNX export / pruning / generic PyTorch+ONNX inference scripts predate the `experiments`
branch and aren't in this working tree — see `README.md`'s "Structured Pruning + ONNX
Deployment (historical)" section for what they produced and where they're preserved.

## Architecture

3-stage pipeline: **HGNetV2 backbone → HybridEncoder (neck) → DFINETransformer (decoder)**

- `src/zoo/dfine/` — all D-FINE-specific code (the innovations live here)
- `src/zoo/dfine/dfine_utils.py` — FDR: `weighting_function`, `distance2bbox`, `bbox2distance`
- `src/zoo/dfine/dfine_criterion.py` — losses including GO-LSD (`loss_ddf`) and FDR (`loss_fgl`)
- `src/zoo/dfine/dfine_decoder.py` — transformer decoder with MSDeformableAttention
- `src/zoo/dfine/hybrid_encoder.py` — RT-DETR neck: AIFI (attention on coarsest scale) + CCFF
- `src/nn/backbone/hgnetv2.py` — HGNetV2 backbone

Config system uses `__include__` YAML composition. The base VisDrone config
(`configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml`) pulls in 5 base files and overrides
specific values; per-experiment configs under `experiments/*/config.yml` typically
`__include__` the current-best experiment's config and override just what changes.

## Key Parameters

- `-t / --tuning` — load weights, reset optimizer (domain transfer)
- `-r / --resume` — load weights + optimizer state (resume interrupted run)
- `reg_max=32` — distribution bins per edge (128 total per box)
- `eval_idx=-1` — use final decoder layer at inference
- For 1-GPU training, override batch size: `-u train_dataloader.total_batch_size=8`

## Lab Notebook

`D-FINE/PROJECT_NOTES/` — all documentation and results live here. Keep it updated.
Always read the relevant notes before starting any task to understand current project state.

- `00_progress.md` — step-by-step log of every experiment, decision, and result (the primary lab notebook)
- `06_bugs_and_fixes.md` — bugs encountered and fixes applied (BUG-001 → BUG-049)
- `10_next_experiments.md` — ablations: crop-zoom, NWD regression loss, rare-class CopyPaste
- `11_ablation_study_runpod.md` — RunPod ablation campaign log (P2 conv-head, resolution unlock, calibration)
- `12_tracking.md` — multi-object tracker comparison
- `SESSION_HANDOFF_*.md` — dated session handoff notes
