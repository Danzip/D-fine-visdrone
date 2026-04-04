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

# Train
python train.py -c configs/dfine/dfine_hgnetv2_s_coco.yml --device cuda:0

# Fine-tune from checkpoint
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --device cuda:0 --tuning path/to/pretrained.pth

# Eval only
python train.py -c configs/dfine/dfine_hgnetv2_s_coco.yml \
    --device cuda:0 --test-only --resume path/to/checkpoint.pth

# Override any config value from CLI
python train.py -c config.yml -u epochs=50 train_dataloader.total_batch_size=8

# ONNX export
python tools/deployment/export_onnx.py \
    --config configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --checkpoint outputs/best.pth --output outputs/model.onnx

# PyTorch inference
python tools/inference/torch_inf.py -c configs/... -r checkpoint.pth \
    --input image.jpg --device cuda:0

# ONNX inference
python tools/inference/onnx_inf.py --onnx outputs/model.onnx --input image.jpg
```

## Architecture

3-stage pipeline: **HGNetV2 backbone → HybridEncoder (neck) → DFINETransformer (decoder)**

- `src/zoo/dfine/` — all D-FINE-specific code (the innovations live here)
- `src/zoo/dfine/dfine_utils.py` — FDR: `weighting_function`, `distance2bbox`, `bbox2distance`
- `src/zoo/dfine/dfine_criterion.py` — losses including GO-LSD (`loss_ddf`) and FDR (`loss_fgl`)
- `src/zoo/dfine/dfine_decoder.py` — transformer decoder with MSDeformableAttention
- `src/zoo/dfine/hybrid_encoder.py` — RT-DETR neck: AIFI (attention on coarsest scale) + CCFF
- `src/nn/backbone/hgnetv2.py` — HGNetV2 backbone

Config system uses `__include__` YAML composition. The S-model config
(`configs/dfine/dfine_hgnetv2_s_coco.yml`) pulls in 5 base files and overrides specific values.

## Key Parameters

- `-t / --tuning` — load weights, reset optimizer (domain transfer)
- `-r / --resume` — load weights + optimizer state (resume interrupted run)
- `reg_max=32` — distribution bins per edge (128 total per box)
- `eval_idx=-1` — use final decoder layer at inference
- For 1-GPU training, override batch size: `-u train_dataloader.total_batch_size=8`

## Lab Notebook

`D-FINE/PROJECT_NOTES/` — all documentation and results live here. Keep it updated.
Always read the relevant notes before starting any task to understand current project state.

- `00_progress.md` — current step status and results
- `01_repo_structure.md` — architecture deep-dive
- `02_coco_baseline.md` — COCO baseline results
- `03_visdrone_dataset.md` — VisDrone dataset preparation
- `04_finetuning_config.md` — fine-tuning configuration
- `05_wsl2_aws_kubernetes.md` — WSL2 migration + AWS/Kubernetes plan
- `06_aws_kubernetes_setup.md` — AWS setup log (what actually happened)
- `06_bugs_and_fixes.md` — bugs encountered and fixes applied
