# Step 4 — Fine-tuning Config

## Config File

`configs/dfine/dfine_hgnetv2_s_visdrone.yml`

## Parameter Decisions

### num_classes: 10
COCO has 80 classes. VisDrone has 10. The classification head (a linear layer of
shape `[hidden_dim=256, num_classes+1]`) is re-initialised when we load the COCO
checkpoint with `--tuning` flag. All other weights (backbone, encoder, most of decoder)
are kept from the pretrained model.

### remap_mscoco_category: False
COCO category IDs are non-contiguous (they skip some numbers in the original 91-class
COCO list). D-FINE applies a remapping to make them 0-79 when training on COCO.
VisDrone IDs are already contiguous 1-10, so no remapping needed.

### Learning Rates

| Parameter group | LR | Reasoning |
|----------------|-----|-----------|
| Backbone (weights) | 0.000025 | Lowest. Backbone holds low-level features (edges, textures) from ImageNet+COCO pretraining. We want to preserve these — they transfer well to VisDrone. Too high LR here causes "catastrophic forgetting". |
| Backbone (BN/norms) | 0.000025 | Same as backbone weights. |
| Encoder/decoder (norms/bias) | 0.0 weight decay | No decay on norm parameters (standard practice — norms are scale/shift, not "weights"). |
| All other encoder/decoder | 0.00005 (global LR) | 4× lower than COCO from-scratch (0.0002). The encoder/decoder need to adapt more than the backbone (VisDrone features are different) but still benefit from pretrained cross-scale fusion. |

**Why not freeze the backbone entirely?**
Freezing would be appropriate if we had very little data (< 1000 images). With 6471
training images, letting the backbone adapt slowly (via low LR) gives better results
because VisDrone's aerial features genuinely differ from COCO's ground-level features.

### Epochs: 72 (= 60 + 12)

D-FINE has a two-stage training schedule:
1. **Epochs 0–59 (60 epochs):** Full augmentation (random crop, zoom-out, photometric
   distortion, multi-scale resize). This is the main learning phase.
2. **Epochs 60–71 (12 epochs):** Augmentation stopped, multi-scale disabled.
   Model trains on fixed 640×640 with only horizontal flip. EMA is reset with
   `ema_restart_decay=0.9999`. This "fine-grained" phase stabilises the distribution
   predictions and pushes mAP ~0.5-1.0 points higher.

Why not more epochs? VisDrone train set is 6,471 images vs COCO's 118,000. Risk of
overfitting increases after ~70-80 epochs on a dataset this size. We monitor val mAP
via W&B — if it plateaus or drops, training can be stopped early.

### Batch Size: 8

The D-FINE paper uses total_batch_size=32 across 4 GPUs = 8 per GPU.
We have 1 GPU, so batch_size=8 exactly matches the per-GPU batch the paper used.
This is important: the gradient updates are identical in scale.

Memory estimate for D-FINE-S at batch=8, 640×640:
- Activations: ~3.5GB
- Model weights + gradients + optimizer states: ~1.5GB
- Total: ~5GB → safely within 8GB VRAM

### Input Resolution: 640×640

VisDrone images are 1360×765 to 2000×1500. Resizing to 640×640 means:
- A 30px object on a 2000px-wide image becomes 30×(640/2000) = **9.6px** — very tiny
- This is unavoidable given memory constraints
- Alternative: 1280×1280 uses 4× more memory (quadratic in spatial dim) → ~12GB needed
- For future work: tiling (process overlapping crops) would preserve resolution

### W&B Visualisation

`use_wandb: True` enables:
1. **Metrics panel:** AP50:95, AP50, AP75, APsmall, APmedium, APlarge logged each epoch
2. **Loss panel:** train losses (loss_vfl, loss_bbox, loss_giou, loss_fgl, loss_ddf)
3. **Per-class images:** 5 images × 10 classes = 50 W&B images logged each epoch
   - Shows GT boxes (one overlay) and predicted boxes (second overlay) on same image
   - Scrub through epochs with W&B step slider to see training progress
   - Image selection: for each class, the 5 val images with most instances of that class

## Training Command

```bash
cd D-FINE
python train.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    --tuning weight/dfine_s_coco.pth \
    --device cuda:0 \
    --use-amp \
    --seed 42
```

Flags explained:
- `-t / --tuning`: load weights, reset optimizer state (correct for domain transfer)
- `--use-amp`: mixed precision (FP16 forward pass, FP32 optimizer) — ~2× faster, ~2× less memory
- `--seed 42`: reproducibility

**Note:** `--tuning` vs `--resume`:
- `--tuning`: loads model weights only. Optimizer state (momentum, adam m/v) is reset.
  Use this for domain transfer — you want fresh optimizer momentum for the new domain.
- `--resume`: loads model weights AND optimizer state. Use this if training was interrupted.

## Expected Training Timeline

At batch_size=8, ~6471 train images:
- Iterations per epoch: ceil(6471 / 8) = 810
- Time per iteration (D-FINE-S, RTX 4060, AMP): ~100-150ms
- Time per epoch: ~810 × 0.125s ≈ 1.7 minutes
- Total for 72 epochs: ~2 hours

## What We Expect to See

**Early epochs (0-10):**
- Loss drops sharply — model adapts classification head to 10 classes
- mAP likely very low (5-15%) — model still confused by aerial domain

**Mid training (10-40):**
- Loss decreases steadily but slower
- mAP climbs significantly — backbone features adapting to aerial viewpoint
- W&B images: predictions getting better per class

**Late training (40-60):**
- Loss may plateau or slightly oscillate
- mAP should be approaching its peak
- `loss_fgl` (distribution refinement loss) becomes smaller — boxes getting tighter

**Phase 2 (60-72):**
- After augmentation stops, mAP should tick up another 0.5-1 AP
- EMA model is reset and re-accumulated

**Target mAP:** VisDrone is significantly harder than COCO.
- COCO-pretrained baseline (no fine-tuning): estimated 10-20% mAP on VisDrone
- After fine-tuning: published results with similar setups achieve 25-35% mAP
- Improvement demonstrates value of domain transfer
