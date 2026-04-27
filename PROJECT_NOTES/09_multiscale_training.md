# Multi-Scale Resolution Training — Complete Log

## Context

All 4 previous high-resolution attempts failed (see `00_progress.md` Step 7b).
The root cause identified: **resolution collapse** — the model locks into 640×640 spatial priors
during VisDrone fine-tuning, then can't recover when jumped to a new resolution.

The fix: **never lock into a single scale**. Use multi-scale training from the COCO checkpoint
throughout the entire VisDrone fine-tune, so the model adapts to varying resolutions continuously.

---

## Experiment D — Object Size Distribution Analysis (2026-04-18)

**Script:** `tools/demo/analyze_box_sizes.py`

Simulates letterbox resize at 6 resolutions and reports what fraction of VisDrone training
boxes fall below key pixel thresholds (stride-8 = 8px feature floor, stride-16 = 16px, etc).

| Resolution | % boxes < 16px | Median W | Median H |
|------------|----------------|----------|----------|
| 640×640    | 77.0%          | 9px      | 8px      |
| 800×800    | 69.5%          | 11px     | 10px     |
| 960×960    | 60.8%          | 14px     | 12px     |
| 1024×1024  | 56.1%          | 15px     | 13px     |
| 1280×736   | 52.3%          | 18px     | 10px     |
| 1280×1280  | 45.2%          | 19px     | 17px     |

**Key insight:** 1280×1280 square gives the best sub-16px reduction (77%→45%, −31.8pp).
1024×1024 gives −20.9pp. Even 960px gives −16.2pp vs 640px baseline.

---

## Key Code Changes

### 1. Adaptive batch in collate (`src/data/dataloader.py`)

Added `adaptive_batch` parameter to `BatchImageCollateFunction`. When True, after choosing
a random scale, the collate subsamples the batch to keep total pixel count constant:

```python
n = max(1, int(len(images) * (self.base_size / sz) ** 2))
```

This means at base_size=1024: sz=768→batch=4, sz=1024→batch=2, sz=1280→batch=1.
No OOM at large scales; no wasted capacity at small scales.

### 2. Gradient accumulation (`src/solver/det_engine.py`)

Added `accum_steps` kwarg to `train_one_epoch`. Loss is divided by accum_steps before
backward; optimizer.step() fires every N batches. EMA also only updates on optimizer steps.

```python
loss = sum(loss_dict.values()) / accum_steps
scaler.scale(loss).backward()
if do_step:
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
```

With accum_steps=4 and batch=2: effective batch=8 at memory cost of batch=2.

### 3. eval_spatial_size must match val resolution

The AIFI encoder pre-caches positional embeddings at `eval_spatial_size` during model init.
During training, pos_embed is built dynamically from the actual feature map `(h, w)`.
During eval, the cached embed is used — if it doesn't match the val resize, crash:

```
RuntimeError: The size of tensor a (1024) must match the size of tensor b (400)
```

Fix: always set `eval_spatial_size` in config to match val_dataloader Resize.
(640×640 default from base include was causing the crash at 1024px.)

---

## Run 1 — ms1280 (2026-04-18 → 2026-04-19)

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_ms1280.yml`

| Setting | Value |
|---------|-------|
| Start checkpoint | `weight/dfine_s_coco.pth` (COCO pretrained) |
| Multi-scale range | [768, 1280] px — `generate_scales(1024, 3)` |
| Dataset Resize | 1024×1024 (letterbox) |
| eval_spatial_size | [1024, 1024] |
| batch | 2 (→1 at 1280px via adaptive_batch) |
| accum_steps | 4 (effective batch = 8) |
| LR | 5e-5 global, 2.5e-5 backbone, cosine T_max=80 |
| AMP | True (mandatory at this resolution) |
| Epochs | 80 (stage 1: 0–63, stage 2: 64–79) |
| VRAM peak | 4.9 GB |

### Results

| Epoch | AP50:95 | Notes |
|-------|---------|-------|
| 0 | 0.023 | Expected — COCO heads reset for 10 VisDrone classes |
| 20 | ~0.15 | Climbing steadily |
| 40 | ~0.22 | Approaching 640px baseline |
| 63 | **0.2509** | Stage 1 best → saved as `best_stg1.pth` |
| 68 | **0.2553** | Stage 2 peak → saved as `best_stg2.pth` |

Stage 2 (epochs 64–79) reloads best_stg1.pth + resets EMA but LR was already ~3e-6.
Despite the dead LR, EMA reset gave a +0.004 boost. Run killed at epoch 68 to start cont.

**Gain vs baseline: +0.024 (0.231 → 0.255)**

---

## Run 2 — ms1280_cont (2026-04-19 → 2026-04-21)

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_cont.yml`

Key differences from Run 1:
- Start: `best_stg2.pth` (AP=0.2553) via `--tuning` (fresh optimizer + LR)
- `stop_epoch: 9999` — never enters stage 2; pure multi-scale stage 1 forever
- `epochs: 500` — run until manual stop at plateau
- `T_max: 500` — cosine LR decays over 500 epochs (stays meaningful for long time)
- Same augmentation as ms1280 + `CopyPasteSmallObjects` added

### Full AP trajectory

| Epoch | AP50:95 | Notes |
|-------|---------|-------|
| 0 | 0.2490 | Fresh LR dip; weights already good |
| 10 | 0.2594 | |
| 30 | 0.2699 | |
| 50 | 0.2784 | |
| 60 | 0.2813 | |
| 80 | 0.2884 | |
| 90 | 0.2901 | First time above 0.29 |
| 107 | 0.2938 | Checkpoint saved (for SWA) |
| 119 | 0.2952 | Checkpoint saved (for SWA) |
| 125 | 0.2964 | |
| **131** | **0.2966** | **Best — saved as best_stg1.pth** |

Rate of improvement slowing to ~0.0001/epoch by epoch 125. Model approaching plateau.

### Crash — BUG-017 (2026-04-21, epoch 132)

Training crashed at the start of epoch 132 in the DataLoader worker:

```
File "src/data/transforms/_transforms.py", line 226, in __call__
    image, target = inputs
ValueError: too many values to unpack (expected 2)
```

Root cause: `CopyPasteSmallObjects.__call__` received a 3-tuple `(image, target, dataset)`
from `stop_epoch_forward` in `container.py` but tried to unpack only 2 values.
See BUG-017 in `06_bugs_and_fixes.md` for full details and fix.

Best checkpoint (`best_stg1.pth`, epoch 131, AP=0.2966) was already saved before the crash.

**Total gain vs 640px baseline: +0.066 (0.231 → 0.2966)**
**Gain purely from resolution + multi-scale training: +28%**

---

## Why Multi-Scale Finally Worked (vs the 4 failed resolution attempts)

All 4 previous attempts (Step 7b) jumped from a single-scale checkpoint to a new fixed resolution.
This triggers a collapse cycle:
- Wrong proposals (tuned to 640px spatial grid) → noisy gradients
- Noisy gradients corrupt backbone+encoder → worse proposals
- Cycle is self-reinforcing → AP never recovers from cold start

Multi-scale from COCO avoids this because:
1. COCO pretraining has no VisDrone-specific scale priors to protect
2. Multi-scale range [768–1280] means the model sees many resolutions every epoch, never locking into one
3. Scale-invariant features emerge naturally from the gradient mixture
4. Adaptive batch keeps the pixel budget constant → no OOM at large scales

The continued run (ms1280_cont) further shows that the model keeps improving for 130+ epochs
with no plateau, suggesting the multi-scale regime is still extracting signal from the data
far longer than the 72-epoch single-scale baseline.

---

## Run 3 — ms1280_mosaic (2026-04-22 → 2026-04-23)

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_mosaic.yml`

### Why this run

ms1280_cont was approaching plateau (~0.2966, gain < 0.0001/epoch). To push past this ceiling
we added two new signal sources:

1. **Mosaic4 augmentation (p=0.5):** combines 4 training images into a 2×2 grid before
   the rest of the pipeline. Each training sample now sees context from 4 different scenes.
   - Benefit for aerial data: VisDrone scenes are visually similar (parking lots, roads).
   Mosaic creates artificial diversity without needing more data.
   - Benefit for small objects: a 32×32 object that fills ~3% of its original image fills
   ~6% of the mosaic quadrant — more gradient signal per step.
   - Why p=0.5 (not 1.0): at p=1.0, random crop always acts on mosaic composites, losing
   single-image fine-grained localization signal. 50/50 mix balances diversity vs precision.

2. **Repeat Factor Sampling (RFS, threshold=0.5):** oversample images containing
   rare classes proportional to `sqrt(t / freq)`. Effective oversampling rates:
   - awning-tricycle (rarest): 1.68×
   - tricycle: 1.38×
   - bus: 1.27×
   - bicycle: 1.08×
   - other classes: 1.0× (no oversampling)
   Motivation: awning-tricycle and tricycle AP were badly lagging in per-class evals.

3. **Two-phase schedule:**
   - Phase 1 (epochs 0–79): full augmentation + multi-scale [768–1280] + adaptive batch
   - Phase 2 (epochs 80–95): no augmentation, fixed 1024px, EMA restart (stage 2)
   Stage 2 acts as a fine-grained polishing pass after augmentation has fully trained the model.

**Start checkpoint:** `output/dfine_hgnetv2_s_visdrone_ms1280_cont/best_stg1.pth` (AP=0.2966)
via `--tuning` (fresh optimizer + LR at 5e-5 global, 2.5e-5 backbone).

### Results

| Epoch | AP50:95 | Notes |
|-------|---------|-------|
| 0 | 0.2920 | Fresh LR dip from 0.2966 start |
| 2 | 0.2943 | Recovering |
| 4 | 0.2980 | Back to pre-run level |
| **5** | **0.3000** | **First time AP ≥ 0.30** |
| 7 | 0.3004 | |
| 8 | 0.3007 | |
| 9 | 0.3010 | |
| 10 | 0.3010 | |
| **11** | **0.3015** | **Best — saved as best_stg1.pth** |
| 12 | 0.3015 | Tied best |

Still climbing at epoch 12. Crashed mid-epoch 13 (see below).

**Best checkpoint:** `output/dfine_hgnetv2_s_visdrone_ms1280_mosaic/best_stg1.pth`

### Crash — epoch 13 (2026-04-23)

Log cuts off mid-epoch 13 (step 2900/4075) with no Python traceback — confirmed OOM kill.
Root cause: BUG-018. WSL2 is capped at 7.56 GB RAM; Mosaic's 4-image-per-sample pipeline
causes a per-step RAM spike across 4 DataLoader workers that pushes total usage above the cap.
The Linux OOM killer fires SIGKILL on the Python process — no exception, just null bytes at the
end of the log file (unfilled pre-allocated write buffer, the classic SIGKILL fingerprint).
See `06_bugs_and_fixes.md` BUG-018 for full diagnosis and fix.

The best checkpoint (epoch 11, AP=0.3015) was already saved before the crash.

**To resume:**
```bash
cd D-FINE
source venv/Scripts/activate
python train.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_mosaic.yml \
    --device cuda:0 \
    --resume output/dfine_hgnetv2_s_visdrone_ms1280_mosaic/last.pth
```

### Why mosaic worked when ms1280_cont was plateauing

ms1280_cont had seen all 6,471 training images ~131 times with the same augmentation set.
The remaining gradient signal was small. Mosaic injected new signal by changing what the model
sees per step — not more data, but more varied combinations of existing data.

The immediate recovery to above-0.3 within 5 epochs (despite the LR reset dip) suggests the
model architecture had headroom that standard augmentation wasn't unlocking.

**Total gain vs 640px baseline: +0.071 (0.231 → 0.3015)**
**Gain over ms1280_cont ceiling: +0.005 (0.2966 → 0.3015) in 12 epochs**

---

## Experiment Results — No-Retraining Ablations (2026-04-22)

All experiments below run on `best_stg1.pth` (epoch 131, AP=0.2966 at 1024px eval).
Copy-paste augmentation was already active during training (in config at line 84).

### Eval resolution comparison

| Eval resolution | AP50:95 | AP-small | AP-medium | AP-large |
|-----------------|---------|----------|-----------|----------|
| 1024×1024 (default) | **0.2966** | 0.208 | 0.411 | 0.589 |
| 1280×1280 | 0.296 | **0.212** | 0.407 | 0.584 |

**Verdict:** No meaningful gain from 1280px eval. Multi-scale training already calibrates the
model well across scales. 1024px is the right eval setting — keep it.

Implementation note: required fixing `_solver.py` `load_state_dict` to use `_matched_state`
with `strict=False` for model and EMA keys, so shape-mismatched anchor buffers
(`decoder.anchors`, `decoder.valid_mask`) are re-initialized rather than failing.

### SAHI inference comparison

| Config | AP50:95 | AP50 | AP-small | AP-medium | AP-large |
|--------|---------|------|----------|-----------|----------|
| Standard eval (1024px) | **0.2966** | **0.479** | **0.208** | **0.411** | **0.589** |
| SAHI 1024px slices→1024px | 0.280 | 0.466 | 0.199 | 0.387 | 0.552 |
| SAHI 640px slices→1024px | 0.283 | 0.479 | 0.208 | 0.383 | 0.527 |

**Verdict:** SAHI hurts across the board. Same pattern as with the old 640px model.
Multi-scale training handles scale variation internally — slicing fragments large/medium objects
and NMS introduces merge errors that cost more than slicing gains. SAHI is off the table.

Reference: old model (AP=0.231) SAHI result was AP=0.225 (also a net loss).

### TTA (Test-Time Augmentation) — COMPLETE (2026-04-22)

Implemented `tools/inference/tta_inf.py`: runs inference at N scales + horizontal flip,
merges with WBF (Weighted Box Fusion). Required two code fixes to enable multi-scale inference:

- `hybrid_encoder.py`: AIFI pos_embed cache falls back to dynamic generation when feature
  map size doesn't match `eval_spatial_size` (line 455-464)
- `dfine_decoder.py`: Same fix for anchor/valid_mask cache in `_get_decoder_input` (line 760-766)

| Config | AP50:95 | AP-small |
|--------|---------|----------|
| Standard eval (1024px) | **0.2966** | **0.208** |
| TTA: 1024px + hflip | 0.276 | 0.184 |
| TTA: 768+1024+1280 + hflip | 0.280 | 0.191 |

**Verdict:** TTA hurts. Multi-scale training + hflip augmentation already internalises what
TTA tries to add. WBF coordinate averaging introduces noise; confidence averaging drops
well-calibrated boxes. TTA is off the table for this model.

### SWA (Stochastic Weight Averaging) — COMPLETE (2026-04-22)

Implemented `tools/inference/swa_avg.py`: averages N checkpoint EMA modules into a single
merged checkpoint, then evals normally.

| Config | AP50:95 |
|--------|---------|
| best_stg1 (epoch 131) | **0.2966** |
| SWA: ep107 + ep119 + ep131 | 0.295 |
| SWA: ep119 + ep131 | 0.296 |

**Verdict:** Neutral. SWA needs post-plateau checkpoints to find the basin centre.
The model was still climbing at epoch 131 — averaging in earlier checkpoints only
pulls down the best weights. Re-test SWA after the model converges (plateau < 0.001/10ep).

### Complete No-Retraining Ablation Summary

| Experiment | AP50:95 | Delta | Verdict |
|------------|---------|-------|---------|
| Baseline 1024px eval | **0.2966** | — | Best |
| Eval at 1280px | 0.296 | −0.001 | No gain |
| SAHI 1024px slices | 0.280 | −0.017 | Hurts |
| SAHI 640px→1024px | 0.283 | −0.014 | Hurts |
| TTA 3-scale+hflip | 0.280 | −0.017 | Hurts |
| TTA 1024px+hflip | 0.276 | −0.021 | Hurts |
| SWA ep107+119+131 | 0.295 | −0.002 | Negligible |
| SWA ep119+131 | 0.296 | ±0 | Neutral |

**Key insight:** All inference-time tricks fail because the model already implicitly learns
scale-invariance, flip-invariance, and multi-scale detection through its training augmentations.
The only proven path forward is retraining with new augmentation diversity.

### Next Steps

1. **Resume ms1280_mosaic training** — crashed mid-epoch 13 (VS Code reset). Resume from `last.pth`.
   Current best: AP=0.3015 at epoch 11. Phase 1 runs to epoch 79 — still 66 epochs of augmented
   training left. Then phase 2 (epochs 80–95) polishes with no augmentation + EMA restart.
   Target: 0.31+ after phase 2.

2. **Post-plateau SWA** — once the model plateaus in phase 1, average the last 5 checkpoints.
   Expected: +0.002–0.005 AP. Re-test with more checkpoints in the plateau region (didn't have
   enough at ep131 of ms1280_cont because it was still climbing).

3. **D-FINE-M** — same config, larger backbone. Expected +3–5 AP on top of current ceiling.
   Blocked by VRAM: D-FINE-M with batch=2 at 1024px likely OOMs on RTX 4060 8GB.
   Would need AWS G4dn (T4, 16GB) or gradient accumulation with batch=1.

---

## Run 4 — ms1280_mosaic v2: AR-Preserving Pipeline (2026-04-24 → in progress)

**Config:** `configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_mosaic.yml` (same file, updated)

**Why a new run instead of resume:**
Between the epoch-13 crash and now, the entire training pipeline was redesigned to be
aspect-ratio-aware. Resuming from `last.pth` would inherit the old square-canvas geometry.
Starting from `best_stg1.pth` of ms1280_cont (AP=0.2966) via `--tuning` gives a clean restart.

### What changed from Run 3

**1. No more square canvases.** Every batch now draws both a resolution AND an aspect ratio:
```python
sz = random.choice(scales)        # e.g. 1280
ar_key = random.choice(["4:3", "16:9"])   # e.g. "16:9"
canvas_h, canvas_w = _canvas_dims(sz, ar_key)  # (704, 1280) — 32-aligned
```
VisDrone images (all 16:9 and 4:3) now fill the canvas without a 25% black bar waste.

**2. AR-aware batch size table.** Probe was rerun against real VisDrone images (not random
tensors). Two batches probed per resolution: one at 4:3 canvas, one at 16:9. Table format:
`{sz: {"4:3": n, "16:9": n}}`. 16:9 batches are ~20% larger (fewer pixels tall).

**3. Mosaic moved to collate with same-AR sampling.**
- Removed `{type: Mosaic}` from dataset transforms pipeline
- Added `mosaic_p: 0.5` to `BatchImageCollateFunction`
- `set_dataset()` builds `_ar_groups = {"4:3": [3302 idxs], "16:9": [3169 idxs]}`
- Mosaic tiles are fetched from the same-AR pool → direct resize to `(tile_h, tile_w)` with
  no padding, no black bars between tiles
- Tiles resize directly with `F.interpolate` (normalized cxcywh coords are resize-invariant)

**4. Dual eval.** Two val passes per epoch:
- Primary: `[704, 1280]` (16:9, 32-aligned) → drives `best_stat` and checkpointing
- Secondary: `[960, 1280]` (4:3, 32-aligned) → logged as `eval_43/*` on W&B

**5. Unified 5-element letterbox.** All paths store `[scale, pad_top, pad_left, canvas_h, canvas_w]`
and postprocessor/det_engine use the same unwind formula.

**Bug fixed before this run:** BUG-019 — `eval_spatial_size: [720, 1280]` → 720/32=22.5, caused
FPN upsample crash. Fixed to `[704, 1280]` which matches `_canvas_dims(1280, "16:9")` exactly.

### Results

| Epoch | AP50:95 | Notes |
|-------|---------|-------|
| — | — | Training started 2026-04-24, in progress |

**Start checkpoint:** `output/dfine_hgnetv2_s_visdrone_ms1280_cont/best_stg1.pth` (AP=0.2966)
