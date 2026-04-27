# Inference-Time Eval Ablations — Deep Dive

All experiments below were run on the **ms1280_cont best checkpoint** (epoch 131, AP=0.2966,
evaluated at 1024×1024). This was the plateau checkpoint at the time.

The question behind all of these: can we extract more AP from the existing weights without
retraining? Short answer: no. Long answer follows.

---

## Summary Table

| Method | AP50:95 | AP50 | AP-small | Delta | Verdict |
|--------|---------|------|----------|-------|---------|
| Standard eval (1024×1024) | **0.2966** | **0.479** | **0.208** | — | Best |
| Eval at 1280×1280 | 0.296 | 0.478 | 0.212 | −0.001 | No gain |
| SAHI 1024px slices (cont model) | 0.280 | 0.466 | 0.199 | −0.017 | Hurts |
| SAHI 640px slices → 1024px (cont model) | 0.283 | 0.479 | 0.208 | −0.014 | Hurts |
| SAHI 640px slices (original 640px model) | 0.225 | 0.404 | 0.153 | −0.006 | Hurts |
| TTA: 1024px + hflip (WBF) | 0.276 | — | 0.184 | −0.021 | Hurts |
| TTA: 3-scale (768+1024+1280) + hflip (WBF) | 0.280 | — | 0.191 | −0.017 | Hurts |
| SWA: ep107 + ep119 + ep131 | 0.295 | — | — | −0.002 | Negligible |
| SWA: ep119 + ep131 | 0.296 | — | — | ±0 | Neutral |

---

## 1. Higher Eval Resolution (1280×1280 vs 1024×1024)

**Hypothesis:** The model was trained at scales up to 1280px. Evaluating at 1280px instead of
1024px might benefit small objects — a 16px object at 1024px becomes 20px at 1280px.

**Result:** AP50:95 = 0.296 (−0.001 vs 0.2966 baseline). AP-small improved from 0.208 → 0.212
(+0.004), but AP-medium and AP-large both dropped slightly.

**Why it didn't help overall:**

The model's AIFI encoder caches positional embeddings at `eval_spatial_size`. At 1024px, the
stride-32 feature map is 32×32 = 1024 tokens. At 1280px it's 40×40 = 1600 tokens. During
training, all resolutions in [768–1280] were seen — so 1600 tokens is not foreign. But the
cosine LR schedule decayed from 5e-5 toward 1e-6 over 500 epochs: by epoch 131, the model
has been fine-tuned at LR ≈ 2e-5 for hundreds of epochs, so the 1024px context has received
far more gradient updates than 1280px. The model is simply more calibrated to 1024px inference.

The AP-small micro-gain from more pixels is real, but doesn't offset the slight drop in
localization precision for medium/large objects (which were already well-calibrated at 1024px).

**When it would work:** If training at 1280px base_size instead of 1024px, or if LR were kept
higher for more epochs at the upper end of the scale range. The ms1280 runs used base_size=1024;
1280px was the *ceiling* of the range, not the centre.

---

## 2. SAHI — Slicing Aided Hyper Inference

**Concept:** Split each image into overlapping tiles, run the detector on each tile independently,
then merge all predictions with NMS. Objects that were tiny relative to the full image become
larger relative to the tile — closer to the model's training resolution.

Two variants were tested:

### 2a. SAHI on the original 640px model (Step 6b, 2026-03-26)

**Config:** `tools/inference/sahi_inf.py`, slice_size=640, overlap=0.2, full val set.

**Result:** AP=0.225 (−0.006 vs 0.231 baseline). AP-small +0.011, but AP-medium −0.022, AP-large −0.074.

### 2b. SAHI on ms1280_cont (2026-04-22)

Two slice configs tested against the 1024px model:

| SAHI config | AP50:95 | AP50 | AP-small | Notes |
|------------|---------|------|----------|-------|
| 1024px slices | 0.280 | 0.466 | 0.199 | −0.017 overall |
| 640px slices → 1024px postprocess | 0.283 | 0.479 | 0.208 | −0.014 overall |

**Why SAHI consistently hurts:**

**Problem 1 — Object fragmentation.** VisDrone images are densely packed. A car or van spans
200–600px in the original 1400×1050 image. At 1024px letterbox, that's 150–450px — well within
the model's detectable range. When you tile into 1024×1024 slices with 20% overlap, many of
these medium/large objects straddle tile boundaries. Each tile half generates a partial detection;
NMS is forced to either merge them (coordinate average → wrong box) or suppress one (miss).
The AP-large collapse (−0.074 in the 640px experiment, consistent losses in the 1024px ones)
is entirely this fragmentation effect.

**Problem 2 — NMS merge errors.** SAHI uses a global NMS pass after collecting all tile
predictions. For small objects near tile edges (which appear in 2 overlapping tiles), the model
produces two nearly-identical boxes. If IOU > NMS threshold, one is suppressed — usually
correctly. But in dense VisDrone scenes (53–70 objects/image), legitimate adjacent objects
have IOU > threshold; SAHI's NMS merges them into a single detection, dropping recall.

**Problem 3 — The model already handles the scale.** The ms1280_cont model was trained on
[768–1280] multi-scale with a stride-8 feature head. At 1024px, a 16px object produces a
2×2 stride-8 response — weak but present. SAHI at 1024px makes that same 16px object a 23px
object in the tile — marginally better, but the model is already calibrated to detect 2×2
responses. The marginal benefit doesn't outweigh fragmentation cost.

**AP-small at 640px slices == baseline (0.208 = 0.208):** This is interesting — slicing to
640px exactly recovers the small-object AP that 1024px slicing loses (1024px slicing gives
0.199 AP-small). At 640px slice size, tiny objects have a slightly different input context.
But overall AP still drops because medium/large objects fragment worse with smaller tiles.

**When SAHI would work:** On a single-scale model trained only at 640px with no small-object
augmentation. In that setting, 16px objects are near the model's detection floor, and slicing
to make them 23px crosses a threshold. The multi-scale training has already closed that gap
internally. SAHI is a pre-training-era workaround.

---

## 3. TTA — Test-Time Augmentation

**Concept:** Run inference at multiple resolutions + horizontal flip, fuse all predictions
with Weighted Box Fusion (WBF). WBF averages coordinates weighted by confidence scores,
unlike NMS which discards all but the highest-confidence box.

Implementation: `tools/inference/tta_inf.py`. Fixes required:
- `hybrid_encoder.py` line 455–464: AIFI pos_embed cache falls back to dynamic generation
  when feature map size doesn't match `eval_spatial_size`
- `dfine_decoder.py` line 760–766: same fix for anchor/valid_mask cache

Two variants:

| TTA config | AP50:95 | AP-small | Delta |
|-----------|---------|----------|-------|
| 1024px + hflip | 0.276 | 0.184 | −0.021 |
| 768+1024+1280 + hflip | 0.280 | 0.191 | −0.017 |

**Why TTA hurts:**

**Problem 1 — WBF coordinate averaging degrades precise boxes.** WBF merges N box predictions
into one by computing a weighted average of coordinates. For a well-localized 20px object,
the 1024px prediction might be `[423, 517, 443, 537]`. The 768px prediction of the same object
might be `[317, 388, 332, 403]` (same object, different scale). WBF normalizes to image space
and averages — the merged box ends up slightly wrong on both axes. IoU at threshold 0.75 is
extremely sensitive to this; a 2-pixel error on a 20px box can drop IoU from 0.80 to 0.68.
AP75 takes the largest hit.

**Problem 2 — The model already has flip-invariance baked in.** The training pipeline includes
`RandomHorizontalFlip` at every epoch. By epoch 131, the model's feature representations are
already approximately equivariant to horizontal flip. Running inference twice and averaging adds
minimal new information — just coordinate noise from the averaging step.

**Problem 3 — Scale diversity is already internalized.** Same argument as SAHI: the [768–1280]
multi-scale training means the model has seen every object at every size in that range. Running
inference at 768px and 1280px and averaging with 1024px doesn't reveal new features; the model
already learned to handle all three scales.

**Why the 3-scale variant is slightly better than 1-scale+hflip:** Adding 768+1280 slightly
reduces the hflip averaging noise (3 coordinate sources averaging out) and recovers AP-small
somewhat (0.184 → 0.191). But neither recovers the overall AP.

**When TTA works:** On models trained at a single fixed scale without flip augmentation.
The classic TTA paper results (e.g., on COCO) use ImageNet-pretrained models that never saw
scale jitter during training. Those models are genuinely surprised by 1.5× scale changes.
D-FINE-S after ms1280_cont training is not.

---

## 4. SWA — Stochastic Weight Averaging

**Concept:** Average the weights from multiple training checkpoints. In loss landscape terms,
SGD/Adam explores a bowl-shaped valley and oscillates around the minimum. SWA averages several
points in that valley to land closer to the centre (lower loss basin, better generalization).

Implementation: `tools/inference/swa_avg.py`. Averages the EMA module weights (not the raw
model weights — the EMA is what gets evaluated).

Two variants, both applied to ms1280_cont checkpoints:

| Config | AP50:95 | Notes |
|--------|---------|-------|
| best_stg1 epoch 131 | **0.2966** | Baseline |
| SWA: ep107 + ep119 + ep131 | 0.295 | −0.002 |
| SWA: ep119 + ep131 | 0.296 | ±0 |

**Why SWA was neutral/negative here:**

SWA works by finding the basin centre of a converged, oscillating model. It requires that
the model has already reached a plateau and is *oscillating* around a minimum — the checkpoints
to average should be at roughly the same loss level, just from different sides of the basin.

At epoch 131, the AP was **still climbing**: 0.2964 at epoch 125, 0.2966 at epoch 131. The model
had not converged. Averaging ep107 (AP=0.2938) with ep131 (AP=0.2966) pulls the average toward
a worse point — ep107 is not the "other side of the basin" but simply an earlier, worse checkpoint.

The ep119+ep131 average is essentially neutral because ep119 (AP=0.2952) and ep131 are close
enough that averaging adds minimal noise and cancels out very little.

**When SWA would work here:** Apply SWA after the model plateaus — when 10 consecutive epochs
show gain < 0.001. Take the last 5 checkpoints from that plateau. Expected gain: +0.003–0.008 AP.
The ms1280_mosaic run has a planned phase 2 (epoch 80–95) which runs at fixed resolution with no
augmentation — that is exactly the kind of fine-grained convergence phase where SWA should work.

---

## 5. NMS Threshold Tuning (not yet tested — from SOTA analysis)

DroneScan-YOLO (2026, same AP range) found +0.010 AP from tuning:
- `conf_threshold`: 0.5 (default) → 0.01
- `iou_threshold`: 0.45 (default) → 0.40

The rationale: in dense VisDrone scenes, many legitimate detections have confidence 0.05–0.20
(partially occluded, tiny objects). Default threshold 0.5 discards these. Lowering to 0.01
recovers recall without much precision penalty because VisDrone images contain many real objects
at every location.

Lowering IoU threshold from 0.45 → 0.40 is less obvious but matters in dense grids: legitimately
adjacent objects (bikes in a rack, cars in a row) can have true IoU of 0.35–0.42. Standard NMS
suppresses them; relaxed NMS keeps them.

**This should be tested next** — it's free (no retraining, 5 minutes to eval) and potentially
+0.5–1.0 AP.

**Script to implement:**
```python
# In postprocessor call or eval script:
model.postprocessor.num_top_queries = 300  # also increase from 100
model.postprocessor.conf_threshold = 0.01
# NMS iou threshold is in postprocessor config
```

---

## Why None of the Inference-Time Tricks Work on This Model (General Principle)

The pattern across all experiments is the same: **every inference-time trick tries to add
something the training process already internalized**.

| Trick | Assumes model hasn't learned... | This model has learned it via... |
|-------|--------------------------------|-----------------------------------|
| Higher eval resolution | ...to detect at high res | multi-scale [768–1280] training |
| SAHI slicing | ...to detect tiny objects | copy-paste aug + high res training |
| TTA scale | ...scale invariance | multi-scale training |
| TTA hflip | ...flip invariance | `RandomHorizontalFlip` every epoch |
| SWA | ...converged to basin centre | model wasn't converged yet |
| WBF | ...precise localization | FDR distribution heads are well-calibrated |

The general lesson: inference-time tricks add value proportional to the gap between
what the model learned and what those tricks provide. With 130+ epochs of multi-scale,
multi-augmentation training, that gap is very small. The only remaining lever is retraining
with signal the model hasn't seen — new augmentation diversity (mosaic), new data, or
architectural changes (P2 head, NWD loss).

---

## What to Try Next (Ranked by Expected Value / Effort)

| Experiment | Expected AP gain | Effort | Notes |
|-----------|-----------------|--------|-------|
| NMS threshold tuning (conf=0.01, iou=0.40) | +0.5–1.0 | 5 min | Free, try immediately |
| Post-plateau SWA after mosaic phase 2 | +0.003–0.008 | 30 min | After phase 2 converges |
| Mosaic p=1.0 (vs current p=0.5) | minor | 1 run | DroneScan used p=1.0; may help or hurt |
| Size-adaptive loss weighting in criterion | +0.5–1.5 | medium | Modify dfine_criterion.py |
| P2 detection head | +1–3 | high | Architecture change to HybridEncoder + decoder |
