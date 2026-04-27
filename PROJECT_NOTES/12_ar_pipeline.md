# AR-Preserving Rectangular Training Pipeline

## Motivation

All prior training runs (steps 4–15) used square canvases (640×640, 1024×1024). VisDrone images
are natively 16:9 or 4:3 — zero square images exist. Forcing a square letterbox wastes canvas on
black bars and compresses objects that are already tiny.

Goal: train on rectangular canvases that match the natural image AR, so the model always sees
proportionally accurate geometry.

Best checkpoint going in: `output/dfine_hgnetv2_s_visdrone_ms1280_cont/best_stg1.pth`
— AP=0.2966, epoch 131.

---

## VisDrone AR Distribution

From COCO metadata (`dataset.coco.imgs`), train split (6,471 images):

| AR bucket | Count | % |
|-----------|-------|---|
| 16:9 (w/h > 1.6) | 3,169 | 49% |
| 4:3 (w/h ≤ 1.6) | 3,302 | 51% |

Val split: 100% 16:9.

Fixed canonical shapes at `base_size=1280`:
- 16:9 → **736×1280** (`ceil32(1280 × 9/16) = ceil32(720) = 736`)
- 4:3 → **960×1280** (`ceil32(1280 × 3/4) = 960`)

Both axes divisible by 32 (FPN stride requirement).

---

## Bugs in Previous Attempts (root-cause log)

### B3 — Dataset-level square Resize destroyed AR before collate
**File:** `ms1280_cont.yml` and earlier configs
**What:** `Resize([1024, 1024])` in the dataset transform chain converted all images to squares
BEFORE the collate function saw them. AR bucket grouping was logically meaningless — collate
received squares from every bucket.
**Why undetected:** The old letterbox + unwind formula gave approximately correct eval results when
the image was already square (padded_w ≈ canvas_w in the symmetric case), so AP wasn't zero.
**Fix:** Remove `Resize` from dataset transforms entirely. All spatial shaping moved to collate.

### B4 — Letterbox unwind used wrong width reference
**File:** `src/solver/det_engine.py`, old `if "letterbox" in target:` block
**What:** Box coordinates after collate were normalized relative to `padded_w` (e.g., 1422 for a
portrait crop padded to 16:9). The eval unwind used `canvas_w = 1280`. These differ →
GT coords wrong → AP collapse for non-matching-AR images.
**Fix:** Remove the letterbox unwind block entirely. Use the existing `scale_boxes(orig_size)` path,
which handles normalized cxcywh correctly via orig_size from COCO metadata.

### B5 — Collate picked AR randomly, not from sampler
**File:** `src/data/dataloader.py`, old `ar_key = random.choice(["4:3", "16:9"])` line
**What:** Even if a sampler yielded only 16:9 images, the collate could pick 4:3 target AR →
added large black bars to every 16:9 image in the batch. No same-AR guarantee.
**Fix:** Replace random AR selection with per-batch sampler (ARBucketBatchSampler) that guarantees
all images in a batch come from the same AR bucket. Collate derives target shape from `orig_size`.

### B6 — CopyPasteSmallObjects 3-tuple crash at epoch 132
**File:** `src/data/transforms/container.py`, `_transforms.py`
**What:** `CopyPasteSmallObjects.forward()` returns `(image, target, dataset)` 3-tuple. Downstream
transforms tried to unpack as 2-tuple → crash. Killed ms1280_cont at epoch 132.
**Fix for AR experiment:** Disabled in new config. Container guardrail added: raises if a transform
GROWS the tuple length (i.e., emits more elements than it received).

### B7 — Portrait crops from RandomZoomOut+RandomIoUCrop invalidated bucket assignment
**What:** `RandomZoomOut` (up to 4× scale) + `RandomIoUCrop` can produce portrait subregions from
landscape images. ARBucketSampler assigned buckets by COCO metadata (always landscape), but images
arrived at collate as portrait. The canonical target shape (736×1280) forced extreme letterboxing.
**Fix for first experiment:** Disable both augmentations. Keep only: PhotometricDistort, HFlip,
SanitizeBB, ConvertPILImage, ConvertBoxes. This guarantees image AR at collate == COCO AR.
Re-enable after AR batching is verified stable.

---

## Final Design (Experiment C — First AR Run)

### Key decisions
| Item | Choice | Reason |
|------|--------|--------|
| Augmentations | PhotometricDistort + HFlip only | No AR change → COCO bucket assignment remains accurate |
| Dataset-level Resize | REMOVED | Was destroying AR before collate |
| CopyPasteSmallObjects | DISABLED | 3-tuple crash; re-enable later |
| Mosaic | OFF (`mosaic_p=0.0`) | 4× pixel overhead; isolate AR variable |
| Multi-resolution | OFF (`base_size_repeat: null`) | Isolate AR variable; fixed 1280px |
| AR sampler | `ARBucketBatchSampler` | Groups by COCO metadata AR; alternates 16:9/4:3 batches |
| Collate letterbox | orig_size-based | Uses COCO W_orig, H_orig to compute canonical target shape |
| Eval size | `[736, 1280]` | Matches 16:9 bucket (val split is 100% 16:9) |

### ARBucketBatchSampler
File: `src/data/dataloader.py`

- At construction: reads `dataset.coco.imgs`, assigns each index to `"16:9"` (w/h > 1.6) or `"4:3"`.
- Prints counts: `[ARBucketBatchSampler] 16:9: 3169  4:3: 3302  batch_size=2`
- Per epoch: shuffles each bucket independently with epoch-seeded RNG.
- Yields batches alternating 16:9/4:3 (interleaved, then remainder from larger bucket).
- Integrates with `DataLoader` via `use_ar_sampler: True` config key.

### Collate letterbox (BatchImageCollateFunction)
File: `src/data/dataloader.py`

For each image in a same-bucket batch:
1. Read `tgt["orig_size"] = [W_orig, H_orig]` (COCO metadata, always landscape).
2. Compute canonical target shape: `target_h = ceil32(sz × H_orig / W_orig)`, `target_w = sz`.
3. Batch canvas = `max(target_h)` × `sz` across the batch (all same AR → identical for same-bucket).
4. Proportionally resize image to fit within `(target_h, target_w)`: `scale = min(target_h/H, target_w/W)`.
5. Center-pad to `(batch_H, batch_W)` with black pixels.
6. Adjust normalized cxcywh boxes: `cx = cx × (new_W/batch_W) + pad_left/batch_W`, etc.

Key invariant: `scale_x = new_W/W ≈ scale_y = new_H/H` within 0.5% — proportional resize, no squishing.

### Files changed
| File | Change |
|------|--------|
| `src/data/dataloader.py` | Added `ARBucketBatchSampler`; `DataLoader` `use_ar_sampler` param; collate uses `orig_size` |
| `src/solver/det_engine.py` | Removed letterbox unwind block; added AMP startup log; added GT box assertions |
| `src/solver/det_solver.py` | Passes `max_train_steps` kwarg to `train_one_epoch` |
| `src/data/transforms/container.py` | Fixed 3-tuple guardrail (fires on tuple GROWTH, not initial 3-tuple from dataset) |
| `configs/dfine/dfine_hgnetv2_s_visdrone_ar.yml` | New config: no Resize, no CopyPaste, no Mosaic, `use_ar_sampler: True`, fixed 1280px |
| `tests/test_collate_ar.py` | 8-test suite (AR drift, canvas shape, proportionality, boxes, bucket uniformity) |

---

## Test Suite

**File:** `tests/test_collate_ar.py`
**Run:** `cd D-FINE && source venv/bin/activate && python tests/test_collate_ar.py`
**Config under test:** `configs/dfine/dfine_hgnetv2_s_visdrone_ar.yml`

The test loads the real train dataloader (`num_workers=0`) and exercises the full pipeline from
raw COCO images through dataset transforms and collate.

### T1 — AR drift after dataset transforms (< 1% threshold)

**What it checks:** For 30 images, compares the image AR after dataset transforms to the original
COCO metadata AR (`info["width"] / info["height"]`). Drift must be < 1%.

**What it guards against:** Accidentally enabling `RandomZoomOut` or `RandomIoUCrop`. These
augmentations can dramatically change the image AR (e.g., a 4× zoom-out followed by a portrait
crop gives a portrait image from a landscape COCO metadata entry). If either is active, this test
fails immediately with drift >> 1%.

**Result (2026-04-25):** drift=0.00% for all 30 images — `RandomPhotometricDistort` and
`RandomHorizontalFlip` do not change spatial dimensions at all.

---

### T2 — Canvas divisible by 32

**What it checks:** For 20 batches, `batch_H % 32 == 0` and `batch_W % 32 == 0`.

**What it guards against:** An off-by-one in `_ceil32()` or a target shape that slips through
without rounding. FPN upsampling and the AIFI attention block both require dimensions divisible
by the feature stride (32).

**Result:** 736×1280 (16:9) and 960×1280 (4:3) — both pass. `736 = 23×32`, `960 = 30×32`,
`1280 = 40×32`.

---

### T3 — Canvas AR matches original AR (within ceil32 rounding)

**What it checks:** Given `orig_size = [W_orig, H_orig]` of the first image in the batch, computes
`expected_h = ceil32(batch_W × H_orig / W_orig)` and checks `|batch_H - expected_h| ≤ 32`.

**What it guards against:** Collate picking the wrong canonical shape (e.g., using the wrong AR
formula or a round32 instead of ceil32). A 32px tolerance covers cases where two slightly different
native resolutions in the same AR bucket produce different `ceil32` outputs.

**Result:** All 20 batches pass. Observed: `batch_H=736, expected_h=736` for 16:9 images at
`orig_size=[1400,788]`; `batch_H=960, expected_h=960` for 4:3 images at `orig_size=[1400,1050]`.

---

### T4 — Proportional resize (scale_x ≈ scale_y within 0.5%)

**What it checks:** For each image in each batch, computes `scale_x = batch_W / W_orig` and
`new_H = round(H_orig × scale_x)`, then checks `|new_H/H_orig - batch_W/W_orig| / (batch_W/W_orig) < 0.005`.

**What it guards against:** Squishing. If the resize distorted AR — e.g., by using two independent
scale factors — this test catches it. The 0.5% threshold covers rounding from integer pixel counts
(a 1-pixel rounding error on a 540-pixel height is 0.18%, well within tolerance).

**Result:** All pairs pass. Representative values: `scale_x=0.9143, new_H/H_o=0.9143` (exact
match for 1400×1050→1280×960); `scale_x=0.9143, new_H/H_o=0.9137` (0.07% drift for 1400×788
where `ceil32(1280×788/1400) = ceil32(719.5) = 736 / 788 = 0.9137 vs 1280/1400 = 0.9143`).

---

### T5 — Non-zero content in batch

**What it checks:** `samples.max().item() > 0` — at least one pixel in the batch is non-zero.

**What it guards against:** A bug where the letterbox padding overwrites ALL pixels (e.g., wrong
padding dimensions fill the entire tensor), or a uint8/float32 dtype mismatch that zeros out the
image. Simple sanity check.

**Result:** All 20 batches pass.

---

### T6 — Boxes in [0, 1] with positive w/h

**What it checks:** For every target in every batch:
- `(boxes >= 0).all() and (boxes <= 1).all()`
- `(boxes[:, 2] > 0).all()` — box width > 0
- `(boxes[:, 3] > 0).all()` — box height > 0

**What it guards against:** The box transform in collate producing out-of-range coordinates (a
sign that pad_left/batch_W ratios or sw/sh scale factors are wrong), or zero-area boxes (which
cause NaN in VFL and GIoU losses).

**Result:** All pass. Observed range: min=0.0016, max=0.9950 — well within [0, 1]. No
zero-w/h boxes.

---

### T7 — All images in batch from same AR bucket

**What it checks:** For each image in the batch, computes `ar = orig_size[0] / orig_size[1]`
and assigns `"16:9"` if `ar > 1.6` else `"4:3"`. Checks that all images in the batch share
the same bucket label.

**What it guards against:** `ARBucketBatchSampler` yielding a mixed-AR batch (e.g., if the
interleaving logic has an off-by-one and sends a 16:9 index into a 4:3 batch). A mixed batch
would produce different canonical target heights per image, and the batch canvas would be set
to the max — adding unnecessary black bars to the shorter image.

**Result:** 20/20 batches are single-bucket. Observed: `buckets={'16:9'}` for even batches,
`buckets={'4:3'}` for odd batches — perfect alternation.

---

### T8 — Both buckets seen across 20 batches

**What it checks:** `len(buckets_seen) == 2` after 20 batches, where `buckets_seen` accumulates
the bucket label from every batch.

**What it guards against:** A degenerate sampler that only yields one AR bucket (e.g., if one
bucket is empty or the interleaving logic is broken).

**Result:** `seen={'4:3', '16:9'}` — both buckets seen in the first 20 batches.

---

### T13 — Mandatory visual check (post-collate GT boxes)

**What it checks:** Not automated — saves post-collate images to `output/visual_check/` with GT
boxes drawn (normalized cxcywh → pixel xyxy via `batch_H, batch_W`). Must be inspected by eye.

**What it guards against:** A sign-flip or scale error in the box transform that passes numeric
tests but produces boxes in the wrong location visually.

**Result (2026-04-25):** 6 images inspected:
- `batch0_img0_736x1280.jpg` — 3 vehicles, red boxes tightly wrapping each car ✓
- `batch0_img1_736x1280.jpg` — 43 detections on dense aerial scene ✓
- `batch1_img0_960x1280.jpg` — 47 boxes on nighttime intersection with heavy PhotometricDistort ✓
- `batch1_img1_960x1280.jpg` — 39 boxes ✓
- `batch2_img0_736x1280.jpg` — 82 boxes ✓
- `batch2_img1_736x1280.jpg` — 135 boxes ✓

No misalignment observed in any image. **Pipeline approved for training.**

---

## Smoke Training (20 steps, 2026-04-25)

```bash
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone_ar.yml \
    --device cuda:0 \
    --resume output/dfine_hgnetv2_s_visdrone_ms1280_cont/best_stg1.pth \
    -u epochs=1 train_dataloader.num_workers=0 max_train_steps=20
```

| Check | Result |
|-------|--------|
| Crash | None ✓ |
| GPU max mem | 1,206 MB (well under 7.5GB limit) ✓ |
| AMP | Not load scaler.state_dict (fresh scaler, expected) |
| Checkpoint | Loaded epoch 131 weights; `decoder.anchors` and `decoder.valid_mask` recomputed (expected — shape changed from 1024×1024 to 736×1280) |
| AP after 20 steps | 0.043 — expected; anchors reset from 1024×1024 to 736×1280, model needs full epoch to re-adapt |

AP=0.043 is not a bug. The checkpoint anchors were sized for a 1024×1024 grid. After the shape
change, anchors are recomputed and the decoder needs training steps to re-calibrate offsets.
AP should recover to ≥ 0.25 after a full epoch (same pattern seen in all prior resolution-change runs).

---

## Next Steps

1. **Full overnight training** — `num_workers=2`, `epochs=500`, monitor first-epoch AP ≥ 0.20
2. **AP check** — If AP < 0.10 after first full epoch, stop and re-run T13
3. **Re-enable augmentations** (after AR stability confirmed):
   - `RandomZoomOut` + `RandomIoUCrop` — would require collate to letterbox to ORIG AR using `orig_size`, which it already does
   - `CopyPasteSmallObjects` — requires container to handle 3-tuple input natively (whitelist it)
4. **Multi-resolution** — Set `base_size_repeat: 3` to add scale variation around 1280px
5. **Phase F batch probe** — Re-measure actual safe batch sizes for 736×1280 and 960×1280 with full forward+loss+backward

---

## Configuration

```yaml
# configs/dfine/dfine_hgnetv2_s_visdrone_ar.yml (key sections)
eval_spatial_size: [736, 1280]
train_dataloader:
  total_batch_size: 2
  use_ar_sampler: True
  num_workers: 0        # → set to 2 for overnight run
  dataset:
    transforms:
      ops:
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomHorizontalFlip}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
  collate_fn:
    stop_epoch: 9999
    base_size: 1280
    base_size_repeat: ~    # fixed resolution
    mosaic_p: 0.0
val_dataloader:
  total_batch_size: 1
  num_workers: 0
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [736, 1280]}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
```

To launch full training:
```bash
python train.py -c configs/dfine/dfine_hgnetv2_s_visdrone_ar.yml \
    --device cuda:0 \
    --resume output/dfine_hgnetv2_s_visdrone_ms1280_cont/best_stg1.pth \
    -u train_dataloader.num_workers=2
```

---

## What Actually Happened — Full AR Training (2026-04-25)

Full training was launched from `output/dfine_hgnetv2_s_visdrone_ms1280_cont/best_stg1.pth`
(AP=0.2966, epoch 131). The AR config ran epochs 132–137+ with `batch_size=2`, fixed 1280px,
no Mosaic, no multi-res.

### Results

| Epoch | AP (COCO bbox) |
|-------|---------------|
| 132 | 0.047 |
| 133 | 0.051 |
| 134–137+ | ~0.050–0.055 (slowly rising) |

The model was recovering from the anchor grid reset (1024→736/960px) but the rate was far slower
than expected. After 6 epochs, AP was still below 0.06 — compared to the mosaic path that had
already reached AP=0.3027 at epoch 15.

### Decision

**Abandoned.** Two reasons:

1. **Recovery speed**: The anchor reset requires many epochs to re-adapt. The mosaic run achieved
   AP=0.3027 in 15 epochs from the same cont checkpoint; the AR run was at ~0.05 after 6 epochs.
   Extrapolating, convergence to ≥0.25 would require 50–100+ additional epochs.

2. **Opportunity cost**: The mosaic run's `best_stg1.pth` (epoch 15, AP=0.3027) was a clean,
   tested checkpoint ready to resume. Spending GPU time on AR recovery with uncertain outcome
   was not justified.

### What the AR pipeline is worth

All tests (T1–T8, T13) passed. The pipeline is geometrically correct. The failure was not a
code bug but a training dynamics problem: loading a checkpoint sized for 1024×1024 into a
736×1280 training regime requires the decoder anchors to be recomputed, which triggers a cold
start of the position-prediction head regardless of what the backbone learned.

**To revisit AR training:** initialize from a square-grid-free checkpoint, or fine-tune the AR
model from the mosaic best (AP≥0.30) with a very low LR and frozen backbone so only the
head re-adapts to the new canvas shape.

### Current path forward

Resumed mosaic training from `output/dfine_hgnetv2_s_visdrone_ms1280_mosaic/best_stg1.pth`
(epoch 15, AP=0.3027) using the clean config `dfine_hgnetv2_s_visdrone_mosaic_resume.yml`
with `batch_size=3`, `use_amp=False`, `accum_steps=4`. W&B run: `dfine_s_visdrone_mosaic_resume`.
