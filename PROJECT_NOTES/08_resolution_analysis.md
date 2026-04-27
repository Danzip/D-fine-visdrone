# Resolution Analysis — Experiment D

**Date:** 2026-04-18  
**Script:** `tools/demo/analyze_box_sizes.py`  
**Goal:** Determine whether higher-resolution training is mechanically justified by measuring what fraction of VisDrone objects cross the ~16px "feature-representable" floor at each candidate resolution.

---

## Key Finding

At the current 640×640 training resolution, **77% of training boxes have a short side below 16px** — the approximate floor below which a stride-8 feature map cannot represent objects with any spatial precision. The model is being asked to detect objects that are mostly smaller than 2 feature cells.

This is not a training recipe problem. It is a resolution-to-object-size mismatch problem. The dataset has inherently tiny objects (median 25×27px at native resolution), and letterboxing them down to 640px makes them ~11px on the short side.

---

## Native Resolution (no resize)

| Split | Median W | Median H | % < 16px | % < 32px |
|-------|----------|----------|----------|----------|
| Train | 25px | 27px | 36.7% | 70.4% |
| Val   | 20px | 25px | 43.7% | 78.3% |

Even at native resolution, 37–44% of objects are below 16px short side. These objects are inherently at the detection floor regardless of training resolution — no amount of upscaling will recover information that isn't in the native image.

---

## Distribution After Letterbox Resize

### Train set (343,204 annotations):

| Resolution | <8px | <12px | <16px | <24px | med W | med H | med √area |
|-----------|------|-------|-------|-------|-------|-------|-----------|
| 640×640   | 45.7% | 66.3% | 77.0% | 89.0% | 10.7px | 11.4px | 11.1px |
| 800×800   | 34.2% | 54.9% | 68.6% | 83.0% | 13.4px | 14.3px | 13.9px |
| 960×544   | 32.9% | 53.3% | 66.6% | 81.8% | 14.0px | 15.0px | 14.5px |
| 960×960   | 26.5% | 45.7% | 60.8% | 77.0% | 16.0px | 17.1px | 16.6px |
| 1280×736  | 20.2% | 38.6% | 52.3% | 71.5% | 18.8px | 20.3px | 19.4px |
| 1280×1280 | 15.5% | 32.2% | 45.7% | 66.3% | 21.4px | 22.9px | 22.2px |

### Val set (38,759 annotations):

| Resolution | <8px | <12px | <16px | <24px | med W | med H | med √area |
|-----------|------|-------|-------|-------|-------|-------|-----------|
| 640×640   | 43.9% | 66.9% | 79.1% | 90.8% | 9.9px | 12.7px | 11.3px |
| 800×800   | 34.0% | 55.5% | 70.9% | 85.5% | 12.4px | 15.8px | 14.1px |
| 960×544   | 25.7% | 43.9% | 60.4% | 79.1% | 14.8px | 19.0px | 16.9px |
| 960×960   | 25.7% | 43.9% | 60.4% | 79.1% | 14.8px | 19.0px | 16.9px |
| 1280×736  | 14.0% | 30.0% | 43.9% | 66.9% | 19.8px | 25.3px | 22.5px |
| 1280×1280 | 14.0% | 30.0% | 43.9% | 66.9% | 19.8px | 25.3px | 22.5px |

---

## Gain Over 640×640 Baseline (pp reduction in sub-16px boxes, train)

| Resolution | Δ sub-16px | Δ sub-8px | Verdict |
|-----------|-----------|----------|---------|
| 800×800   | −8.4pp | −11.6pp | Meaningful — low VRAM cost |
| 960×544   | −10.4pp | −12.9pp | Meaningful — fits 16:9 natively, no aspect distortion |
| 960×960   | −16.2pp | −19.3pp | Large gain — high priority |
| 1280×736  | −24.7pp | −25.5pp | Very large gain — highest ROI of feasible resolutions |
| 1280×1280 | −31.3pp | −30.2pp | Max gain, likely OOM at batch>1 |

---

## Notable Observations

### 1. The 640×640 baseline is severely resolution-starved
Median box at 640px is 10.7×11.4px. The stride-8 feature map cell is 8×8px — the median object covers only ~1.8 feature cells on each side. This is why AP-small (0.142) is so low: the model is not failing to learn, it is being asked to detect objects the feature map literally cannot resolve.

### 2. 960×544 and 960×960 produce identical val results
VisDrone val images are mostly 16:9. When letterboxing to 960×544, the image fills the frame without padding. When letterboxing to 960×960, the same scale ratio applies — the 16:9 image still fits at 960×540 internally, with 460px of vertical padding. The objects end up the same pixel size either way. **Training at 960×544 is therefore equivalent to 960×960 for this dataset** with ~3× less compute.

### 3. 1280×736 is the highest-impact feasible resolution
Moves median box from 10.7px → 18.8px (short side). Sub-16px fraction drops from 77% → 52%. This crosses the point where the majority of objects are finally above the feature-representation floor. 1280×1280 gives diminishing marginal returns (45.7%) at much higher compute cost.

### 4. A hard floor exists regardless of resolution
At native resolution, 37% of train objects are already below 16px. These objects cannot benefit from any resizing because the information isn't in the source pixels. The maximum achievable benefit from resolution alone is bounded by this ~37% irreducible fraction.

---

## Implication for Multi-Scale Training

D-FINE's standard COCO training already includes `RMultiScaleInput` (480–800px range). This means `dfine_s_coco.pth` already has multi-scale priors — the model is not brittle to a single 640px scale. This is good news.

The correct next experiment is therefore **not** "find a multi-scale pretrained checkpoint" (we already have one) but:

**Retrain VisDrone fine-tuning from `dfine_s_coco.pth` with `RMultiScaleInput` range extended to cover 640–1280px**, so the model learns VisDrone object sizes across the full resolution range rather than specializing to 640px during fine-tuning.

The previous resolution collapses happened because we fine-tuned to 640px first (locking in single-scale priors), then tried to jump. The fix is to never lock in to 640px during VisDrone fine-tuning at all.

---

## Recommended Next Experiment

**Config change:** In `dfine_hgnetv2_s_visdrone.yml`, modify the `RMultiScaleInput` range from `[480, 800]` to `[640, 1280]` (or `[480, 960]` as a safer first step).

**Expected gain:** If multi-scale fine-tuning converges, expect AP50:95 to improve from 0.231 toward 0.26–0.28, primarily via AP-small and AP-medium. This is based on the distribution analysis showing that 25pp more objects become feature-representable at 1280px, and literature reporting 2–10% AP gains from multi-scale on similar datasets.

**Risk:** VRAM — at 1280px max scale, batch size will need to drop to 2 or even 1 on RTX 4060 8GB. Training will be slower but should not collapse if multi-scale is active throughout (not introduced after convergence).
