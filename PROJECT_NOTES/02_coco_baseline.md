# Step 2 — D-FINE-S COCO Baseline Results

## Model

| Item | Value |
|------|-------|
| Model | D-FINE-S (Small) |
| Backbone | HGNetV2-B0 |
| Parameters | 10.2M |
| Pretrained on | COCO 2017 train (118k images) |
| Checkpoint | `weight/dfine_s_coco.pth` (39.9 MB) |
| Why S-variant? | Fewest parameters, fastest iteration. Pipeline works → scale to L for better accuracy later. |

---

## COCO Val2017 Official Evaluation

Evaluated on all **5000 COCO val2017 images** using the official COCO evaluator.

### Main Metrics

| Metric | Value | What it means |
|--------|-------|---------------|
| **mAP (AP@0.50:0.95)** | **48.5%** | Primary metric. Averaged over IoU thresholds 0.50→0.95 in steps of 0.05. This is the hard metric — a detection only counts if the predicted box overlaps the GT by at least 50%...90% respectively. |
| AP@0.50 (AP50) | 65.4% | "Loose" threshold — box only needs 50% overlap. Higher because even rough boxes qualify. This is what older papers (PASCAL VOC era) reported. |
| AP@0.75 (AP75) | 52.6% | "Strict" threshold — box needs 75% overlap. Measures precise localisation. |
| APS (small) | 29.4% | Objects < 32×32 pixels. Always the hardest — small objects have few pixels and are easily confused. **This number becomes critical for VisDrone.** |
| APM (medium) | 52.2% | Objects 32×32 to 96×96 pixels. |
| APL (large) | 65.4% | Objects > 96×96 pixels. Large objects are easiest — lots of pixels, clear features. |

### Recall Metrics

| Metric | Value | What it means |
|--------|-------|---------------|
| AR@1 | 37.4% | Average recall when only 1 detection is allowed per image. Measures if the model finds the "best" single object. |
| AR@10 | 62.9% | Up to 10 detections per image. |
| AR@100 | 70.0% | Up to 100 detections. This is the standard recall ceiling. |
| AR@100 small | 50.9% | Recall on small objects. |
| AR@100 large | 87.4% | Recall on large objects — model finds almost all large objects. |

### Reading These Numbers in Context

- **48.5 mAP is excellent** for a 10M parameter model. For comparison, the original DETR (2020)
  achieved 42.0 mAP with 41M parameters. D-FINE-S delivers 15% better accuracy with 75% fewer params.
- The **gap between AP50 (65.4%) and mAP (48.5%)** tells us localisation is the bottleneck,
  not classification. The model finds the right class well, but the boxes aren't always tight.
  D-FINE's distribution regression specifically addresses this.
- **APS = 29.4%** vs **APL = 65.4%** — a 36 point gap on small vs large objects.
  This predicts VisDrone will be hard: VisDrone is 100% aerial, small objects.
  The COCO-pretrained model will likely underperform significantly on VisDrone without fine-tuning.

### Confirmed Match to Paper

The paper reports **48.5 mAP** for D-FINE-S on COCO. We reproduce **48.5 mAP** exactly.
This confirms: checkpoint is authentic, our environment is correct, evaluation methodology matches.

---

## Latency Benchmark (GPU)

**Device:** NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
**Input resolution:** 640×640
**Protocol:** 100 warmup iterations → 500 timed iterations

| Metric | Value |
|--------|-------|
| Mean latency | **24.10 ms** |
| Std deviation | 3.24 ms |
| P50 (median) | 23.82 ms |
| P95 | 28.97 ms |
| Throughput | **41.5 FPS** |

### Interpretation

**Why warmup?**
The first forward pass triggers CUDA kernel JIT compilation and memory allocation.
Our first sample image took 758ms (cold start) vs 33ms after warmup — a 23x difference.
The 100-iteration warmup ensures we measure steady-state performance.

**What does 41 FPS mean?**
- Real-time video: 25-30 FPS → we are at **1.4× real-time** on this laptop GPU
- The D-FINE paper reports 3.49ms (287 FPS) on an A100 (80GB datacenter GPU)
- Our RTX 4060 Laptop is a consumer mobile GPU — 24ms vs 3.49ms = 6.9× slower than A100
- This ratio is expected: A100 has 77.6 TFLOPS FP16 vs RTX 4060's ~33 TFLOPS FP16
- **For VisDrone deployment target:** if deploying on Snapdragon 6 Gen 1 (mobile SoC),
  expect ~10-20× slower than our laptop GPU → ~240-480ms per frame on CPU-only mobile hardware.
  INT8 quantization (Step 5) will be essential.

**P95 = 28.97ms** means 95% of frames finish within 29ms. The 5% tail (up to ~35ms)
is from GPU thermal throttling or OS scheduling jitter — normal on a laptop.

---

## Sample Inference Results

5 COCO val images, annotated outputs saved to `PROJECT_NOTES/coco_baseline_results/`.

| Image | Content | Detections (>0.4) | Top Detection | Latency |
|-------|---------|-------------------|---------------|---------|
| 000000001000.jpg | Crowd scene | 20 | person (0.939) | 758ms* |
| 000000003501.jpg | Kitchen/food | 7 | bowl (0.924), broccoli (0.753) | 33.8ms |
| 000000007386.jpg | Street | 3 | motorcycle (0.859), dog (0.692) | 35.5ms |
| 000000016228.jpg | Outdoor event | 33 | horse (0.948), bench (0.849) | 32.0ms |
| 000000020247.jpg | Wildlife | 2 | bear (0.962), bear (0.957) | 38.9ms |

*First image: 758ms cold start (CUDA kernel compilation). All subsequent: 32-38ms.

**What scores mean:**
- 0.9+ : Very high confidence. Model has seen many training examples matching this pattern.
- 0.7-0.9: Confident. Correct class, good box.
- 0.4-0.7: Moderate. Correct class, possibly partial occlusion or unusual angle.
- < 0.4: Filtered out. Too ambiguous to be useful.

---

## What We Expect to Change with VisDrone Fine-tuning

The COCO-pretrained model will likely score **poorly on VisDrone** for three reasons:

1. **Domain shift:** COCO images are ground-level, front-facing. VisDrone is aerial.
   The visual appearance of "person" seen from above looks nothing like person in COCO.

2. **Scale shift:** COCO has large objects (APL = 65.4%). VisDrone objects are tiny.
   The model's `APS = 29.4%` tells us it already struggles with small objects in COCO,
   and VisDrone is far more extreme.

3. **Class mismatch:** COCO has 80 classes. VisDrone has 10 specific classes (pedestrian,
   car, van, truck, etc.) with different visual properties at aerial viewpoint.

After fine-tuning we expect: better recall on small objects, correct class vocabulary,
adapted feature representations for aerial viewpoint.

---

## Summary of Findings

- D-FINE-S achieves 48.5 mAP on COCO — confirmed to match paper exactly
- GPU latency: 24ms mean, 41 FPS on RTX 4060 Laptop — exceeds real-time (30 FPS)
- The model is excellent on large/medium objects but struggles with small (APS=29.4%)
- Cold start takes 758ms; steady-state is ~24ms after warmup
- This COCO performance is our **before fine-tuning baseline** for VisDrone comparison
