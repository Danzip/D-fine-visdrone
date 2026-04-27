# Step 3 — VisDrone Dataset

## Dataset Overview

VisDrone2019-DET is an object detection benchmark collected from drone (UAV) footage.
Unlike COCO (ground-level, varied scenes), every VisDrone image is an aerial top-down view.
This creates a genuine domain gap that makes COCO-pretrained models struggle.

| Property | VisDrone | COCO (for comparison) |
|----------|----------|----------------------|
| Viewpoint | Aerial / top-down | Ground level |
| Typical object size | 10-60px | 100-400px |
| Avg objects/image | 53-70 | ~7 |
| Classes | 10 (vehicles + pedestrians) | 80 (diverse) |
| Image resolution | 1360x765 to 2000x1500 | 640x480 typical |

---

## Dataset Statistics

### Training Set

| Metric | Value |
|--------|-------|
| Images | 6,471 |
| Total annotations | 343,204 |
| Avg objects per image | **53.0** |
| Boxes under 32px max-side | **46.3%** |
| Boxes under 64px max-side | **78.0%** |
| Median box area (% of image) | **0.046%** |

### Validation Set

| Metric | Value |
|--------|-------|
| Images | 548 |
| Total annotations | 38,759 |
| Avg objects per image | **70.7** |
| Boxes under 32px max-side | **53.1%** |
| Boxes under 64px max-side | **85.7%** |
| Median box area (% of image) | **0.055%** |

---

## Class Distribution (Training)

| Class | Count | % of total | Notes |
|-------|-------|------------|-------|
| car | 144,866 | 42.2% | Dominant class — vehicles seen from above |
| pedestrian | 79,337 | 23.1% | Individual walking people |
| motor | 29,647 | 8.6% | Motorcycles |
| people | 27,059 | 7.9% | Crowds / groups (vs individual pedestrian) |
| van | 24,956 | 7.3% | Larger passenger vehicle |
| truck | 12,875 | 3.8% | Heavy goods vehicle |
| bicycle | 10,480 | 3.1% | Pedal cycles |
| bus | 5,926 | 1.7% | Large bus |
| tricycle | 4,812 | 1.4% | Three-wheeled vehicle |
| awning-tricycle | 3,246 | 0.9% | Covered tricycle / tuk-tuk |

**Class imbalance:** car (42%) is 45x more frequent than awning-tricycle (0.9%).
This is important for training — the model will naturally bias toward car.
The VisFocal loss (VFL) handles this somewhat by weighting by IoU quality.

---

## Why VisDrone is Hard — Key Challenges

### 1. Object Density

VisDrone has 53-70 objects per image vs ~7 in COCO. This is extreme.
A single 1400x1050 image may have 245 annotated objects (seen in our samples).
The transformer decoder uses 300 queries — scenes with 245 objects mean almost every
query needs to match something. False negatives are inevitable.

### 2. Tiny Objects

**46-53% of boxes have max-side under 32px.** In COCO, the "small" category
(objects < 32x32 pixels) only makes up 15% of annotations, and D-FINE-S achieves
APS=29.4% on those. For VisDrone, the situation is worse:

- Objects are smaller relative to image size
- Objects are from aerial viewpoint — features differ from COCO ground-level
- Car seen from directly above looks very different from car in COCO side/front view

A 10×10 pixel car contains only 100 pixels. The model has almost no texture
information to work with — it must rely on shape and local context.

### 3. Domain Gap from COCO Pretraining

The COCO model learned features for:
- Pedestrians: front-facing, full-body, ~200-500px tall
- Cars: side-view or front-view, clearly recognizable shape

In VisDrone:
- Pedestrians: top-down blob, 15-25px, no face/body ratio visible
- Cars: rectangle seen from directly above, no windshield/wheels visible

The convolutional backbone's learned filters are optimised for ground-level features.
Fine-tuning is needed to adapt them.

### 4. Class Distinction Ambiguity

"pedestrian" vs "people" is ambiguous — both are humans walking.
The distinction is: "people" = group/crowd, "pedestrian" = individual.
From aerial view, distinguishing a tightly-packed crowd from individuals requires
fine-grained spatial reasoning.

Similarly "tricycle" vs "awning-tricycle" differ only by the presence of a covering,
which may not be visible from all altitudes.

---

## Annotation Format

**VisDrone format** (per-image .txt files):
```
bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
```

**COCO format** (single JSON per split) — what D-FINE expects:
```json
{
  "images": [{"id": 0, "file_name": "img.jpg", "width": 1360, "height": 765}],
  "annotations": [{"id": 0, "image_id": 0, "category_id": 4,
                    "bbox": [x, y, w, h], "area": float, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "pedestrian"}, ...]
}
```

Conversion script: `tools/visdrone2coco.py`
Output:
- `dataset/visdrone/annotations/instances_train.json` (37MB, 343k annotations)
- `dataset/visdrone/annotations/instances_val.json` (4MB, 38k annotations)

**Skipped during conversion:**
- `score=0`: ignored/distractor regions (explicitly excluded from eval)
- `category=0`: unknown class

---

## Sample Image Analysis

From 10 random training images (visualised in `PROJECT_NOTES/visdrone_samples/`):

| Image | Size | Objects | Tiny (<32px) | Dominant Class |
|-------|------|---------|--------------|----------------|
| 9999985... | 1400x1050 | 27 | 74% | car (10) |
| 0000303... | 1360x765 | **245** | 82% | motor (89), pedestrian (74) |
| 0000134... | 1920x1080 | 162 | 57% | car (62), pedestrian (30) |
| 9999999... | 2000x1500 | 43 | 7% | car (36) |
| 9999950... | 1400x1050 | 17 | 65% | car (8), people (6) |
| 9999943... | 1400x1050 | 54 | 28% | car (25), pedestrian (10) |
| 9999942... | 1400x1050 | 73 | 41% | car (22), pedestrian (18) |
| 0000339... | 1360x765 | 77 | 68% | car (56), motor (8) |
| 9999998... | 2000x1500 | 14 | 0% | van (5), car (5) |
| 0000281... | 1360x765 | 29 | 0% | car (18), people (4) |

**Altitude variance:** The dataset spans different drone altitudes. At high altitude
(image 9999998: avg 223px boxes), objects are relatively large. At low altitude
(image 0000303: avg 27px boxes, 245 objects), it's extremely dense and tiny.
The model needs to handle both.

---

## COCO vs VisDrone Comparison

| Property | COCO (our baseline) | VisDrone |
|----------|---------------------|----------|
| mAP expected (pretrained, no finetune) | 48.5% | ~10-20% (estimated) |
| Primary challenge | 80-class diversity | Tiny objects, aerial domain |
| Objects/image | ~7 | 53-70 |
| "Small" objects (< 32px) | ~15% | 46-53% |
| Domain | Ground level | Aerial |

The large drop in expected mAP without fine-tuning is primarily due to:
1. Feature distribution shift (aerial vs ground)
2. Scale shift (most VisDrone objects fall in COCO's "small" category where it scores 29.4%)
3. Class mismatch (COCO-trained model has no "motor" or "awning-tricycle" concept)

---

## Dataset Paths

```
dataset/visdrone/
├── VisDrone2019-DET-train/
│   ├── images/                  <- 6,471 JPG images
│   └── annotations/             <- per-image .txt files (VisDrone format)
├── VisDrone2019-DET-val/
│   ├── images/                  <- 548 JPG images
│   └── annotations/             <- per-image .txt files (VisDrone format)
└── annotations/
    ├── instances_train.json     <- converted COCO format (37MB)
    ├── instances_val.json       <- converted COCO format (4MB)
    └── dataset_stats.json       <- raw statistics
```
