"""
SAHI (Slicing Aided Hyper Inference) wrapper for D-FINE.

What SAHI does:
  Instead of running the detector on the full image (where small objects become tiny),
  it cuts the image into overlapping 640x640 patches, runs D-FINE on each patch,
  then merges detections with NMS. Small objects stay large relative to their patch.

Usage — single image:
  python tools/inference/sahi_inf.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    -r output/dfine_hgnetv2_s_visdrone/best_stg1.pth \
    --input path/to/image.jpg \
    --output sahi_result.jpg

Usage — evaluate AP on VisDrone val set:
  python tools/inference/sahi_inf.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
    -r output/dfine_hgnetv2_s_visdrone/best_stg1.pth \
    --eval \
    --ann dataset/visdrone/VisDrone2019-DET-val/annotations_coco.json \
    --img-dir dataset/visdrone/VisDrone2019-DET-val/images
"""

import argparse
import os
import sys
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


class DFineDetectionModel(DetectionModel):
    """SAHI DetectionModel wrapper for D-FINE."""

    def load_model(self):
        cfg = YAMLConfig(self.config_path, resume=self.model_path)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)

        class _Deploy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)

        self.model = _Deploy().to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])

    def set_model(self, model: Any, **kwargs):
        self.model = model

    def perform_inference(self, image: np.ndarray):
        """Run D-FINE on one patch (numpy HWC uint8). Stores raw output."""
        pil = Image.fromarray(image)
        w, h = pil.size
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            labels, boxes, scores = self.model(tensor, orig_size)

        self._original_predictions = [
            labels[0].cpu().numpy(),
            boxes[0].cpu().numpy(),   # xyxy absolute
            scores[0].cpu().numpy(),
        ]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        labels, boxes, scores = self._original_predictions
        shift = shift_amount_list[0]
        full_shape = full_shape_list[0] if full_shape_list else None

        preds = []
        for label, box, score in zip(labels, boxes, scores):
            if score < self.confidence_threshold:
                continue
            cat_id = int(label)
            cat_name = VISDRONE_CLASSES[cat_id] if cat_id < len(VISDRONE_CLASSES) else str(cat_id)
            preds.append(ObjectPrediction(
                bbox=box.tolist(),
                score=float(score),
                category_id=cat_id,
                category_name=cat_name,
                shift_amount=shift,
                full_shape=full_shape,
            ))

        self._object_prediction_list_per_image = [preds]


def load_dfine_sahi(config_path, checkpoint_path, device="cuda:0",
                    confidence_threshold=0.3, image_size=640):
    model = DFineDetectionModel(
        config_path=config_path,
        model_path=checkpoint_path,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        device=device,
    )
    return model


def infer_image(model, image_path, output_path,
                slice_size=640, overlap=0.2):
    result = get_sliced_prediction(
        image_path, model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        verbose=1,
    )
    result.export_visuals(
        export_dir=os.path.dirname(os.path.abspath(output_path)),
        file_name=os.path.splitext(os.path.basename(output_path))[0],
    )
    print(f"Detected {len(result.object_prediction_list)} objects")
    for p in result.object_prediction_list[:15]:
        print(f"  {p.category.name:20s}  score={p.score.value:.2f}  "
              f"box={[round(x) for x in p.bbox.to_xyxy()]}")


def evaluate_coco(model, ann_json, img_dir, slice_size=640, overlap=0.2):
    import json
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with open(ann_json) as f:
        images = json.load(f)["images"]

    print(f"Running SAHI on {len(images)} val images...")
    all_results = []

    for i, img_info in enumerate(images):
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        result = get_sliced_prediction(
            img_path, model,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            verbose=0,
        )
        for p in result.object_prediction_list:
            x1, y1, x2, y2 = p.bbox.to_xyxy()
            all_results.append({
                "image_id": img_info["id"],
                "category_id": p.category.id,  # VisDrone annotations are 0-indexed
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": p.score.value,
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(images)}]")

    pred_path = "output/sahi_val_predictions.json"
    os.makedirs("output", exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(all_results, f)

    coco_gt = COCO(ann_json)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = ["AP50:95", "AP50", "AP75", "AP-small", "AP-medium", "AP-large"]
    print("\n=== SAHI Results ===")
    for name, val in zip(names, coco_eval.stats[:6]):
        print(f"  {name:12s}: {val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",    required=True)
    parser.add_argument("-r", "--resume",    required=True)
    parser.add_argument("-d", "--device",    default="cuda:0")
    parser.add_argument("--threshold",       type=float, default=0.3)
    parser.add_argument("--slice-size",      type=int,   default=640)
    parser.add_argument("--image-size",      type=int,   default=None,
                        help="resize each slice to this before model inference "
                             "(default: same as --slice-size)")
    parser.add_argument("--overlap",         type=float, default=0.2)
    parser.add_argument("--input",           type=str,   default=None)
    parser.add_argument("--output",          type=str,   default="output/sahi_result.jpg")
    parser.add_argument("--eval",            action="store_true")
    parser.add_argument("--ann",             type=str,   default=None)
    parser.add_argument("--img-dir",         type=str,   default=None)
    args = parser.parse_args()

    image_size = args.image_size if args.image_size is not None else args.slice_size
    model = load_dfine_sahi(
        args.config, args.resume, args.device,
        confidence_threshold=args.threshold,
        image_size=image_size,
    )

    if args.eval:
        evaluate_coco(model, args.ann, args.img_dir,
                      slice_size=args.slice_size, overlap=args.overlap)
    elif args.input:
        infer_image(model, args.input, args.output,
                    slice_size=args.slice_size, overlap=args.overlap)
    else:
        print("Provide --input <image> or --eval")
