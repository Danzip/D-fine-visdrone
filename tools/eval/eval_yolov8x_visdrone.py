"""
Evaluate the YOLOv8-X VisDrone checkpoint (dfine_app_server/models/best.pt,
source: mshamrai HuggingFace) on our own VisDrone val split, using the same
faster_coco_eval evaluator D-FINE's training/eval uses, so the resulting AP
is directly comparable to our own numbers instead of trusting the model
card's unverified 0.47 AP50 figure.

Usage:
    python tools/eval/eval_yolov8x_visdrone.py
"""

import json
from pathlib import Path

from faster_coco_eval import COCO, COCOeval_faster
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT.parent / "dfine_app_server" / "models" / "best.pt"
ANN_FILE = ROOT / "dataset" / "visdrone" / "annotations" / "instances_val.json"
IMG_DIR = ROOT / "dataset" / "visdrone" / "VisDrone2019-DET-val" / "images"
IMGSZ = 640  # matches this checkpoint's own train_args.imgsz


def main():
    coco_gt = COCO(str(ANN_FILE))
    model = YOLO(str(WEIGHTS))

    results = []
    for image_id, info in coco_gt.imgs.items():
        img_path = IMG_DIR / info["file_name"]
        pred = model.predict(str(img_path), imgsz=IMGSZ, conf=0.001, iou=0.65, verbose=False)[0]
        boxes = pred.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            results.append({
                "image_id": image_id,
                "category_id": int(box.cls.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(box.conf.item()),
            })

    print(f"Collected {len(results)} detections over {len(coco_gt.imgs)} images.")

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    # VisDrone scenes can have 100-800 objects; default maxDets=100 caps recall on dense
    # images. Match det_engine.py's maxDets=[1,10,500] so this is directly comparable to
    # D-FINE's own eval, not just same-metric but different-maxDets.
    coco_eval.params.maxDets = [1, 10, 500]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(f"\nYOLOv8-X (best.pt) on VisDrone val ({len(coco_gt.imgs)} images):")
    print(f"  AP50:95     : {coco_eval.stats[0]:.4f}")
    print(f"  AP50        : {coco_eval.stats[1]:.4f}")
    print(f"  AP75        : {coco_eval.stats[2]:.4f}")
    print(f"  AP-small    : {coco_eval.stats[3]:.4f}")
    print(f"  AP-medium   : {coco_eval.stats[4]:.4f}")
    print(f"  AP-large    : {coco_eval.stats[5]:.4f}")


if __name__ == "__main__":
    main()
