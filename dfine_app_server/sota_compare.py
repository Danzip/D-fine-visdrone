"""
Side-by-side comparison: YOLOv8-X vs D-FINE-S on the same image.

Saves two files:
  <image>_yolo.jpg   — YOLOv8-X detections
  <image>_dfine.jpg  — D-FINE-S detections

Usage:
    python sota_compare.py --image /path/to/image.jpg
    python sota_compare.py --image /path/to/image.jpg --threshold 0.25
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort
from PIL import Image

# ── shared config ─────────────────────────────────────────────────────────────
YOLO_PATH  = Path(__file__).parent / "models" / "best.pt"
ONNX_PATH  = Path(__file__).parent / "../D-FINE/output/pruning_recovery/best_recovery.onnx"
INPUT_SIZE = (640, 640)

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

COLORS = [
    (229, 57, 53),
    (255, 112, 67),
    (253, 216, 53),
    (67, 160, 71),
    (0, 172, 193),
    (30, 136, 229),
    (142, 36, 170),
    (216, 27, 96),
    (255, 143, 0),
    (109, 76, 65),
]


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(img_bgr, boxes, labels, scores, title):
    out = img_bgr.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[int(label) % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{VISDRONE_CLASSES[int(label)] if int(label) < len(VISDRONE_CLASSES) else str(label)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(out, f"{title}  —  {len(boxes)} detections", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


# ── YOLOv8 ───────────────────────────────────────────────────────────────────
def run_yolo(image_path, threshold):
    model = YOLO(str(YOLO_PATH))
    results = model(image_path, conf=threshold, iou=0.45, verbose=False)[0]
    boxes  = results.boxes.xyxy.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()
    return boxes, labels, scores


# ── D-FINE ────────────────────────────────────────────────────────────────────
def run_dfine(image_path, threshold):
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # letterbox — same as training pipeline
    scale = min(INPUT_SIZE[0] / orig_w, INPUT_SIZE[1] / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (INPUT_SIZE[0] - new_w) // 2
    pad_top  = (INPUT_SIZE[1] - new_h) // 2
    padded = Image.new("RGB", INPUT_SIZE, (0, 0, 0))
    padded.paste(img, (pad_left, pad_top))

    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]

    orig_size = np.array([[INPUT_SIZE[1], INPUT_SIZE[0]]], dtype=np.int64)
    labels_out, boxes_out, scores_out = session.run(
        None, {"images": arr, "orig_target_sizes": orig_size}
    )
    labels_out = labels_out[0]
    boxes_out  = boxes_out[0]
    scores_out = scores_out[0]

    mask = scores_out >= threshold
    labels_out = labels_out[mask]
    boxes_out  = boxes_out[mask]
    scores_out = scores_out[mask]

    # undo letterbox padding → original image coords
    boxes_final = []
    for box in boxes_out:
        x1 = max(0.0, (box[0] - pad_left) / scale)
        y1 = max(0.0, (box[1] - pad_top)  / scale)
        x2 = min(float(orig_w), (box[2] - pad_left) / scale)
        y2 = min(float(orig_h), (box[3] - pad_top)  / scale)
        boxes_final.append([x1, y1, x2, y2])

    return np.array(boxes_final), labels_out, scores_out, orig_w, orig_h


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    # handle Windows paths from WSL
    image_path = args.image.replace("c:/", "/mnt/c/").replace("C:/", "/mnt/c/").replace("\\", "/")
    stem = str(Path(image_path).with_suffix(""))

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    # ── YOLOv8 ──
    print("Running YOLOv8-X VisDrone...")
    yolo_boxes, yolo_labels, yolo_scores = run_yolo(image_path, args.threshold)
    print(f"  {len(yolo_boxes)} detections")
    yolo_vis = draw_boxes(img_bgr, yolo_boxes, yolo_labels, yolo_scores,
                          "YOLOv8-X VisDrone (SOTA, AP50=0.470)")
    yolo_out = stem + "_yolo.jpg"
    cv2.imwrite(yolo_out, yolo_vis)
    print(f"  Saved: {yolo_out}")

    # ── D-FINE ──
    print("Running D-FINE-S VisDrone...")
    dfine_boxes, dfine_labels, dfine_scores, orig_w, orig_h = run_dfine(image_path, args.threshold)
    print(f"  {len(dfine_boxes)} detections")
    dfine_vis = draw_boxes(img_bgr, dfine_boxes, dfine_labels, dfine_scores,
                           "D-FINE-S VisDrone (ours, AP50=0.389)")
    dfine_out = stem + "_dfine.jpg"
    cv2.imwrite(dfine_out, dfine_vis)
    print(f"  Saved: {dfine_out}")

    print("\nDone. Open both files to compare.")


if __name__ == "__main__":
    main()
