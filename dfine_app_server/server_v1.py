"""
D-FINE VisDrone inference server.
Accepts an image + model choice, returns detections as JSON.

Run:
    cd /home/danziv/projects/DFine/dfine_app_server
    source ../D-FINE/venv/bin/activate
    uvicorn server_v1:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import numpy as np
import onnxruntime as ort
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# ── config ────────────────────────────────────────────────────────────────────
ONNX_PATH  = "../D-FINE/output/pruning_recovery/best_recovery.onnx"
YOLO_PATH  = Path(__file__).parent / "models" / "best.pt"
INPUT_SIZE = (640, 640)   # (W, H)
SCORE_THRESHOLD = 0.3

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

AVAILABLE_MODELS = {
    "dfine":  "D-FINE-S (ours, 10M params, AP50=0.389)",
    "yolov8": "YOLOv8-X (SOTA, 68M params, AP50=0.470)",
}

# ── load models ───────────────────────────────────────────────────────────────
dfine_session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(f"D-FINE loaded. Providers: {dfine_session.get_providers()}")

yolo_model = YOLO(str(YOLO_PATH))
print("YOLOv8-X loaded.")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="D-FINE VisDrone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── preprocessing (letterbox — matches training pipeline) ─────────────────────
def letterbox(image: Image.Image):
    orig_w, orig_h = image.size
    img = image.convert("RGB")
    target_w, target_h = INPUT_SIZE
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (target_w - new_w) // 2
    pad_top  = (target_h - new_h) // 2
    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    padded.paste(img, (pad_left, pad_top))
    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr, orig_w, orig_h, pad_left, pad_top, scale


# ── inference helpers ─────────────────────────────────────────────────────────
def run_dfine(image: Image.Image, threshold: float):
    tensor, orig_w, orig_h, pad_left, pad_top, scale = letterbox(image)
    pad_h, pad_w = INPUT_SIZE[1], INPUT_SIZE[0]
    orig_size = np.array([[pad_h, pad_w]], dtype=np.int64)

    labels, boxes, scores = dfine_session.run(
        None, {"images": tensor, "orig_target_sizes": orig_size}
    )
    labels, boxes, scores = labels[0], boxes[0], scores[0]

    mask = scores >= threshold
    labels, boxes, scores = labels[mask], boxes[mask], scores[mask]

    detections = []
    for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        x1 = max(0.0, (box[0] - pad_left) / scale)
        y1 = max(0.0, (box[1] - pad_top)  / scale)
        x2 = min(float(orig_w), (box[2] - pad_left) / scale)
        y2 = min(float(orig_h), (box[3] - pad_top)  / scale)
        detections.append({
            "label": VISDRONE_CLASSES[int(label)] if int(label) < len(VISDRONE_CLASSES) else str(label),
            "label_id": int(label),
            "score": round(float(score), 4),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1)},
        })
    return detections, orig_w, orig_h


def run_yolo(image: Image.Image, threshold: float):
    orig_w, orig_h = image.size
    results = yolo_model(image, conf=threshold, iou=0.45, verbose=False)[0]
    boxes  = results.boxes.xyxy.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()

    detections = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        detections.append({
            "label": VISDRONE_CLASSES[label] if label < len(VISDRONE_CLASSES) else str(label),
            "label_id": int(label),
            "score": round(float(score), 4),
            "box": {"x1": round(box[0], 1), "y1": round(box[1], 1),
                    "x2": round(box[2], 1), "y2": round(box[3], 1)},
        })
    return detections, orig_w, orig_h


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models": AVAILABLE_MODELS}


@app.get("/models")
def models():
    return AVAILABLE_MODELS


@app.post("/detect")
async def detect(
    file:      UploadFile = File(...),
    model:     str        = Form("dfine"),
    threshold: float      = Form(0.3),
):
    data  = await file.read()
    image = Image.open(io.BytesIO(data))

    if model == "yolov8":
        detections, orig_w, orig_h = run_yolo(image, threshold)
    else:
        detections, orig_w, orig_h = run_dfine(image, threshold)

    return JSONResponse({
        "model": model,
        "image_width":  orig_w,
        "image_height": orig_h,
        "detections":   detections,
    })
