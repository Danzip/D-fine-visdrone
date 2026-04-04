"""
YOLOv8-X VisDrone inference server (clean, working version).

Run:
    cd /home/danziv/projects/DFine/dfine_app_server
    source ../D-FINE/venv/bin/activate
    uvicorn server_yolo:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

YOLO_PATH = Path(__file__).parent / "models" / "best.pt"
SCORE_THRESHOLD = 0.3

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

model = YOLO(str(YOLO_PATH))
print("YOLOv8-X loaded.")

app = FastAPI(title="YOLOv8 VisDrone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8x-visdrone"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    print(f"Received file: name={file.filename!r} size={len(data)} bytes content_type={file.content_type!r}")

    # Write to temp file — YOLO reads it with its own pipeline (same as sota_compare.py)
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    img = Image.open(io.BytesIO(data))
    orig_w, orig_h = img.size
    print(f"Image size: {orig_w}x{orig_h}, temp file: {tmp_path}")

    results = model(tmp_path, conf=SCORE_THRESHOLD, iou=0.45, verbose=False)[0]
    Path(tmp_path).unlink(missing_ok=True)

    boxes  = results.boxes.xyxy.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()

    detections = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        detections.append({
            "label": VISDRONE_CLASSES[label] if label < len(VISDRONE_CLASSES) else str(label),
            "label_id": int(label),
            "score": round(float(score), 4),
            "box": {
                "x1": round(box[0], 1), "y1": round(box[1], 1),
                "x2": round(box[2], 1), "y2": round(box[3], 1),
            },
        })

    return JSONResponse({
        "image_width":  orig_w,
        "image_height": orig_h,
        "detections":   detections,
    })
