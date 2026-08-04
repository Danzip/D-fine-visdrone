"""
ByteTrack multi-object tracking on video, using a trained D-FINE checkpoint.

Runs D-FINE per-frame detection (same val-time preprocessing/postprocessing as
training: plain resize to eval_spatial_size, no letterbox — see postprocessor.py)
and feeds detections into supervision's ByteTrack implementation.

Two additions on top of vanilla ByteTrack, both aimed at the ID-fragmentation
problem seen on dense, panning aerial footage (see PROJECT_NOTES for the
uav0000137 baseline: 184 GT tracks -> 1039 predicted tracks):

1. Low-confidence detections (0.1-conf_high) are now actually passed to the
   tracker instead of being discarded before it ever sees them. ByteTrack's
   own core.py hard-codes a >0.1 floor for its second-stage ("low score")
   recovery association — the previous version filtered at conf_high (0.3)
   *before* handing detections to the tracker, silently disabling that
   recovery pass entirely.

2. Lightweight global motion compensation (GMC): ORB features + RANSAC affine
   estimate the frame-to-frame camera transform, composed into a running
   "world" reference frame. Detections are warped into world space before
   tracking (so a panning/rotating camera looks stationary to the tracker's
   constant-velocity Kalman model) and warped back to pixel space for
   drawing. This is a simpler external approximation of what BoT-SORT's GMC
   does internally (which warps each track's Kalman mean/covariance directly)
   — supervision's ByteTrack doesn't expose that hook, so this is bolted on
   from the outside instead. Good enough to remove the dominant panning
   component; expect some residual drift over very long sequences since the
   world-frame reference accumulates per-frame affine estimates.

Usage:
    python tools/tracking/track_video.py \
        -c experiments/e6_1280/config.yml \
        -r output/runpod_results/e6_1280_best_ep46.pth \
        --video path/to/input.mp4 \
        --output output/tracking/tracked.mp4 \
        --device cuda:0
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision

from src.core import YAMLConfig
from src.solver import TASKS

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def build_model(config, checkpoint, device):
    cfg = YAMLConfig(config, resume=checkpoint, device=device)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    postprocessor = solver.postprocessor.deploy()  # tuple output instead of list[dict]
    input_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])  # [H, W]
    return model, postprocessor, solver.device, input_size


@torch.no_grad()
def detect(model, postprocessor, frame_bgr, input_size, device, conf_floor, nms_iou=None):
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)

    outputs = model(tensor)
    orig_sizes = torch.tensor([[w, h]], dtype=torch.float32, device=device)
    labels, boxes, scores = postprocessor(outputs, orig_sizes)
    labels, boxes, scores = labels[0], boxes[0], scores[0]

    # Keep everything down to ByteTrack's own low-score floor (0.1, hard-coded
    # in its core.py) so the second-stage recovery association has detections
    # to work with. Confirmed-track display filtering happens inside the
    # tracker itself (track_activation_threshold), not here.
    mask = scores >= conf_floor
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    # D-FINE is end-to-end (no NMS in the model) — at a low conf_floor on a
    # dense scene, multiple queries can converge on the same physical object.
    # Measured on uav0000137: 500/500 top queries clearing 0.1 on every
    # frame, vs ~104 GT objects/frame — that gap feeds every downstream
    # tracker far more "detections" per frame than there are real objects,
    # which dominates tracker runtime (see PROJECT_NOTES/12_tracking.md).
    # Per-class NMS (not class-agnostic): duplicate boxes on one object share
    # its class, but e.g. a pedestrian on/near a bicycle can legitimately
    # overlap across classes and shouldn't suppress each other.
    if nms_iou is not None and boxes.shape[0] > 0:
        keep = torchvision.ops.batched_nms(boxes, scores, labels, nms_iou)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    boxes = boxes.cpu().numpy().astype(np.float32)
    scores = scores.cpu().numpy().astype(np.float32)
    labels = labels.cpu().numpy().astype(int)
    return boxes, scores, labels


class MotionCompensator:
    """Estimates frame-to-frame camera motion (ORB + RANSAC affine) and
    composes it into a running transform to a shared 'world' frame anchored
    at frame 0. warp_to_world/warp_to_pixel move box coordinates between
    that world frame (stable under camera panning, used for tracking) and
    the current frame's pixel space (used for display)."""

    def __init__(self, max_features=500):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.H_accum = np.eye(3, dtype=np.float32)  # current-frame pixels -> world

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is not None and des is not None and len(des) >= 10 and len(self.prev_des) >= 10:
            matches = sorted(self.matcher.match(self.prev_des, des), key=lambda m: m.distance)[:200]
            if len(matches) >= 10:
                src = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                dst = np.float32([kp[m.trainIdx].pt for m in matches])
                # dst (current frame) -> src (previous frame)
                M, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    M3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
                    self.H_accum = self.H_accum @ M3

        self.prev_kp, self.prev_des = kp, des
        return self.H_accum.copy()

    @staticmethod
    def _warp_boxes(boxes_xyxy, H):
        if len(boxes_xyxy) == 0:
            return boxes_xyxy
        # warp all 4 corners per box, then take the axis-aligned bounding box
        # of the warped quad (H may include a small rotation component)
        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        corners = np.stack([
            np.stack([x1, y1], axis=1), np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1), np.stack([x1, y2], axis=1),
        ], axis=1).reshape(-1, 2)  # (4N, 2)
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        warped = (H @ np.hstack([corners, ones]).T).T
        warped = (warped[:, :2] / warped[:, 2:3]).reshape(-1, 4, 2)
        out = np.concatenate([warped.min(axis=1), warped.max(axis=1)], axis=1)
        return out.astype(np.float32)

    def warp_to_world(self, boxes_xyxy, H):
        return self._warp_boxes(boxes_xyxy, H)

    def warp_to_pixel(self, boxes_xyxy, H):
        return self._warp_boxes(boxes_xyxy, np.linalg.inv(H))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf-low", type=float, default=0.1,
                         help="floor passed to the tracker; ByteTrack's own low-score recovery pass needs >0.1")
    parser.add_argument("--nms-iou", type=float, default=None,
                         help="per-class NMS IoU threshold applied before tracking (off by default); "
                              "e.g. 0.6 to cut duplicate-query detections on dense scenes")
    parser.add_argument("--conf-high", type=float, default=0.3,
                         help="track_activation_threshold: min score to spawn/confirm a new track")
    parser.add_argument("--min-consecutive-frames", type=int, default=3,
                         help="hysteresis: frames a track must match before it's emitted, filters noise tracks")
    parser.add_argument("--lost-track-buffer", type=int, default=30,
                         help="frames a track survives with no match before being dropped (occlusion tolerance)")
    parser.add_argument("--gmc", dest="gmc", action="store_true", default=True,
                         help="enable ORB-based global motion compensation (default on)")
    parser.add_argument("--no-gmc", dest="gmc", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    model, postprocessor, device, input_size = build_model(args.config, args.checkpoint, args.device)
    print(f"Model loaded. Input size (H,W)={input_size}, device={device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=args.conf_high,
        lost_track_buffer=args.lost_track_buffer,
        minimum_consecutive_frames=args.min_consecutive_frames,
    )
    motion_comp = MotionCompensator() if args.gmc else None
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)
    trace_annotator = sv.TraceAnnotator(trace_length=30, thickness=1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_lengths = {}
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        boxes, scores, labels = detect(model, postprocessor, frame, input_size, device, args.conf_low, args.nms_iou)

        if motion_comp is not None:
            H = motion_comp.update(frame)
            track_boxes = motion_comp.warp_to_world(boxes, H)
        else:
            track_boxes = boxes

        detections = sv.Detections(xyxy=track_boxes, confidence=scores, class_id=labels)
        detections = tracker.update_with_detections(detections)

        if motion_comp is not None and len(detections) > 0:
            detections.xyxy = motion_comp.warp_to_pixel(detections.xyxy, H)

        for tid in detections.tracker_id:
            track_lengths[int(tid)] = track_lengths.get(int(tid), 0) + 1

        labels_text = [
            f"#{tid} {CLASS_NAMES[c]} {conf:.2f}"
            for tid, c, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]

        annotated = frame.copy()
        annotated = trace_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels_text)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"frame {frame_idx}/{total_frames}  tracks so far: {len(track_lengths)}")

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.2f} FPS)")
    print(f"Unique tracks: {len(track_lengths)}")
    if track_lengths:
        lengths = list(track_lengths.values())
        print(f"Track length (frames): mean={np.mean(lengths):.1f}  median={np.median(lengths):.0f}  max={max(lengths)}")
        short = sum(1 for l in lengths if l <= 2)
        print(f"Tracks lasting <=2 frames (likely ID-switch/noise): {short} ({100 * short / len(lengths):.0f}%)")
    print(f"Output saved: {args.output}")


if __name__ == "__main__":
    main()
