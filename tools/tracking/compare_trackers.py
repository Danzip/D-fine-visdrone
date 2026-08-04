"""
Compare our fixed ByteTrack (supervision + external GMC, see track_video.py)
against BoT-SORT, StrongSORT, OC-SORT and DeepOCSORT (via the `boxmot`
package) on the exact same D-FINE detections.

D-FINE inference (the expensive part — 1280x1280 per frame) runs exactly
once; every tracker is then replayed against the identical cached detection
stream, so differences in the results are attributable to the tracker, not
to detection noise between runs.

Usage:
    python tools/tracking/compare_trackers.py \
        -c experiments/e6_1280/config.yml \
        -r output/runpod_results/e6_1280_best_ep46.pth \
        --video output/tracking/input_uav0000137.mp4 \
        --output-dir output/tracking/compare \
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

from boxmot.trackers.registry import create_tracker
from tools.tracking.track_video import CLASS_NAMES, MotionCompensator, build_model, detect

# out-of-the-box (library defaults) except frame_rate, which every one of
# these trackers accepts as a **kwargs passthrough even if unused
REID_WEIGHTS = "osnet_x0_25_msmt17.pt"  # smallest zoo model, fine for a first comparison pass
BOXMOT_TRACKERS = ["ocsort", "botsort", "strongsort", "deepocsort"]


def cache_detections(model, postprocessor, video_path, input_size, device, conf_low, nms_iou=None, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_dets = []
    det_counts = []
    idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and idx >= max_frames:
            break
        boxes, scores, labels = detect(model, postprocessor, frame, input_size, device, conf_low, nms_iou)
        all_dets.append((boxes, scores, labels))
        det_counts.append(len(boxes))
        idx += 1
        if idx % 50 == 0:
            print(f"  detected frame {idx}...")
    cap.release()
    print(f"Cached detections for {idx} frames in {time.time() - t0:.1f}s "
          f"(dets/frame: mean={np.mean(det_counts):.1f} min={min(det_counts)} max={max(det_counts)})")
    return all_dets, fps, w, h


def summarize(track_lengths, frame_count, elapsed):
    if not track_lengths:
        return {"unique_tracks": 0, "median_len": 0, "mean_len": 0.0, "short_pct": 0.0, "fps": frame_count / elapsed}
    lengths = list(track_lengths.values())
    short = sum(1 for l in lengths if l <= 2)
    return {
        "unique_tracks": len(lengths),
        "median_len": float(np.median(lengths)),
        "mean_len": float(np.mean(lengths)),
        "short_pct": 100 * short / len(lengths),
        "fps": frame_count / elapsed,
    }


def run_sv_bytetrack(all_dets, video_path, fps, output_path, conf_high, min_consec, lost_buf):
    """Our fixed supervision ByteTrack + external GMC (see track_video.py)."""
    tracker = sv.ByteTrack(
        frame_rate=fps, track_activation_threshold=conf_high,
        lost_track_buffer=lost_buf, minimum_consecutive_frames=min_consec,
    )
    motion_comp = MotionCompensator()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)
    trace_annotator = sv.TraceAnnotator(trace_length=30, thickness=1)

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_lengths = {}
    t0 = time.time()
    for boxes, scores, labels in all_dets:
        ret, frame = cap.read()
        if not ret:
            break
        H = motion_comp.update(frame)
        track_boxes = motion_comp.warp_to_world(boxes, H)
        detections = sv.Detections(xyxy=track_boxes, confidence=scores, class_id=labels)
        detections = tracker.update_with_detections(detections)
        if len(detections) > 0:
            detections.xyxy = motion_comp.warp_to_pixel(detections.xyxy, H)

        for tid in detections.tracker_id:
            track_lengths[int(tid)] = track_lengths.get(int(tid), 0) + 1
        labels_text = [f"#{tid} {CLASS_NAMES[c]} {conf:.2f}"
                       for tid, c, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated = frame.copy()
        annotated = trace_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels_text)
        writer.write(annotated)
    elapsed = time.time() - t0
    cap.release()
    writer.release()
    return summarize(track_lengths, len(all_dets), elapsed)


def run_boxmot_tracker(tracker_name, all_dets, video_path, fps, output_path, device):
    needs_reid = tracker_name in ("botsort", "strongsort", "deepocsort")
    tracker = create_tracker(
        tracker_name,
        reid_weights=REID_WEIGHTS if needs_reid else None,
        device=str(device),
        half=False,
        tracker_kwargs={"frame_rate": int(round(fps))},
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)
    trace_annotator = sv.TraceAnnotator(trace_length=30, thickness=1)

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_lengths = {}
    t0 = time.time()
    for boxes, scores, labels in all_dets:
        ret, frame = cap.read()
        if not ret:
            break
        if len(boxes) > 0:
            dets = np.concatenate([boxes, scores[:, None], labels[:, None].astype(np.float32)], axis=1)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        res = tracker.update(dets, frame)
        xyxy = res.xyxy.astype(np.float32) if len(res) else np.empty((0, 4), dtype=np.float32)
        ids = res.id if len(res) else np.empty((0,), dtype=int)
        confs = res.conf.astype(np.float32) if len(res) else np.empty((0,), dtype=np.float32)
        clses = res.cls if len(res) else np.empty((0,), dtype=int)

        for tid in ids:
            track_lengths[int(tid)] = track_lengths.get(int(tid), 0) + 1

        detections = sv.Detections(xyxy=xyxy, confidence=confs, class_id=clses, tracker_id=ids)
        labels_text = [f"#{tid} {CLASS_NAMES[c]} {conf:.2f}" for tid, c, conf in zip(ids, clses, confs)]
        annotated = frame.copy()
        annotated = trace_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels_text)
        writer.write(annotated)
    elapsed = time.time() - t0
    cap.release()
    writer.release()
    return summarize(track_lengths, len(all_dets), elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-low", type=float, default=0.1)
    parser.add_argument("--conf-high", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=None,
                         help="per-class NMS IoU threshold applied before tracking (off by default)")
    parser.add_argument("--min-consecutive-frames", type=int, default=3)
    parser.add_argument("--lost-track-buffer", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, postprocessor, device, input_size = build_model(args.config, args.checkpoint, args.device)
    print(f"Model loaded. Input size (H,W)={input_size}, device={device}")

    print("Running D-FINE detection once, caching results for all trackers...")
    all_dets, fps, w, h = cache_detections(
        model, postprocessor, args.video, input_size, device, args.conf_low, args.nms_iou, args.max_frames
    )

    results = {}

    print("\n=== sv-ByteTrack (fixed: low-conf recovery + GMC) ===")
    results["bytetrack (ours)"] = run_sv_bytetrack(
        all_dets, args.video, fps, os.path.join(args.output_dir, "bytetrack.mp4"),
        args.conf_high, args.min_consecutive_frames, args.lost_track_buffer,
    )
    print(results["bytetrack (ours)"])

    for name in BOXMOT_TRACKERS:
        print(f"\n=== {name} (boxmot, library defaults) ===")
        results[name] = run_boxmot_tracker(
            name, all_dets, args.video, fps, os.path.join(args.output_dir, f"{name}.mp4"), device,
        )
        print(results[name])

    print("\n\n=== Summary ===")
    header = f"{'tracker':<20} {'tracks':>7} {'median_len':>11} {'mean_len':>9} {'short%':>7} {'fps':>6}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<20} {r['unique_tracks']:>7} {r['median_len']:>11.0f} {r['mean_len']:>9.1f} "
              f"{r['short_pct']:>6.0f}% {r['fps']:>6.2f}")


if __name__ == "__main__":
    main()
