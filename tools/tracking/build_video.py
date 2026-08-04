"""One-off: assemble a VisDrone-MOT frame sequence into an mp4 for tracker testing."""
import argparse
import glob
import os

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--frames-dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--fps", type=float, default=25.0)
args = parser.parse_args()

frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
assert frame_paths, f"No frames found in {args.frames_dir}"

first = cv2.imread(frame_paths[0])
h, w = first.shape[:2]

os.makedirs(os.path.dirname(args.output), exist_ok=True)
writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
for p in frame_paths:
    writer.write(cv2.imread(p))
writer.release()

print(f"Wrote {len(frame_paths)} frames ({w}x{h} @ {args.fps}fps) -> {args.output}")
