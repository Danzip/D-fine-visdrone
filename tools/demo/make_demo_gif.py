"""
Generate demo.gif from VisDrone val images using the pruned ONNX model.
Picks a diverse set of images, runs inference, draws colored boxes, saves GIF.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

COLORS = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE",
    "#30B0C7", "#32ADE6", "#007AFF", "#5856D6", "#AF52DE",
]


def letterbox(image: Image.Image, size: int = 640):
    w, h = image.size
    ratio = min(size / w, size / h)
    nw, nh = int(w * ratio), int(h * ratio)
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_w = (size - nw) // 2
    pad_h = (size - nh) // 2
    canvas.paste(resized, (pad_w, pad_h))
    return canvas, ratio, pad_w, pad_h


def run(sess, im_pil, threshold=0.35):
    canvas, ratio, pad_w, pad_h = letterbox(im_pil)
    orig_size = torch.tensor([[canvas.size[1], canvas.size[0]]])
    im_data = T.ToTensor()(canvas).unsqueeze(0)
    labels, boxes, scores = sess.run(
        None, {"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
    )
    labels, boxes, scores = labels[0], boxes[0], scores[0]
    keep = scores > threshold
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
    # undo letterbox
    boxes = boxes.copy().astype(float)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
    return labels, boxes, scores


def annotate(im_pil, labels, boxes, scores, scale=1.0):
    out = im_pil.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, int(14 * scale)))
    except Exception:
        font = ImageFont.load_default()

    for lbl, box, scr in zip(labels, boxes, scores):
        color = COLORS[int(lbl) % len(COLORS)]
        x1, y1, x2, y2 = box
        lw = max(2, int(3 * scale))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        text = f"{CLASSES[int(lbl)]} {scr:.2f}"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 2), text, fill="white", font=font)
    return out


def pick_images(val_dir: Path, n: int, seed: int = 42) -> list[Path]:
    all_imgs = sorted(val_dir.glob("*.jpg"))
    # Sample from evenly-spaced buckets so we get scene variety
    step = max(1, len(all_imgs) // n)
    candidates = [all_imgs[i * step] for i in range(n * 4) if i * step < len(all_imgs)]
    random.seed(seed)
    return random.sample(candidates, min(n, len(candidates)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output", default="demo.gif")
    parser.add_argument("--n-frames", type=int, default=12)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=800, help="output frame width")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    val_dir = Path(args.val_dir)
    images = pick_images(val_dir, args.n_frames)
    print(f"Selected {len(images)} images")

    frames = []
    for i, img_path in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {img_path.name}")
        im = Image.open(img_path).convert("RGB")
        labels, boxes, scores = run(sess, im, threshold=args.threshold)
        scale = args.width / im.width
        annotated = annotate(im, labels, boxes, scores, scale=scale)
        # resize to target width
        th = int(im.height * scale)
        annotated = annotated.resize((args.width, th), Image.LANCZOS)
        frames.append(annotated)

    # All frames must be same size — crop/pad to median height
    heights = sorted(f.height for f in frames)
    target_h = heights[len(heights) // 2]
    target_w = args.width
    padded = []
    for f in frames:
        canvas = Image.new("RGB", (target_w, target_h), (20, 20, 20))
        y_off = (target_h - f.height) // 2
        canvas.paste(f, (0, max(0, y_off)))
        padded.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=256))

    duration_ms = 1000 // args.fps
    padded[0].save(
        args.output,
        save_all=True,
        append_images=padded[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Saved {args.output}  ({len(padded)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
