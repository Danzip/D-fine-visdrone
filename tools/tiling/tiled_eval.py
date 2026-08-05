"""
SAHI-style tiled-inference evaluation (standalone CLI).

Thin wrapper around src/solver/tiled_eval.py's evaluate_tiled() -- the same
function det_solver.py calls during training when a config has a
`tiled_eval:` block, so there is exactly one implementation of the
tile/infer/merge/NMS/score logic, not two that could drift apart.

For each *original* (untiled) image in --ann/--img-dir: slices it into the
same overlapping-tile grid used by build_tiled_visdrone.py
(src/data/tiling.py), runs each tile through the model, maps predicted boxes
back to full-image coordinates, merges duplicate detections from overlapping
tiles with class-wise NMS, and scores the merged per-image result against
the real ground truth (faster_coco_eval, maxDets=[1,10,500] to match
train.py's own eval convention).

Works on any checkpoint -- run it against a checkpoint that was never
trained on tiles to get a "SAHI-without-tile-training" control, or against
a tile-trained checkpoint for the real result.

Usage:
    python tools/tiling/tiled_eval.py -c <config.yml> -r <checkpoint.pth> \
        --ann dataset/visdrone/annotations/instances_val.json \
        --img-dir dataset/visdrone/VisDrone2019-DET-val/images \
        --tile 640 --overlap 0.5 --device cuda:0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.tiled_eval import evaluate_tiled


def build_solver(config_path: str, checkpoint_path: str, device):
    update = {"resume": checkpoint_path}
    if device:
        update["device"] = device
    cfg = YAMLConfig(config_path, **update)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()  # builds model/postprocessor, loads the checkpoint (shape-matched)
    return solver


def run(args):
    solver = build_solver(args.config, args.resume, args.device)
    module = solver.ema.module if solver.ema else solver.model

    stats, _ = evaluate_tiled(
        module,
        solver.postprocessor,
        solver.device,
        args.ann,
        args.img_dir,
        tile=args.tile,
        overlap=args.overlap,
        nms_iou=args.nms_iou,
        threshold=args.threshold,
    )

    ap = stats["coco_eval_bbox"]
    print(f"\nTiled eval ({args.tile}px tiles, {args.overlap * 100:.0f}% overlap) "
          f"on {Path(args.resume).name}:")
    print(f"  AP50:95     : {ap[0]:.4f}")
    print(f"  AP50        : {ap[1]:.4f}")
    print(f"  AP75        : {ap[2]:.4f}")
    print(f"  AP-small    : {ap[3]:.4f}")
    print(f"  AP-medium   : {ap[4]:.4f}")
    print(f"  AP-large    : {ap[5]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)
    run(parser.parse_args())
