"""Full-image, tiled + NMS-merged evaluation.

Shared by tools/tiling/tiled_eval.py (standalone CLI) and, when a
`tiled_eval:` block is present in the experiment config, by the training
loop itself (det_solver.py) -- so checkpoint selection during training is
driven by the metric the model is actually meant to be good at (tiled
inference + NMS merge on full images), not a whole-image resize proxy.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision
from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image

from ..data.tiling import get_tile_boxes


def _image_to_tensor(im: Image.Image, device) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def evaluate_tiled(
    model,
    postprocessor,
    device,
    ann_file,
    img_dir,
    tile: int = 640,
    overlap: float = 0.5,
    nms_iou: float = 0.5,
    threshold: float = 0.3,
):
    """Slice every image in `ann_file`/`img_dir` into overlapping tiles, run
    the model per tile, map boxes back to full-image coordinates, NMS-merge
    duplicate detections per image, and score against the real (untiled)
    ground truth.

    Returns (stats, coco_eval): `stats` is `{"coco_eval_bbox": [AP50:95, AP50,
    AP75, APsmall, APmedium, APlarge, ...]}`, matching det_engine.evaluate()'s
    stats shape for drop-in use by the training loop. `coco_eval` is the raw
    COCOeval_faster object (not the CocoEvaluator wrapper det_engine.evaluate()
    returns) -- callers that only need `.eval`-style artifact saving should
    treat it as optional/None-checked.
    """
    was_training = model.training
    model.eval()

    coco_gt = COCO(str(ann_file))
    img_dir = Path(img_dir)
    results = []

    for image_id, info in coco_gt.imgs.items():
        img_path = img_dir / info["file_name"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size

            boxes_all, scores_all, labels_all = [], [], []
            for tx0, ty0, tx1, ty1 in get_tile_boxes(w, h, tile=tile, overlap=overlap):
                tile_im = im.crop((tx0, ty0, tx1, ty1))
                tile_w, tile_h = tile_im.size
                if (tile_w, tile_h) != (tile, tile):
                    tile_im = tile_im.resize((tile, tile))

                outputs = model(_image_to_tensor(tile_im, device))
                orig_sizes = torch.tensor([[tile_w, tile_h]], device=device)
                result = postprocessor(outputs, orig_sizes)[0]

                keep = result["scores"] >= threshold
                boxes = result["boxes"][keep].cpu()
                boxes[:, [0, 2]] += tx0
                boxes[:, [1, 3]] += ty0
                boxes_all.append(boxes)
                scores_all.append(result["scores"][keep].cpu())
                labels_all.append(result["labels"][keep].cpu())

            if not any(b.numel() for b in boxes_all):
                continue
            boxes_all = torch.cat(boxes_all)
            scores_all = torch.cat(scores_all)
            labels_all = torch.cat(labels_all)

            keep_idx = []
            for label in labels_all.unique():
                idx = (labels_all == label).nonzero(as_tuple=True)[0]
                nms_keep = torchvision.ops.nms(boxes_all[idx], scores_all[idx], nms_iou)
                keep_idx.append(idx[nms_keep])
            keep_idx = torch.cat(keep_idx)

            for i in keep_idx.tolist():
                x1, y1, x2, y2 = boxes_all[i].tolist()
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(labels_all[i].item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(scores_all[i].item()),
                    }
                )

    if was_training:
        model.train()

    if not results:
        return {"coco_eval_bbox": [0.0] * 12}, None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.maxDets = [1, 10, 500]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {"coco_eval_bbox": coco_eval.stats.tolist()}, coco_eval
