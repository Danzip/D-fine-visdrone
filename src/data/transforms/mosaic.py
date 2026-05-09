"""
Mosaic4 augmentation for D-FINE.

Combines the current image with 3 randomly sampled images into a 2×2 grid.
Each tile is resized to half the target output size, then assembled into a
full-size mosaic. Bounding boxes from all 4 tiles are translated to their
final positions and sanitized.

This is compatible with D-FINE's 3-argument transform signature:
    forward(image, target, dataset) -> (image, target, dataset)

Insert early in the transform pipeline, before other geometric augmentations.
"""

import random

import torch
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from ...core import register
from .._misc import SanitizeBoundingBoxes, convert_to_tv_tensor

torchvision.disable_beta_transforms_warning()


@register()
class Mosaic(T.Transform):
    """4-image mosaic augmentation.

    Args:
        output_size: side length of the square output image (e.g. 1024)
        p:           probability of applying mosaic (otherwise pass through)
        fill:        background fill value for the canvas (default 114)
    """

    def __init__(self, output_size: int = 1024, p: float = 0.5, fill: int = 114) -> None:
        super().__init__()
        self.output_size = output_size
        self.p = p
        self.fill = fill
        self.tile_size = output_size // 2
        self._resize_tile = T.Resize(size=[self.tile_size, self.tile_size])
        self._sanitize = SanitizeBoundingBoxes(min_size=1)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target, dataset = inputs

        if torch.rand(1).item() >= self.p:
            return image, target, dataset

        # Gather 4 images: current + 3 random
        extra_idxs = random.choices(range(len(dataset)), k=3)
        images = [image]
        targets = [target]
        for idx in extra_idxs:
            im, tgt = dataset.load_item(idx)
            images.append(im)
            targets.append(tgt)

        # Resize each tile to half the output size
        resized_imgs, resized_tgts = [], []
        for im, tgt in zip(images, targets):
            im, tgt = self._resize_tile(im, tgt)
            resized_imgs.append(im)
            resized_tgts.append(tgt)

        # Build canvas
        ts = self.tile_size
        canvas = Image.new(mode=resized_imgs[0].mode, size=(ts * 2, ts * 2), color=self.fill)

        # Tile positions: top-left, top-right, bottom-left, bottom-right
        offsets_xy = [(0, 0), (ts, 0), (0, ts), (ts, ts)]
        for im, (ox, oy) in zip(resized_imgs, offsets_xy):
            canvas.paste(im, (ox, oy))

        # Translate bounding boxes to mosaic coordinates
        offset_tensors = [
            torch.tensor([[ox, oy, ox, oy]], dtype=torch.float32)
            for ox, oy in offsets_xy
        ]
        merged_boxes = []
        merged_labels = []
        for tgt, off in zip(resized_tgts, offset_tensors):
            if "boxes" in tgt and len(tgt["boxes"]) > 0:
                boxes = tgt["boxes"].float()  # BoundingBoxes xyxy pixel
                merged_boxes.append(boxes + off)
                merged_labels.append(tgt["labels"])

        if merged_boxes:
            all_boxes = torch.cat(merged_boxes, dim=0)
            all_labels = torch.cat(merged_labels, dim=0)
        else:
            all_boxes = torch.zeros((0, 4), dtype=torch.float32)
            all_labels = torch.zeros(0, dtype=torch.int64)

        # Keep fields from first target, replace boxes/labels
        out_target = {k: v for k, v in resized_tgts[0].items() if k not in ("boxes", "labels")}
        h, w = ts * 2, ts * 2
        out_target["boxes"] = convert_to_tv_tensor(
            all_boxes, key="boxes", box_format="xyxy", spatial_size=[h, w]
        )
        out_target["labels"] = all_labels

        canvas, out_target = self._sanitize(canvas, out_target)

        return canvas, out_target, dataset
