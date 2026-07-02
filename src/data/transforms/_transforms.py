"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2._utils import _get_fill

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class LetterboxResize:
    """
    Fit an image into a target canvas without squishing.

    size=int  → classic square letterbox: resize longest side to `size`,
                center-pad shorter side to produce a `size×size` canvas.

    size=[h,w] → AR-aware rectangular letterbox: pad image to match target
                 AR (canvas_w/canvas_h), then resize to exactly (h, w).
                 Same logic as _pad_to_ar_and_resize in the collate.

    Stores unified 5-element letterbox in target["letterbox"]:
        [scale, pad_top, pad_left, padded_h, padded_w]
    where 'padded' refers to the intermediate AR-matched size before resize.
    Unwind: orig_x = norm_x * padded_w - pad_left; orig_y = norm_y * padded_h - pad_top
    Then divide by scale (1.0 for rectangular path, resize_scale for square path).

    Works on PIL images or float tensors (C×H×W).
    Boxes must be in normalized cxcywh format.
    """

    def __init__(self, size) -> None:
        if isinstance(size, int):
            self.target_h = self.target_w = size
            self.square = True
        else:
            self.target_h, self.target_w = int(size[0]), int(size[1])
            self.square = False

    def __call__(self, image, target=None):
        import torch
        import torch.nn.functional as TF
        from PIL import Image as PILImage

        # Unpack sample tuple passed by Compose.default_forward: (img, target[, dataset, ...])
        _extra = ()
        if isinstance(image, (tuple, list)):
            parts = image
            image = parts[0]
            target = parts[1] if len(parts) > 1 else None
            _extra = tuple(parts[2:])

        th, tw = self.target_h, self.target_w

        if isinstance(image, PILImage.Image):
            w, h = image.size   # PIL: (width, height)
        else:
            _, h, w = image.shape

        if self.square:
            # ── Square path: resize longest side → sz, center-pad to sz×sz ──
            sz = th
            scale = sz / max(h, w)
            new_h, new_w = round(h * scale), round(w * scale)
            pad_top  = (sz - new_h) // 2
            pad_left = (sz - new_w) // 2

            if isinstance(image, PILImage.Image):
                image = image.resize((new_w, new_h), PILImage.BILINEAR)
                canvas = PILImage.new(image.mode, (sz, sz), 0)
                canvas.paste(image, (pad_left, pad_top))
                image = canvas
            else:
                image = TF.interpolate(
                    image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
                ).squeeze(0)
                pad_bottom = sz - new_h - pad_top
                pad_right  = sz - new_w - pad_left
                image = TF.pad(image, [pad_left, pad_right, pad_top, pad_bottom], value=0.0)

            if target is not None:
                if "boxes" in target and len(target["boxes"]):
                    boxes = target["boxes"].clone()
                    # Boxes were normalized to (w, h); after uniform resize they stay [0,1];
                    # after center-pad to (sz, sz) adjust for shifted origin.
                    boxes[:, 0] = (boxes[:, 0] * new_w + pad_left) / sz
                    boxes[:, 1] = (boxes[:, 1] * new_h + pad_top)  / sz
                    boxes[:, 2] =  boxes[:, 2] * new_w / sz
                    boxes[:, 3] =  boxes[:, 3] * new_h / sz
                    target["boxes"] = boxes
                # Unwind: orig = (norm * sz - pad) / scale
                target["letterbox"] = torch.tensor(
                    [scale, float(pad_top), float(pad_left), float(sz), float(sz)],
                    dtype=torch.float32,
                )
                return (image, target) + _extra
            return (image,) + _extra if _extra else image

        else:
            # ── Rectangular path: pad to target AR, resize to (th, tw) ──────
            target_ar = tw / th
            current_ar = w / h

            if current_ar < target_ar - 1e-4:
                # Image taller than target → pad sides
                padded_w = round(h * target_ar)
                padded_h = h
                pad_left = (padded_w - w) // 2
                pad_right = padded_w - w - pad_left
                pad_top = pad_bottom = 0
            elif current_ar > target_ar + 1e-4:
                # Image wider than target → pad top/bottom
                padded_h = round(w / target_ar)
                padded_w = w
                pad_top = (padded_h - h) // 2
                pad_bottom = padded_h - h - pad_top
                pad_left = pad_right = 0
            else:
                padded_h, padded_w = h, w
                pad_top = pad_bottom = pad_left = pad_right = 0

            if pad_top or pad_bottom or pad_left or pad_right:
                if isinstance(image, PILImage.Image):
                    canvas = PILImage.new(image.mode, (padded_w, padded_h), 0)
                    canvas.paste(image, (pad_left, pad_top))
                    image = canvas
                else:
                    image = TF.pad(image, [pad_left, pad_right, pad_top, pad_bottom], value=0.0)

            if isinstance(image, PILImage.Image):
                image = image.resize((tw, th), PILImage.BILINEAR)
            else:
                image = TF.interpolate(
                    image.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
                ).squeeze(0)

            if target is not None:
                if "boxes" in target and len(target["boxes"]):
                    boxes = target["boxes"].clone()
                    # Boxes normalized to (w, h); adjust for AR padding, resize is uniform.
                    boxes[:, 0] = (boxes[:, 0] * w + pad_left) / padded_w
                    boxes[:, 1] = (boxes[:, 1] * h + pad_top)  / padded_h
                    boxes[:, 2] =  boxes[:, 2] * w / padded_w
                    boxes[:, 3] =  boxes[:, 3] * h / padded_h
                    target["boxes"] = boxes
                # Unwind: orig_x = norm_x * padded_w - pad_left  (scale=1.0, no divide)
                target["letterbox"] = torch.tensor(
                    [1.0, float(pad_top), float(pad_left), float(padded_h), float(padded_w)],
                    dtype=torch.float32,
                )
                return (image, target) + _extra
            return (image,) + _extra if _extra else image


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill(self._fill, type(inpt))
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt


@register()
class CopyPasteSmallObjects:
    """Copy-paste augmentation for small objects (YOLOv8-style, same-image).

    For each chosen small object:
      1. Crop the pixel region from the image.
      2. Optionally flip it horizontally (50% chance).
      3. Find a random paste location where the new box has IoA < ioa_thresh
         with every existing box (tries up to max_attempts random locations).
      4. Paste the crop onto the image.
      5. Append the new box [x1,y1,x2,y2] and its label to the target.

    Ground truth is updated atomically: boxes and labels lists are extended
    together so every pasted pixel has exactly one corresponding annotation.

    Insert AFTER ConvertPILImage (float32 [0,1] tensor) and BEFORE ConvertBoxes
    (boxes are still xyxy pixel coords at this point).

    Args:
        max_size:      max side length (px) to treat an object as "small"
        num_paste:     max number of objects to paste per image
        p:             probability of applying the transform at all
        ioa_thresh:    max allowed intersection-over-paste-area before rejecting a location
        max_attempts:  random location tries before giving up on one object
    """

    def __init__(
        self,
        max_size: int = 32,
        num_paste: int = 10,
        p: float = 0.5,
        ioa_thresh: float = 0.30,
        max_attempts: int = 10,
        rare_labels: list = None,
        rare_boost: float = 3.0,
        scale_jitter: list = None,
    ) -> None:
        self.max_size = max_size
        self.num_paste = num_paste
        self.p = p
        self.ioa_thresh = ioa_thresh
        self.max_attempts = max_attempts
        # R3: rare-class targeting — rare_labels get rare_boost× sampling weight;
        # scale_jitter=[lo, hi] resizes each crop by a random factor before pasting.
        self.rare_labels = set(rare_labels) if rare_labels else None
        self.rare_boost = rare_boost
        self.scale_jitter = tuple(scale_jitter) if scale_jitter else None

    @staticmethod
    def _max_ioa(candidate: torch.Tensor, existing: torch.Tensor) -> float:
        """Max intersection-over-candidate-area between candidate box and all existing boxes."""
        # candidate: [4]  existing: [N, 4]  — xyxy pixel coords
        if existing.shape[0] == 0:
            return 0.0
        ix1 = torch.maximum(candidate[0], existing[:, 0])
        iy1 = torch.maximum(candidate[1], existing[:, 1])
        ix2 = torch.minimum(candidate[2], existing[:, 2])
        iy2 = torch.minimum(candidate[3], existing[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        cand_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
        if cand_area <= 0:
            return 1.0
        return (inter / cand_area).max().item()

    def __call__(self, *inputs):
        if torch.rand(1).item() >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        inputs = inputs if len(inputs) > 1 else inputs[0]
        # Support 3-tuple (image, target, dataset) from Compose pipeline
        extra = inputs[2:] if len(inputs) > 2 else ()
        image, target = inputs[0], inputs[1]

        # boxes: BoundingBoxes xyxy pixel coords [N, 4]
        boxes = target["boxes"]
        labels = target["labels"]

        if len(boxes) == 0:
            return (image, target) + extra

        # Select small objects
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        small_mask = (
            (widths <= self.max_size) & (heights <= self.max_size)
            & (widths > 1) & (heights > 1)
        )
        small_idx = torch.where(small_mask)[0]
        if len(small_idx) == 0:
            return (image, target) + extra

        C, H, W = image.shape
        image = image.clone()  # avoid modifying the original tensor in-place

        n_paste = min(self.num_paste, len(small_idx))
        if self.rare_labels is not None:
            weights = torch.tensor(
                [self.rare_boost if int(labels[i]) in self.rare_labels else 1.0
                 for i in small_idx],
                dtype=torch.float32,
            )
            chosen = small_idx[torch.multinomial(weights, n_paste, replacement=False)]
        else:
            chosen = small_idx[torch.randperm(len(small_idx))[:n_paste]]

        # Running list of all boxes (original + newly pasted) used for IoA checks
        all_boxes_list = boxes.tolist()
        new_boxes = []
        new_labels = []

        for idx in chosen:
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[:, y1:y2, x1:x2].clone()
            oh, ow = crop.shape[1], crop.shape[2]
            if W <= ow or H <= oh:
                continue

            # Optional horizontal flip (50%)
            if torch.rand(1).item() < 0.5:
                crop = crop.flip(-1)

            # R3: random scale jitter (kept within image bounds, min 2px)
            if self.scale_jitter is not None:
                lo, hi = self.scale_jitter
                s = lo + (hi - lo) * torch.rand(1).item()
                nh = max(2, min(H - 1, int(round(oh * s))))
                nw = max(2, min(W - 1, int(round(ow * s))))
                if (nh, nw) != (oh, ow):
                    crop = torch.nn.functional.interpolate(
                        crop.unsqueeze(0), size=(nh, nw),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                    oh, ow = nh, nw

            # Find a non-overlapping paste location
            placed = False
            current_boxes = torch.tensor(all_boxes_list, dtype=boxes.dtype) if all_boxes_list else torch.zeros((0, 4), dtype=boxes.dtype)
            for _ in range(self.max_attempts):
                px = torch.randint(0, W - ow, (1,)).item()
                py = torch.randint(0, H - oh, (1,)).item()
                candidate = torch.tensor([px, py, px + ow, py + oh], dtype=boxes.dtype)
                if self._max_ioa(candidate, current_boxes) < self.ioa_thresh:
                    placed = True
                    break

            if not placed:
                continue

            # Paste pixel data
            image[:, py:py + oh, px:px + ow] = crop

            # Record ground truth for pasted object
            new_box = [px, py, px + ow, py + oh]
            all_boxes_list.append(new_box)
            new_boxes.append(torch.tensor(new_box, dtype=boxes.dtype))
            new_labels.append(labels[idx])

        if new_boxes:
            canvas_size = getattr(boxes, _boxes_keys[1])
            all_boxes_tensor = torch.cat([boxes, torch.stack(new_boxes)], dim=0)
            # Re-wrap as BoundingBoxes so downstream transforms treat it correctly
            all_boxes_tensor = convert_to_tv_tensor(
                all_boxes_tensor, key="boxes", box_format="xyxy", spatial_size=canvas_size
            )
            target["boxes"] = all_boxes_tensor
            target["labels"] = torch.cat([labels, torch.stack(new_labels)])

        return (image, target) + extra
