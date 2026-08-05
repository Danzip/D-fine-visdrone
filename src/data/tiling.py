"""Tile geometry shared by the offline tiled-dataset builder
(tools/tiling/build_tiled_visdrone.py) and the sliced-inference eval script
(tools/tiling/tiled_eval.py), so train-time and eval-time tiling are
guaranteed to use the identical grid.
"""

from typing import Dict, List, Tuple


def get_tile_boxes(
    img_w: int, img_h: int, tile: int = 640, overlap: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """Overlapping tile grid covering a (img_w, img_h) image.

    Each tile is `tile` x `tile` (or smaller, if the image itself is smaller
    than `tile` in that dimension). Adjacent tiles overlap by `overlap`
    (stride = tile * (1 - overlap)). The last tile in each row/column is
    anchored to the image edge instead of overshooting it.

    Returns a list of (x0, y0, x1, y1) absolute pixel boxes.
    """
    stride = max(1, round(tile * (1 - overlap)))

    def starts(size: int) -> List[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile + 1, stride))
        if s[-1] != size - tile:
            s.append(size - tile)
        return s

    boxes = []
    for y0 in starts(img_h):
        for x0 in starts(img_w):
            boxes.append((x0, y0, min(x0 + tile, img_w), min(y0 + tile, img_h)))
    return boxes


def remap_boxes_to_tile(
    anns: List[Dict], tile_box: Tuple[int, int, int, int], min_visible_ratio: float = 0.2
) -> List[Dict]:
    """Clip COCO-style annotations (dicts with an [x,y,w,h] 'bbox' key, absolute
    pixels) to a tile and remap them to tile-local coordinates. An annotation is
    dropped if less than `min_visible_ratio` of its original area survives the
    clip. Other keys on each ann dict are shallow-copied through unchanged
    (except 'bbox'/'area', which are overwritten with the clipped values).
    """
    x0, y0, x1, y1 = tile_box
    out = []
    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        ix0, iy0 = max(bx, x0), max(by, y0)
        ix1, iy1 = min(bx + bw, x1), min(by + bh, y1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        orig_area = max(1e-6, bw * bh)
        if (iw * ih) / orig_area < min_visible_ratio:
            continue
        new_ann = dict(ann)
        new_ann["bbox"] = [ix0 - x0, iy0 - y0, iw, ih]
        new_ann["area"] = iw * ih
        out.append(new_ann)
    return out
