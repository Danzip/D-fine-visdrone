"""
Offline tiled-dataset builder for SAHI-style train-time tiling.

Slices each VisDrone train image into overlapping tiles (640x640 / 50%
overlap by default) and remaps ground-truth boxes into tile-local
coordinates, producing a new COCO-format train split. Val is intentionally
left untiled here -- tools/tiling/tiled_eval.py tiles val images on the fly
at inference time, so eval always runs against the real, un-cropped ground
truth (matching how the model will actually be used at inference).

Not committed to git -- regenerate this on the training machine/pod after
`git pull` (see .gitignore for dataset/visdrone_tiled/).

Usage:
    python tools/tiling/build_tiled_visdrone.py \
        --ann dataset/visdrone/annotations/instances_train.json \
        --img-dir dataset/visdrone/VisDrone2019-DET-train/images \
        --out-dir dataset/visdrone_tiled/train \
        --tile 640 --overlap 0.5 --min-visible 0.2
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.tiling import get_tile_boxes, remap_boxes_to_tile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--min-visible", type=float, default=0.2)
    args = p.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ann) as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_anns = []
    next_img_id = 0
    next_ann_id = 0

    for img_info in coco["images"]:
        img_path = img_dir / img_info["file_name"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            src_anns = anns_by_image.get(img_info["id"], [])

            for tx0, ty0, tx1, ty1 in get_tile_boxes(w, h, tile=args.tile, overlap=args.overlap):
                tile_anns = remap_boxes_to_tile(src_anns, (tx0, ty0, tx1, ty1), args.min_visible)

                tile_im = im.crop((tx0, ty0, tx1, ty1))
                tile_w, tile_h = tile_im.size
                tile_name = f"{Path(img_info['file_name']).stem}_{tx0}_{ty0}.jpg"
                tile_im.save(out_img_dir / tile_name, quality=95)

                new_images.append(
                    {"id": next_img_id, "file_name": tile_name, "width": tile_w, "height": tile_h}
                )
                for ann in tile_anns:
                    new_anns.append(
                        {
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],
                            "area": ann["area"],
                            "iscrowd": ann.get("iscrowd", 0),
                        }
                    )
                    next_ann_id += 1
                next_img_id += 1

    out_ann_path = out_dir / "instances_train_tiled.json"
    with open(out_ann_path, "w") as f:
        json.dump({"images": new_images, "annotations": new_anns, "categories": coco["categories"]}, f)

    n_src_images = len(coco["images"])
    n_src_boxes = len(coco["annotations"])
    print(f"Tiled {n_src_images} source images -> {len(new_images)} tiles "
          f"({len(new_images) / n_src_images:.1f}x)")
    print(f"Boxes: {n_src_boxes} source -> {len(new_anns)} tile-local "
          f"({len(new_anns) / max(1, n_src_boxes):.2f}x)")
    print(f"Wrote {out_ann_path}")
    print(f"Wrote {len(new_images)} tile images to {out_img_dir}")


if __name__ == "__main__":
    main()
