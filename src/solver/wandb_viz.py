"""
W&B Per-Class Visualization for VisDrone Fine-tuning.

Called once per epoch after evaluation. For each of the 10 VisDrone classes,
picks 5 val images that contain that class, runs inference, and logs to W&B
showing GT boxes vs predicted boxes side by side in the same image.

W&B renders boxes as overlays on the image with a toggle in the UI.
The step slider lets you scrub through epochs to see training progress.

Usage (called from det_solver.py):
    from .wandb_viz import WandbVisualizer
    viz = WandbVisualizer(val_dataset, model, postprocessor, device, class_names)
    viz.log_epoch(epoch)
"""

import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# VisDrone class map: category_id (1-10) -> name
VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


class WandbVisualizer:
    """
    Logs 3 val images per class to TensorBoard (and optionally W&B), showing GT vs predictions.

    Pre-selection: at init, for each class pick the 3 val images with the most instances.

    TensorBoard: uses add_images() so each class is ONE panel (10 panels total).
    Step slider shows history; latest step is shown by default. Scrollable, not cluttered.
    """

    def __init__(
        self,
        ann_json_path: str,
        images_dir: str,
        model: nn.Module,
        postprocessor: nn.Module,
        device: str,
        n_per_class: int = 3,
        score_threshold: float = 0.3,
        input_size: tuple = (640, 640),
    ):
        self.images_dir = images_dir
        self.model = model
        self.postprocessor = postprocessor
        self.device = device
        self.n_per_class = n_per_class
        self.score_threshold = score_threshold
        self.input_size = input_size

        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
        ])

        # Load annotations and build per-class image selection
        self._load_and_select(ann_json_path)

    def _load_and_select(self, ann_json_path: str):
        """Pre-select 5 images per class with most instances of that class."""
        with open(ann_json_path) as f:
            coco = json.load(f)

        # Build lookups
        self.id_to_image = {img["id"]: img for img in coco["images"]}

        # Count annotations per (image_id, category_id)
        counts = defaultdict(lambda: defaultdict(int))  # counts[image_id][cat_id]
        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            counts[ann["image_id"]][ann["category_id"]] += 1
            ann_by_image[ann["image_id"]].append(ann)
        self.ann_by_image = dict(ann_by_image)

        # For each class, rank images by count of that class, pick top-N
        self.class_image_ids = {}
        for cat_id in VISDRONE_CLASSES:
            ranked = sorted(
                [iid for iid in counts if counts[iid][cat_id] > 0],
                key=lambda iid: -counts[iid][cat_id],
            )
            self.class_image_ids[cat_id] = ranked[: self.n_per_class]

        # Unique set of image IDs we will ever need
        self.selected_ids = set()
        for ids in self.class_image_ids.values():
            self.selected_ids.update(ids)

        print(
            f"[WandbViz] Pre-selected {len(self.selected_ids)} unique val images "
            f"({self.n_per_class} per class × {len(VISDRONE_CLASSES)} classes)"
        )

    @torch.no_grad()
    def _run_inference(self, image_id: int):
        """Load image, run model, return (pil_image, pred_labels, pred_boxes, pred_scores)."""
        img_info = self.id_to_image[image_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        im = Image.open(img_path).convert("RGB")
        orig_w, orig_h = im.size

        tensor = self.transform(im).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([[orig_w, orig_h]], dtype=torch.float32).to(self.device)

        outputs = self.model(tensor)
        results = self.postprocessor(outputs, orig_size)
        # postprocessor returns list of dicts [{"labels":..., "boxes":..., "scores":...}]
        result = results[0]
        labels = result["labels"]
        boxes = result["boxes"]
        scores = result["scores"]

        mask = scores > self.score_threshold
        return (
            im,
            labels[mask].cpu().numpy().tolist(),
            boxes[mask].cpu().numpy().tolist(),
            scores[mask].cpu().numpy().tolist(),
        )

    def _make_wandb_image(self, image_id: int, pil_image, pred_labels, pred_boxes, pred_scores):
        """
        Create a wandb.Image with GT and prediction box overlays.

        W&B box format:
            {"position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
             "class_id": int, "scores": {"confidence": float}}

        Both GT and predictions use the same class_labels dict, so the legend
        shows class names. GT boxes show in one colour, predictions in another
        (W&B handles colouring per box-set automatically).
        """
        import wandb

        class_labels = {k: v for k, v in VISDRONE_CLASSES.items()}

        # GT boxes
        gt_anns = self.ann_by_image.get(image_id, [])
        gt_box_data = []
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            gt_box_data.append({
                "position": {
                    "minX": float(x),
                    "minY": float(y),
                    "maxX": float(x + w),
                    "maxY": float(y + h),
                },
                "class_id": ann["category_id"],
                "domain": "pixel",
            })

        # Prediction boxes
        pred_box_data = []
        for lbl, box, scr in zip(pred_labels, pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            pred_box_data.append({
                "position": {
                    "minX": float(x1),
                    "minY": float(y1),
                    "maxX": float(x2),
                    "maxY": float(y2),
                },
                "class_id": int(lbl),
                "scores": {"confidence": float(scr)},
                "domain": "pixel",
            })

        return wandb.Image(
            np.array(pil_image),
            boxes={
                "ground_truth": {
                    "box_data": gt_box_data,
                    "class_labels": class_labels,
                },
                "predictions": {
                    "box_data": pred_box_data,
                    "class_labels": class_labels,
                },
            },
            caption=self.id_to_image[image_id]["file_name"],
        )

    def log_epoch(self, epoch: int):
        """
        Run inference on all pre-selected images and log to W&B.
        Organised as: viz/class_name -> list of 5 wandb.Image objects.
        """
        import wandb

        # Cache inference results (one pass per unique image)
        inference_cache = {}
        for image_id in self.selected_ids:
            try:
                inference_cache[image_id] = self._run_inference(image_id)
            except Exception as e:
                print(f"[WandbViz] Warning: failed on image {image_id}: {e}")

        # Build per-class W&B image lists
        log_dict = {"epoch": epoch}
        for cat_id, cat_name in VISDRONE_CLASSES.items():
            wb_images = []
            for image_id in self.class_image_ids.get(cat_id, []):
                if image_id not in inference_cache:
                    continue
                pil_im, pred_labels, pred_boxes, pred_scores = inference_cache[image_id]
                wb_img = self._make_wandb_image(
                    image_id, pil_im, pred_labels, pred_boxes, pred_scores
                )
                wb_images.append(wb_img)

            if wb_images:
                log_dict[f"viz/{cat_name}"] = wb_images

        wandb.log(log_dict, step=epoch)
        print(
            f"[WandbViz] Logged {len(self.selected_ids)} images to W&B "
            f"(epoch {epoch}, threshold={self.score_threshold})"
        )

    @torch.no_grad()
    def log_epoch_tensorboard(self, writer, global_step: int):
        """
        Run inference on pre-selected images and log GT+pred box overlays to TensorBoard.

        Uses add_images() (plural) — all N images for a class are stacked into ONE panel
        (tag: val/<class_name>). This gives 10 panels total instead of 30, keeping the
        Images tab scrollable and uncluttered. The step slider shows history; latest is
        shown by default.

        Green = ground-truth, Red = predictions above score_threshold.
        Call writer.flush() after to ensure TensorBoard picks up the data immediately.
        """
        from torchvision.utils import draw_bounding_boxes
        import torchvision.transforms.functional as TF

        # One inference pass per unique image, cached for reuse across classes
        inference_cache = {}
        for image_id in self.selected_ids:
            try:
                inference_cache[image_id] = self._run_inference(image_id)
            except Exception as e:
                print(f"[WandbViz/TB] Warning: failed on image {image_id}: {e}")

        n_logged = 0
        for cat_id, cat_name in VISDRONE_CLASSES.items():
            class_imgs = []
            for image_id in self.class_image_ids.get(cat_id, []):
                if image_id not in inference_cache:
                    continue
                pil_im, pred_labels, pred_boxes, pred_scores = inference_cache[image_id]

                # [3, H, W] uint8 — draw_bounding_boxes requires uint8
                img_uint8 = TF.to_tensor(pil_im)  # float [0,1]
                img_uint8 = (img_uint8 * 255).to(torch.uint8)

                # --- Ground-truth boxes (green) ---
                gt_anns = self.ann_by_image.get(image_id, [])
                if gt_anns:
                    gt_boxes = torch.tensor(
                        [[a["bbox"][0], a["bbox"][1],
                          a["bbox"][0] + a["bbox"][2],
                          a["bbox"][1] + a["bbox"][3]] for a in gt_anns],
                        dtype=torch.float32,
                    )
                    gt_labels = [VISDRONE_CLASSES.get(a["category_id"], "?") for a in gt_anns]
                    img_uint8 = draw_bounding_boxes(
                        img_uint8, gt_boxes, labels=gt_labels, colors="green", width=2
                    )

                # --- Prediction boxes (red) ---
                if pred_boxes:
                    pred_box_t = torch.tensor(pred_boxes, dtype=torch.float32)
                    pred_lbl_strs = [
                        f"{VISDRONE_CLASSES.get(int(l), '?')} {s:.2f}"
                        for l, s in zip(pred_labels, pred_scores)
                    ]
                    img_uint8 = draw_bounding_boxes(
                        img_uint8, pred_box_t, labels=pred_lbl_strs, colors="red", width=2
                    )

                # Resize to fixed display size so images can be stacked (VisDrone has varying resolutions)
                img_display = TF.resize(img_uint8, list(self.input_size))
                class_imgs.append(img_display.float() / 255.0)  # [C, H, W] float

            if class_imgs:
                # Stack into [N, C, H, W] — one panel per class in TensorBoard
                stacked = torch.stack(class_imgs)
                writer.add_images(f"val/{cat_name}", stacked, global_step=global_step)
                n_logged += len(class_imgs)

        writer.flush()
        print(
            f"[WandbViz/TB] Logged {n_logged} annotated images to TensorBoard "
            f"(step {global_step}, threshold={self.score_threshold})"
        )
