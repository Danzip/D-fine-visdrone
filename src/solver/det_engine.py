"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
import torchvision
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    print(f"[train] AMP enabled: {scaler is not None}")
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    accum_steps: int = kwargs.get("accum_steps", 1)  # gradient accumulation steps
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    wandb_viz = kwargs.get("wandb_viz", None)
    viz_step_interval = kwargs.get("viz_step_interval", 500)
    max_train_steps = kwargs.get("max_train_steps", None)

    n_batches = len(data_loader)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * n_batches + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=n_batches)

        is_last_batch = (i + 1 == n_batches)
        do_step = ((i + 1) % accum_steps == 0) or is_last_batch

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # GT integrity assertions (every batch)
        for _tgt in targets:
            if "boxes" in _tgt and len(_tgt["boxes"]):
                _b = _tgt["boxes"]
                assert not torch.isnan(_b).any(), "NaN in target boxes"
                assert not torch.isinf(_b).any(), "Inf in target boxes"
                assert (_b >= -0.01).all() and (_b <= 1.01).all(), \
                    f"Box out of normalized range: min={_b.min():.4f} max={_b.max():.4f}"
                assert (_b[:, 2] > 0).all(), "Zero/negative box width in batch"
                assert (_b[:, 3] > 0).all(), "Zero/negative box height in batch"

        # Verbose sanity dump — first 5 batches of epoch 0 only
        if epoch == 0 and i < 5 and dist_utils.is_main_process():
            assert not torch.isnan(samples).any(), f"NaN in image tensor at batch {i}"
            assert not torch.isinf(samples).any(), f"Inf in image tensor at batch {i}"
            n_boxes = [len(t.get("boxes", [])) for t in targets]
            labels_flat = torch.cat([t["labels"] for t in targets if len(t.get("labels", []))])
            boxes_flat = torch.cat([t["boxes"] for t in targets if len(t.get("boxes", []))])
            print(
                f"[sanity batch {i}]  img={tuple(samples.shape)}  "
                f"n_boxes={n_boxes}  "
                f"labels=[{int(labels_flat.min())},{int(labels_flat.max())}]  "
                f"box_min={boxes_flat.min():.3f}  box_max={boxes_flat.max():.3f}"
            )

        try:
            if scaler is not None:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(samples, targets=targets)

                if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                    print(outputs["pred_boxes"])
                    state = model.state_dict()
                    new_state = {}
                    for key, value in model.state_dict().items():
                        new_key = key.replace("module.", "")
                        state[new_key] = value
                    new_state["model"] = state
                    dist_utils.save_on_master(new_state, "./NaN.pth")

                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)

                loss = sum(loss_dict.values()) / accum_steps
                scaler.scale(loss).backward()
                # Sync + release freed graph memory back to driver before the next
                # forward pass (different resolution/size). Prevents fragmentation OOM.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if do_step:
                    if max_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                outputs = model(samples, targets=targets)
                loss_dict = criterion(outputs, targets, **metas)

                loss: torch.Tensor = sum(loss_dict.values()) / accum_steps
                loss.backward()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if do_step:
                    if max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()

        except torch.OutOfMemoryError:
            print(f"[OOM] epoch={epoch} batch={i} — skipping batch and continuing")
            optimizer.zero_grad()
            if scaler is not None:
                scaler.update()
            torch.cuda.empty_cache()
            continue

        # ema — update only on optimizer steps to stay consistent with effective batch
        if ema is not None and do_step:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if dist_utils.is_main_process() and global_step % 10 == 0:
            if writer:
                writer.add_scalar("Loss/total", loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f"Loss/{k}", v.item(), global_step)
            if use_wandb:
                import wandb
                step_log = {"step": global_step, "epoch": epoch, "train/loss": loss_value.item()}
                for j, pg in enumerate(optimizer.param_groups):
                    step_log[f"train/lr_pg{j}"] = pg["lr"]
                for k, v in loss_dict_reduced.items():
                    step_log[f"train/{k}"] = v.item()
                wandb.log(step_log, step=global_step)

        # Per-step image visualization every viz_step_interval steps
        if (
            wandb_viz is not None
            and dist_utils.is_main_process()
            and global_step % viz_step_interval == 0
        ):
            torch.cuda.empty_cache()
            if writer:
                try:
                    wandb_viz.log_epoch_tensorboard(writer, global_step)
                except Exception as e:
                    print(f"[WandbViz/TB] Step image logging failed at step {global_step}: {e}")
            if use_wandb:
                try:
                    wandb_viz.log_epoch(global_step)
                except Exception as e:
                    print(f"[WandbViz/W&B] Step image logging failed at step {global_step}: {e}")

        if max_train_steps is not None and i + 1 >= max_train_steps:
            print(f"[train] Stopping early at {i + 1} steps (max_train_steps={max_train_steps})")
            break

    if use_wandb:
        wandb_log = {"epoch": epoch, "train/epoch_loss": np.mean(losses)}
        for j, pg in enumerate(optimizer.param_groups):
            wandb_log[f"train/epoch_lr_pg{j}"] = pg["lr"]
        for k, meter in metric_logger.meters.items():
            if k != "lr":
                wandb_log[f"train/epoch_{k}"] = meter.global_avg
        wandb.log(wandb_log, step=global_step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt_boxes = scale_boxes(
                target["boxes"],
                (target["orig_size"][1], target["orig_size"][0]),
                (samples[idx].shape[-1], samples[idx].shape[-2]),
            )
            gt.append({"boxes": gt_boxes, "labels": target["labels"]})
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
