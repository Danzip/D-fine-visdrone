"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import os
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb

            # Persist run ID across watchdog restarts so all epochs land in one W&B run.
            _run_id_file = self.output_dir / "wandb_run_id.txt"
            _run_id = None
            if _run_id_file.exists():
                _run_id = _run_id_file.read_text().strip()
                print(f"[W&B] Resuming run {_run_id}")

            _run = wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
                id=_run_id,
                resume="allow",
            )
            if not _run_id_file.exists():
                _run_id_file.write_text(_run.id)
                print(f"[W&B] New run {_run.id} — saved to {_run_id_file}")

            wandb.watch(self.model)
            wandb.define_metric("*", step_metric="epoch")

        # Image visualizer disabled — caused 400+ GB disk usage (BUG-028).
        wandb_viz = None
        if False and (self.use_wandb or self.writer):
            try:
                from .wandb_viz import WandbVisualizer
                val_cfg = args.yaml_cfg.get("val_dataloader", {}).get("dataset", {})
                ann_file = val_cfg.get("ann_file", "")
                img_folder = val_cfg.get("img_folder", "")
                if ann_file and img_folder and os.path.exists(ann_file):
                    import torch.nn as nn
                    class _DeployModel(nn.Module):
                        def __init__(self, model, postprocessor):
                            super().__init__()
                            self.model = model
                            self.postprocessor = postprocessor
                        def forward(self, images):
                            return self.model(images)
                    _deploy = _DeployModel(
                        self.ema.module if self.ema else self.model,
                        self.postprocessor,
                    )
                    # Use eval_spatial_size [H, W] so visualizer matches model input.
                    _eval_sz = args.yaml_cfg.get("eval_spatial_size", [640, 640])
                    wandb_viz = WandbVisualizer(
                        ann_json_path=ann_file,
                        images_dir=img_folder,
                        model=_deploy,
                        postprocessor=self.postprocessor,
                        device=self.device,
                        input_size=tuple(_eval_sz),
                    )
                    dest = []
                    if self.use_wandb:
                        dest.append("W&B")
                    if self.writer:
                        dest.append("TensorBoard")
                    print(f"[WandbViz] Visualizer ready. Will log 3 images/class every 500 steps to {' + '.join(dest)}.")
            except Exception as e:
                print(f"[WandbViz] Could not initialise visualizer: {e}")

        # BUG-043: stats() (calflops) crashes on the msfd 4-feat/3-level encoder
        # split at profile time — profiling is cosmetic, never fatal.
        try:
            n_parameters, model_stats = stats(self.cfg)
            print(model_stats)
        except Exception as e:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[stats] profiler failed ({type(e).__name__}: {e}) — continuing; params={n_parameters}")
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        # Startup sanity log — verify clean mosaic + square multi-res, no AR
        if dist_utils.is_main_process():
            collate = self.train_dataloader.collate_fn
            print("=" * 60)
            print("[STARTUP] Training configuration")
            _train_ops = [o.get("type","") for o in (args.yaml_cfg.get("train_dataloader",{}).get("dataset",{}).get("transforms",{}).get("ops",[]))]
            _mosaic_on = "Mosaic" in _train_ops
            print(f"  mosaic enabled:   {_mosaic_on}  (in dataset transforms)")
            print(f"  square multi-res: {collate.scales is not None}")
            if collate.scales is not None:
                print(f"  scale range:      {min(collate.scales)}–{max(collate.scales)}  ({len(collate.scales)} steps)")
            print(f"  adaptive_batch:   {getattr(collate, 'adaptive_batch', False)}")
            _ar = getattr(self.train_dataloader, '_ar_batch_sampler', None)
            print(f"  AR batching:      {'ENABLED (' + type(_ar).__name__ + ')' if _ar else 'DISABLED'}")
            print(f"  batch_size:       {self.train_dataloader.batch_size}")
            print(f"  num_workers:      {self.train_dataloader.num_workers}")
            print(f"  accum_steps:      {args.yaml_cfg.get('accum_steps', 1)}")
            print(f"  resume:           {self.cfg.resume or 'None (fresh start)'}")
            print(f"  start_epoch:      {start_epoch}")
            print(f"  total_epochs:     {args.epochs}")
            print(f"  output_dir:       {self.output_dir}")
            print("=" * 60)

        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # BUG-045: stop_epoch=0 means "no stage split, single-scale from the
            # start" (used by clean-finetune/polish configs) — but epoch==0 is
            # also true on iteration 0 of any run, so this unconditionally fired
            # a full state reload (LR scheduler, optimizer, last_epoch) from
            # best_stg1.pth on EVERY run with stop_epoch=0, even brand new ones.
            # Guard: only a genuine stage-1->stage-2 transition (stop_epoch > 0)
            # should trigger the reload.
            if self.train_dataloader.collate_fn.stop_epoch > 0 and epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
                wandb_viz=wandb_viz,
                viz_step_interval=500,
                accum_steps=args.yaml_cfg.get("accum_steps", 1),
                max_train_steps=args.yaml_cfg.get("max_train_steps", None),
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            torch.cuda.empty_cache()
            # Primary eval: 16:9 canvas — drives best_stat and checkpoint saving
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            # Secondary eval: 4:3 canvas — informational, logged with "43_" prefix
            if self.val_dataloader_43 is not None:
                test_stats_43, _ = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader_43,
                    self.evaluator,
                    self.device,
                    epoch,
                    use_wandb=False,   # suppress duplicate epoch-level wandb log
                )
                if self.use_wandb and dist_utils.is_main_process():
                    import wandb
                    log_43 = {"epoch": epoch}
                    for k, v in test_stats_43.items():
                        for i, val in enumerate(v):
                            log_43[f"eval_43/{k}_{i}"] = val
                    wandb.log(log_43)
                if self.writer and dist_utils.is_main_process():
                    for k, v in test_stats_43.items():
                        for i, val in enumerate(v):
                            self.writer.add_scalar(f"Test_43/{k}_{i}", val, epoch)
                print(f"[Eval 4:3]  {test_stats_43}")

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        stg2_path = self.output_dir / "best_stg2.pth"
                        stg1_path = self.output_dir / "best_stg1.pth"
                        rollback_path = stg2_path if stg2_path.exists() else stg1_path
                        state = torch.load(str(rollback_path), map_location="cpu")
                        # Use EMA weights (not live-model weights) for both copies.
                        # EMA is the smoothed best; live weights at save time are noisier.
                        ema_weights = state["ema"]["module"] if "ema" in state else state["model"]
                        live_module = dist_utils.de_parallel(self.model)
                        stat, _ = self._matched_state(live_module.state_dict(), ema_weights)
                        live_module.load_state_dict(stat, strict=False)
                        ema_module = dist_utils.de_parallel(self.ema).module
                        stat, _ = self._matched_state(ema_module.state_dict(), ema_weights)
                        ema_module.load_state_dict(stat, strict=False)
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay} (EMA weights from {rollback_path.name})")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

                # Log per-class GT vs prediction visualisations to W&B
                if wandb_viz is not None:
                    try:
                        wandb_viz.log_epoch(epoch)
                    except Exception as e:
                        print(f"[WandbViz] W&B logging failed at epoch {epoch}: {e}")

            # TensorBoard image logging is now done every 500 steps inside train_one_epoch

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
