#!/usr/bin/env python3
"""
D-FINE pruning recovery training — 10 epochs to recover AP after pruning.

Loads the pruned model checkpoint produced by prune_dfine.py, then trains for
10 epochs using only task losses (no sparsity loss). Constant flat LR = 1e-5.

The goal is to allow remaining neurons to redistribute the capacity that was
held by pruned neurons, recovering AP toward the pre-pruning level.

Usage (from D-FINE/):
  python tools/pruning/recovery_train.py \\
      -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \\
      --pruned-checkpoint output/pruning/best_pruned.pth \\
      --output-dir output/pruning_recovery \\
      --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver.det_engine import evaluate, train_one_epoch

# ── Recovery hyperparameters ──────────────────────────────────────────────
LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_NORM = 0.1
RECOVERY_EPOCHS = 10
USE_AMP = True
# ─────────────────────────────────────────────────────────────────────────


def get_decoder(model: nn.Module):
    m = model.module if hasattr(model, "module") else model
    return m.decoder.decoder  # DFINE → DFINETransformer → TransformerDecoder


def get_ffn_dims(model: nn.Module):
    return [layer.linear1.out_features for layer in get_decoder(model).layers]


def apply_ffn_dims(model: nn.Module, ffn_dims: list):
    """Resize FFN layers to match the pruned dims before loading state dict."""
    device = next(model.parameters()).device
    for layer, new_dim in zip(get_decoder(model).layers, ffn_dims):
        if layer.linear1.out_features != new_dim:
            d_model = layer.linear1.in_features
            layer.linear1 = nn.Linear(d_model, new_dim).to(device)
            layer.linear2 = nn.Linear(new_dim, d_model).to(device)


def get_ap(stats: dict) -> float:
    if "coco_eval_bbox" in stats:
        return stats["coco_eval_bbox"][0]
    k = next(iter(stats))
    return stats[k][0]


def main():
    parser = argparse.ArgumentParser(description="D-FINE pruning recovery training")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--pruned-checkpoint", required=True,
                        help="checkpoint saved by prune_dfine.py (contains ffn_dims)")
    parser.add_argument("--output-dir", default="output/pruning_recovery")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=RECOVERY_EPOCHS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Build model from config ───────────────────────────────────────────
    dist_utils.setup_distributed(0, "builtin", seed=42)
    cfg = YAMLConfig(args.config, **{"lr_scheduler": None})
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    evaluator = cfg.evaluator

    val_loader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    train_loader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=True)

    # ── Load pruned weights ───────────────────────────────────────────────
    ckpt = torch.load(args.pruned_checkpoint, map_location="cpu")
    ffn_dims = ckpt["ffn_dims"]
    pruning_ap = ckpt.get("ap", None)
    total_pruned = ckpt.get("total_pruned", "?")

    apply_ffn_dims(model, ffn_dims)
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(ckpt["model"])

    print(f"Loaded pruned model: ffn_dims={ffn_dims}, pruning_ap={pruning_ap}, total_pruned={total_pruned}")
    print(f"Recovery: {args.epochs} epochs, constant LR={LR}\n")

    # ── W&B setup ─────────────────────────────────────────────────────────
    use_wandb = cfg.yaml_cfg.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.yaml_cfg.get("project_name", "D-FINE"),
                name=f"recovery_{Path(args.pruned_checkpoint).stem}",
                config={
                    "lr": LR,
                    "ffn_dims": ffn_dims,
                    "total_pruned": total_pruned,
                    "pruning_ap": pruning_ap,
                    "recovery_epochs": args.epochs,
                },
            )
        except Exception as e:
            print(f"W&B init failed: {e}")
            use_wandb = False

    # ── Flat optimizer — no LR schedule ──────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    # ── Baseline eval of pruned model ─────────────────────────────────────
    print("=== Eval of pruned model before recovery ===")
    stats, _ = evaluate(model, criterion, postprocessor, val_loader, evaluator, device, 0, use_wandb)
    ap_before = get_ap(stats)
    print(f"AP50:95 (pruned, before recovery): {ap_before:.4f}")

    best_ap = ap_before
    best_epoch = 0
    best_path = output_dir / "best_recovery.pth"

    # Save initial pruned state as starting best
    torch.save({"model": (model.module if hasattr(model, "module") else model).state_dict(),
                "ffn_dims": ffn_dims, "epoch": 0, "ap": ap_before}, best_path)

    # ── Recovery training loop ────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Recovery epoch {epoch}/{args.epochs} ---")

        # Use existing train_one_epoch (no sparsity loss)
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            use_wandb=use_wandb,
            epochs=args.epochs,
            max_norm=MAX_NORM,
            print_freq=100,
            scaler=scaler,
        )

        stats, _ = evaluate(model, criterion, postprocessor, val_loader, evaluator, device, epoch, use_wandb)
        ap = get_ap(stats)
        print(f"  AP50:95 = {ap:.4f}  (best so far = {best_ap:.4f})")

        if use_wandb:
            import wandb
            wandb.log({"recovery/epoch": epoch, "recovery/ap50_95": ap})

        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch
            torch.save(
                {
                    "model": (model.module if hasattr(model, "module") else model).state_dict(),
                    "ffn_dims": ffn_dims,
                    "epoch": epoch,
                    "ap": ap,
                },
                best_path,
            )
            print(f"  [best] New best AP={ap:.4f} — saved to {best_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Recovery training complete")
    print(f"  FFN dims:         {ffn_dims}")
    print(f"  Neurons pruned:   {total_pruned}")
    print(f"  AP before pruning (reported at stop): {pruning_ap}")
    print(f"  AP after pruning (before recovery):   {ap_before:.4f}")
    print(f"  AP after recovery (best):             {best_ap:.4f} (epoch {best_epoch})")
    print(f"  Best checkpoint:  {best_path}")
    print(f"{'='*60}")

    if use_wandb:
        import wandb
        wandb.finish()

    dist_utils.cleanup()


if __name__ == "__main__":
    main()
