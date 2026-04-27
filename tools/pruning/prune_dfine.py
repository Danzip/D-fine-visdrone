#!/usr/bin/env python3
"""
D-FINE structured pruning — interleaved train+prune loop.

Architecture: D-FINE-S visdrone config has 3 decoder layers, each with
  FFN: linear1 (256→1024) + linear2 (1024→256)
  Self-attn: 8 heads (head_dim=32)
Total prunable FFN neurons: 3 × 1024 = 3,072

Each epoch:
  1. Train with task losses + group-lasso sparsity on decoder FFN neurons/attn heads
  2. Prune bottom N_PRUNE (default 31 = ~1% of 3072) FFN neurons globally
  3. Evaluate AP50:95
  4. AP >= AP_FLOOR (0.216): save checkpoint, continue
  5. AP < AP_FLOOR: restore previous checkpoint, stop loop

After this loop finishes run recovery training:
  python tools/pruning/recovery_train.py \\
      -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \\
      --pruned-checkpoint output/pruning/best_pruned.pth \\
      --output-dir output/pruning_recovery \\
      --device cuda:0

Usage (from D-FINE/):
  python tools/pruning/prune_dfine.py \\
      -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \\
      --checkpoint output/dfine_hgnetv2_s_visdrone/best_stg1.pth \\
      --output-dir output/pruning \\
      --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver.det_engine import evaluate

# ── Pruning hyperparameters ───────────────────────────────────────────────
LR = 1e-5           # constant flat LR for all params throughout pruning
WEIGHT_DECAY = 1e-4
MAX_NORM = 0.1      # gradient clip norm
LAMBDA_SPARSE = 2e-4  # group-lasso weight; scales sparsity term relative to task losses
N_PRUNE_PER_EPOCH = 31  # ~1% of 3072 initial FFN neurons
AP_FLOOR = 0.216    # stop if AP50:95 drops below this
MAX_EPOCHS = 100    # safety ceiling; loop exits early via AP_FLOOR
USE_AMP = True
# ─────────────────────────────────────────────────────────────────────────


def get_decoder(model: nn.Module):
    """Return TransformerDecoder (the module that owns .layers).
    Path: model [DFINE] → .decoder [DFINETransformer] → .decoder [TransformerDecoder]
    Unwraps DataParallel/DistributedDataParallel if needed.
    """
    m = model.module if hasattr(model, "module") else model
    return m.decoder.decoder  # DFINETransformer.decoder = TransformerDecoder


def group_lasso_loss(model: nn.Module) -> torch.Tensor:
    """Structured sparsity loss: group lasso on decoder FFN neurons + self-attn heads.

    FFN: sum of L2 norms of linear1 row vectors.
         Row n = neuron n's input weights; driving its norm to zero = safe to remove.

    Self-attn heads: sum of Frobenius norms of each head's output-projection block.
         Head h contributes out_proj.weight[:, h*hd:(h+1)*hd]; low norm = low contribution.

    Note: this loss alone does not remove anything; it shapes the weight landscape so
    that the bottom-N neurons by norm are cleanly separable from the rest.
    """
    terms = []
    for layer in get_decoder(model).layers:
        # FFN group lasso: one group per neuron (row of linear1.weight)
        terms.append(layer.linear1.weight.norm(dim=1).sum())

        # Self-attn head group lasso: one group per head (column block of out_proj)
        d_model = layer.self_attn.embed_dim
        n_heads = layer.self_attn.num_heads
        head_dim = d_model // n_heads
        op_w = layer.self_attn.out_proj.weight  # [d_model, d_model]
        for h in range(n_heads):
            terms.append(op_w[:, h * head_dim:(h + 1) * head_dim].norm())

    return sum(terms)


def get_ffn_dims(model: nn.Module):
    """Current FFN intermediate dims per decoder layer."""
    return [layer.linear1.out_features for layer in get_decoder(model).layers]


def get_neuron_importances(model: nn.Module):
    """Return list of (importance, layer_idx, neuron_idx) sorted ascending by importance."""
    importances = []
    for i, layer in enumerate(get_decoder(model).layers):
        norms = layer.linear1.weight.detach().norm(dim=1)  # [dim_ff]
        for j, norm in enumerate(norms):
            importances.append((norm.item(), i, j))
    importances.sort(key=lambda x: x[0])
    return importances


def prune_ffn_neurons(model: nn.Module, to_prune: dict):
    """Physically remove FFN neurons by restructuring linear1 and linear2.

    to_prune: {layer_idx: [neuron_indices_to_remove]}

    linear1.weight[n, :] = neuron n's input weights  (shape [dim_ff, d_model])
    linear2.weight[:, n] = neuron n's output weights  (shape [d_model, dim_ff])
    Removing neuron n = dropping row n from linear1 and column n from linear2.
    """
    device = next(model.parameters()).device
    for layer_idx, remove in to_prune.items():
        layer = get_decoder(model).layers[layer_idx]
        remove_set = set(remove)
        dim_ff = layer.linear1.out_features
        keep = [i for i in range(dim_ff) if i not in remove_set]

        if len(keep) < 16:
            print(f"  [skip] layer {layer_idx}: would leave only {len(keep)} neurons — too few")
            continue

        # Rebuild linear1 keeping selected rows
        new_l1 = nn.Linear(layer.linear1.in_features, len(keep)).to(device)
        new_l1.weight = nn.Parameter(layer.linear1.weight.data[keep].clone())
        new_l1.bias = nn.Parameter(layer.linear1.bias.data[keep].clone())
        layer.linear1 = new_l1

        # Rebuild linear2 keeping selected columns; bias is unchanged
        new_l2 = nn.Linear(len(keep), layer.linear2.out_features).to(device)
        new_l2.weight = nn.Parameter(layer.linear2.weight.data[:, keep].clone())
        new_l2.bias = nn.Parameter(layer.linear2.bias.data.clone())
        layer.linear2 = new_l2


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Flat AdamW at constant LR. Must be rebuilt after each pruning step
    because prune_ffn_neurons replaces nn.Linear objects (old params are stale)."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )


def train_one_epoch(model, criterion, loader, optimizer, device, epoch, lambda_sparse=2e-4):
    model.train()
    criterion.train()
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    total_task, total_sparse, n_batches = 0.0, 0.0, 0

    for i, (samples, targets) in enumerate(loader):
        samples = samples.to(device)
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]
        metas = dict(
            epoch=epoch, step=i,
            global_step=epoch * len(loader) + i,
            epoch_step=len(loader),
        )
        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type="cuda"):
                outputs = model(samples, targets=targets)
            with torch.autocast(device_type="cuda", enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            task_loss = sum(loss_dict.values())
            # group_lasso_loss operates on fp32 weights — safe outside autocast
            sparse_loss = lambda_sparse * group_lasso_loss(model)
            scaler.scale(task_loss + sparse_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            task_loss = sum(loss_dict.values())
            sparse_loss = lambda_sparse * group_lasso_loss(model)
            (task_loss + sparse_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()

        total_task += task_loss.item()
        total_sparse += sparse_loss.item()
        n_batches += 1

        if i % 100 == 0:
            print(
                f"  [epoch {epoch} step {i}/{len(loader)}] "
                f"task={task_loss.item():.4f}  sparse={sparse_loss.item():.6f}"
            )

    return total_task / n_batches, total_sparse / n_batches


def get_ap(stats: dict) -> float:
    """Extract AP50:95 from evaluate() return value."""
    if "coco_eval_bbox" in stats:
        return stats["coco_eval_bbox"][0]
    # fallback: first value of first key
    k = next(iter(stats))
    return stats[k][0]


def update_notes_table(log: list, notes_path: str, stopped: bool = False, recovery_ap: float = None):
    """Rewrite the Results table in 07_pruning.md with current log entries.

    log: list of dicts with keys epoch, ap, total_pruned, ffn_dims
    stopped: whether the pruning loop just hit the AP floor
    recovery_ap: if set, fill in the After recovery row
    """
    notes_path = Path(notes_path)
    if not notes_path.exists():
        return

    text = notes_path.read_text()

    # Build new table content
    rows = ["| Epoch | FFN dims | Neurons pruned | AP50:95 |",
            "|---|---|---|---|"]
    for entry in log:
        ep = entry["epoch"]
        label = f"{ep} (baseline)" if ep == 0 else str(ep)
        dims = str(entry["ffn_dims"])
        pruned = entry["total_pruned"]
        ap = f"{entry['ap']:.4f}"
        rows.append(f"| {label} | {dims} | {pruned} | {ap} |")

    if stopped and len(log) > 1:
        last = log[-1]
        rows.append(f"| **STOP** | {last['ffn_dims']} | {last['total_pruned']} | {last['ap']:.4f} |")

    if recovery_ap is not None:
        last = log[-1]
        rows.append(f"| After recovery | {last['ffn_dims']} | {last['total_pruned']} | {recovery_ap:.4f} |")
    else:
        rows.append("| After recovery | (same) | (same) | TBD |")

    new_table = "\n".join(rows)

    # Replace everything between "## Results" header and the next "## " header (or EOF)
    import re
    pattern = r"(## Results\n\n\*\(to be filled in after run\)\*\n\n).*?(\n\n## |\Z)"
    replacement_block = f"## Results\n\n{new_table}\n\n"

    new_text, count = re.subn(pattern, replacement_block, text, flags=re.DOTALL)
    if count == 0:
        # Table already exists — replace from the header row onward
        pattern2 = r"(## Results\n\n)\|.*?(\n\n## |\Z)"
        new_text, count2 = re.subn(pattern2, f"## Results\n\n{new_table}\n\n", text, flags=re.DOTALL)
        if count2 == 0:
            return  # Can't find the section; skip silently

    notes_path.write_text(new_text)



def save_pruning_checkpoint(path: Path, model: nn.Module, epoch: int, ap: float, total_pruned: int):
    """Save model weights + FFN dims so we can reconstruct the pruned architecture later."""
    m = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": m.state_dict(),
            "ffn_dims": get_ffn_dims(model),
            "epoch": epoch,
            "ap": ap,
            "total_pruned": total_pruned,
        },
        path,
    )


def load_base_checkpoint(model: nn.Module, path: str):
    """Load weights from a fine-tuning checkpoint (prefers EMA weights)."""
    ckpt = torch.load(path, map_location="cpu")
    if "ema" in ckpt and ckpt["ema"] is not None:
        state = ckpt["ema"]["module"]
        print(f"Loaded EMA weights from {path}")
    elif "model" in ckpt:
        state = ckpt["model"]
        print(f"Loaded model weights from {path}")
    else:
        state = ckpt
        print(f"Loaded raw state dict from {path}")
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(state, strict=False)


def main():
    parser = argparse.ArgumentParser(description="D-FINE structured pruning")
    parser.add_argument("-c", "--config", required=True, help="path to visdrone config yml")
    parser.add_argument("--checkpoint", required=True, help="fine-tuned checkpoint to prune from")
    parser.add_argument("--output-dir", default="output/pruning", help="where to save checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-prune", type=int, default=N_PRUNE_PER_EPOCH,
                        help="FFN neurons to remove per epoch (default 31 = ~1%% of 3072)")
    parser.add_argument("--ap-floor", type=float, default=AP_FLOOR,
                        help="stop pruning if AP50:95 drops below this (default 0.216)")
    parser.add_argument("--lambda-sparse", type=float, default=LAMBDA_SPARSE,
                        help="group-lasso weight (default 2e-4)")
    parser.add_argument("--notes", type=str,
                        default="PROJECT_NOTES/07_pruning.md",
                        help="path to notes file for auto-updating results table")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ── Build model + supporting objects from config ──────────────────────
    dist_utils.setup_distributed(0, "builtin", seed=42)
    cfg = YAMLConfig(
        args.config,
        # Disable LR schedule — we use a flat constant LR
        **{"lr_scheduler": None}
    )
    # Disable pretrained backbone re-download
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    evaluator = cfg.evaluator

    val_loader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    train_loader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=True)

    # ── Load fine-tuned weights ───────────────────────────────────────────
    load_base_checkpoint(model, args.checkpoint)

    # ── W&B setup ─────────────────────────────────────────────────────────
    use_wandb = cfg.yaml_cfg.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.yaml_cfg.get("project_name", "D-FINE"),
                name=f"pruning_{Path(args.checkpoint).stem}",
                config={
                    "lr": LR,
                    "lambda_sparse": args.lambda_sparse,
                    "n_prune_per_epoch": args.n_prune,
                    "ap_floor": args.ap_floor,
                },
            )
        except Exception as e:
            print(f"W&B init failed: {e}")
            use_wandb = False

    # ── Baseline eval ─────────────────────────────────────────────────────
    print("\n=== Baseline evaluation before pruning ===")
    stats, _ = evaluate(model, criterion, postprocessor, val_loader, evaluator, device, 0, use_wandb)
    baseline_ap = get_ap(stats)
    print(f"Baseline AP50:95: {baseline_ap:.4f}")

    # ── Pruning state ─────────────────────────────────────────────────────
    optimizer = make_optimizer(model)
    total_pruned = 0
    prev_checkpoint_path = output_dir / "epoch_000_baseline.pth"
    save_pruning_checkpoint(prev_checkpoint_path, model, 0, baseline_ap, 0)

    pruning_log = [{"epoch": 0, "ap": baseline_ap, "total_pruned": 0, "ffn_dims": get_ffn_dims(model)}]
    update_notes_table(pruning_log, args.notes)

    print(f"\nInitial FFN dims: {get_ffn_dims(model)}")
    print(f"Pruning {args.n_prune} neurons/epoch until AP < {args.ap_floor}\n")

    # ── Main pruning loop ─────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}  |  total_pruned={total_pruned}  |  FFN dims={get_ffn_dims(model)}")
        print(f"{'='*60}")

        # 1. Train with sparsity loss
        task_loss, sparse_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            lambda_sparse=args.lambda_sparse,
        )
        print(f"  avg task_loss={task_loss:.4f}  avg sparse_loss={sparse_loss:.6f}")

        # 2. Prune bottom N neurons globally
        importances = get_neuron_importances(model)
        to_remove = importances[:args.n_prune]

        # Group removals by layer
        to_prune_by_layer = {}
        for _, layer_idx, neuron_idx in to_remove:
            to_prune_by_layer.setdefault(layer_idx, []).append(neuron_idx)

        print(f"  Pruning {args.n_prune} neurons: {dict((k, len(v)) for k, v in to_prune_by_layer.items())}")
        prune_ffn_neurons(model, to_prune_by_layer)
        total_pruned += args.n_prune
        print(f"  New FFN dims: {get_ffn_dims(model)}")

        # 3. Rebuild optimizer (old parameter references are stale after pruning)
        optimizer = make_optimizer(model)

        # 4. Evaluate
        stats, _ = evaluate(model, criterion, postprocessor, val_loader, evaluator, device, epoch, use_wandb)
        ap = get_ap(stats)
        print(f"  AP50:95 = {ap:.4f}  (floor={args.ap_floor})")

        log_entry = {
            "epoch": epoch,
            "ap": ap,
            "total_pruned": total_pruned,
            "ffn_dims": get_ffn_dims(model),
            "task_loss": task_loss,
            "sparse_loss": sparse_loss,
        }
        pruning_log.append(log_entry)
        update_notes_table(pruning_log, args.notes)

        if use_wandb:
            import wandb
            wandb.log({
                "pruning/epoch": epoch,
                "pruning/ap50_95": ap,
                "pruning/total_pruned": total_pruned,
                "pruning/task_loss": task_loss,
                "pruning/sparse_loss": sparse_loss,
                **{f"pruning/ffn_dim_layer{i}": d for i, d in enumerate(get_ffn_dims(model))},
            })

        # 5. AP check
        if ap >= args.ap_floor:
            # Good — save this as the new "previous good" checkpoint
            ckpt_path = output_dir / f"epoch_{epoch:03d}_ap{ap:.4f}.pth"
            save_pruning_checkpoint(ckpt_path, model, epoch, ap, total_pruned)
            # Keep a rolling "best_pruned" pointer
            best_path = output_dir / "best_pruned.pth"
            save_pruning_checkpoint(best_path, model, epoch, ap, total_pruned)
            prev_checkpoint_path = ckpt_path
            print(f"  [ok] Saved checkpoint: {ckpt_path.name}")
        else:
            print(f"\n  [STOP] AP {ap:.4f} < floor {args.ap_floor}")
            print(f"  Restoring previous checkpoint: {prev_checkpoint_path.name}")
            # Reload the last good checkpoint
            # (previous checkpoint has different FFN dims — need to reconstruct)
            prev_ckpt = torch.load(prev_checkpoint_path, map_location="cpu")
            # Rebuild model architecture with correct dims
            from src.core import YAMLConfig as _YC
            cfg2 = _YC(args.config)
            if "HGNetv2" in cfg2.yaml_cfg:
                cfg2.yaml_cfg["HGNetv2"]["pretrained"] = False
            model = cfg2.model.to(device)
            _apply_ffn_dims(model, prev_ckpt["ffn_dims"])
            m = model.module if hasattr(model, "module") else model
            m.load_state_dict(prev_ckpt["model"])
            print(f"  Restored model with FFN dims: {prev_ckpt['ffn_dims']}, AP={prev_ckpt['ap']:.4f}")
            update_notes_table(pruning_log, args.notes, stopped=True)
            break

    # ── Save log ──────────────────────────────────────────────────────────
    log_path = output_dir / "pruning_log.json"
    with open(log_path, "w") as f:
        json.dump(pruning_log, f, indent=2)
    print(f"\nPruning complete. Log saved to {log_path}")
    print(f"Best pruned model: {output_dir / 'best_pruned.pth'}")
    print(f"\nNext step — recovery training:")
    print(f"  python tools/pruning/recovery_train.py \\")
    print(f"      -c {args.config} \\")
    print(f"      --pruned-checkpoint {output_dir / 'best_pruned.pth'} \\")
    print(f"      --output-dir output/pruning_recovery \\")
    print(f"      --device {args.device}")

    if use_wandb:
        import wandb
        wandb.finish()

    dist_utils.cleanup()


def _apply_ffn_dims(model: nn.Module, ffn_dims: list):
    """Resize FFN layers to match saved dims (needed when restoring a pruned checkpoint)."""
    device = next(model.parameters()).device
    for layer, new_dim in zip(get_decoder(model).layers, ffn_dims):
        if layer.linear1.out_features != new_dim:
            d_model = layer.linear1.in_features
            layer.linear1 = nn.Linear(d_model, new_dim).to(device)
            layer.linear2 = nn.Linear(new_dim, d_model).to(device)


if __name__ == "__main__":
    main()
