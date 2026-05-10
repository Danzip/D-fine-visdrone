"""YOLOv8-style lightweight conv detection head for P2 (stride-4) features.

Replaces MSDeformableAttention at the finest feature level with two stacked
depthwise-separable conv blocks followed by separate cls/reg branches.
Uses FCOS-style "center-inside" assignment: each anchor whose center falls
inside a GT box is a candidate positive; among candidates for each GT, the
one with highest predicted IoU is selected.

Output keys added to the model output dict:
  p2_logits  : [B, H*W, num_classes]   raw cls logits (no sigmoid)
  p2_boxes   : [B, H*W, 4]             normalized cxcywh in (0, 1)
  p2_anchors : [H*W, 2]                normalized anchor center (cx, cy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


class _DWBlock(nn.Module):
    """Depthwise-separable conv → BN → SiLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


@register()
class P2ConvHead(nn.Module):
    """Lightweight YOLOv8-style detection head for P2 neck features.

    Architecture:
        DWBlock(in_ch → hidden) → DWBlock(hidden → hidden)
        ├── cls branch: DWBlock → Conv1×1 → [B, C, H, W]
        └── reg branch: DWBlock → Conv1×1 → [B, 4, H, W]
    """

    __share__ = ["num_classes"]

    def __init__(self, in_channels: int = 256, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            _DWBlock(in_channels, hidden_dim),
            _DWBlock(hidden_dim, hidden_dim),
        )
        self.cls_branch = nn.Sequential(
            _DWBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        self.reg_branch = nn.Sequential(
            _DWBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias init for cls head: prior prob = 0.01 → logit = -log((1-p)/p)
        prior = 0.01
        bias_val = -((1 - prior) / prior) ** 0.5  # approximately log-odds
        if hasattr(self.cls_branch[-1], "bias") and self.cls_branch[-1].bias is not None:
            nn.init.constant_(self.cls_branch[-1].bias, bias_val)

    def forward(self, p2_feat: torch.Tensor):
        """
        Args:
            p2_feat: [B, in_channels, H, W]  P2 neck feature map (stride 4).
        Returns:
            cls_logits:    [B, H*W, num_classes]
            pred_boxes:    [B, H*W, 4]  normalized cxcywh
            anchor_points: [H*W, 2]     normalized cell centers (cx, cy)
        """
        B, _, H, W = p2_feat.shape
        x = self.stem(p2_feat)

        cls = self.cls_branch(x)   # [B, C, H, W]
        reg = self.reg_branch(x)   # [B, 4, H, W]

        # Anchor centers: (i + 0.5) / grid_size  (normalized)
        gy, gx = torch.meshgrid(
            torch.arange(H, dtype=p2_feat.dtype, device=p2_feat.device),
            torch.arange(W, dtype=p2_feat.dtype, device=p2_feat.device),
            indexing="ij",
        )
        anchor_points = torch.stack(
            [(gx + 0.5) / W, (gy + 0.5) / H], dim=-1
        ).reshape(-1, 2)  # [H*W, 2]

        # Decode boxes: cx/cy = sigmoid(reg) constrained to neighbour cells,
        # w/h = sigmoid(reg) — tiny-object friendly since they rarely exceed 0.5.
        reg = reg.flatten(2).permute(0, 2, 1)    # [B, HW, 4]
        pred_boxes = torch.sigmoid(reg)            # [B, HW, 4]  all in (0, 1)

        cls_logits = cls.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        return cls_logits, pred_boxes, anchor_points


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def p2_head_loss(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    anchor_points: torch.Tensor,
    targets: list,
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
    w_cls: float = 1.0,
    w_reg: float = 5.0,
    w_iou: float = 2.0,
) -> dict:
    """FCOS-style TAL loss for P2 dense predictions.

    Assignment: for each GT box, all anchor points whose center falls inside
    the GT are candidates; the candidate with the highest predicted IoU
    (clamped to [0,1]) is selected as the single positive for that GT.
    Remaining candidate anchors get a soft target equal to their predicted IoU.

    Args:
        pred_logits:   [B, N, C]   raw cls logits
        pred_boxes:    [B, N, 4]   normalized cxcywh
        anchor_points: [N, 2]      normalized (cx, cy)
        targets:       list[dict]  with 'labels' [M] and 'boxes' [M,4] cxcywh
        num_classes:   int
    Returns:
        dict with keys loss_p2_cls, loss_p2_reg, loss_p2_iou  (unscaled)
    """
    device = pred_logits.device
    B, N, C = pred_logits.shape

    loss_cls_total = pred_logits.new_zeros(1)
    loss_reg_total = pred_logits.new_zeros(1)
    loss_iou_total = pred_logits.new_zeros(1)
    num_pos_total = 0

    ax = anchor_points[:, 0]  # [N]
    ay = anchor_points[:, 1]  # [N]

    for i in range(B):
        gt_boxes  = targets[i]["boxes"].to(device)    # [M, 4] cxcywh normalized
        gt_labels = targets[i]["labels"].to(device)   # [M]
        M = gt_boxes.shape[0]

        pred_b = pred_boxes[i]    # [N, 4]
        pred_l = pred_logits[i]   # [N, C]

        # --- classification target (all-negative baseline) ---
        tgt_cls = torch.zeros(N, C, device=device)

        if M > 0:
            gcx = gt_boxes[:, 0]  # [M]
            gcy = gt_boxes[:, 1]
            gw  = gt_boxes[:, 2]
            gh  = gt_boxes[:, 3]

            # inside[n, m] = anchor n is inside GT m
            inside = (
                (ax[:, None] >= (gcx - gw / 2)[None]) &
                (ax[:, None] <= (gcx + gw / 2)[None]) &
                (ay[:, None] >= (gcy - gh / 2)[None]) &
                (ay[:, None] <= (gcy + gh / 2)[None])
            )  # [N, M]

            if inside.any():
                # IoU between every anchor's predicted box and every GT
                pred_xyxy = box_cxcywh_to_xyxy(pred_b)          # [N, 4]
                gt_xyxy   = box_cxcywh_to_xyxy(gt_boxes)        # [M, 4]
                iou_mat, _ = box_iou(pred_xyxy, gt_xyxy)        # [N, M]

                # Masked IoU: only inside anchors are candidates
                iou_inside = iou_mat * inside.float()            # [N, M]

                # For each GT, pick the single best anchor (highest IoU inside)
                best_anchor_per_gt = iou_inside.argmax(dim=0)   # [M]
                best_iou_per_gt = iou_inside.max(dim=0).values  # [M]

                # Build pos mask: selected (anchor, GT) pairs with IoU > 0
                valid_gt = best_iou_per_gt > 0                   # [M]
                pos_anchors = best_anchor_per_gt[valid_gt]       # [K]
                pos_gt_idx  = torch.where(valid_gt)[0]           # [K]
                pos_iou     = best_iou_per_gt[valid_gt]          # [K]

                K = pos_anchors.shape[0]
                if K > 0:
                    num_pos_total += K

                    # Soft cls target: IoU at positive positions
                    gt_cls_labels = gt_labels[pos_gt_idx]         # [K]
                    tgt_cls[pos_anchors, gt_cls_labels] = pos_iou.detach()

                    # Regression loss on positive anchors
                    pos_pred  = pred_b[pos_anchors]               # [K, 4]
                    pos_gt    = gt_boxes[pos_gt_idx]              # [K, 4]

                    loss_reg_total = loss_reg_total + F.l1_loss(pos_pred, pos_gt, reduction="sum")

                    pos_pred_xyxy = box_cxcywh_to_xyxy(pos_pred)
                    pos_gt_xyxy   = box_cxcywh_to_xyxy(pos_gt)
                    giou = generalized_box_iou(pos_pred_xyxy, pos_gt_xyxy)  # [K, K]
                    loss_iou_total = loss_iou_total + (1 - giou.diagonal()).sum()

        # Varifocal cls loss: L = |q - σ(p)|^γ * BCE(p, q)
        # q = tgt_cls (0 for bg, IoU for fg), p = pred_l
        p_sig = pred_l.sigmoid()                                   # [N, C]
        q     = tgt_cls                                            # [N, C]
        bce   = F.binary_cross_entropy_with_logits(pred_l, q, reduction="none")
        vfl_w = torch.where(q > 0, q, alpha * p_sig.pow(gamma))
        loss_cls_total = loss_cls_total + (vfl_w * bce).sum()

    denom = max(num_pos_total, 1)
    return {
        "loss_p2_cls": w_cls * loss_cls_total / (N * B),
        "loss_p2_reg": w_reg * loss_reg_total / denom,
        "loss_p2_iou": w_iou * loss_iou_total / denom,
    }
