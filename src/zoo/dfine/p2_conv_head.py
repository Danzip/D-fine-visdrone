"""YOLOv8-style lightweight conv detection head for P2 (stride-4) features.

Replaces MSDeformableAttention at the finest feature level with two stacked
depthwise-separable conv blocks followed by separate cls/reg branches.

Box decoding: cx/cy are anchor-relative offsets in CELL units
(YOLO-style (sigmoid*2-0.5)/grid + anchor), so every anchor predicts within
±0.5 cells of its own center. w/h use plain sigmoid with bias init at ~2% of
image (median VisDrone box size).

Assignment: top-K (K=10) candidates per GT — all inside-anchor candidates ranked by
predicted IoU, top-10 each get a soft cls target equal to their IoU with the GT.
Gives 10x more gradient per GT than top-1 at init when IoU is near-zero everywhere.

Output keys added to the model output dict:
  p2_logits  : [B, H*W, num_classes]   raw cls logits (no sigmoid)
  p2_boxes   : [B, H*W, 4]             normalized cxcywh, anchor-decoded
  p2_anchors : [H*W, 2]                normalized anchor center (cx, cy)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou

_TOP_K = 10


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
        # FIX-2: correct log-odds bias for cls head (prior=0.01 → logit≈-4.6)
        prior = 0.01
        bias_val = math.log(prior / (1.0 - prior))
        if hasattr(self.cls_branch[-1], "bias") and self.cls_branch[-1].bias is not None:
            nn.init.constant_(self.cls_branch[-1].bias, bias_val)
        # FIX-5: zero-init the final cls conv weight.
        # kaiming_normal_(fan_out=10) gives weight std≈0.45 → output std≈5.0 (128 inputs).
        # sigmoid(N(-4.6, 5.0)) > 0.5 for ~18% of 256K predictions, flooding postprocessor
        # top-K with random FPs and giving AP=0 throughout the 50-epoch warmup.
        nn.init.zeros_(self.cls_branch[-1].weight)
        # BUG-039 companion: init w/h bias so initial predictions are ~2% of image
        # (median VisDrone box) instead of sigmoid(0)=50% of image. Keeps initial
        # L1 in a sane range and lets IoU-based top-K assignment rank candidates
        # meaningfully from the first iterations.
        with torch.no_grad():
            self.reg_branch[-1].bias[2:].fill_(math.log(0.02 / 0.98))

    def forward(self, p2_feat: torch.Tensor):
        """
        Args:
            p2_feat: [B, in_channels, H, W]  P2 neck feature map (stride 4).
        Returns:
            cls_logits:    [B, H*W, num_classes]
            pred_boxes:    [B, H*W, 4]  normalized cxcywh, anchor-decoded
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
        ).reshape(-1, 2)  # [HW, 2]  (cx, cy)

        reg = reg.flatten(2).permute(0, 2, 1)    # [B, HW, 4]

        # FIX-1 + BUG-039: anchor-relative decoding so each cell predicts near its
        # own center. The offset must be in CELL units (divide by grid size) —
        # without the division it spans ±0.5 of the whole image (~±80 cells),
        # which made tiny-box regression unstable (loss_p2_reg ~43 in msfd_640).
        # cx/cy range: anchor ± 0.5 cells (matches YOLO).
        cx = (torch.sigmoid(reg[..., 0]) * 2 - 0.5) / W + anchor_points[:, 0]
        cy = (torch.sigmoid(reg[..., 1]) * 2 - 0.5) / H + anchor_points[:, 1]
        wh = torch.sigmoid(reg[..., 2:])          # (0, 1) — fine for tiny objects
        pred_boxes = torch.clamp(
            torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1), wh], dim=-1), 0.0, 1.0
        )  # [B, HW, 4]

        cls_logits = cls.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        return cls_logits, pred_boxes, anchor_points


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _elementwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU of paired boxes (xyxy): boxes1[i] vs boxes2[i] → [P].

    generalized_box_iou builds the full P×P matrix — prohibitive when P is
    thousands of positives; only the diagonal is needed.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-9)
    lt_c = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = (wh_c[:, 0] * wh_c[:, 1]).clamp(min=1e-9)
    return iou - (area_c - union) / area_c

def p2_head_loss(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    anchor_points: torch.Tensor,
    targets: list,
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> dict:
    """Top-K TAL loss for P2 dense predictions.

    Assignment: for each GT box, all anchor points whose center falls inside
    the GT are candidates; the top-K (K=10) by predicted IoU each become a
    positive, receiving a soft cls target equal to their IoU with the GT.

    Args:
        pred_logits:   [B, N, C]   raw cls logits
        pred_boxes:    [B, N, 4]   normalized cxcywh (anchor-decoded)
        anchor_points: [N, 2]      normalized (cx, cy)
        targets:       list[dict]  with 'labels' [M] and 'boxes' [M,4] cxcywh
        num_classes:   int
    Returns:
        dict with keys loss_p2_cls, loss_p2_reg, loss_p2_iou  (unscaled)
    """
    device = pred_logits.device
    B, N, C = pred_logits.shape

    # fp32 accumulators — AMP runs BCE in fp16, summing 256K terms overflows fp16 max (65504)
    loss_cls_total = pred_logits.new_zeros(1, dtype=torch.float32)
    loss_reg_total = pred_logits.new_zeros(1, dtype=torch.float32)
    loss_iou_total = pred_logits.new_zeros(1, dtype=torch.float32)
    num_pos_total = 0

    ax = anchor_points[:, 0]  # [N]
    ay = anchor_points[:, 1]  # [N]

    for i in range(B):
        gt_boxes  = targets[i]["boxes"].to(device)    # [M, 4] cxcywh normalized
        gt_labels = targets[i]["labels"].to(device)   # [M]
        M = gt_boxes.shape[0]

        pred_b = pred_boxes[i]    # [N, 4]
        pred_l = pred_logits[i]   # [N, C]

        tgt_cls = torch.zeros(N, C, device=device)

        if M > 0:
            gcx = gt_boxes[:, 0]
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
                pred_xyxy = box_cxcywh_to_xyxy(pred_b)    # [N, 4]
                gt_xyxy   = box_cxcywh_to_xyxy(gt_boxes)  # [M, 4]
                iou_mat, _ = box_iou(pred_xyxy, gt_xyxy)  # [N, M]

                # FIX-3 (vectorized): top-K assignment for ALL GTs at once.
                # The old per-GT python loop with .item() syncs took minutes on
                # dense VisDrone images (M up to ~900) — see BUG-041.
                cand_iou = torch.where(inside, iou_mat, iou_mat.new_zeros(()))
                k = min(_TOP_K, N)
                topk_iou, topk_idx = cand_iou.topk(k, dim=0)   # [k, M]
                valid = topk_iou > 0                            # [k, M]

                if valid.any():
                    pos_anchors = topk_idx[valid]                       # [P]
                    pos_gt_idx  = valid.nonzero(as_tuple=True)[1]       # [P] GT col
                    pos_iou     = topk_iou[valid]                       # [P]
                    num_pos_total += pos_anchors.numel()

                    # Soft cls target: amax handles anchors shared by multiple GTs
                    flat_idx = pos_anchors * C + gt_labels[pos_gt_idx]
                    tgt_cls.view(-1).scatter_reduce_(
                        0, flat_idx, pos_iou.detach(), reduce="amax"
                    )

                    # Regression losses on positive anchors (elementwise)
                    pos_pred = pred_b[pos_anchors]       # [P, 4]
                    pos_gt   = gt_boxes[pos_gt_idx]      # [P, 4]

                    loss_reg_total = loss_reg_total + F.l1_loss(
                        pos_pred, pos_gt, reduction="sum"
                    ).float()

                    giou = _elementwise_giou(
                        box_cxcywh_to_xyxy(pos_pred), box_cxcywh_to_xyxy(pos_gt)
                    )
                    loss_iou_total = loss_iou_total + (1 - giou).float().sum()

        # Varifocal cls loss: w = q for positives, alpha*p^gamma for negatives
        p_sig = pred_l.sigmoid()
        q     = tgt_cls
        bce   = F.binary_cross_entropy_with_logits(pred_l, q, reduction="none")
        vfl_w = torch.where(q > 0, q, alpha * p_sig.pow(gamma))
        loss_cls_total = loss_cls_total + (vfl_w * bce).float().sum()

    # BUG-038: return UNSCALED losses (as the docstring always said) — the
    # criterion multiplies by weight_dict. The old internal w_cls/w_reg/w_iou
    # scaling stacked with weight_dict, making reg effectively 25× and iou 4×.
    denom = max(num_pos_total, 1)
    return {
        "loss_p2_cls": loss_cls_total / (N * B),
        "loss_p2_reg": loss_reg_total / denom,
        "loss_p2_iou": loss_iou_total / denom,
    }
