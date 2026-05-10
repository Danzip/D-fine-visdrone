"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register

__all__ = [
    "DFINE",
]


@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
        "p2_head",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        p2_head: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.p2_head = p2_head

    def forward(self, x, targets=None):
        x = self.backbone(x)
        feats = self.encoder(x)

        if self.p2_head is not None:
            # feats[0] = P2 neck → conv head; feats[1:] = P3/P4/P5 → transformer
            out = self.decoder(feats[1:], targets)
            cls_logits, pred_boxes, anchor_points = self.p2_head(feats[0])
            out["p2_logits"]  = cls_logits       # [B, HW, C]
            out["p2_boxes"]   = pred_boxes        # [B, HW, 4]
            out["p2_anchors"] = anchor_points     # [HW, 2]
        else:
            out = self.decoder(feats, targets)

        return out

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
