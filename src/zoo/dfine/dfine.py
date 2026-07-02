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

        if self.p2_head is not None:
            # Encoder only sees P3/P4/P5 (x[1:]) — keeping the original 3-level encoder
            # unchanged.  P2 (x[0], 64ch backbone features) goes straight to the conv head,
            # bypassing the encoder's bottom-up PAN which would otherwise propagate random
            # P2 weights into P3/P4/P5 and corrupt the transformer inputs.
            feats = self.encoder(x[1:])
            out = self.decoder(feats, targets)
            cls_logits, pred_boxes, anchor_points = self.p2_head(x[0])
            out["p2_logits"]  = cls_logits       # [B, HW, C]
            out["p2_boxes"]   = pred_boxes        # [B, HW, 4]
            out["p2_anchors"] = anchor_points     # [HW, 2]
        else:
            feats = self.encoder(x)
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
