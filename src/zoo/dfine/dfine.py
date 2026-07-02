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
        "p2_fusion",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        p2_head: nn.Module = None,
        p2_fusion: nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.p2_head = p2_head
        self.p2_fusion = p2_fusion

    def forward(self, x, targets=None):
        x = self.backbone(x)

        if self.p2_head is not None:
            # Encoder only sees P3/P4/P5 (x[1:]) — keeping the original 3-level encoder
            # unchanged.  Raw backbone P2 (x[0], 64ch) bypasses the encoder's bottom-up
            # PAN (random P2 weights would corrupt the transformer inputs).
            feats = self.encoder(x[1:])
            out = self.decoder(feats, targets)
            # Cheap-FPN option: fuse raw P2 with upsampled neck-P3 semantics
            p2_feat = self.p2_fusion(x[0], feats[0]) if self.p2_fusion is not None else x[0]
            cls_logits, pred_boxes, anchor_points = self.p2_head(p2_feat)
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
