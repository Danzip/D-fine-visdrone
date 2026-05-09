"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from ..core import register

__all__ = ["AdamW", "SGD", "Adam", "MultiStepLR", "CosineAnnealingLR", "OneCycleLR", "LambdaLR", "WarmupCosineHoldLR"]


SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)


@register()
class WarmupCosineHoldLR(LRScheduler):
    """3-phase epoch-level LR schedule:
      1. Linear warmup: epochs [0, warmup_epochs) — warmup_start_lr → base_lr
      2. Cosine decay:  epochs [warmup_epochs, warmup_epochs+cosine_epochs) — base_lr → eta_min
      3. Hold:          epochs [warmup_epochs+cosine_epochs, ...) — stays at eta_min

    warmup_start_lr defaults to eta_min/10 (an absolute floor, not relative to base_lr).
    """

    def __init__(self, optimizer, warmup_epochs: int, cosine_epochs: int,
                 eta_min: float = 1e-7, warmup_start_lr: float = None, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else eta_min / 10.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            t = ep / max(1, self.warmup_epochs - 1)
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * t
                    for base_lr in self.base_lrs]
        ep_cos = ep - self.warmup_epochs
        if ep_cos < self.cosine_epochs:
            factor = 0.5 * (1.0 + math.cos(math.pi * ep_cos / self.cosine_epochs))
            return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]
        return [self.eta_min for _ in self.base_lrs]
