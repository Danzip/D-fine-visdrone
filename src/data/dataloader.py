"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import random

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision

from ..core import register

torchvision.disable_beta_transforms_warning()


__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]


def _build_rfs_sampler(dataset, threshold: float):
    """Repeat Factor Sampling (LVIS-style) sampler for class-imbalanced datasets.

    For each category c:
        f_c  = (# images containing c) / (total images)
        r_c  = max(1, sqrt(threshold / f_c))

    For each image i:
        weight_i = max(r_c for c in categories present in image i)

    Images with only common classes get weight=1.0.
    Images containing rare classes get weight > 1.0 → oversampled.

    Args:
        dataset:   CocoDetection instance (must have .coco and .ids attributes)
        threshold: frequency threshold t; typical values 0.001–0.01
    """
    from torch.utils.data import WeightedRandomSampler

    coco = dataset.coco
    img_ids = dataset.ids
    n_imgs = len(img_ids)

    # Count images per category
    cat_img_count = {}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        cats = {a["category_id"] for a in coco.loadAnns(ann_ids)}
        for cat in cats:
            cat_img_count[cat] = cat_img_count.get(cat, 0) + 1

    # Per-category repeat factor
    cat_repeat = {
        cat: max(1.0, math.sqrt(threshold / (count / n_imgs)))
        for cat, count in cat_img_count.items()
    }

    # Per-image weight = max repeat factor of classes it contains
    weights = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        cats = {a["category_id"] for a in coco.loadAnns(ann_ids)}
        w = max((cat_repeat.get(c, 1.0) for c in cats), default=1.0)
        weights.append(w)

    # Number of samples = round(sum of weights) so effective dataset size scales naturally
    num_samples = round(sum(weights))
    print(f"[RFS] threshold={threshold}, effective dataset size: {n_imgs} → {num_samples} "
          f"({num_samples/n_imgs:.1f}× oversampling)")

    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


class ARBucketBatchSampler:
    """Groups dataset images into 16:9 and 4:3 buckets by COCO metadata AR.

    Yields same-AR batches so each batch contains images from only one AR bucket.
    Batches alternate 16:9/4:3 with epoch-seeded shuffling within each bucket.
    """

    def __init__(self, dataset, batch_size: int):
        coco = dataset.coco
        img_ids = dataset.ids

        self._16_9 = []
        self._4_3 = []
        for idx, img_id in enumerate(img_ids):
            info = coco.imgs[img_id]
            ar = info["width"] / info["height"]
            if ar > 1.6:
                self._16_9.append(idx)
            else:
                self._4_3.append(idx)

        self.batch_size = batch_size
        self._epoch = 0
        print(f"[ARBucketBatchSampler] 16:9: {len(self._16_9)}  "
              f"4:3: {len(self._4_3)}  batch_size={batch_size}")

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self._epoch)
        bs = self.batch_size

        a = self._16_9[:]
        b = self._4_3[:]
        rng.shuffle(a)
        rng.shuffle(b)

        # Full batches only (drop last incomplete)
        batches_a = [a[i:i + bs] for i in range(0, (len(a) // bs) * bs, bs)]
        batches_b = [b[i:i + bs] for i in range(0, (len(b) // bs) * bs, bs)]

        # Interleave then shuffle so bucket order is unpredictable across epochs
        all_batches = []
        for i in range(max(len(batches_a), len(batches_b))):
            if i < len(batches_a):
                all_batches.append(batches_a[i])
            if i < len(batches_b):
                all_batches.append(batches_b[i])
        rng.shuffle(all_batches)

        for batch in all_batches:
            yield from batch

    def __len__(self):
        bs = self.batch_size
        return (len(self._16_9) // bs + len(self._4_3) // bs) * bs


@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __init__(self, dataset, collate_fn, shuffle=False,
                 repeat_factor_threshold=None, use_ar_sampler=False, **kwargs):
        sampler = None
        self._ar_batch_sampler = None

        if use_ar_sampler:
            batch_size = kwargs.pop("batch_size", 1)
            kwargs.pop("drop_last", None)  # batch_sampler owns drop behaviour
            ar_bs = ARBucketBatchSampler(dataset, batch_size)
            self._ar_batch_sampler = ar_bs
            super().__init__(
                dataset=dataset,
                collate_fn=collate_fn,
                batch_sampler=ar_bs,
                **kwargs,
            )
        else:
            if repeat_factor_threshold is not None and repeat_factor_threshold > 0:
                sampler = _build_rfs_sampler(dataset, repeat_factor_threshold)
                shuffle = False

            super().__init__(
                dataset=dataset,
                collate_fn=collate_fn,
                shuffle=shuffle,
                sampler=sampler,
                **kwargs,
            )

        bs = self.batch_size
        sampler_name = (type(self._ar_batch_sampler).__name__ if self._ar_batch_sampler
                        else type(sampler).__name__ if sampler else 'default')
        print(f"[DataLoader] effective batch_size={bs}  workers={self.num_workers}  sampler={sampler_name}")

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
        if self._ar_batch_sampler is not None:
            self._ar_batch_sampler.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image"""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        adaptive_batch=False,
        batch_size_table_path=None,
        mosaic_p=0.0,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = (
            generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        self.adaptive_batch = adaptive_batch
        self.mosaic_p = mosaic_p
        # batch_size_table_path: accepted for config compatibility, ignored.
        # The mosaic config sets this to ~ (null). Use adaptive_batch instead.

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if "masks" in targets[0]:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")
                raise NotImplementedError("")

            if self.adaptive_batch and sz != self.base_size:
                n = max(1, int(len(images) * (self.base_size / sz) ** 2))
                n = min(n, len(images))
                images = images[:n]
                targets = targets[:n]

        return images, targets
