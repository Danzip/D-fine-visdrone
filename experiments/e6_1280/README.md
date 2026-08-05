# Experiment: E6 — 1280×1280 resolution unlock via score-head bias reset

## Status: DONE — current overall best (AP=0.344)

## What this tests

Whether the standing best (P2ConvHead + P2FusionLite + NWD, `msfd_1024`
lineage, capped at 1024×1024) has headroom left simply from more resolution,
without any further architecture/loss changes.

4 earlier direct attempts to jump resolution from an already-adapted
checkpoint had all flatlined at AP 0.11-0.13 (`PROJECT_NOTES/11_ablation_study_runpod.md`
W3). Root cause found here: it was never AIFI's position embeddings (already
resolution-agnostic, recomputed per forward pass) — it was
`enc_score_head`/`dec_score_head` biases staying miscalibrated for the ~2x
denser anchor grid at 1280px vs 1024px after a naive tuning-load. Fix:
`reset_score_head_bias: True` resets those biases fresh
(`bias_init_with_prob(0.01)`) instead of silently inheriting the
old-resolution-tuned value.

## Starting checkpoint

`output/runpod_results/msfd_1024_best_ep109.pth` (AP=0.3219) — not
`msfd_1024_polish2` (AP=0.3226, technically higher). Chosen because
`msfd_1024`'s own multi-scale phase already included genuine 1280×1280
batches (`generate_scales(1024, 3)` reaches up to 1280), so this checkpoint
retains more exposure to that resolution than one that's been
single-scale-locked for longer (polish2's extra 50 single-scale epochs).
Deliberately chosen to reduce (not eliminate) the historical
resolution-jump collapse risk. Full rationale in `config.yml`'s header.

## Results

| Epoch | AP50:95 | AP50 | AP-small | AP-medium | AP-large | Notes |
|-------|---------|------|----------|-----------|----------|-------|
| 46/50 (best) | 0.344 | 0.549 | 0.257 | 0.453 | 0.626 | `output/runpod_results/e6_1280_best_ep46.pth` — current overall best, re-verified 2026-08-05 via `train.py --test-only` |
| 49/50 (last) | — | — | — | — | — | `output/runpod_results/e6_1280_last_ep49.pth`, not the peak checkpoint |

Up from the pre-E6 standing best of 0.3226 (+0.0214, +6.6% relative) — the
largest single-experiment jump since the original NWD/P2 architecture work.
Within ~0.012 of the DroneScan-YOLO paper-reported SOTA figure (0.356,
unverified — see README.md's Limitations section). Superseded `r6_o2m`
(one-to-many matching, implemented but not yet run as of this writing) as
the tuning base for any future work on this lineage.
