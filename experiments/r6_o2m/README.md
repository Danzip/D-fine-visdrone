# Experiment: R6 — one-to-many auxiliary matching (H-DETR-lite)

## Status: IMPLEMENTED + LOCALLY VALIDATED — not yet launched on a full run

## What this tests

The decoder has only 300 queries with strict one-to-one Hungarian matching;
VisDrone scenes can have up to ~800 objects, and dense-scene recall looks
query-bottlenecked (AR-500=0.507 vs AR-100=0.367 in one eval — see
`PROJECT_NOTES/11_ablation_study_runpod.md` W4). The matcher already had a
`get_top_k_matches`/`return_topk` one-to-many mechanism, but it was dead
code (never wired into the criterion) and had a real bug (masked matched
queries using target-index values misread as query indices across the whole
batch, instead of scoping to the current image). Fixed the bug
(`src/zoo/dfine/matcher.py`, BUG-048) and wired it into `DFINECriterion` as
a new **additive** auxiliary loss (`o2m_k`, `o2m_loss_weight`) on the final
decoder layer only — on top of, not replacing, the existing one-to-one loss.
Zero effect on eval/inference; matching only changes training-time gradient.

## Starting checkpoint

`output/runpod_results/e6_1280_best_ep46.pth` (AP=0.344, current best).
`reset_score_head_bias: False` deliberately — this isn't a resolution jump,
resetting would destroy E6's already-correct calibration.

## Verification so far (local, RTX 4060, 2026-07-07)

- Unit test on the fixed matcher: each GT matched to exactly `k` distinct
  queries, no query reuse, empty-target images handled cleanly.
- 100 real training iterations on the actual pod-bound config, including a
  dense-scene batch (173+103 boxes in one batch) — all `loss_*_o2m` terms
  (vfl, bbox, giou, nwd) finite and stable, no crash, no NaN, GPU memory
  within budget.

## Results

| Epoch | AP50:95 | Notes |
|-------|---------|-------|
| — | — | not yet launched — implementation is done and locally validated, but the full 30-epoch run on `e6_1280`'s checkpoint hasn't executed. Config: `o2m_k: 3`, `o2m_loss_weight: 0.5`, 30 epochs |
