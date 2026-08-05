# SAHI-style tiled training — closing the train/inference mismatch

## Hypothesis

An earlier inference-only SAHI ablation (slice=1024px/640px, overlap=0.2, on
the epoch-131 checkpoint) hurt accuracy: 29.7% → 28.0-28.3% AP50:95 (see
`00_progress.md` Step 14, README.md's Inference-Time Ablation table). The
model had never seen tiled/cropped input during training, only full images.
Hypothesis: train on the same overlapping tiles the eval will use, so the
model isn't out-of-distribution at inference time.

Built from scratch (not reusing the old, `master`-only `sahi_inf.py`):
- `src/data/tiling.py` — shared tile-grid geometry (640x640, 50% overlap /
  320px stride) and GT box remapping/clipping, used identically by the
  offline train-set builder and the eval path.
- `tools/tiling/build_tiled_visdrone.py` — offline-tiles the VisDrone train
  set into a new COCO-format split (not committed — regenerated locally,
  see `.gitignore`).
- `src/solver/tiled_eval.py` (`evaluate_tiled`) — for each **original,
  untiled** image: tile → per-tile inference → map boxes back to full-image
  coordinates → pool all tiles' boxes for that image → class-wise NMS once
  over the pooled set → score the merged result against the real ground
  truth. A tile is only ever a prediction mechanism, never the scored unit.
  Wired into `det_solver.py` via a `tiled_eval:` config block, so this is
  the eval used for **both** per-epoch training validation (checkpoint
  selection) **and** `--test-only` — not a whole-image-resize proxy.
- `tools/tiling/tiled_eval.py` — thin CLI wrapper around the same function,
  for one-off eval runs against any checkpoint.
- `experiments/sahi_tiled/config.yml` — `__include__`s `msfd_1024` (today's
  strongest architecture: P2ConvHead + P2FusionLite + NWD matcher), but
  **tunes from a fresh COCO-pretrained checkpoint**
  (`weight/dfine_s_coco.pth`), not an already VisDrone-adapted one — cleaner
  test of whether tiling itself is the lever. This exact combination has no
  precedent in this project (every prior P2/NWD run started from an
  already-adapted checkpoint), so warmup is long as a hedge against the
  early-collapse pattern in `11_ablation_study_runpod.md` (W3).

## Verification (2026-08-05, local RTX 4060, before any RunPod launch)

1. **Tiling geometry, by hand:** VisDrone images are one of 3 fixed sizes.
   1920x1080 → 5x3=15 tiles, 1360x765 → 4x2=8 tiles, 960x540 → 2x1=2 tiles
   (height 540<640, single row). Code matches hand calculation.
2. **Box remapping, visually:** ran `build_tiled_visdrone.py` on a 6-image
   subset (6 → 30 tiles, 363 → 928 tile-local boxes — the >1x box growth is
   expected, the same object appears in every tile it overlaps). Drew boxes
   on 3 sample tiles and inspected — pixel-perfect alignment against the
   source objects, including across the 50%-overlap seam between adjacent
   tiles (`tools/tiling/build_tiled_visdrone.py` output spot-checked
   directly, not just trusted).
3. **Eval path, end-to-end:** `tools/tiling/tiled_eval.py` run against
   `output/runpod_results/e6_1280_best_ep46.pth` (today's best, never
   trained on tiles) — full 548-image val set:

   ```
   python tools/tiling/tiled_eval.py -c experiments/e6_1280/config.yml \
       -r output/runpod_results/e6_1280_best_ep46.pth \
       --ann dataset/visdrone/annotations/instances_val.json \
       --img-dir dataset/visdrone/VisDrone2019-DET-val/images \
       --tile 640 --overlap 0.5 --device cuda:0
   ```

## Results so far

| Run | AP50:95 | AP50 | AP75 | AP-small | AP-medium | AP-large |
|-----|---------|------|------|----------|-----------|----------|
| `e6_1280`, standard whole-image eval | 0.344 | 0.549 | 0.356 | 0.257 | 0.453 | 0.626 |
| `e6_1280` + tiled eval (640px/50%), **no tile training** | 0.146 | 0.363 | 0.090 | 0.108 | 0.215 | 0.242 |

**This is the missing apples-to-apples control** — the earlier documented
SAHI numbers (28.0-28.3% vs 29.7%) are from a different, older checkpoint at
different tile/overlap settings (20% overlap, not 50%) and aren't valid for
comparing against today's best. Measured fresh instead.

The drop here (-0.198 AP50:95, -58% relative) is far larger than the earlier
ablation's ~1.7-2.1pt drop. AP-large fell hardest (0.626→0.242) — consistent
with large objects fragmenting or duplicating across the denser 50%-overlap
tile grid, which the model (trained only on full 1280px images with a wide
receptive field) has never had to handle. This makes the train/inference
mismatch hypothesis sharper to test than the original ablation did: if
train-time tiling closes most of a 0.198 AP50:95 gap, that's a strong,
legible result either way.

## Next: full training run

Not yet launched. Plan (RunPod, large-VRAM GPU, batch size tuned up from
the config's placeholder `32`):
1. Sanity gate: 2-3 epochs, confirm loss decreases, checkpoint saves.
2. Confirm the per-epoch tiled-eval validation signal (now the actual
   training-loop eval, not a separate post-hoc step) produces sane numbers.
3. Full run for as long as the time budget allows; report final
   AP50:95/AP50/AP-small/AP-medium/AP-large via `tiled_eval.py` against the
   original untiled val set, plus a standard whole-image-at-640 number as a
   cheap sanity cross-check.
4. Update this table, `experiments/sahi_tiled/README.md`, and the main
   `README.md` Results/Model-comparison tables with the final number.
