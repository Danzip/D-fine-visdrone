# Experiment: R1+R2+R3 bundled — zoom-crop + NWD regression loss + rare-class CopyPaste (plain, non-P2 architecture)

## Status: DONE — did not beat standing best

## What this tests

Three augmentation/loss ideas bundled into one run, on the simpler plain
3-level (non-P2) architecture, run in parallel with `plain_r2r3_nozoom` to
isolate R1's individual effect:
- **R1 (crop-zoom / SAHI-style train-time slicing):** train on 640² windows
  cropped from ≥960-scaled images — objects get 1.5-2x more pixels during
  learning, zero deploy cost (unlike SAHI at inference time).
- **R2 (NWD regression loss):** NWD was already in the matcher cost + SAL
  weighting; this adds it to the box regression loss itself (previously
  L1+GIoU only, scale-blind).
- **R3 (rare-class CopyPaste 2.0):** paste rare classes (bicycle, tricycle,
  awning-tricycle) preferentially with scale jitter, instead of any
  small object.

## Starting checkpoint

`configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml`'s own lineage (plain
architecture, not the P2/msfd branch).

## Results

| Run | AP50:95 | vs standing best (0.3226) | Notes |
|-----|---------|---------------------------|-------|
| `plain_r1r2r3` (R1+R2+R3 bundled) | 0.3183 | -0.0043 | `output/runpod_results/plain_r1r2r3_last.pth` |

Bundle underperformed the standing best. R2/R3's isolated effects weren't
separately measured (no R2-only or R3-only control run) — see
`plain_r2r3_nozoom` for the R1-isolation control, which shows R1 (crop-zoom)
itself was ~neutral, meaning R2+R3 together are what's actually costing the
~0.4 AP here, not R1.
