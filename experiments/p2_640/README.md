# Experiment: p2_640 — standard DETR-style P2 (full MSDeformableAttention on 4 levels) at 640×640

## Status: NEVER LAUNCHED

## What this tests

The "v1" approach to a P2 (stride-4) detection level: add P2 as a genuine
4th transformer level with full MSDeformableAttention (`return_idx: [0,1,2,3]`,
`num_levels: 4`, fewer sample points at P2 to limit compute), as opposed to
`msfd_1024`'s "v2" approach (transformer stays 3-level; P2 handled by a
YOLOv8-style conv-only head, no attention). See
`PROJECT_NOTES/11_ablation_study_runpod.md` §2 for the full v1-vs-v2
comparison table.

## Starting checkpoint

`output/dfine_hgnetv2_s_visdrone_nwd/best_stg1_dfine_s_visdrone_nwd_sqrt.pth`
(AP=0.321), planned.

## Results

| Epoch | AP50:95 | Notes |
|-------|---------|-------|
| — | — | never launched — user killed the planned p2_640-vs-msfd_640 tournament runner (2.6x the compute of `msfd_1024` alone) before it started; `msfd_1024` (conv-only P2, cheaper) was run instead and became the flagship P2 lineage (see `experiments/msfd_1024/README.md`) |
