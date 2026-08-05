# Experiment: msfd_1024 polish — ultra-low-LR stabilization pass

## Status: DONE — superseded by `msfd_1024_polish2` (same run, continued)

## What this tests

`msfd_1024` finished at AP=0.3219 (ep109) but hit BUG-044 (stage-2 LR
scheduler reset) mid-climb. `msfd_1024_polish` is a targeted low-LR
continuation from that checkpoint to let it finish converging cleanly,
augmentations off, rather than restarting the whole 110-epoch run.

## Starting checkpoint

`output/runpod_results/msfd_1024_best_ep109.pth` (AP=0.3219).

## Results

This run continues directly into `msfd_1024_polish2` (same watchdog-managed
session) — see `experiments/msfd_1024_polish2/README.md` for the final
number (AP=0.3226). No separate intermediate checkpoint from `polish` alone
is referenced elsewhere in the project notes; treat `polish`/`polish2` as one
continuous experiment split across two directories.
