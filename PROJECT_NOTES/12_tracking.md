# ByteTrack Multi-Object Tracking (2026-08-03)

## What

Added multi-object tracking on top of the existing D-FINE detector, using
[supervision](https://github.com/roboflow/supervision)'s ByteTrack
implementation (`supervision==0.29.1`). Detection stays exactly as trained
(D-FINE's own postprocessor, plain resize to `eval_spatial_size`, no
letterbox — matches val-time preprocessing) — ByteTrack just adds temporal
identity on top of the per-frame boxes.

**Files:**
- `tools/tracking/track_video.py` — loads a D-FINE checkpoint via the same
  `YAMLConfig` + `solver.eval()` pattern as `tools/calibration/calibrate_scores.py`
  (handles EMA-vs-model weights and shape-matched loading for a different
  `eval_spatial_size` automatically), runs it frame-by-frame on a video, feeds
  detections to ByteTrack, writes an annotated mp4 with track IDs + motion trails.
- `tools/tracking/build_video.py` — one-off helper, assembles a VisDrone-MOT
  frame sequence into an mp4 for testing (this project's DET subset has no
  video/sequence data of its own).
- `requirements.txt` — added `supervision>=0.29.0`.

## Test data

VisDrone here is the DET subset (independent images, no motion). Downloaded
the official **VisDrone2019-MOT-val** set (real drone video sequences, same
domain/labeling convention as the DET data D-FINE was trained on) via
`gdown` from the official Google Drive link in the
[VisDrone-Dataset repo](https://github.com/VisDrone/VisDrone-Dataset) —
extracted to `dataset/visdrone_mot/` (not committed, same pattern as
`dataset/visdrone/`).

Test sequence: **`uav0000137_00458_v`** — 233 frames, 2688×1512, a busy
street intersection (~104 objects/frame average, 184 unique ground-truth
tracks total per the MOT annotation file). Chosen deliberately as a stress
test: dense, small objects, panning camera.

Model: **e6_1280** checkpoint (`output/runpod_results/e6_1280_best_ep46.pth`,
config `experiments/e6_1280/config.yml`) — current standing-best detector,
AP=0.344, 1280×1280 input.

## Environment note

The Bash tool in this session is Windows Git Bash browsing the repo through
the `\\wsl.localhost\...` UNC path — it cannot execute this project's
Linux-native venv (`venv/bin/python`, built inside real WSL2 for CUDA
passthrough). All Python here had to go through `wsl.exe -e bash -lc "..."`
from PowerShell instead; a plain `source venv/bin/activate` in the Bash tool
silently resolves to an unrelated Windows Python install with no error. See
memory note `project-wsl-execution-path` if this bites again.

## Baseline run — vanilla ByteTrack, naive threshold

First pass: `sv.ByteTrack(frame_rate=fps, minimum_consecutive_frames=3)`,
detections pre-filtered to `score >= 0.3` before ever reaching the tracker.

| Metric | Value |
|---|---|
| Unique tracks | 1,039 (vs. 184 GT — **5.6x inflation**) |
| Median track length | 18 / 233 frames |
| Mean track length | 37.5 frames |
| Tracks lasting <=2 frames | 5% |
| Speed | 4.69 FPS |

Detection quality itself was excellent (dense, correct boxes even in
clutter) — the problem was purely temporal identity persistence.

## Fixes (user-requested: hysteresis + the two suggested directions)

**1. Confirmed hysteresis was already active** (`minimum_consecutive_frames=3`,
now exposed as `--min-consecutive-frames`). This prevents noisy
single-frame detections from ever spawning a track, but does *not* stop
already-confirmed real tracks from fragmenting under occlusion/camera
motion — it only explains the small 5%/2-frame tail, not the bulk of the
1,039-vs-184 inflation.

**2. Bug fix: low-confidence detections were never reaching the tracker.**
ByteTrack's core association algorithm (`supervision`'s
`byte_tracker/core.py`) hard-codes a second "low score" recovery pass using
detections with `score > 0.1` — this is the actual point of ByteTrack (BYTE
= use every detection box, not just confident ones, for association). The
original code filtered to `score >= 0.3` *before* constructing
`sv.Detections`, so that recovery pass never had anything below 0.3 to work
with — silently disabled. Fixed: detections are now passed through down to
`--conf-low` (default 0.1), and `--conf-high` (default 0.3) is passed as
`track_activation_threshold` so ByteTrack itself decides the high/low split
internally, same as it does in the reference implementation.

**3. Camera motion compensation (GMC).** `MotionCompensator` class in
`track_video.py`: ORB features + RANSAC-estimated affine transform between
consecutive frames, composed into a running transform to a shared "world"
frame anchored at frame 0. Detection boxes are warped into world space
before being handed to the tracker (so a panning/rotating drone camera
looks stationary to ByteTrack's constant-velocity Kalman model) and warped
back to pixel space for display.

**Known limitation, stated explicitly rather than glossed over:** this is a
simpler *external* approximation of proper GMC, not what BoT-SORT does
natively (which warps each track's Kalman mean/covariance directly inside
the tracker). `supervision`'s `ByteTrack` doesn't expose that hook, so this
warps detections in and predictions back out instead. Expect some residual
drift over very long sequences since the world-frame reference accumulates
per-frame affine estimates — fine over 233 frames, not verified for
much longer sequences.

## Result — same test sequence, all three fixes active

| Metric | Baseline | Fixed | Delta |
|---|---|---|---|
| Unique tracks | 1,039 | **670** | -35% |
| Track inflation vs. 184 GT | 5.6x | **3.6x** | |
| Median track length | 18 | **52** | 2.9x |
| Mean track length | 37.5 | **77.1** | 2.1x |
| Tracks <=2 frames | 5% | **2%** | |
| Speed | 4.69 FPS | 6.44 FPS | |

Outputs: `output/tracking/tracked_uav0000137.mp4` (baseline, kept for
comparison), `output/tracking/tracked_uav0000137_v2.mp4` (fixed).
`output/tracking/` is not committed (matches the existing `output/` pattern).

**Still 3.6x inflated vs. ground truth — real room left.** Suspected
remaining cause: ID switches between visually-similar objects in tight
clusters (e.g. grouped pedestrians/cyclists at the crosswalk), which motion
compensation and hysteresis can't fix — that needs appearance-based
Re-ID matching. `supervision==0.29.1`'s `ByteTrack` is motion-only (no
appearance embeddings) — see the "ByteTrack vs DeepSORT" question below.

## ByteTrack vs. DeepSORT vs. hybrids

Asked whether ByteTrack has DeepSORT built in, or if there's a hybrid:

- **ByteTrack**: pure motion (Kalman filter + IoU + Hungarian matching), no
  appearance/Re-ID model. Its innovation is using *every* detection
  (including low-score ones) for association via a two-stage match, not
  appearance matching. This is what `supervision`'s `ByteTrack` implements —
  confirmed no other tracker module exists in the installed package
  (`venv/lib/python3.12/site-packages/supervision/tracker/` has only
  `byte_tracker/`).
- **DeepSORT**: SORT (Kalman + Hungarian, ByteTrack's ancestor) + a CNN
  Re-ID embedding, matched via cosine distance, specifically to survive
  occlusion/appearance-based re-identification that pure motion can't.
- **Hybrids that combine both ideas exist, but not in `supervision`:**
  BoT-SORT (ByteTrack + internal GMC + optional Re-ID — closest thing to
  "ByteTrack + DeepSORT"), StrongSORT (DeepSORT + stronger Re-ID backbone +
  BYTE-style matching), OC-SORT (motion-only, improved occlusion handling,
  no appearance), DeepOCSORT (OC-SORT + appearance).

  Implemented and measured all four via `boxmot==19.0.0` (`create_tracker`
  factory, common `.update(dets, img) -> TrackResults` interface across
  trackers) — see the comparison below.

## Tracker comparison — BoT-SORT / StrongSORT / OC-SORT / DeepOCSORT vs. our fixed ByteTrack

`tools/tracking/compare_trackers.py`: runs D-FINE detection **once** per
frame (the expensive 1280x1280 pass), caches the raw detections, then
replays every tracker against the identical cached stream — so differences
below are attributable to the tracker, not detection noise between runs.
All boxmot trackers used out-of-the-box library defaults (no hand-tuning)
plus `osnet_x0_25_msmt17.pt` (smallest zoo Re-ID model) where appearance
matching is used. Same test sequence (`uav0000137_00458_v`, 233 frames,
184 GT tracks).

| Tracker | Unique tracks | vs. 184 GT | Median len | Mean len | <=2 frames | FPS |
|---|---|---|---|---|---|---|
| ByteTrack (ours: low-conf fix + GMC) | 670 | 3.64x | 52 | 77.1 | 2% | 10.8 |
| OC-SORT (motion-only) | 660 | 3.59x | 13 | 32.2 | 13% | 13.8 |
| **BoT-SORT** (GMC + Re-ID) | **362** | **1.97x** | **56** | 80.9 | 6% | 7.2 |
| **StrongSORT** (Re-ID) | **335** | **1.82x** | **62** | 78.0 | 6% | 3.1 |
| DeepOCSORT (motion + Re-ID) | 664 | 3.61x | 18 | 44.8 | 15% | 4.2 |

**Headline finding: appearance (Re-ID) matching roughly halves track
fragmentation on this dense aerial scene** — BoT-SORT and StrongSORT both
land at <2x GT inflation, vs. ~3.6x for every motion-only tracker
(including our own GMC-augmented ByteTrack). This confirms the hypothesis
from the earlier writeup: the remaining fragmentation after GMC was mostly
ID switches in tight clusters (grouped pedestrians/cyclists at the
crosswalk) that motion alone can't disambiguate but appearance can.

**DeepOCSORT underperforming despite having Re-ID is a real, unexplained
result, not a bug we found** — it did no better than the motion-only
trackers here (664 tracks, worse median length than even OC-SORT). Not
root-caused (out of scope for this pass); plausible factors going in order
of suspicion: its default `det_thresh=0.5` is the highest of all five
(cuts recall harder before Re-ID even gets a chance to help), its
association-weight blending (`w_association_emb=0.75`, `aw_param=0.5`) may
be tuned for eye-level MOT17-style pedestrian footage rather than this
aerial/top-down viewing angle, or its CMC (`cmc_off=False`, method
unspecified/default) may be less effective than BoT-SORT's here. Would
need a targeted ablation (toggle `embedding_off`/`cmc_off` individually) to
actually attribute this — not done.

**Speed spread is real and matters for any live/near-real-time use**:
StrongSORT's per-frame full-image Re-ID embedding pass makes it the
slowest by a wide margin (3.1 FPS) despite being the most accurate here.
BoT-SORT gets most of the same accuracy gain (1.97x vs. 1.82x GT
inflation) at more than 2x the speed (7.2 vs. 3.1 FPS) — **BoT-SORT is the
practical pick** if this needs to run closer to real time; StrongSORT if
accuracy is the only axis that matters.

Outputs: `output/tracking/compare/{bytetrack,ocsort,botsort,strongsort,deepocsort}.mp4`
(not committed, same pattern as the rest of `output/`).

## Why tracker FPS is so low despite only ~600-700 total unique tracks

Asked why StrongSORT (3.1 FPS) and the others are so slow given the video
only produced ~600-700 *distinct track IDs total* over 233 frames. That
total is the wrong number to look at — it says nothing about the per-frame
cost. Profiled with `cProfile` (40 frames, `strongsort`/`botsort`):

```
detections/frame: mean=500.0 min=500 max=500
```

**The real driver: every single frame is saturating the detector's
`num_top_queries=500` cap.** `e6_1280`'s config caps top-k at 500; at
`conf_low=0.1` on this dense scene, all 500 top-scoring queries clear that
floor on literally every frame — vs. ~104 GT objects/frame. Each tracker is
matching up to 500 candidate boxes against ~100+ simultaneously-active
tracks, every frame, for 233 frames (~116,500 detection-instances total)
— that's the actual per-frame compute load, not the "600 tracks created
over the whole video" figure.

Where the time goes, profiled per-tracker:

- **StrongSORT (3.1 FPS)** — bottleneck is the **matching cascade, not
  Re-ID**: `scipy.linalg.solve_triangular` (Kalman gating / Mahalanobis
  distance) is 44% of total runtime (4.53s/10.2s), called 5,257 times over
  39 frames — once per active track per frame, computed sequentially in
  pure Python/scipy, not batched across tracks. Re-ID embedding extraction
  is smaller (41%) than the matching cost.
- **BoT-SORT (7.2 FPS)** — bottleneck is **Re-ID embedding extraction**:
  `resolve_batch_embeddings` (crop + OSNet forward per detection) is 57% of
  runtime (2.92s/5.1s); of that, `get_crops` alone (CPU-side `cv2.resize`
  per box, not even GPU work) is 31% of total runtime by itself. BoT-SORT's
  matching is vectorized and cheap, which is why it's ~2.3x faster than
  StrongSORT despite both doing Re-ID.

Both are research reference implementations (per-track sequential Kalman
gating; per-box CPU crop loop) — neither is written to scale gracefully to
500 dets/frame, they assume a more typical few-dozen-per-frame load.

## NMS experiment — cutting detections/frame before tracking

D-FINE is end-to-end (no NMS in the model itself, matching that
`DFINEPostProcessor.forward` just does sigmoid + topk over flattened
(query, class) scores — see `src/zoo/dfine/postprocessor.py`). At a low
`conf_low` on a dense scene, multiple queries can converge on the same
physical object, inflating detections-per-frame well past the true object
count. Added `--nms-iou` to `detect()` in `track_video.py` (used by both
`track_video.py` and `compare_trackers.py`): per-class (`torchvision.ops.
batched_nms`), applied after the `conf_low` filter and before the tracker
ever sees the detections. Per-class, not class-agnostic, deliberately —
duplicate boxes on one object share its class, but e.g. a pedestrian
on/near a bicycle can legitimately overlap across classes and shouldn't
suppress each other.

Re-ran the full 5-tracker comparison with `--nms-iou 0.6`, same video,
same cached-detection-once methodology:

**Detections/frame: 500.0 (saturated) -> mean 374.2 (min 339, max 400) — a
real 25% cut.** But that's still far above the ~104 GT objects/frame — the
gap isn't purely duplicate boxes on the same instance; VisDrone-MOT's
ground truth only tracks "relevant" moving/notable objects, and D-FINE at
`conf_low=0.1` also correctly detects plenty of legitimate but
un-annotated objects (e.g. individual bikes in the parked bike-share racks
visible on the left edge of the frame) that were never given GT track IDs.
So NMS removed real duplicates, but "374 vs 104" isn't all duplicate-driven
error — some of that gap is GT annotation scope, not detector noise.

| Tracker | Tracks (no NMS) | Tracks (NMS 0.6) | Δ tracks | FPS (no NMS) | FPS (NMS 0.6) | Δ FPS |
|---|---|---|---|---|---|---|
| ByteTrack (ours) | 670 | 606 | -9.6% | 10.8 | 11.3 | +4% |
| OC-SORT | 660 | 630 | -4.5% | 13.8 | 13.6 | -1% |
| BoT-SORT | 362 | 359 | -0.8% | 7.2 | 7.4 | +3% |
| StrongSORT | 335 | 328 | -2.1% | 3.1 | 3.1 | ~0% |
| DeepOCSORT | 664 | 571 | -14.0% | 4.2 | 4.3 | +3% |

**Result: NMS did what it was supposed to (cut raw detections 25%) but did
NOT meaningfully fix the speed problem — FPS moved by ~±3% across the
board, within noise.** This contradicts the "detection count drives
runtime" hypothesis from the profiling section above, and the profiling
data explains why in hindsight:

- **StrongSORT's bottleneck (Kalman gating cascade) scales with active
  *track* count, not raw detection count.** `gating_distance` is called
  once per confirmed track per frame (vectorized across whatever
  detections remain in that round), so cutting detections 25% without
  cutting the ~100+ simultaneously-active tracks barely touches the
  dominant cost. Track count barely moved either (only -0.8% to -9.6%
  across trackers) because most of those 500→374 detections being removed
  were near-duplicates of objects that were already going to become (or
  already were) a single track either way — NMS mostly removed redundant
  *evidence* for tracks that existed regardless.
- **BoT-SORT's embedding cost is more fixed-overhead-bound than
  compute-bound at this scale** — GPU kernel launch / crop-loop overhead
  per call doesn't scale linearly down with a 25% smaller batch of crops.

**Practical takeaway:** NMS is a reasonable detector-hygiene improvement
(real duplicates removed, cleaner detection stream) but is not the lever
for tracker speed on this workload — the trackers' own per-track-count
scaling (StrongSORT) or fixed-overhead (BoT-SORT) dominate. Actual speed
fixes would mean touching the tracker internals (batching StrongSORT's
gating across tracks, which `boxmot`'s reference implementation doesn't do)
or accepting the current FPS for offline/near-real-time-but-not-live use.
`--nms-iou` is left in `track_video.py`/`compare_trackers.py` as an
available flag (default off) since it's a legitimate detection-quality
improvement independent of the speed question.
