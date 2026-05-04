# Bugs, Issues & Fixes Log

All issues encountered during development, their root cause, and resolution status.

---

## BUG-001 — W&B service process fails to start on Windows

**Status:** ⚠️ WORKAROUND (not fully fixed — fix planned via WSL2)

**Symptom:**
Running `python train.py` with `use_wandb: True` on Windows causes W&B to hang or crash
on startup. Training never begins.

**Root Cause:**
W&B spawns a background service daemon process at startup. On Windows, Python's
`multiprocessing` uses `spawn` start method (not `fork`), which causes the W&B
service process to fail silently.

**Workaround applied** (`det_solver.py`):
```python
os.environ.setdefault("WANDB_START_METHOD", "thread")
```
Forces W&B to run its service in a thread instead of a subprocess. Partially works
but W&B features (media logging, live charts) are unreliable in thread mode.

**Proper fix:** Run training under WSL2 where Linux process spawning works correctly.
See `05_wsl2_aws_kubernetes.md`.

**Config:** `use_wandb: False` in `dfine_hgnetv2_s_visdrone.yml` — W&B disabled until WSL2.

---

## BUG-002 — TensorBoard only showed first event file, ignored newer runs

**Status:** ✅ FIXED (delete old event files before each run)

**Symptom:**
TensorBoard launched with `--logdir output/.../summary/` only displayed data from the
first training run (up to step ~2100). Subsequent runs created new `.tfevents` files in
the same directory but TensorBoard never loaded them — even while training was actively
writing new data.

**Root Cause:**
Every `python train.py` call creates a new `.tfevents` file in the same directory.
After several runs, multiple event files piled up and TensorBoard showed the oldest one.
There is exactly **one event file per run** — old ones serve no purpose and cause confusion.

**Fix — archive old event files before starting a new run:**
```bash
# Move old events to a timestamped archive folder, then start fresh
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p output/dfine_hgnetv2_s_visdrone/summary/archive/$timestamp
mv output/dfine_hgnetv2_s_visdrone/summary/events.out.tfevents.* \
   output/dfine_hgnetv2_s_visdrone/summary/archive/$timestamp/
python train.py ...
```

To review an old run, point TensorBoard at the archive:
```bash
tensorboard --logdir output/dfine_hgnetv2_s_visdrone/summary/archive/20260324_210000
```

**Note:** `--reload_multifile true` was tried but did not work in practice.
**Note:** Do NOT delete event files — they contain full loss curves, LR history, and image snapshots for every logged step.

---

## BUG-003 — TensorBoard Images tab was empty despite logging calls succeeding

**Status:** ✅ FIXED

**Symptom:**
Loss and LR curves appeared correctly in TensorBoard but the Images tab was completely
empty. No errors visible in the training log (exception was silently caught).

**Root Cause:**
`wandb_viz.py` → `_run_inference()` called the postprocessor incorrectly:

```python
# WRONG — postprocessor returns List[Dict], not a tuple
labels, boxes, scores = self.postprocessor(outputs, orig_size)
```

`DFINEPostProcessor.forward()` returns `List[{"labels": ..., "boxes": ..., "scores": ...}]`
(one dict per image in the batch). Trying to unpack a 1-element list into 3 variables
raises `ValueError: not enough values to unpack`, which was caught by the outer
`try/except` in `log_epoch_tensorboard`, silently swallowing the error.

**Fix applied** (`wandb_viz.py`):
```python
results = self.postprocessor(outputs, orig_size)
result = results[0]  # batch size = 1
labels = result["labels"]
boxes = result["boxes"]
scores = result["scores"]

mask = scores > self.score_threshold
return (
    im,
    labels[mask].cpu().numpy().tolist(),
    boxes[mask].cpu().numpy().tolist(),
    scores[mask].cpu().numpy().tolist(),
)
```

---

## BUG-004 — `torch.stack` crash when batching images for TensorBoard

**Status:** ✅ FIXED

**Symptom:**
After fixing BUG-003, image logging still failed at step 500/1000 with:
```
stack expects each tensor to be equal size,
but got [3, 765, 1360] at entry 0 and [3, 540, 960] at entry 1
```

**Root Cause:**
VisDrone val images have varying resolutions (not all 1920×1080). When stacking
3 images per class into a single `[N, C, H, W]` tensor for `add_images()`, PyTorch
requires all images to have the same spatial dimensions.

**Fix applied** (`wandb_viz.py` → `log_epoch_tensorboard`):
After drawing bounding boxes, resize each image to `self.input_size` (640×640) before
appending to the stack:
```python
img_display = TF.resize(img_uint8, list(self.input_size))
class_imgs.append(img_display.float() / 255.0)
```
Boxes are drawn at original resolution first (coordinates are correct), then the whole
annotated image is resized for display — no coordinate scaling issues.

---

## BUG-005 — TensorBoard image panels cluttered (30 separate panels)

**Status:** ✅ FIXED

**Symptom:**
Original implementation used `writer.add_image()` once per image, creating
10 classes × 3 images = 30 separate panels in the TensorBoard Images tab.
Impossible to browse without excessive scrolling.

**Fix applied** (`wandb_viz.py`):
Switched to `writer.add_images()` (plural) which takes a `[N, C, H, W]` tensor.
All 3 images for a class are stacked and logged under a single tag `val/<class_name>`.
Result: 10 panels total (one per class), each showing 3 images side-by-side.
Step slider allows scrubbing through history; latest step shown by default.

---

## BUG-006 — Image visualization triggered per epoch instead of per N steps

**Status:** ✅ FIXED

**Symptom:**
`log_epoch_tensorboard` was called at end-of-epoch in `det_solver.py`. With ~1617
steps/epoch at ~15 min/epoch, you had to wait 15 minutes for any image feedback.

**Fix applied:**
- Moved visualization trigger into `train_one_epoch` in `det_engine.py`
- Fires every `viz_step_interval=500` global steps (~5 minutes at observed training speed)
- Also triggers at `global_step == 0` for immediate feedback on run start
- Removed end-of-epoch call from `det_solver.py`

**Step math:** 2100 steps observed in 30 min → ~70 steps/min → 5 min ≈ 350 steps.
Rounded up to 500 for a clean number with some buffer.

---

## BUG-007 — CUDA OOM during overnight run

**Status:** ✅ KNOWN — requires `total_batch_size=4` override

**Symptom:**
Overnight run crashed at epoch 9 step 0 with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.06 GiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
Of the allocated memory 12.96 GiB is allocated by PyTorch
```

**Root Cause:**
The overnight run command omitted the `-u train_dataloader.total_batch_size=4` override.
The config default is `total_batch_size=8`. On the RTX 4060 Laptop (8GB VRAM),
batch_size=8 works for most epochs but eventually fragments memory enough to OOM.
The `12.96 GiB allocated` exceeds the 8GB physical VRAM — Windows was spilling to
shared system RAM (slower but tolerated for a while before failing).

**Fix:**
Always use `total_batch_size=4` on this machine with `use_amp: False`.

**Note — AMP changes the equation entirely:**
The 640×640 run used `use_amp: False` (fp32 activations). Enabling AMP (`use_amp: True`)
stores activations in fp16, roughly halving activation memory. With AMP, batch=8 at
640×640 would likely fit in 8GB and the OOM would never have occurred.

The 1280×736 run uses `use_amp: True` with batch=4 and peaks at 6.7GB — even though
images are 2.3× larger — because fp16 activations compensate:
  2.3× more pixels × 0.5 (fp16) = 1.15× more VRAM vs 640×640 fp32 batch=4.

Rule of thumb for this GPU:
  - fp32 (no AMP): max safe batch ≈ 4 at 640×640
  - fp16 (AMP):    max safe batch ≈ 8 at 640×640, or 4 at 1280×736

---

---

## BUG-008 — `PadToSize` uses removed `get_spatial_size` API (torchvision 0.20)

**Status:** ✅ FIXED

**Symptom:**
Running training with `PadToSize` transform raises:
```
AttributeError: module 'torchvision.transforms.v2.functional' has no attribute 'get_spatial_size'
```

**Root Cause:**
`PadToSize._get_params()` in `src/data/transforms/_transforms.py` called
`F.get_spatial_size(flat_inputs[0])`. This function was removed in torchvision 0.20
and replaced with `F.get_size()` (returns `[H, W]`).

Additionally, `PadToSize._transform()` used `self._fill[type(inpt)]` which fails
because torchvision 0.20's `T.Pad._fill` stores fills under an `'others'` key
(not per-type keys). Fixed by importing and using `_get_fill(self._fill, type(inpt))`.

**Fix applied** (`src/data/transforms/_transforms.py`):
```python
# Added import:
from torchvision.transforms.v2._utils import _get_fill
# In _get_params:
sp = F.get_size(flat_inputs[0])     # was: F.get_spatial_size(...)
# In _transform:
fill = _get_fill(self._fill, type(inpt))  # was: self._fill[type(inpt)]
```

---

## BUG-009 — `profiler_utils.stats()` uses square input for non-square models

**Status:** ✅ FIXED

**Symptom:**
Training at 1280×736 crashes immediately with:
```
RuntimeError: The size of tensor a (1600) must match the size of tensor b (920) at non-singleton dimension 1
```
in `hybrid_encoder.py`. The crash is in `calflops.calculate_flops` which runs a model
forward pass to count FLOPs, not in the actual training step.

**Root Cause:**
`profiler_utils.stats()` computes input shape as `(1, 3, base_size, base_size)` —
always square. With `base_size=1280` and a model configured for 1280×736, `calflops`
feeds a 1280×1280 image. The stride-32 feature map is 40×40=1600 elements. But the
model's cached eval pos_embed was built for 40×23=920 elements (from `eval_spatial_size=[736, 1280]`).
Since `calflops` runs the model in eval mode, the 920-element cached pos_embed is used
against the 1600-element feature map → shape mismatch.

**Fix applied** (`src/misc/profiler_utils.py`):
```python
eval_size = cfg.yaml_cfg.get("eval_spatial_size")
if eval_size:
    input_shape = (1, 3, eval_size[0], eval_size[1])   # [H, W] directly
else:
    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)          # original square fallback
```

---

## BUG-010 — `stop_epoch=0` in collate_fn triggers D-FINE stage1→stage2 reload at epoch 0

**Status:** ✅ FIXED

**Symptom:**
With `collate_fn.stop_epoch: 0`, training crashes at the start of epoch 0:
```
FileNotFoundError: 'output/dfine_hgnetv2_s_visdrone_1280/best_stg1.pth'
```

**Root Cause:**
`det_solver.py` line 109 contains D-FINE's two-stage training boundary logic:
```python
if epoch == self.train_dataloader.collate_fn.stop_epoch:
    self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
```
Setting `stop_epoch=0` to "disable multi-scale jitter immediately" accidentally
made this fire at epoch 0, trying to load a checkpoint that doesn't exist yet.

Multi-scale jitter is actually controlled by `scales=None` (when `base_size_repeat=None`),
not `stop_epoch`. The correct way to keep all epochs in "stage 1" mode without
triggering the boundary: set `stop_epoch` to a value > total epochs.

**Fix applied** (`configs/dfine/dfine_hgnetv2_s_visdrone_1280.yml`):
```yaml
collate_fn:
  stop_epoch: 200   # > 50 epochs → stage2 boundary never fires
  base_size_repeat: ~  # null → scales=None → multi-scale jitter disabled
```

---

## BUG-011 — WandbViz uses wrong input_size for non-640×640 models

**Status:** ✅ FIXED

**Symptom:**
When training at 1280×736, WandbViz logs `0 annotated images` and prints dozens of:
```
[WandbViz] Warning: failed on image N: The size of tensor a (400) must match tensor b (920)
```
400 = 20×20 (stride-32 features at 640×640), 920 = 40×23 (correct for 736×1280).

**Root Cause:**
`WandbVisualizer` default `input_size=(640, 640)` — the visualizer was resizing images
to 640×640 before inference, producing a stride-32 feature map of 20×20=400 elements.
But the model's cached eval pos_embed expects 40×23=920. Shape mismatch → inference fails.

**Fix applied** (`src/solver/det_solver.py`):
```python
_eval_sz = args.yaml_cfg.get("eval_spatial_size", [640, 640])
wandb_viz = WandbVisualizer(
    ...
    input_size=tuple(_eval_sz),   # pass [H, W] from config instead of hardcoded 640×640
)
```

---

## BUG-017 — CopyPasteSmallObjects crashes with 3-arg pipeline tuple

**Status:** ✅ FIXED

**Symptom:**
Training crashed at the start of epoch 132 (ms1280_cont run) in DataLoader worker 0:

```
File "src/data/transforms/_transforms.py", line 226, in __call__
    image, target = inputs
ValueError: too many values to unpack (expected 2)
```

**Where:** `CopyPasteSmallObjects.__call__` in `src/data/transforms/_transforms.py`

**Root cause — how the transform pipeline passes data:**

`coco_dataset.py` calls `self._transforms(img, target, self)` — passing 3 positional args
so transforms can read `dataset.epoch` for epoch-gated policies.

`container.py:stop_epoch_forward` packs these as `sample = (img, target, dataset)` (a 3-tuple),
then calls each transform as `transform(sample)` — passing the 3-tuple as a single argument.

Inside `CopyPasteSmallObjects.__call__(*inputs)`:
- `inputs = ((img, target, dataset),)` — a 1-tuple wrapping the 3-tuple
- `inputs = inputs if len(inputs) > 1 else inputs[0]` → `inputs = (img, target, dataset)` ✓
- Old code: `image, target = inputs` → **ValueError** — 3 values, 2 variables

The bug only fired when `stop_epoch_forward` was active (i.e., the pipeline had an epoch-gated
policy). `default_forward` passes args differently and was not affected.

**Why it only appeared at epoch 132:**
The ms1280_cont config has `policy: epoch: 9999` — the policy ops never trigger, so
`stop_epoch_forward` still runs but always takes the `else` branch (no ops are skipped).
Every transform gets called, including `CopyPasteSmallObjects`. It crashed on first call
of epoch 132 — we got lucky it had run 131 full epochs without the stop-epoch policy kicking in.
(The policy only skips ops when `cur_epoch >= policy_epoch`, but all transforms were still called
via `transform(sample)` the entire time.)

**Fix applied** (`src/data/transforms/_transforms.py`):
```python
# Before:
inputs = inputs if len(inputs) > 1 else inputs[0]
image, target = inputs

# After:
inputs = inputs if len(inputs) > 1 else inputs[0]
extra = inputs[2:] if len(inputs) > 2 else ()   # preserve dataset ref
image, target = inputs[0], inputs[1]
```

Return sites updated to `return (image, target) + extra` to pass the dataset ref downstream.

---

## BUG-018 — WSL2 OOM killer silently kills training mid-epoch (Mosaic run)

**Status:** ⚠️ WORKAROUND KNOWN — fix requires `.wslconfig` change

**Symptom:**
ms1280_mosaic training stopped mid-epoch 13 (step 2900/4075, ~71% through) with no Python
traceback, no error message, and no W&B crash report. The train log simply ends.

Last line of `output/train_mosaic.log` is a block of null bytes (`\x00\x00\x00...`).
`output/dfine_hgnetv2_s_visdrone_ms1280_mosaic/log.txt` stopped being written at **03:14:28**.
W&B background thread kept sending data until **03:34:34** — 20 minutes after the main process died.

**Root cause — WSL2 RAM cap + Mosaic memory spike:**

WSL2 total system RAM: **7.56 GB** (`MemTotal: 7,723,200 kB` in `/proc/meminfo`).

Standard training memory layout at steady state:
- PyTorch process (model weights, optimizer state, activations): ~3–4 GB
- 4 DataLoader workers × 1 image each: ~200 MB
- CUDA driver + misc: ~500 MB
- Headroom: ~2–3 GB

Mosaic changes the DataLoader budget significantly:
- Each worker now loads and composites **4 images** before passing the batch forward
- A 1024×1024 float32 mosaic composite = 1024 × 1024 × 3 × 4 bytes = **12 MB per worker**
- With `num_workers: 4`, peak in-flight CPU RAM for images = 4 workers × 4 images × ~3 MB (PIL) + 4 composites = ~60–80 MB extra per step
- But the main spike comes from the worker **process fork**: each of 4 workers gets a copy-on-write clone of the entire Python process; when Mosaic performs large temporary allocations (4 PIL images + intermediate numpy arrays), the copy-on-write pages materialize in RAM for all workers simultaneously

This pushes total RAM from ~5–6 GB to above 7.56 GB at peak, triggering the OOM killer.

**How the null bytes confirm SIGKILL:**

The null bytes are the unfilled portion of a pre-allocated write buffer. When the OS writes
to a file, it pre-allocates a page-aligned buffer. If the process is hard-killed (SIGKILL)
mid-write, the kernel flushes the pre-allocated block to disk with the unwritten portion zeroed.
A clean Python exit (exception, sys.exit) always flushes stdout and closes file handles — no nulls.
Null bytes = SIGKILL = OOM killer, not a code bug.

**Why W&B survived 20 min after the main process died:**

W&B's internal sync daemon runs as a background thread (using `WANDB_START_METHOD=thread`).
When the main Python process was killed, the W&B thread was mid-flush and survived in a zombie
state, draining its queue to the W&B API until the thread was also torn down at 03:34.

**Fix 1 — Give WSL2 more RAM (recommended):**

Create or edit `C:\Users\<yourname>\.wslconfig`:
```ini
[wsl2]
memory=12GB
processors=8
```
Then from PowerShell: `wsl --shutdown`, reopen terminal. Verify with `cat /proc/meminfo`.

Your physical machine almost certainly has 16+ GB RAM; the default WSL2 cap is 50% of physical
or 8 GB (whichever is lower). Setting 12 GB leaves 4 GB for Windows and is safe.

**Fix 2 — Reduce DataLoader workers (fallback if .wslconfig isn't changed):**

In `configs/dfine/include/dataloader.yml`:
```yaml
num_workers: 2   # was 4 — halves per-step worker RAM overhead for Mosaic
```

This slows throughput by ~15–20% (more DataLoader stall time) but keeps within the 7.56 GB cap.

**Fix 3 — Both (belt and suspenders):**

Apply Fix 1 (12 GB WSL2) and keep num_workers=4. If the machine ever runs other heavy processes
concurrently, add `pin_memory: False` to the dataloader config as an additional safety valve.

---

## DESIGN-001 — n_per_class reduced from 5 to 3

**Status:** ✅ INTENTIONAL CHANGE

**Reason:**
5 images × 10 classes = 50 inference passes per visualization step.
Reduced to 3 to cut visualization overhead. 30 images still gives good coverage
of each class (top-3 images by instance count, pre-selected at init time).

---

## BUG-019 — eval_spatial_size [720,1280] causes FPN upsample mismatch

**Status:** ✅ FIXED

**Symptom:**
Training crashed immediately after checkpoint load with:
```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 46 but got size 45 for tensor number 1 in the list.
```
Crash site: `hybrid_encoder.py` line 483, inside `torch.concat([upsample_feat, feat_low], dim=1)`.

**Root Cause:**
`eval_spatial_size: [720, 1280]` was used as the FLOPs profiler input shape.
At H=720: stride-16 features have height 45px (720/16=45), but stride-32 features have
height 22px (720/32=22.5 → floor=22). Upsampling stride-32 by 2× gives 44px, not 45.
The FPN concatenation of 44 and 45 fails with a size mismatch.

This would also crash actual val inference if those images reached the encoder at H=720.

**Rule:** All canvas dimensions (H and W) must be divisible by 32.
720/32 = 22.5 ✗. 704/32 = 22 ✓. 960/32 = 30 ✓. 1280/32 = 40 ✓.

**Fix:**
Changed `eval_spatial_size` and `val_dataloader LetterboxResize` from `[720, 1280]` to
`[704, 1280]` in `configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_mosaic.yml`.

`_canvas_dims(1280, "16:9")` already returns `(704, 1280)` — the training collate was using
704, but the config had 720. Now both match exactly.

**Verification:**
`_canvas_dims` always produces 32-aligned short sides by construction:
```python
short = max(32, round(s / 32) * 32)
```
All scale ladder values (768..1280 step 32) are also 32-aligned.
Val targets [704, 1280] and [960, 1280] are both fully 32-aligned.

---

## BUG-020 — train.py --use-amp default=False silently overrides YAML use_amp=True

**Status:** ✅ FIXED

**Symptom:**
Training runs in FP32 even though `use_amp: True` is set in the YAML config.
Config dump at startup shows `'use_amp': False` in both outer cfg and yaml_cfg.
Memory usage is ~2× higher than expected; OOM occurs at smaller batch sizes.

**Root Cause:**
`train.py` uses `action="store_true"` for `--use-amp` with no explicit `default`.
`argparse` defaults `store_true` actions to `False` when the flag is absent.
The startup code collects all argparse values with `v is not None` as the filter,
but `False is not None` so `use_amp=False` enters `update_dict` and overwrites the YAML.

**Fix:**
```python
# train.py
parser.add_argument("--use-amp", action="store_true", default=None, ...)
parser.add_argument("--test-only", action="store_true", default=None, ...)
```
With `default=None`, the unset flag stays `None` and is excluded by the `v is not None` filter.

---

## BUG-021 — LetterboxResize.__call__ crashes with 3-tuple sample from Compose.default_forward

**Status:** ✅ FIXED

**Symptom:**
Val dataloader crashes with `AttributeError: 'tuple' object has no attribute 'shape'`
at `_transforms.py:83` (`_, h, w = image.shape`) when resuming training.

**Root Cause:**
`CocoDetection.__getitem__` calls `self._transforms(img, target, self)` (3 args).
`Compose.default_forward` packs these as `sample = (img, target, dataset)` (3-tuple)
and calls `transform(sample)` for each transform. torchvision v2 transforms
(`ConvertPILImage`, `ConvertBoxes`) handle the tuple via their `_transformed_types`
dispatch. `LetterboxResize` is a plain class — it receives the 3-tuple as `image`,
then fails at `image.shape`.

**Fix:**
Added tuple-unpack at the top of `LetterboxResize.__call__`:
```python
_extra = ()
if isinstance(image, (tuple, list)):
    parts = image
    image = parts[0]
    target = parts[1] if len(parts) > 1 else None
    _extra = tuple(parts[2:])
```
Both return paths now repack `_extra` if present.

---

## BUG-022 — CUDA OOM during multi-scale mosaic training (batch_size_table wrong)

**Status:** ✅ FIXED (2026-04-25)

**Symptom:**
Training crashes mid-epoch with `torch.OutOfMemoryError` or `CUDA driver error: out of memory`
in loss_boxes/generalized_box_iou or decoder forward pass.

**Root Cause:**
The `batch_size_table.json` was probed in inference mode (no targets, no DN branch).
Training mode activates the Denoising (DN) branch (+100 extra queries per image), roughly
doubling decoder memory usage vs inference. 4:3 aspect ratio canvases (e.g. 960×1280) are
also ~2.5× more memory-intensive than 16:9 (704×1280) at the same `sz` due to AIFI
self-attention being O(tokens²) in memory (30×40=1200 tokens vs 22×40=880 tokens).
The original table values (up to 15 per batch at 768px) far exceeded safe limits.

**Additional factors:**
- `use_amp: False` (BUG-020) doubled memory, masking the real problem
- VRAM fragmentation across multi-scale batches exacerbates peak usage

**Fix:**
1. Re-probed batch sizes in training mode (with targets); reduced table substantially
   (e.g. 1280px: 4:3=2, 16:9=2-3; 768px: 4:3=3, 16:9=4)
2. Added `torch.cuda.synchronize() + torch.cuda.empty_cache()` after every backward call
   in `det_engine.py` to release fragmented VRAM before the next forward pass

---

## BUG-023 — Letterbox unwind used wrong width reference (B4)

**Status:** ✅ FIXED (2026-04-25)

**Symptom:**
GT box evaluation produced wrong pixel coordinates → near-zero AP for non-matching-AR images even
though training loss was finite.

**Root Cause:**
`det_engine.py` had a block:
```python
if "letterbox" in target:
    scale, pad_top, pad_left, canvas_h, canvas_w = ...
    b[:, [0, 2]] = b[:, [0, 2]] * canvas_w - pad_left   # BUG: canvas_w used
```
After letterbox in collate, boxes were normalized relative to `padded_w` (post-padding,
pre-resize width). For a portrait crop padded to 16:9: `padded_w = round(H × 1.78) ≈ 1422`.
The unwind used `canvas_w = 1280`. These differ → GT x-coords off by 11% → AP collapse.

**Fix:** Removed the entire letterbox unwind block. The correct path uses
`scale_boxes(target["boxes"], (orig_H, orig_W), (img_H, img_W))` via `orig_size`.

---

## BUG-024 — ARBucketSampler 3-tuple container guardrail too eager

**Status:** ✅ FIXED (2026-04-25)

**Symptom:**
```
ValueError: Transform 'RandomPhotometricDistort' received a 3-tuple input (expected 2).
```
Crash on first training batch after adding the container guardrail.

**Root Cause:**
`CocoDetection.__getitem__` legitimately calls `self._transforms(img, target, self)` — passing the
dataset as a third argument. The guardrail checked `len(sample) > 2` BEFORE calling each transform,
so it fired on the very first transform even though the 3-tuple was from the dataset, not from a
transform.

**Fix:** Changed guardrail to fire AFTER calling the transform, checking if the transform GREW the
tuple length (`cur_len > prev_len and cur_len > 2`). The initial 3-tuple from the dataset is
allowed through; only a transform that expands the count is flagged.

---

## BUG-025 — Dataset-level Resize([1024,1024]) destroyed AR before collate (B3)

**Status:** ✅ FIXED (2026-04-25) for AR experiment

**Symptom:**
ARBucketSampler bucket assignment was correct (from COCO metadata), but all images arrived at
collate as 1024×1024 squares. AR-aware collate was a no-op since AR information was already gone.

**Root Cause:**
`Resize([1024,1024])` was the last dataset-level spatial transform. It runs after all color
augmentation and converts every image to a square before collate sees it.

**Fix:** Removed `Resize` from dataset transforms in the new `dfine_hgnetv2_s_visdrone_ar.yml`.
All spatial shaping now happens entirely in `BatchImageCollateFunction`.

---

## BUG-026 — Mosaic run AP collapsed epoch 15→16 (cause unknown)

**Status:** ⚠️ UNEXPLAINED — no action needed (best checkpoint saved before collapse)

**Symptom:**
W&B run `l7dygiqx` (`dfine_s_visdrone_ms1280_mosaic`) reached AP=0.3027 at epoch 15, then
collapsed to AP≈0.000143 at epoch 16. Training continued degrading rather than recovering.

**What was verified:**
- `best_stg1.pth` saved epoch 15 weights before the collapse; checkpoint is clean.
- No NaN in loss at epoch 15 (W&B loss curve was smooth).
- `use_amp: False` — AMP cannot explain the collapse.
- Mosaic transform order was correct (Mosaic first, then RandomIoUCrop).
- The collapse happened inside the original run, not as a result of the AR experiment.

**Hypotheses (none confirmed):**
- LR spike — if `CosineAnnealingLR` hit the warmup/decay inflection near epoch 15 with
  an unfortunate LR, it could push the model into a bad basin. `T_max=80` so epoch 15 is
  not the end of cosine decay, but accumulated gradient scale might have spiked.
- Mosaic extreme crop — `RandomIoUCrop` at `p=0.8` may have coincidentally produced
  near-empty images for an entire epoch's batches, starving supervision.
- Single-epoch fluke — EMA decay restores the pre-collapse weights; the next epoch may
  have recovered, but training was not continued from this checkpoint.

**Mitigation:** Resume from `best_stg1.pth` (epoch 15), which pre-dates the collapse.
This is the checkpoint used for all subsequent training.

---

## BUG-027 — CUDA OOM from VRAM fragmentation during mosaic resume (2026-04-27)

**Status:** ✅ FIXED

**Symptom:**
Mosaic resume run crashed repeatedly after epoch 23 (best AP=0.3056) with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 434.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 614.00 MiB is free.
```
614 MB free but 434 MB contiguous unavailable — allocator fragmentation, not total exhaustion.
Crash site: AIFI self-attention softmax (O(tokens²) memory, worst case at large scales).

**Root cause — two compounding factors:**

1. **`use_amp: False` (fp32 training)** — The resume config was kept in fp32 to match the
   original mosaic run (which used fp32 due to BUG-020). fp32 activations use ~2× the VRAM
   of AMP (fp16). This left almost no headroom at large scales.

2. **VRAM fragmentation** — Multi-scale training allocates tensors of different sizes every
   step (768px → 1280px). Over many steps, the allocator accumulates fragmented free blocks
   that can't satisfy a large contiguous allocation even when total free memory is sufficient.

**WSL was NOT crashing** — the `.wslconfig` `memory=12GB` fix from BUG-018 was already active
(confirmed: `/proc/meminfo` shows MemTotal=12GB). The process dying from CUDA OOM made the
terminal appear frozen, but WSL itself was fine.

**Does nohup help?** No. `nohup` only prevents SIGHUP (terminal disconnect). It cannot stop
CUDA OOM (Python exception) or kernel SIGKILL.

**Fix 1 — Enable AMP (`use_amp: True`):**
Changed in `configs/dfine/dfine_hgnetv2_s_visdrone_mosaic_resume.yml`.
fp16 activations halve VRAM usage. Switching mid-training is safe: Adam optimizer moments
are fp32 regardless; GradScaler starts fresh at scale=65536 and auto-adjusts within a few steps.

**Fix 2 — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:**
Set in `train_watchdog.sh`. Allows the allocator to use non-contiguous memory segments
instead of requiring large contiguous blocks — directly addresses the fragmentation crash.
PyTorch explicitly recommends this in the OOM error message.

---

## TOOLING-001 — Watchdog auto-restart script (2026-04-27)

**File:** `D-FINE/train_watchdog.sh`

**Problem:** Multi-scale mosaic training crashes intermittently (CUDA OOM, WSL blip, etc.).
Manual restart required each time, losing overnight progress.

**Solution:** A bash loop that auto-resumes from `last.pth` on any non-zero exit:

```bash
while true; do
    python train.py -c $CONFIG --resume last.pth ...
    [ $? -eq 0 ] && break
    sleep 15   # brief pause before restart
done
```

Also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on every launch.

**Usage (run in tmux so terminal disconnect doesn't kill it):**
```bash
tmux new -s train
bash train_watchdog.sh
# Ctrl+B then D to detach; tmux attach -t train to check back in
```

**Per-batch OOM guardrail in `src/solver/det_engine.py`:**
The training loop wraps each forward+backward in a `try/except torch.OutOfMemoryError`.
If a single batch OOMs (e.g. an unlucky large-scale draw with many objects), training skips
that batch, zeros gradients, calls `torch.cuda.empty_cache()`, and continues — instead of
crashing the whole run. Also, `torch.cuda.synchronize() + torch.cuda.empty_cache()` is called
after every backward pass to release freed graph memory back to the driver before the next
forward (different resolution → different tensor sizes → fragmentation risk).

**Why batch_size=2 (not 4) with accum_steps=4:**
Effective batch = batch_size × accum_steps = 2 × 4 = **8**, matching the original mosaic run
(epochs 0–15, AP=0.3027). Increasing to batch_size=4 doubles the effective batch to 16,
changing gradient dynamics and requiring a ~2× LR adjustment — not safe mid-training.
The `adaptive_batch` guardrail in `BatchImageCollateFunction` already handles high-res OOM
by subsampling the batch down to 1 image at 1280px regardless of nominal batch_size.

---

## BUG-028 — Disk full from W&B + TensorBoard logs crushing training sessions (2026-04-29)

**Status:** ✅ FIXED

**Symptom:**
Multiple training sessions crashed mid-run with no useful error message. Root cause identified
post-mortem: 400+ GB consumed by accumulated W&B local sync cache + TensorBoard event files
(which embed full images logged by `WandbVisualizer.log_epoch_tensorboard` every 500 steps).

**Root cause breakdown:**
1. `src/solver/det_engine.py` called `save_samples()` at the start of training and eval,
   writing `.webp` images to `output/{run}/train_samples/` and `output/{run}/val_samples/`.
2. `WandbVisualizer.log_epoch_tensorboard()` was called every 500 steps, embedding 30 full
   1024px images into TensorBoard event files. Over 80+ epochs × 1600 steps/epoch, this
   accumulated gigabytes of image data in the `.tfevents` files.
3. W&B local sync directory also caches run data locally before uploading.

**Fix applied (2026-04-29):**
- Removed both `save_samples()` call sites from `det_engine.py` (train + eval loops).
- Disabled `wandb_viz.log_epoch_tensorboard()` call in `det_engine.py` — TensorBoard now
  receives only scalar metrics, not images. W&B cloud image logging (`wandb_viz.log_epoch`)
  is kept since it goes to W&B servers, not local disk.
- Expanded `try/except` in train loop: non-OOM exceptions are now caught, printed, and appended
  to `output/{run}/error_log.txt`, then training continues (instead of crashing silently or
  requiring a full watchdog restart for trivial errors).

**Prevention:**
- Periodically `du -sh ~/wandb/ output/*/summary/` to monitor log size.
- W&B local cache can be purged with `wandb sync --clean` after runs complete.

---

## TOOLING-002 — Expanded error handling in training loop (2026-04-29)

**File:** `src/solver/det_engine.py`

**Problem:** The per-batch try/except only caught `torch.OutOfMemoryError`. Any other exception
(e.g. a NaN assertion, a shape mismatch on an unusual batch, a DataLoader race) would either
propagate to `sys.exit(1)` (the loss-is-nan check) or crash with a traceback that scrolled
off the terminal and was lost.

**Fix:**
Added `except Exception as e` after the OOM handler. Non-OOM errors:
1. Print full traceback to stdout immediately.
2. Append to `output/{run}/error_log.txt` with epoch and batch index.
3. Zero grad, update scaler, empty CUDA cache, then `continue` — same recovery as OOM.

This means a single bad batch (e.g. corrupted sample from augmentation) can no longer crash
a multi-day training run.

---

## BUG-029 — CUDA driver OOM bypasses Python OOM handler, crashes process (2026-04-30)

**Status:** ✅ FIXED

**Symptom:**
Training crashed every epoch overnight. `error_log.txt` showed:
```
RuntimeError: CUDA driver error: out of memory
```
Watchdog restarted from `last.pth` each time, so progress was preserved but each epoch
required 1–2 restarts, fragmenting W&B into ~10 separate runs.

**Root cause:**
`torch.OutOfMemoryError` is raised by PyTorch's Python-level allocator. But when VRAM
fragmentation causes the CUDA *driver* itself to fail (not PyTorch's allocator), it raises
`RuntimeError` with "out of memory" in the message instead. Our `except torch.OutOfMemoryError`
handler didn't catch this — the exception propagated and killed the process.

**Fix (`src/solver/det_engine.py`):**
Changed handler to catch both:
```python
except (torch.OutOfMemoryError, RuntimeError) as e:
    if not (isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()):
        raise  # re-raise non-OOM RuntimeErrors
```
Non-OOM RuntimeErrors are re-raised so they still surface in the general `except Exception`
handler and get logged to `error_log.txt`.

---

## BUG-030 — W&B creates new run on every watchdog restart (2026-04-30)

**Status:** ✅ FIXED

**Symptom:**
Each crash → watchdog restart → new `wandb.init()` → new run ID. Overnight training
scattered metrics across ~10 separate W&B runs. No continuous loss/AP curve visible.

**Root cause:**
`det_solver.py` called `wandb.init()` without `id=` or `resume=`, so W&B always created
a fresh run. After a crash, all context of the previous run was lost.

**Fix (`src/solver/det_solver.py`):**
On first start, save the W&B run ID to `output/{run}/wandb_run_id.txt`. On every subsequent
restart, read that ID and pass `id=run_id, resume="allow"` to `wandb.init()`. W&B resumes
the same run seamlessly — metrics append to the existing curves instead of starting fresh.

---

## BUG-031 — CUDA CachingAllocator internal assertion kills process (2026-05-01)

**Status:** ✅ MITIGATED (unrecoverable — process-level abort, not catchable in Python)

**Symptom:**
Training died silently at epoch 79 with no watchdog restart. `error_log.txt` showed:
```
RuntimeError: !handles_.at(i) INTERNAL ASSERT FAILED at
"../c10/cuda/CUDACachingAllocator.cpp":393, please report a bug to PyTorch.
```

**Root cause:**
This is a C++ `assert()` inside PyTorch's CUDA memory allocator. It calls `abort()` at the
C level — the process dies before Python's `except` block is ever reached. No amount of
Python-level exception handling can catch it. Triggered by rapid allocation/deallocation of
tensors at different resolutions (mosaic variable-size batches), stressing the allocator.
Known PyTorch 2.5 bug; patched in later releases.

**Why watchdog didn't restart:**
Watchdog wasn't running — it had been stopped at some earlier point and training was running
as a bare `python train.py` process with no auto-resume loop.

**Mitigation:**
- Always run training via `train_watchdog.sh` in a tmux session (see TOOLING-001).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the watchdog — this changes
  the allocator strategy and significantly reduces fragmentation-induced assertion failures.
- Removed `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` from the per-batch hot
  path (BUG-032). Those calls were originally meant to flush memory between variable-size
  batches but were causing ~50% GPU utilization instead. `empty_cache()` is retained in the
  `except` block for OOM recovery only.

---

## BUG-032 — Per-batch synchronize+empty_cache halves GPU utilization (2026-05-01)

**Status:** ✅ FIXED

**Symptom:**
GPU utilization hovered at ~50% per W&B metrics despite data loading being only 3% of
batch time (data=0.020s vs time=0.70s per batch).

**Root cause:**
`det_engine.py` called `torch.cuda.synchronize()` then `torch.cuda.empty_cache()` after
every single `backward()` pass. `synchronize()` forces the CPU to block until all GPU
kernels finish — this serializes the CPU-GPU pipeline and stalls the GPU between batches.
`empty_cache()` adds a driver round-trip on top. Together they were eating ~15-25% of
each batch's time in synchronization overhead.

**Fix (`src/solver/det_engine.py`):**
Removed both calls from the hot path in both the AMP and non-AMP branches. Retained
`empty_cache()` only in the `except` block where it belongs (OOM recovery). The GPU now
runs continuously without forced stalls between batches.

**Note:**
The original calls were added to prevent fragmentation OOM (BUG-031 type crashes) from
variable-resolution batches. With them removed, such crashes are marginally more likely —
but `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the watchdog mitigates this, and
the watchdog auto-resumes from `last.pth` if a crash does occur.

---

## BUG-033 — Allocator crash leaves zombie process holding GPU, blocks watchdog restart (2026-05-02)

**Status:** ✅ FIXED

**Symptom:**
After a `!handles_.at(i) INTERNAL ASSERT FAILED` crash (BUG-031), the training process
didn't fully release its CUDA context. It appeared as a running process in `ps` and still
held ~7349 MiB of VRAM in `nvidia-smi`. The watchdog restarted a new python process, but
it couldn't acquire the GPU — training was silently stuck with no W&B updates for hours.

**Root cause:**
`abort()` from a C++ assertion terminates the Python process but may leave the CUDA driver
context in a partially-released state under WSL2. The GPU memory lingers until the OS
reclaims it, which can take arbitrarily long.

**Fix (`train_watchdog.sh`):**
Added `fuser -k /dev/nvidia* 2>/dev/null || true` before each training restart. This sends
SIGKILL to any process still holding a file descriptor on the nvidia device nodes, forcing
the CUDA context to be released before the new training process starts. A 3-second sleep
gives the driver time to clean up before python launches.

---

## TOOLING-003 — PyTorch upgrade 2.5.1 → 2.6.0+cu124 (2026-05-02)

**Motivation:**
`!handles_.at(i) INTERNAL ASSERT FAILED` crashes (BUG-031) were happening every 1-2 epochs,
repeatedly stalling training. PyTorch 2.5.1 has known CUDA allocator bugs in high-churn
variable-size allocation patterns (mosaic training). Upgrading was the right fix.

**CUDA compatibility constraint:**
- NVIDIA driver: 555.97 → supports CUDA up to 12.5
- cu126 wheels require driver 560+, so we are limited to cu124 wheels
- Latest cu124 wheel: `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`
- CUDA runtime and cuDNN are **bundled inside the wheel** — system CUDA driver is untouched

**Code changes required:**
- `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler` (deprecated, removed in 2.6)
  - `src/optim/amp.py`: changed import and alias
  - `src/core/_config.py`: updated import + instantiation to `GradScaler('cuda')`
  - `src/solver/det_engine.py`: updated import

**Install command:**
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

**Smoke test:** 5-step resume passed cleanly. AMP enabled, no crashes.

**Next upgrade path:** To get 2.7+ (cu126+), need to update Windows GPU driver to 560+ first.
Driver updated 2026-05-02 (596.36, CUDA 13.2) — upgraded further to torch==2.11.0+cu128, torchvision==0.26.0+cu128.

---

## TOOLING-004 — NWD + size-adaptive loss + maxDets eval fix (2026-05-02)

**Changes:**

**1. NWD in Hungarian cost matrix (`src/zoo/dfine/matcher.py`):**
GIoU cost is non-zero for non-overlapping boxes (it does give gradient), but it is not
size-relative — a 4px displacement on a 4px object costs the same as on a 40px object.
NWD (Normalized Wasserstein Distance) measures distance in (cx, cy, w/2, h/2) space,
making it proportional to box size. Better assignment for VisDrone tiny objects.
`cost_nwd: 2, nwd_constant: 0.5` in new config. Backward compatible (defaults to 0).

**2. Size-adaptive loss weighting (`src/zoo/dfine/dfine_criterion.py`):**
After Hungarian matching, each matched pair's L1, GIoU, and FGL losses are multiplied
by `1/(w*h + ε)`, normalized to mean=1. 77% of VisDrone boxes are <16px — without this,
large objects dominate gradients and tiny objects are under-optimized.
Enabled via `size_adaptive: True` in config. Default False.

**3. maxDets=500 in COCO eval (`src/solver/det_engine.py`):**
Default COCO maxDets=100 limits evaluation to top-100 predictions per image.
VisDrone images can have 100-800 objects — this silently caps what can be scored.
Changed to [1, 10, 500]. AP impact: negligible for D-FINE (DETR has no NMS so the
101st-300th predictions are low-confidence noise; YOLO benefits more from this).
Semantically correct regardless.

**4. num_top_queries=500 in new config:**
More prediction slots available for COCO eval matching.

**Eval results on best_stg2.pth (epoch 92):**
- maxDets=100: AP=0.316, AP50=0.506, AP-small=0.225
- maxDets=500: AP=0.316, AP50=0.507, AP-small=0.226  ← essentially identical
- Note: both are higher than the 0.3142 seen during training; training log uses running
  model state while --test-only uses the saved EMA weights, which typically score slightly higher.

**New training config:** `configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml`
**New watchdog:** `train_watchdog_nwd.sh`
Starts from `best_stg2.pth` via `--tuning` (resets optimizer for new loss landscape).

---

## BUG-034 — W&B image logging used global_step instead of epoch (2026-05-02)

**Status:** ✅ FIXED

**Symptom:**
W&B showed image panels on a step-based x-axis (0, 500, 1000...) while all scalar
metrics used epoch. Caused "step N < current step M" warnings on watchdog restarts
because the resumed run's global_step reset to 0 while W&B already had data at step 222000+.

**Root cause:**
`det_engine.py` called `wandb_viz.log_epoch(global_step)` — passing the global step
counter instead of the epoch number. `wandb_viz.log_epoch` calls `wandb.log(..., step=epoch)`,
so passing global_step made images land at step 500, 1000, etc. rather than epoch 1, 2, etc.
The trigger also fired every 500 steps mid-epoch rather than once per epoch.

**Fix (`src/solver/det_engine.py`):**
- Changed trigger from `global_step % viz_step_interval == 0` → `is_last_batch`
- Changed call from `wandb_viz.log_epoch(global_step)` → `wandb_viz.log_epoch(epoch)`
All W&B metrics now consistently use epoch as x-axis.


---

## BUG-035 — 1/area size-adaptive weighting caused AP regression (2026-05-02)

**Status:** ✅ FIXED (switched to sqrt(1/area))

**Symptom:**
NWD run with `size_w = 1 / (area + ε)` regressed from AP=0.316 → 0.308 over 20 epochs.
AP-small specifically dropped from 0.225 → 0.217 — the opposite of the intended effect.

**Root cause:**
`1/area` weighting amplifies loss by up to 625× for a 4×4px box vs a 40×40px box.
VisDrone labels for sub-10px objects are inherently noisy (many are annotated at ~1-2px
precision). Over-amplifying those labels gives the model a noisy training signal for the
long tail of tiny objects, drowning out the cleaner medium/large-object gradients.

**Fix (`src/zoo/dfine/dfine_criterion.py`):**
Changed both `loss_boxes` and `loss_local` from:
```python
size_w = 1.0 / (areas + 1e-6)
```
to:
```python
size_w = 1.0 / (areas + 1e-6).sqrt()
```
The sqrt reduces the amplification ratio from 625× to 25× — still emphasizes small objects
but doesn't blow up noise.

**Locations changed:**
- `loss_boxes`: lines 140–145
- `loss_local` (FGL): lines 192–197

---

## INSIGHT-001 — New loss landscape requires lower LR at start (2026-05-03)

**Observation:**
When fine-tuning from best_stg2.pth with NWD matching + sqrt size-adaptive loss (a new loss
landscape), the model stalled for ~80 epochs before starting to improve:

| Epoch range | AP range | LR |
|-------------|----------|----|
| 0–50 | 0.308–0.315 | 0.000025–0.000020 |
| 50–85 | 0.314–0.316 | 0.000020–0.000013 |
| 85–106 | 0.316→0.320 | 0.000013–0.000010 |

The model only started climbing once cosine annealing brought the LR below ~0.000015.

**Root cause:**
The new loss terms (NWD cost reshapes the matching assignments; size-adaptive weighting
amplifies gradients for tiny objects) introduce a different gradient landscape from what
the optimizer momentum was tuned for. At high LR, the gradient signal from the new losses
is too noisy to make consistent progress — the optimizer overshoots. As LR decays, the
step sizes shrink to a level where the new gradient signal is useful.

**Lesson for future runs:**
When introducing a new loss function on top of a trained checkpoint, use a **lower max LR**
from the start — roughly 0.3–0.5× the original fine-tuning LR. This avoids wasting
~50–80 epochs waiting for cosine annealing to do the work.

For this project: instead of lr=0.00005 (the default), start at lr=0.000015–0.00002 when
the loss landscape changes significantly. The warmup can still run from 1e-6.

---

## BUG-036 — D-FINE EMA rollback on non-improvement loaded wrong checkpoint (2026-05-04)

**Status:** ✅ FIXED (`src/solver/det_solver.py`)

**Symptom:**
Three visible V-shaped drops in W&B AP curve during the NWD training run:
- Deep 1 (~ep52-58): AP 0.315 → 0.314 (minor, from CUDA crashes causing training instability)
- Deep 2 (~ep80-88): AP 0.316 → 0.314 (intentional stop_epoch reset + repeated crash-restarts)
- Deep 3 (~ep110-115): AP 0.3209 → 0.315 (non-improvement rollback loaded wrong checkpoint)

**Root cause (deep 3 and part of deep 2):**
D-FINE's `det_solver.py` has a "non-improvement rollback" mechanism: whenever stage-2 eval
doesn't beat the current best, it reloads `best_stg1.pth` and decrements EMA decay:
```python
self.ema.decay -= 0.0001
self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
```
The original D-FINE design assumes short 2-stage training (~80 ep each). `best_stg1.pth`
is the best pre-stop_epoch checkpoint. For our 160+ ep stage-2 run, this means any
non-improvement epoch rolls the model ALL THE WAY BACK to ep<80 weights (AP ~0.314),
destroying 30+ epochs of stage-2 progress.

Additionally, `load_resume_state` loads the **live model weights** from the checkpoint, not
the EMA weights. The EMA is always the better (smoother) version; the live model at save
time is noisier. This makes the regression worse.

**Fix applied:**
1. Rollback target changed from `best_stg1.pth` → `best_stg2.pth` (falls back to stg1 only
   if stg2 doesn't exist yet).
2. Rollback now loads **EMA weights** into both the live model and the EMA module, instead
   of live model weights. Optimizer and LR scheduler are NOT reset — training continues
   from current LR without disruption.

```python
stg2_path = self.output_dir / "best_stg2.pth"
stg1_path = self.output_dir / "best_stg1.pth"
rollback_path = stg2_path if stg2_path.exists() else stg1_path
state = torch.load(str(rollback_path), map_location="cpu")
ema_weights = state["ema"]["module"] if "ema" in state else state["model"]
live_module = dist_utils.de_parallel(self.model)
stat, _ = self._matched_state(live_module.state_dict(), ema_weights)
live_module.load_state_dict(stat, strict=False)
ema_module = dist_utils.de_parallel(self.ema).module
stat, _ = self._matched_state(ema_module.state_dict(), ema_weights)
ema_module.load_state_dict(stat, strict=False)
```

---

## BUG-037 — last.pth only saved for epoch < stop_epoch, causing full stage-2 redo on crash (2026-05-04)

**Status:** ✅ FIXED (`src/solver/det_solver.py`)

**Symptom:**
After any crash past ep80, the watchdog restarted from `last.pth` (ep79 state) and redid
ALL of ep80+. This caused:
- The `stop_epoch=80` EMA reset to fire 4+ separate times
- 30+ epochs of stage-2 training redone from scratch after every crash
- LR scheduler re-running the same cosine range multiple times

Evidence: "Refresh EMA at epoch 80 with decay 0.9999" appears 4 times in the log,
followed by refreshes at ep82, ep88 (successive crash-restart re-entries into stage 2).

**Root cause:**
Original D-FINE code only saves `last.pth` for `epoch < stop_epoch`:
```python
if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
    checkpoint_paths = [self.output_dir / "last.pth"]
```
After ep80, only `best_stg2.pth` (updated on new-best events) is saved. The D-FINE authors
designed 2 short stages — stage 2 crashes were presumably rare/acceptable. For our 320-epoch
training this is catastrophic.

**Fix applied:**
Removed the `epoch < stop_epoch` guard — `last.pth` now saves after every epoch:
```python
if self.output_dir:
    checkpoint_paths = [self.output_dir / "last.pth"]
```

---

## INSIGHT-002 — LR warm restart after loss adaptation: theory and decision (2026-05-03)

**Question:**
After the model adapts to a new loss landscape for 100+ epochs, can a higher LR in phase 2
find a better minimum than continuing at the current low LR?

**Theory (Edge of Stability + SGDR):**

*Why high LR stalls with a new loss (Q1):*
Introducing a new loss term reshapes the local curvature (Hessian) around the current weights.
At high LR, the model enters "Progressive Sharpening" — the Hessian spectral norm grows and
oscillates, preventing descent. The optimizer can only make progress once LR decays to where
step size ≤ 2 / (Hessian sharpness). This is exactly the 80-epoch stall in the NWD run
(LR ~2.5e-5 → stall; LR < 1.5e-5 → improvement). See INSIGHT-001.

*Why a warm restart CAN work after adaptation (Q2):*
After 100+ epochs of adaptation, the landscape around the new loss is flatter and
well-characterized. Literature supports:
- **SGDR** (Loshchilov & Hutter 2016): periodic LR resets escape sharp minima, find broader
  basins. Works 2–4× faster than monotonic schedules.
- **Cyclic LR** (Smith): LR bumps jump out of sharp minima toward better-generalizing regions.
- **Edge of Stability** (Cohen et al. 2022): training at the edge often improves generalization
  despite non-monotonic loss curves.

**Decision for this run:**
Phase 2 uses **safe continuation** — cosine from current LR (~1e-6) down to 1e-7 over 160 ep.
Rationale: model is already at AP=0.320 and climbing; low-risk continuation likely gives
0.322–0.324. A warm restart might reach 0.325–0.330 but risks a regression at ep161–180.

**Candidate follow-up experiment (if phase 2 plateaus early):**
Warm restart run: LR bump to 3e-5, cosine decay to 1e-7 over 160 ep, starting from the
best phase-2 checkpoint. Based on SGDR evidence, the adapted landscape should absorb the
LR spike without the 80-epoch stall we saw in phase 1.
