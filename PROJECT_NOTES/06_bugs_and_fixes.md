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

