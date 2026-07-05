# Ablation Study Plan — RunPod Campaign (2026-07-02)

**Goal:** improve on AP=0.322 (NWD+sqrt best); stretch goal: reach/pass SOTA
(DroneScan-YOLO 0.356, 10M params, 1280×1280).
**Budget:** $17 on RunPod RTX A5000 ($0.27/hr ≈ 63 GPU-hours). (Revised down
from $30 on 2026-07-02; user added +$50 on 2026-07-04, ~$56 total spent to date.)
**Current best (2026-07-05):** `output/runpod_results/polish2_last.pth` —
**AP=0.3226, AP-small=0.2344** (msfd_1024_polish2; see `00_progress.md`
Steps 24-25). Supersedes the NWD+sqrt checkpoint referenced below, which was
the campaign's starting point.
**Status:** msfd/P2 architecture line shelved by user decision (Step 26).
R1+R2+R3 tested on the plain (non-P2) architecture — neither beat polish2
(Step 27). R4 (per-class calibration) tested — see `00_progress.md` Step 28.
Next: E6 (1280 resolution + AIFI PE interpolation).

---

## 1. System stress test — weaknesses found (2026-07-02 code + log audit)

### W1 — AP-small is the whole gap
Best model: AP=0.322, AP-small=0.226, AP-medium=0.430, AP-large=0.625.
77% of VisDrone boxes are <16px. Every AP point to SOTA lives in AP-small.
Per-class: car 0.619 vs bicycle ~0.11, tricycle/awning-tricycle weak — rare classes
with tiny boxes compound.

### W2 — P2ConvHead (msfd) has 3 live bugs; run is dead on arrival
`output/msfd_640`: **AP=0.0 at epochs 1–2** (tuning from a 0.32 checkpoint should
give ~0.31 at epoch 0). Repeated OOM (needs ~14 GB; local card has 8) plus a CUDA
allocator crash. Root causes found in code review:

1. **Double loss weighting** — FIXED 2026-07-02 as **BUG-038** (see
   06_bugs_and_fixes.md). Was: reg effectively **25×**, iou **4×** — matches the
   observed `loss_p2_reg: 43.7`.
2. **Anchor decode image-relative, not cell-relative** — FIXED 2026-07-02 as
   **BUG-039** (+ w/h bias init at logit(0.02)). Was: offsets spanned ±0.5 of
   the whole image instead of ±0.5 cells.
3. **AP=0 during warmup is still unexplained after FIX-5** — zero-init cls conv
   makes all 25,600 P2 scores exactly sigmoid(−4.6)=0.01; the postprocessor
   top-500 fills with tied junk boxes. AP should be low-but-nonzero, yet is 0.0 —
   suspect the eval path (needs a 10-minute debug: eval the tuned checkpoint with
   the P2 merge disabled in the postprocessor; if AP≈0.31 the merge is the culprit).
4. **Design question** — P2ConvHead consumes **raw backbone stage-0 (64ch)**
   with no neck fusion. DroneScan-YOLO's MSFD fuses P2 with upsampled P3 (semantic
   features). Raw stride-4 features are edge/texture-level; classification from
   them is weak. Fix candidate: add a single fusion conv (upsample neck-P3 →
   concat → 1×1) before the head. ~0.1M params.

### W3 — Resolution ceiling is a transformer-specific fragility
4/4 attempts to raise resolution failed (see 00_progress step 7b). The encoder
proposal mechanism (`enc_score_head`) and decoder `sampling_offsets` specialize
to the 640-token grid. Meanwhile SOTA trains at 1280×1280 natively. Mitigation
never tried: **interpolate AIFI positional embeddings + re-init `enc_output`
scoring at the new grid before training** (idea #3 in the 7b post-mortem).

### W4 — Dense-scene recall cap
VisDrone val images have up to ~800 objects; decoder has 300 queries
(num_top_queries=500 at eval). One-to-one matching means recall is structurally
capped in crowded scenes. AR-500 = 0.507 vs AR-100 = 0.367 in the ar_aware_p2
eval — the model finds more when allowed; queries are a bottleneck candidate.

### W-AR — ar_aware post-mortem (added 2026-07-02, from run logs)
ar_aware never beat the checkpoint it started from. Trajectory (local log):
tuned from AP=0.321 → **ep0 collapsed to 0.2625** (canvas change 1024²-multiscale
→ 736×1280 cost ~6 AP instantly) → slow asymptotic recovery: 0.307 @ep10,
0.3158 @ep61, final 0.318 @ep110 (RunPod; ar_aware_p2 variant 0.3181).
110 epochs spent to land **0.3 AP below baseline**. AP-small ticked up
(+0.003) but not enough to matter.

Two lessons, both feeding E6:
1. Dropping Mosaic (AR-aware training can't use it) likely cost more than the
   rectangular canvas gained — Mosaic was worth ~+2 AP historically. Any future
   geometry experiment must keep Mosaic or replace its diversity.
2. The ep0 collapse-and-recover pattern is W3 in miniature — BUT unlike the
   catastrophic 640→1280 attempts (flatlined at 0.125), starting from a
   **multi-scale-trained** checkpoint recovered to within 0.5 AP in ~40 epochs.
   Geometry transfer is feasible from multi-scale weights; E6's PE interpolation
   aims to remove the collapse itself.

### W5 — Infra fragility burns paid GPU-hours
- msfd_1024 crashed with `ZeroDivisionError` in `src/misc/logger.py:62`
  (`global_avg` on empty meter — fires when an epoch aborts before any batch logs).
  FIXED 2026-07-02 as **BUG-040**.
- 14 wandb restarts of the same run in one day (rm375kt9) — watchdog restart loop
  on a fatal (non-transient) error. Watchdog must distinguish OOM/assert (stop)
  from transient CUDA hiccups (restart).
- BUG-036/037 class of resume bugs already fixed — keep `last.pth` verification
  in the launch checklist.

---

## 2. The two P2 head versions

| | v1: `p2_640` (transformer P2) | v2: `msfd_640` / `msfd_1024` (conv P2) |
|---|---|---|
| Approach | P2 as a 4th level inside MSDeformableAttention | Transformer stays 3-level; YOLOv8-style DWConv head on P2 |
| Code | Pure config (`return_idx: [0,1,2,3]`, `num_levels: 4`, `num_points: [2,3,6,3]`) | `src/zoo/dfine/p2_conv_head.py` + hooks in dfine.py, criterion, postprocessor |
| P2 source | HybridEncoder (fused, 4-level neck) | **Raw backbone stage-0, no fusion (W2.4)** |
| Tokens @640 | 25,600 P2 tokens through attention | 25,600 positions through convs only |
| Status | **Never launched** (no output dir) | Launched, **broken** (W2) |
| Risk | Attention cost; may dilute queries | 3 bugs must be fixed first |
| Precedent | Drone-DETR-style | DroneScan-YOLO MSFD-style (their +AP came from this) |

**Verdict:** v1 is launch-ready today. v2 is the SOTA-precedent design but needs
the W2 fixes (~1-2 hours of work) plus a smoke test before it touches paid GPUs.

---

## 3. Is the transformer hurting us?

**Short answer: the decoder is earning its keep; the encoder-proposal coupling is
the liability.**

Evidence FOR the transformer:
- 0.322 with 10M params beats RT-DETR-R50 (0.284, 42M) and VRF-DETR (0.322) —
  the D-FINE decoder + FDR is parameter-efficient on this task.
- NMS-free one-to-one matching gives clean dense-scene predictions (no NMS tuning).
- GO-LSD distillation makes early decoder layers strong (cheap eval_idx ablation below).

Evidence AGAINST:
- Resolution transfer fails structurally (W3) — a pure conv detector retrains at
  1280 trivially; our proposal scoring + sampling offsets can't leave 640/1024.
- Query cap in dense scenes (W4).
- AIFI self-attention runs only on stride-32 (20×20) — it cannot help small
  objects (they live on P3/P2); it costs params/latency for large-object context.
  Whether it helps at all on aerial imagery (few large objects) is untested.
- Drone-DETR gets 0.339 at 640 — so DETR-family isn't the ceiling; but they use
  a small-object-specific encoder. Our gap is encoder/features, not the decoder.

Ablations E0/E4/E5 below answer this empirically for ~$3.

---

## 4. Experiment plan (prioritized)

Cost model: A5000, 1024px multi-scale ≈ 7 min/epoch, 640px ≈ 3.5 min/epoch
(from ar_aware + nwd run logs; ±30%).

**$17 structure: tournament, not parallel.** Both P2 versions run only to the
ep-30 gate; the loser is killed and only the winner gets a full schedule. E4
(queries) is funded only if E0's recall analysis supports it. E7 (combo) is
funded from whatever remains after E6.

| # | Experiment | What it answers | Epochs | Est. cost | Kill criterion |
|---|-----------|-----------------|--------|-----------|----------------|
| E0 | **Local eval suite** (free, before any pod): eval_idx sweep (layers 0..-1) on best ckpt; P2-merge-disabled eval of msfd ckpt (W2.3); per-class × per-size AP matrix; error taxonomy on 50 worst val images | Is the last decoder layer best? What broke msfd eval? Where exactly do we lose AP? | 0 | $0 | — |
| E1 | **Fix W2 bugs** + 3-epoch smoke test of msfd_640 on pod (batch 8) | P2 conv head viable? | 3 | ~$0.5 | ~~epoch-0 AP < 0.30~~ **corrected: < 0.22** — the 0.322 ckpt scores ~0.23 at 640 eval (it earned 0.322 at 1024). PASSED 2026-07-02: ep0 AP=0.2286 ≈ 640-baseline 0.231 ✓. Vectorized loss (BUG-041): 7 min/epoch incl eval on 3090 → full 110-ep 640px run ≈ $2.9. **Framing fix: 640px P2 runs compete vs 0.231 (640 baseline); the winner's mechanism then goes into msfd_1024 to challenge 0.322.** |
| E2a/E3a | **P2 tournament** — p2_640 AND msfd_640 (fixed), each to ep 30 only | Which P2 approach wins? | 2×30 | ~$2.0 | either: AP < 0.29 at ep 30 |
| E2b | **P2 winner, full run** — continue winner from ep-30 checkpoint to 110 | Does P2 lift AP-small at convergence? | +80 | ~$2.6 | AP < baseline at ep 60 |
| E5 | **AIFI off** (`use_encoder_idx: []`) — fine-tune from best, 40 ep | Is encoder attention pulling its weight? | 40 | ~$1.3 | AP drop > 0.5 pt at ep 20 → AIFI stays |
| E6 | **1280×736 with PE interpolation** — implement AIFI pos-emb interpolation + fresh `enc_score_head` at new grid, tune from best | Can we unlock SOTA's resolution advantage? (W3) | 50 | ~$3.5 | AP < 0.25 at ep 15 (vs 0.125 in failed attempt 1 — must clearly beat it) |
| E4 | *(conditional — only if E0 shows recall gap)* **Queries 300→600**, 40 ep | Is dense-scene recall query-capped? (W4) | 40 | ~$1.3 | AP flat vs baseline at ep 20 |
| E7 | *(conditional — only if ≥$3 left and E2b or E6 won)* **Combo run**, full schedule | The SOTA attempt | 110 | ~$3.5 | — |
| — | Reserve for crashes/reruns | | | ~$2.5 | |
| | **Committed core (E0–E6)** | | | **≈ $9.9** | |
| | **Cap incl. conditionals + reserve** | | | **≤ $17** | |

Order: E0 (today, local) → E1 → E2a+E3a tournament → E2b winner + E5 → E6 →
decide E4/E7 with remaining funds. One pod, sequential runs — parallel pods
double the idle-time risk on a small budget.

Note on the cut from $30: what got dropped is *breadth* (both P2 versions at
full length, unconditional E4, big reserve). The SOTA-critical path —
working P2 head + resolution unlock (E6) + one combo attempt — survives intact.

Success criteria:
- Minimum: one config > 0.322 (beat current best)
- Target: > 0.34 (pass Drone-DETR at 640, become 2nd-best known)
- Stretch: ≥ 0.356 (SOTA) — realistically requires E6 (resolution) + a working P2
  head to stack; both must win for this to be reachable.

Honest note: DroneScan-YOLO's edge is 1280 native training + fused P2 + a mature
SAL-NWD recipe. We already have their loss recipe (NWD+sqrt). Stacking a working
P2 (+1–1.5 pt precedent) and 1280 training (+2–3 pt precedent) is the only
credible path past 0.356 — and E6 is the highest-risk item. Expect 0.33–0.35;
treat 0.356+ as possible, not planned.

### Pre-launch checklist (every run)
1. `git pull` on pod; verify commit hash matches local push
2. Epoch-0 eval AP ≥ 0.30 when tuning from best ckpt (catches W2-class bugs)
3. `last.pth` written after epoch 0 (BUG-037 regression check)
4. Watchdog: max 3 restarts then stop + alert (W5)
5. W&B run name unique; delete stale `wandb_run_id.txt` (known RunPod issue)
6. ~~Fix `logger.py:62` ZeroDivisionError guard before first launch~~ done (BUG-040)

---

## 5. Round-2 experiment candidates (researched 2026-07-02, user-requested)

Ten directions beyond the original plan, ranked by (expected gain × precedent) / cost
for THIS project (10M-param edge target, ~$15 remaining). Sources: EFSI-DETR (2026,
AP 33.1/APs 24.8 VisDrone), Dome-DETR (2025), FMFN-YOLO (mAP50 44.5), UAV-DETR,
H-DETR/Co-DETR line, NWD-loss literature.

| # | Experiment | Idea | Cost | Expected | Result |
|---|-----------|------|------|----------|--------|
| R1 | **Crop-zoom training** | Train on 640² windows cropped from ≥960-scaled images (SAHI-style slicing at TRAIN time only — zero deploy cost, unlike SAHI inference which failed here). Objects get 1.5-2× more pixels during learning. | 1 aug class + run | +1–2 APs | ❌ **TESTED 2026-07-05, ~neutral/slightly negative** (-0.0005 AP, isolated via plain_r1r2r3 vs plain_r2r3_nozoom). Not validated. |
| R2 | **NWD/WIoU regression loss** | NWD is currently only in the matcher COST + SAL weighting; the regression loss itself is still L1+GIoU (scale-blind). Swap GIoU→NWD-loss (or Wise-IoU) for matched pairs. | ~20 LoC + run | +0.5–1.5 APs | ⚠️ **TESTED bundled with R3** (msfd_1024 and plain_r1r2r3/nozoom) — bundle underperformed standing best each time; R2's isolated effect not separately measured (no R2-only control run) |
| R3 | **Class-targeted CopyPaste 2.0** | Current CopyPasteSmall pastes any ≤32px object. Paste rare classes preferentially (bicycle AP≈0.11, tricycle, awning-tricycle) with 0.5–1.5× scale jitter. Attacks the per-class tail directly. | ~30 LoC + run | +1–2 AP on rare classes | ⚠️ **TESTED bundled with R2** — see above; isolated effect not separately measured |
| R4 | **Per-class score calibration** | Eval-only: rare classes are systematically under-confident; fit per-class temperature/offset on train-split predictions. Free AP50, no retraining. | eval only, $0 | +0.3–0.8 AP50 | ❌ **TESTED 2026-07-05, net negative** (AP50 0.5238→0.5196). Hypothesis partially confirmed — every predicted-under-confident class gained AP50, but car/van/truck/bus lost more. Not adopted. Full detail: `00_progress.md` Step 28. |
| R5 | **Distill from YOLOv8-X** | The 68M-param teacher (AP50 0.47) already sits in dfine_app_server/models/best.pt. Soft-label + feature distillation into D-FINE-S. Strong precedent. | ~1 day code + run | +1–3 AP | Not yet tried |
| R6 | **One-to-many auxiliary matching (H-DETR-lite)** | During training add a duplicated-GT group so multiple queries get positive gradient per GT; one-to-one at eval. Addresses gradient starvation on dense scenes. | moderate code | +1–2 AP | Not yet tried |
| R7 | **Density-adaptive queries (Dome-DETR-lite)** | Use encoder score mass to pick top-k per image (300→900 on crowded scenes). Answers W4 without paying 900 queries everywhere. | moderate code | +0.5–1 AP dense scenes | Not yet tried |
| R8 | **Frequency-domain P2/P3 enhancement (EFSI-lite)** | High-pass (wavelet/FFT) branch fused into fine levels — the 2026 VisDrone DETR SOTA's core trick (EFSI: AP 33.1, APs 24.8). | bigger code | +1–2 APs | Not yet tried |
| R9 | **Pseudo-label semi-supervised round** | Label VisDrone test-challenge (~1.6K imgs) + unlabeled footage with current best, retrain on union (score>0.6 pseudo-GT). More data is the root fix for the resolution-grid problem too. | cheap code, +1 run | +1–2 AP | Not yet tried |
| R10 | **RepGFPN/BiFPN-P2 neck swap** | Replace CCFF with a modern lightweight fused-P2 neck (DAMO-YOLO RepGFPN). The "cheap FPN" discussion, maximal version. | large change | +1–2 APs, deploy-friendly | Not yet tried |

Original recommended funding order with ~$15: R4 (free) → R2+R3 combined in
one run (~$3) → R1 (~$3) → R5 distillation if budget remains (~$4). R4 done
2026-07-05; R2+R3 bundle (tested twice, with and without R1) underperformed
the standing best both times. R6–R10 remain post-campaign candidates. Current
priority has shifted to E6 (1280 resolution unlock, §4) given the consistent
finding across this project's history that resolution — not slicing,
zoom-crop, or calibration tricks — is the lever that actually moves AP-small.

---

## 6. What's needed to launch (from Daniel)

1. **RunPod access** — either an API key (`RUNPOD_API_KEY`, scoped to pods) so the
   campaign can be managed programmatically, or start the pod manually and share
   the SSH connection string. Notes say A5000 + repo + dataset were working at
   `/workspace/D-fine-visdrone` — confirm the volume still exists (storage may
   have been released; dataset re-upload is ~2 GB).
2. **GitHub push access** — bug fixes (W2, W5) must be pushed to
   `github.com/Danzip/D-fine-visdrone` since the pod pulls from there.
3. **W&B** — key already on pod per notes; confirm project `dfine-visdrone`.
4. **Approve the W2 code fixes** before they're committed (they change
   p2_conv_head.py, dfine_criterion.py, and the msfd configs).
5. **Budget confirmation** — $30 cap; runs are killed per the criteria above, and
   the reserve is not spent without checking in.
