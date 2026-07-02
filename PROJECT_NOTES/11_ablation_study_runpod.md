# Ablation Study Plan — RunPod Campaign (2026-07-02)

**Goal:** improve on AP=0.322 (NWD+sqrt best); stretch goal: reach/pass SOTA
(DroneScan-YOLO 0.356, 10M params, 1280×1280).
**Budget:** $17 on RunPod RTX A5000 ($0.27/hr ≈ 63 GPU-hours). (Revised down
from $30 on 2026-07-02.)
**Current best:** `output/dfine_hgnetv2_s_visdrone_nwd/best_stg1_dfine_s_visdrone_nwd_sqrt.pth`

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
| E1 | **Fix W2 bugs** + 3-epoch smoke test of msfd_640 on pod (batch 8) | P2 conv head viable? | 3 | ~$0.5 | epoch-0 AP < 0.30 (tuned ckpt must not regress) |
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

## 5. What's needed to launch (from Daniel)

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
