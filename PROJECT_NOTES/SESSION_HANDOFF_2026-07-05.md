# Session Handoff — 2026-07-05

> **Superseded (2026-07-06):** E6 finished (AP=0.344, best ep46/50) — see
> `SESSION_HANDOFF_2026-07-06.md` for current status and next steps, and
> `00_progress.md` Step 29 / `06_bugs_and_fixes.md` BUG-047 for the finished
> run and the idle-billing incident that happened right after it completed.
> This file is kept for the design rationale behind E6 (§1) and the
> claude-council install notes (§2), both still accurate.

Read this first when resuming. Full experiment history lives in
`00_progress.md`, `06_bugs_and_fixes.md`, `11_ablation_study_runpod.md` as
usual — this file is just "what's live right now and what to do next."

---

## 1. E6 (1280 resolution unlock) — RUNNING, check on it first

**Pod:** `41rlk08qzcicf4`, RTX 3090 community, `142.169.249.42:36577`
(SSH key: `~/.ssh/id_rsa`). Self-stops on completion or after a 16h deadman
ceiling via `~/.runpod/autostop_launch.sh` — safe to leave alone, but check
whether it's already finished/stopped before assuming it's still running.

**Check status:**
```bash
ssh -p 36577 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa root@142.169.249.42 \
  'tail -5 /workspace/e6_1280_run.log.wrapper'
```
If SSH fails, the pod likely already self-stopped — check RunPod balance/pod
list via `~/.runpod/api_key` + the REST API (`desiredStatus` should be
`EXITED`) and pull the final checkpoint from `/workspace/D-fine-visdrone/output/e6_1280/` before it's gone (community pods lose local disk on deletion).

**What it is:** tuned from `msfd_1024_best_ep109.pth` (AP=0.3219, NOT
polish2 — deliberately chosen for less single-scale specialization since its
own multi-scale phase already included 1280x1280 batches), square 1280x1280
(matching DroneScan-YOLO SOTA precedent), 50 epochs, augs-off tail from
epoch 35. Config: `experiments/e6_1280/config.yml`.

**Root-cause fix used:** NOT "AIFI positional-embedding interpolation" as
originally planned in `11_ablation_study_runpod.md` — AIFI's sin-cos PE is
already resolution-agnostic (recomputed fresh every forward pass). The real
fix is `reset_score_head_bias` (new config flag, `src/solver/_solver.py`),
which resets `enc_score_head`/`dec_score_head` biases to
`bias_init_with_prob(0.01)` at tune-time instead of silently inheriting the
old-resolution-calibrated bias. Verified working both locally and on the pod
(bias reset to -4.5951, correct anchor-buffer skip on load).

**Included from this week's R-series:** R2 (NWD regression loss) + R3
(rare-class CopyPaste) — both inherited from msfd_1024's config. **NOT
included:** R1 (aggressive zoom-crop, tested slightly negative in isolation)
and R4 (per-class score calibration, tested net-negative — eval-only
technique anyway, not baked into any checkpoint).

**Progress as of epoch 16/50 (last checked):**
- Epoch 0 peak: AP=0.342 (up from starting checkpoint's 0.3219 at 1024 eval,
  though note the eval resolution changed too, so it's not perfectly
  apples-to-apples)
- Epochs 0-15: stable plateau, 0.330-0.342, currently reading 0.342 for the
  last 3 completed epochs
- No sign of the historical resolution-collapse pattern (past naive attempts
  flatlined at AP 0.11-0.13) — this run has never been below 0.330
- Gap to SOTA (DroneScan-YOLO, 0.356): currently ~0.014 AP
- LR: warmup finished at epoch 5, now in the 45-epoch cosine decay toward
  1e-7, currently ~9e-6, tracking the predicted schedule correctly

**Pre-authorized contingency (user approved 2026-07-05, no need to re-ask):**
if AP is still clearly climbing (not plateaued) at epoch 50, resume from
`last.pth` with either a constant low LR or a slow additional decay — same
pattern as `msfd_1024_polish`'s 25→50 epoch extension. Don't ask again,
just do it when this gets checked.

**Important caveat, told to the user before they exited 2026-07-05:**
Claude Code doesn't run unattended across a session exit — there's no
mechanism (cron jobs are explicitly session-only) that survives `/exit` and
automatically executes this check at epoch 50. This isn't time-sensitive
though: the run completes its full 50-epoch schedule and the pod self-stops
safely on its own regardless of whether anyone's watching, so there's no
risk of loss or waste from nobody checking in exactly at epoch 50. Whenever
the next session opens (could be hours or days later), check the final
trajectory then — if it was still climbing at epoch 50, resume from the
saved checkpoint at that point, however late.

**Once E6 finishes, in priority order:**
1. **Deployment reality-check** — re-export ONNX + INT8 quantize the E6
   checkpoint, benchmark real latency on the target edge hardware
   (Snapdragon/Jetson). 1280 is ~4x the compute of the last benchmarked
   config (47ms/21FPS at a much cheaper resolution) — a great AP number
   here could still be a non-starter for the actual deployment target. This
   hasn't been checked at all yet and is arguably more important than
   squeezing more AP.
2. **msfd/P2 control experiment** — same 1280 + bias-reset-fix setup, but
   on the plain (non-P2) architecture, tuning the `plain_r2r3_nozoom`
   lineage instead. Settles whether P2's ~10-15% compute overhead earns its
   keep, at either resolution — open question since 2026-07-04.
3. Lower priority: re-run R4 calibration on the E6 checkpoint (free, but
   don't expect a different verdict — same cross-class-competition
   mechanism that made it net-negative before doesn't change with
   resolution). R5 (distillation) and R6-R10 remain unstarted, bigger lift,
   hold off until E6/the control experiment settle the architecture
   question.

**Housekeeping:** rotate the RunPod API key once this pod is stopped/deleted
(it was copied there via the new `autostop_launch.sh` wrapper, with explicit
user authorization — see `06_bugs_and_fixes.md` BUG-046).

---

## 2. claude-council skill — installed, needs a fresh session to activate

Installed via `npx skills add TorpedoD/claude-council` to
`~/.agents/skills/claude-council`, symlinked at `~/.claude/skills/`. It did
**not** show up in the current session's tool registry (installed
mid-session, after the skill list was already loaded) — `/claude-council`
should work in a **new** session without any further action.

Reviewed for safety before installing: clean code, no external network
calls beyond internal agent spawns, explicit shell-injection/XSS defenses in
the orchestration file. One caveat worth remembering: an automated Snyk scan
flagged it "High Risk" (while two other scanners said Safe/0 alerts) —
couldn't pull the exact finding, but it's most likely a static-analyzer flag
on the skill's dynamic-bash-command-construction-from-text pattern, which
is guarded procedurally (explicit escaping instructions in the prompt) not
enforced independently in code. Worth being a little careful that any
future invocation actually follows those escaping steps when framing a
question, rather than treating the skill as unconditionally hardened.

It's built for high-stakes decision pressure-testing (5 advisor personas +
peer review + forced debate + dual-chairman synthesis), not technical
research — good fit for something like "is the 4x compute cost of 1280
worth it" or "should we keep carrying P2/msfd forward," less useful for
routine technical questions.

---

## 3. This week's campaign, in one paragraph

msfd_1024 finished (P2ConvHead+P2FusionLite+R2+R3, AP=0.3219 @ep109,
APs=0.2323 new record) → polish/polish2 harvested the missing LR decay
(BUG-044/045), landing the standing pre-E6 best at AP=0.3226/APs=0.2344 →
msfd/P2 architecture shelved by user decision (control experiment to
attribute its gain properly was explicitly skipped) → plain_r1r2r3 +
plain_r2r3_nozoom tested R1/R2/R3 on the simpler non-P2 architecture, both
underperformed the standing best, R1 isolated to ~neutral/slightly negative
→ an idle-waste incident (~$4.80, WSL2 killing the babysitter on session
close) got root-caused and fixed (BUG-046: `.wslconfig` `vmIdleTimeout=-1` +
new pod-side self-stop wrapper `autostop_launch.sh`) → R4 per-class score
calibration tested, net-negative but hypothesis partially confirmed → E6
(this doc's §1) launched, currently the active/live thread.
