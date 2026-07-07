# Session Handoff — 2026-07-06

Read this first when resuming. Full experiment history lives in
`00_progress.md`, `06_bugs_and_fixes.md`, `11_ablation_study_runpod.md` as
usual — this file is just "what's live right now and what to do next."
Prior handoff: `SESSION_HANDOFF_2026-07-05.md` (superseded, but still has the
E6 design rationale and the claude-council install notes).

---

## 1. E6 (1280 resolution unlock) — FINISHED, AP=0.344

Full 50-epoch schedule completed. **Final AP=0.344**, up from the pre-E6
standing best of 0.3226, ~0.012 off the DroneScan-YOLO SOTA precedent
(0.356) — caveat: that 0.356 is their paper-reported number; their weights
were never found publicly, so this gap is unverified, see `00_progress.md`
SOTA context section. Best checkpoint at epoch 46/50
(`output/runpod_results/e6_1280_best_ep46.pth`), last at epoch 49
(`..._last_ep49.pth`) — both already retrieved to local disk. Best landing
a few epochs before the end suggests the run had plateaued rather than
still climbing, so the pre-authorized resume-with-low-LR contingency was
**not** triggered. Caveat: this reading is from checkpoint epoch numbers,
not a full per-epoch trajectory/wandb pull — worth double-checking if a
resume ever becomes relevant later. Full design rationale (bias-reset fix,
what's included from R2/R3, what's excluded): `SESSION_HANDOFF_2026-07-05.md`
§1 and `00_progress.md` Step 29.

**Incident right after completion:** the pod sat idle for ~10.6h (~$2.33)
because the pod-side `autostop_launch.sh` (built in BUG-046 for exactly
this) didn't fire this time. See `06_bugs_and_fixes.md` BUG-047.

**New standing infra (as of today):** a second, laptop-side watchdog is now
running independently of any pod-side mechanism —
`~/.runpod/watchdog.sh`, started via `~/start_watchdog.sh`
(`setsid nohup ... & disown`, survives terminal close given the
`vmIdleTimeout=-1` fix from BUG-046). Polls the RunPod API every 10 min,
SSHes into every `RUNNING` pod, and force-stops any pod with no `train.py`
process for 2 consecutive checks (~20 min grace). **Verify it's alive before
launching any new pod:** `ps aux | grep '[w]atchdog.sh'` — if it's not
running, restart with `bash ~/start_watchdog.sh` before starting new
training, otherwise a repeat of BUG-047 has no backstop. Log:
`~/.runpod/watchdog.log`; stop events: `~/.runpod/watchdog_stops.log`.

**Next up, in priority order (unchanged from 07-05, none started yet):**
1. **Deployment reality-check** — ONNX export + INT8 quantize
   `e6_1280_best_ep46.pth`, benchmark real latency on the target edge
   hardware (Snapdragon/Jetson). 1280 is ~4x the compute of the last
   benchmarked config (47ms/21FPS at a much cheaper resolution) — this
   hasn't been checked at all and matters more than squeezing more AP.
2. **msfd/P2 vs plain control experiment** at 1280 — settles whether P2's
   ~10-15% compute overhead earns its keep. Open since 2026-07-04.
3. Lower priority: R4 re-run on the E6 checkpoint (don't expect a different
   verdict), R5 distillation, R6-R10 unstarted.

**Housekeeping carried over from BUG-046, still not done:** rotate the
RunPod API key once the E6 pod (`41rlk08qzcicf4`) is confirmed
stopped/deleted — it was copied onto that pod by `autostop_launch.sh`.

---

## 2. claude-council skill

See `SESSION_HANDOFF_2026-07-05.md` §2 — installed, should now be available
in this (new) session via `/claude-council`.
