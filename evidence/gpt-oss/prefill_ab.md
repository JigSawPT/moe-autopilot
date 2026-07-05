# Prefill A/B — is the V2 expert split really "prefill unchanged"?

**Date:** 2026-07-04 · **Machine:** RTX 5090 32 GB + Ryzen 9 9950X3D + 64 GB DDR5 · **Build:** `llama.cpp-aipc` PR build, commit `ab2fbb7a6` build 9875 (CUDA 13.0 toolset, arch 120), `llama-bench.exe` from that build's `build/bin/Release/` output directory.

## Question
The V2 hot/cold expert split is **decode-only** by construction (the split routing is gated on `n_tk ≤ 8`). We have long *claimed* "prefill unchanged" but never measured the explicit A/B. This is that measurement.

## Method
- **Model (incumbent):** Qwen3-Coder-Next MXFP4 (`qwen3next 80B.A3B MXFP4 MoE`, 48 GB), `bench/models.json:qwen3-coder-next-mxfp4`.
- **Flags (both arms):** `-ngl 999 --n-cpu-moe 24 -b 2048 -ub 2048 --mmap 0 -fa on -t 16`. (Brief said `-fa 1`; the PR-build `llama-bench` takes `on|off|auto`, so `-fa on` = the intended flash-attn ON.)
- **Metrics:** `pp2048` (prefill) and `tg128` (decode), `-r 2` (2 reps) per arm. Run **twice** (independent model loads) to confirm reproducibility per method rule #1 (never promote a number without a standalone confirmation).
- **ARM A (control):** no AIPC env.
- **ARM B (treatment):** `AIPC_MOE_HOT_LIST=autopilot\profiles\session.hotlist`, `AIPC_MOE_HOT_N=96`. Load log confirms the split armed: `AIPC split: 72 hot tensors across 24 layers, 96 experts/layer, 4452 MiB of VRAM`.
- **VRAM discipline:** logged before/after each arm; no processes left resident.

### VRAM baselines (MiB used / free / total)
| Point | used | free | total |
|---|---|---|---|
| Session baseline (pre-work) | 1751 | 30437 | 32607 |
| ARM A before / after | 1751 / 1751 | 30437 / 30437 | 32607 |
| ARM B before / after | 1751 / 1710 | 30437 / 30478 | 32607 |

Microbench-style llama-bench frees the model on exit; no residency leaked. (The hot-copy 4452 MiB is allocated inside the process during the run and released on exit — it does not show at the "after" sample.)

## Raw results (tok/s)

| Metric | ARM A run1 | ARM A run2 | ARM B run1 | ARM B run2 | A mean | B mean | Δ (B vs A) |
|---|---|---|---|---|---|---|---|
| **pp2048** | 3153.34 ± 13.32 | 3154.89 ± 16.72 | 2841.15 ± 13.54 | 2842.33 ± 3.60 | **3154.1** | **2841.7** | **−9.90 %** |
| **tg128** | 73.94 ± 0.03 | 74.52 ± 0.37 | 78.03 ± 0.20 | 78.18 ± 0.84 | **74.23** | **78.11** | **+5.22 %** |

Raw JSON: `prefill_ab_A.json`, `prefill_ab_B.json` (run 1); `prefill_ab_A2.json`, `prefill_ab_B2.json` (confirmation run 2). Stderr with load/split logs: `prefill_ab_A.stderr`, `prefill_ab_B.stderr`.

## ⚠️ FINDING — prefill is NOT unchanged: pp2048 drops ~9.9 % in ARM B

**The `>3 %` threshold in the brief is exceeded by ~3.3×, and it reproduces exactly** (run1 −9.90 %, run2 −9.91 %; the two control runs agree to 0.05 %, the two AIPC runs to 0.04 %). This is far outside the ~0.4 % per-run stddev — **not** machine-state noise.

**The expected result was pp2048 statistically identical** (the split is decode-only, guarded at `n_tk ≤ 8`, so prefill should never enter the hot/cold path). The decode side behaved as designed (tg128 **+5.22 %**, ARM B faster). But prefill measurably regresses.

### What is NOT the cause (ruled out by evidence)
- **Not extra graph splits / not the split routing.** Graph split counts are *identical* in both arms: `graph splits = 74 (bs=2048), 50 (bs=1)` in A and B alike. The decode-only guard works — prefill graph structure is unchanged. So the regression is **not** the scheduler carving new split boundaries in prefill.
- **Not VRAM pressure / WDDM paging.** Free VRAM stays > 30 GB in both arms (hot copy is 4.35 GB against 30 GB free); no paging regime (method rule #2 satisfied, working set ≪ 30 GB).
- **Not mmap / cold-cache.** `--mmap 0` in both arms (method rule #3).

### Mechanism ISOLATED — it is the resident hot copies, confirmed by a third arm
A third measurement pins the cause. **ARM C**: hotlist *parsed* but `AIPC_MOE_HOT_N=0` → the code takes the "split inactive" branch (`llama-model.cpp:1681`, `n_hot_req <= 0`), so the list is read but **no hot copies are placed in VRAM**.

| Arm | AIPC env | hot copies in VRAM | pp2048 (tok/s) | Δ vs control |
|---|---|---|---|---|
| A (control) | none | 0 | 3154.1 | — |
| **C** | hotlist + `HOT_N=0` | **0** (split inactive) | **3154.30 ± 15.00** | **+0.006 % (identical)** |
| B (treatment) | hotlist + `HOT_N=96` | **4452 MiB** | 2841.7 | **−9.90 %** |

Raw: `prefill_ab_C_hotn0.json`. **Conclusion: parsing the hot-list costs nothing in prefill (C == A to 0.006 %); the entire −9.9 % regression is caused specifically by the 4452 MiB of duplicate hot expert weights resident in VRAM.** The decode-only *guard* is doing its job (graph structure identical, C confirms the parse path is free) — the penalty is purely a **resource side-effect of holding the hot copies**, even though those copies are *not used for compute* in prefill (the guard forces the CPU expert path). Most likely the 4.35 GB hot-copy allocation reduces the large-`ub=2048` prefill scratch/compute-buffer headroom or shifts CUDA buffers to a less favourable region; the graph-split count (74 both arms) rules out any scheduling change. **This is a real, reproducible regression and is now mechanistically isolated — the unqualified "prefill unchanged" claim in the V2 write-up must be corrected to "prefill routing unchanged, but resident hot copies cost ~10 % prefill throughput."**

## TTFT / end-to-end model (from the measured numbers)
`TTFT ≈ prompt_tokens / pp2048` · `total ≈ TTFT + output_tokens / tg128`. Using the A/B means above.

| Scenario | prompt | out | A: TTFT | A: total | B: TTFT | B: total | ΔTTFT | Δtotal | B verdict |
|---|---|---|---|---|---|---|---|---|---|
| chat | 500 | 300 | 158.5 ms | 4200.0 ms | 176.0 ms | 4016.7 ms | +17.4 ms | **−183.3 ms** | **B wins** (decode dominates) |
| long-context | 16000 | 300 | 5072.8 ms | 9114.3 ms | 5630.4 ms | 9471.2 ms | +557.7 ms | **+356.9 ms** | **B loses** (prefill penalty swamps) |
| agent-loop | 4000 | 100 | 1268.2 ms | 2615.4 ms | 1407.6 ms | 2687.9 ms | +139.4 ms | **+72.5 ms** | **B loses** (prefill-heavy, decode gain too small) |

**Interpretation.** The V2 split is a **net win only when generated tokens dominate the prompt** (chat-like). For prompt-heavy shapes (long-context, agent tool-loops with big context + short outputs) the −9.9 % prefill penalty now *outweighs* the +5.2 % decode gain at the end-to-end level. The break-even is roughly where `output/tg_gain ≈ prompt/pp_loss`: with these numbers, ARM B stops being a total-latency win once `prompt / output ≳ ~2.3`.

## Verdict
- **Decode:** as designed — tg128 **+5.2 %** with the hot-list split (ARM B). ✓
- **Prefill:** **NOT unchanged — pp2048 regresses −9.9 %, reproducible, well outside noise. FINDING flagged.** ✗ The split is *routing*-decode-only, but the resident hot copies impose a prefill cost that the graph-split count does not reveal.
- **End-to-end:** the split's benefit is **conditional on the prompt:output ratio**. Net-positive for generation-heavy chat; net-negative for prompt-heavy long-context / agent-loop workloads. This nuance should feed the Autopilot decision of *when* to arm the hot-list, and should temper the unqualified "prefill unchanged" claim.

### Follow-up
1. ✅ **Done in this task (ARM C above):** hotlist parsed with `HOT_N=0` gives pp2048 identical to control → mechanism isolated to the resident hot copies, not the parse path.
2. Not done (out of scope): sweep `HOT_N ∈ {48, 96, 192}` on pp2048 to see whether the prefill penalty scales with resident hot-copy size (would quantify the resource-contention curve).
