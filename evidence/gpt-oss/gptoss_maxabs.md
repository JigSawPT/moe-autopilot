# gpt-oss-120b: maximum HONEST absolute decode tok/s (2026-07-05)

**HEADLINE — new best absolute:** **47.61 tok/s** at **ncmoe25 / HOT_N22**, corpus hot-list, **63.4 %
coverage**, VRAM peak **29.63 GB** (< 30 GB), split active (`bias=yes`, 8215 MiB, 75 hot tensors /
25 layers), **+31.6 %** over that config's own no-cache baseline (36.18 tok/s). Instrument =
`llama-bench tg128` — the SAME instrument as the prior 44.63.

**Beats the prior best 44.63 tok/s (ncmoe26/HOT_N24) by +6.7 % absolute.** The prior sweep never
tested ncmoe < 26; the win comes from Lever 1's hypothesis being TRUE — a faster no-cache baseline at
one-fewer offloaded layer more than pays for the slightly lower coverage.

**Binding ceiling: VRAM, not routing-flatness.** The optimum sits pinned against the 30 GB WDDM cliff
(the winner's 22 experts already reach 29.63 GB; H23 = 29.90 GB was *slower*, not faster). Routing
flatness is a secondary, real limiter on the adaptive lever (Lever 2) but the absolute number is
capped by the 30 GB budget.

---

## Method / hygiene (the rules ARE the point here)

- **Idle machine re-confirmed before EVERY measurement** (no other GPU compute process) and after the
  last run. Idle GPU baseline **~1.6 GB** this session (higher than the prior sweep's 0.69 GB — a
  co-resident display/system baseline; it shifts *absolute* peak up ~0.9 GB, which is why this VRAM
  budget is tighter than the sweep's and why one attractive config was disqualified — see below).
- **Working VRAM ≤ 30 GB on every reported run.** Peak sampled live (200 ms cadence) for the bench arm;
  read at load-settle for the server arm. Any config that peaked > 30 GB is **disqualified as a
  headline** even if it ran clean (honest 47.6 beats a cliff-risk 48.6).
- **Split ACTIVE (`bias=yes`) confirmed on every ON run** — SWIGLU_OAI dual-chain, never fell back.
  (Server needed `-lv 4` to *surface* the load-time `AIPC split:` INFO line; default server log level
  suppresses it. The split loads identically either way — proven by the 8215 MiB hot-copy VRAM and the
  speed delta. See "Instrument notes".)
- **Standalone**: the model reloads between configs (llama-bench with distinct `-ncmoe`; server killed
  and relaunched per arm; VRAM returns to ~1.6 GB idle between arms). Distinct OFF baseline measured
  per ncmoe.
- **Non-circular**: Lever-1 uses the **corpus** hot-list (`gptoss.hotlist` == `gptoss_corpus_v1.hotlist`,
  from `corpus_v1.txt`, NOT the bench). Lever-2 profiles a **session** disjoint from the **held-out**
  eval prompt (different domains). Coverage is `autopilot.py coverage` over all 36 layers — the same
  metric as the prior sweep (HOT_N24 → 65.9 % reproduces exactly).

Build: `llama.cpp-aipc` build tree (aipc-swiglu-oai, SWIGLU_OAI split) — see the reproduction guide
for the exact build recipe.
Model: `gpt-oss-120b-mxfp4-00001-of-00003.gguf`.

---

## LEVER 1 — best (ncmoe, HOT_N) for ABSOLUTE speed (corpus hot-list, llama-bench tg128)

Bench: `llama-bench -m <gguf> -ngl 999 -ncmoe <N> -b 4096 -ub 4096 -mmp 0 -t 16 -p 0 -n 128 -r 3 -fa 1
-v -o json`. ON sets `AIPC_MOE_HOT_LIST=gptoss.hotlist` + `AIPC_MOE_HOT_N=<H>`; OFF unsets both.
Each ON = 3 internal reps; the winner also run **3× standalone** (9 total decodes).

**OFF baselines (measured, idle):** ncmoe24 **37.42** ± 0.87 · ncmoe25 **36.18** ± 0.69 · ncmoe26
**35.63** ± 0.60. *Fewer offloaded layers ⇒ faster baseline, monotonically* — the premise of Lever 1.
(ncmoe26 reproduces the sweep's 35.10 within noise.)

| config | coverage | OFF tok/s | ON tok/s | %gain | VRAM peak | note |
|---|---|---|---|---|---|---|
| ncmoe26 / H24 | 65.9 % | 35.63 | 45.29 ± 0.10 | +27.1 % | 28.92 GB | prior-best anchor, reproduced (sweep: 44.63) |
| ncmoe26 / H26 | 68.2 % | 35.63 | 46.34 ± 0.28 | +30.1 % | 29.68 GB | new headroom at ncmoe26 |
| ncmoe25 / H21 | 62.1 % | 36.18 | 46.53 ± 1.01 | +28.6 % | 29.34 GB | |
| **ncmoe25 / H22** | **63.4 %** | **36.18** | **47.61** | **+31.6 %** | **29.63 GB** | **WINNER** (reps 47.12 / 47.81 / 47.91, std ≤0.55) |
| ncmoe25 / H23 | 64.7 % | 36.18 | 46.89 ± 0.38 | +29.6 % | 29.90 GB | at the cliff, *slower* than H22 |
| ncmoe24 / H18 | 57.9 % | 37.42 | 46.95 ± 0.68 | +25.5 % | 29.74 GB | fastest baseline arm |
| ncmoe24 / H19 | 59.3 % | 37.42 | 46.98 ± 0.55 | +25.5 % | 29.97 GB | at the cliff edge |
| ~~ncmoe24 / H21~~ | 62.1 % | 37.42 | ~~48.64 ± 0.24~~ | ~~+30.0 %~~ | **30.50 GB** | **DISQUALIFIED (> 30 GB)** |

**The optimum is a genuine interior peak at ncmoe25.** The two knobs trade off:
- ncmoe24: fastest baseline (37.42) but VRAM caps HOT_N at 18–19 → only 57.9–59.3 % coverage → 46.95–46.98.
- ncmoe26: most coverage room (H26 = 68.2 %) but slowest baseline (35.63) → 46.34.
- **ncmoe25: enough baseline (36.18) AND enough coverage (63.4 % at the H22 VRAM cap) → 47.61 — the top.**

**On the disqualified 48.64:** ncmoe24/H21 was the fastest number seen, and its per-run std (0.24) shows
it did not visibly cliff on that run — but it peaked at **30.50 GB**, over the hard rule. Per the
escalation rule ("an honest 44.6 beats a cliff-inflated 50"), it is excluded. Under the prior sweep's
lower 0.69 GB idle baseline it would have peaked ~29.6 GB and been legal; today's ~1.6 GB system
baseline pushes it over. The honest budget-legal winner is ncmoe25/H22 = 47.61.

### VRAM math (validated exact this session)

Hot copies `= (HOT_N + 4) × n_off × 12.64 MiB` — predicted == logged to 0.01 GiB on every run
(e.g. ncmoe25/H22 → 8215 MiB). Model+activations+idle-baseline scales **~1.58 GB per offloaded layer**:
measured OFF peaks ncmoe24 **23.09** / ncmoe25 **21.51** / ncmoe26 **19.93** GB. Total peak =
base(n_off) + hot_copies, exact to 0.00 GB. Budget-legal max HOT_N (≤ 29.8 GB, today's baseline):
ncmoe24→H18, ncmoe25→H22, ncmoe26→H26. **VRAM is the binding constraint at the top** — the winner is
the max HOT_N that fits at its ncmoe, and every attempt to add one more expert (H23) or drop a layer
for a faster baseline (ncmoe24) either went over budget or was slower.

---

## LEVER 2 — adaptive / session hot-list (the honest, non-circular adaptive number)

**Instrument = real `llama-server`** (not llama-bench) on a **held-out** prompt disjoint from the
session used to build the hot-list. Server: `-ngl 999 --n-cpu-moe 25 --no-mmap --poll 100 -c 1536 -b
512 -ub 512 -np 1 -t 16 -fa on --jinja -lv 4`, `reasoning_effort: low`, temp 0, seed 42, `max_tokens
400` (all reps decoded the full 400), `cache_prompt: false`; **1 warmup discarded + 3 measured reps**;
metric = server `predicted_per_second`. VRAM peak 29.71 GB, split `bias=yes` (8215 MiB) confirmed on
both ON arms.

**Workload files (committed):** session = `autopilot/profiles/gptoss_session.txt` (6 prompts: coding
×2, reasoning ×2, general ×2 — realistic daily use); held-out = `autopilot/profiles/gptoss_heldout.txt`
(a UART ring-buffer / producer-consumer systems prompt — a domain NOT in the session). Session hot-list
`gptoss_session.hotlist` profiled at ncmoe25 (`llama-aipc-moe-profile`, ub4096).

**Coverage of the HELD-OUT workload (autopilot.py coverage, HOT_N22, all 36 layers):**

| hot-list | held-out coverage |
|---|---|
| corpus (`gptoss.hotlist`) | **28.0 %** (min 12.8, max 50.0) |
| **session (`gptoss_session.hotlist`)** | **45.1 %** (min 23.9, max 61.1) |

The generic corpus hot-list (project-docs prose) matches a live English systems prompt
poorly (28 %); the workload-matched session hot-list nearly **doubles** held-out coverage to 45 % —
the Coder-Next "generic 33 % → session 68 %" lesson, replicated on gpt-oss.

**Decode tok/s on the held-out prompt (internal server-to-server delta — all 3 arms same server):**

| arm | held-out coverage | decode tok/s | vs OFF |
|---|---|---|---|
| OFF (no cache) | — | 38.03 (38.05 / 38.08 / 37.97) | — |
| corpus-ON | 28.0 % | 40.90 (41.05 / 40.75 / 40.90) | **+7.5 %** |
| **session-ON** | **45.1 %** | **41.85 (41.71 / 41.93 / 41.90)** | **+10.0 %** |

**Adaptive gain (session over corpus): +2.3 % absolute** (41.85 vs 40.90), from +17.1 coverage points.
Honest and non-circular (disjoint session/held-out; split confirmed both arms). Session > corpus, as
predicted.

---

## Coverage → speed cross-check (both levers)

- **Lever 1 (bench):** across the grid, ON tok/s tracks `OFF_baseline(ncmoe) × speedup(coverage)`. A
  fitted model `speedup = 1 / (1 − cold_frac·coverage·eff)` with `eff ≈ 0.343` (stable across all four
  prior-sweep points AND the new points) predicts ncmoe25/H22 at 46.3 and the measured is 47.6 — the
  low-ncmoe multiplier runs a touch *above* the fit (the interior peak is slightly better than a
  single-slope model expects). Direction and magnitude confirmed.
- **Lever 2 (server):** +17.1 held-out-coverage points bought +2.3 % tok/s. The tok/s delta is far
  smaller than the coverage delta because **gpt-oss routing is flat**: both hot-lists already cover the
  high-traffic head experts; the extra experts the session list adds are lower-count tail experts, so
  each added covered expert recovers less cold bandwidth. This is the routing-flatness ceiling — real,
  but secondary to VRAM for the absolute number.

---

## Which ceiling is binding?

- **Absolute tok/s (the headline metric): VRAM-bound.** The winner is pinned at the 30 GB WDDM cliff
  (29.63 GB); the next expert (H23, 29.90 GB) was slower and one attractive config (ncmoe24/H21, 48.64)
  was over budget. More VRAM (a higher-VRAM card, or a higher WDDM budget) would go
  straight to a higher HOT_N and a higher number. On THIS 32 GB card at ≤ 30 GB, **47.61 tok/s is the
  honest ceiling.**
- **Adaptive gain (Lever 2): routing-flatness-bound, not VRAM-bound.** At fixed VRAM the session
  hot-list already lifts held-out coverage 28 → 45 %, but gpt-oss's flat routing converts that into only
  +2.3 % tok/s. A more skewed MoE (Coder-Next) converts coverage to speed far more efficiently; on
  gpt-oss the adaptive lever is genuine but modest.

## Instrument notes (honest)

- **llama-bench tg128 vs llama-server differ by ~13 %** at the same config (bench 47.61 vs server ~41
  for ncmoe25/H22 ON). The bench uses a tiny synthetic context with no KV growth and `-ub 4096`; the
  server carries a real prompt, KV cache, and ran at `-ub 512` (needed to fit ≤ 30 GB with the hot
  copies). Both are honest for their purpose: the bench number is the like-for-like successor to the
  44.63 headline; the server numbers are the honest **adaptive** comparison and the realistic
  single-user serving throughput. They are NOT mixed into one figure.
- The session hot-list's +2.3 % is reported ONLY on the server (real workload). Applying a session
  hot-list to the bench's synthetic tg128 would be meaningless (the hot-list isn't matched to what the
  bench decodes) — so the adaptive lever correctly lives on the server instrument.

## Bottom line

- **New absolute headline: 47.61 tok/s, gpt-oss-120b, ncmoe25/HOT_N22, corpus hot-list (non-circular),
  63.4 % coverage, VRAM 29.63 GB, +31.6 % over its own no-cache baseline** — beats the prior 44.63 by
  +6.7 %, same instrument.
- **Adaptive (honest, server, held-out): session 41.85 vs corpus 40.90 vs OFF 38.03 tok/s** — the
  workload-matched hot-list adds +2.3 % over corpus (held-out coverage 28 → 45 %).
- **Ceiling: VRAM-bound for absolute throughput; routing-flatness-bound for the adaptive gain.**

---

### Raw artifacts
Bench JSON+stderr: `off{24,25,26}.*`, `on25_22{,b,c}.*`, `on25_21.*`, `on25_23.*`, `on24_{18,19,21}.*`,
`on26_26.*`, `sanity_on26_24.*` (each `*.stderr` carries the `AIPC split: … bias=yes … MiB` line for ON
runs). Server JSON path via `srv.sh`; stderr `srv_s_{off,corpus,session}.stderr` (ON arms carry the
`AIPC split:` line under `-lv 4`). Harnesses: `run_bench.ps1`, `srv.sh`. Committed workload +
hot-lists: `autopilot/profiles/gptoss_session.txt`, `gptoss_heldout.txt`, `gptoss_session.hotlist`,
`gptoss_session.json`, `gptoss_heldout.hotlist`, `gptoss_heldout.json`.
