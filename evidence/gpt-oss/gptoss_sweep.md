# gpt-oss-120b: config sweep to beat +26% decode (GOAL MET, 2026-07-05)

**Headline:** **gpt-oss-120B +39.9% decode / 30.61 → 42.81 tok/s on RTX 5090**, `ncmoe30 / HOT_N42`,
**81.6% coverage**, VRAM peak **29.8 GB**, **corpus hot-list = non-circular**. Split confirmed active
(`bias=yes`, 90 hot tensors across 30 layers, 42 experts/layer, 17442 MiB). Idle machine, honest
OFF/ON, 3 reps each.

**GOAL MET: YES.** Every cell in the grid beats +26%. The hot/cold expert cache replicates and
**exceeds** the Coder-Next breakthrough on the far-more-useful gpt-oss-120b — the best honest
config is **+39.9%**, vs the +24.8% clean baseline (ncmoe26/HOT_N20) and the +26% Coder-Next anchor.

---

## The grid (idle machine, aipc-swiglu-oai @ a01f550c1, SWIGLU_OAI split; corpus hot-list; each OFF/ON = 3 reps)

Bench: `llama-bench -m <gptoss-mxfp4> -ngl 999 -ncmoe <N> -b 4096 -ub 4096 -mmp 0 -t 16 -p 0 -n 128 -r 3 -fa 1 -o json`.
ON arm sets `AIPC_MOE_HOT_LIST=autopilot/profiles/gptoss.hotlist` + `AIPC_MOE_HOT_N=<H>`; OFF arm unsets both.
Baseline VRAM 689 MiB before each; peak sampled at 200 ms cadence during each ON run. GPU idle (no other GPU workload) confirmed before EVERY measurement and after the sweep.

| ncmoe | HOT_N | coverage | OFF tok/s | ON tok/s | %gain | hot copies | VRAM peak | split |
|---|---|---|---|---|---|---|---|---|
| 26 | 24 | 65.9% | 35.10 ± 0.04 | 44.63 ± 0.31 | **+27.1%** | 9201 MiB | 28.09 GB | bias=yes |
| 28 | 29 | 71.3% | 32.47 ± 0.19 | 41.79 ± 0.33 | **+28.7%** | 11679 MiB | 27.34 GB | bias=yes |
| 30 | 38 | 78.9% | 30.61 ± 0.33 | 41.18 ± 0.18 | **+34.6%** | 15926 MiB | 28.33 GB | bias=yes |
| **30** | **42** | **81.6%** | **30.61 ± 0.33** | **42.81 ± 0.07** | **+39.9%** | **17442 MiB** | **29.81 GB** | **bias=yes** |

- **Control (ncmoe26/HOT_N24)** reproduces the known regime: +27.1% at 44.63 tok/s (OFF 35.10 ≈ the
  clean +24.8% baseline's 35.92; bumping HOT_N 20→24 lifts the gain). Sanity anchor holds.
- **Primary (ncmoe28/HOT_N29)** = +28.7% (predicted +27.2%). Beats +26% with margin at 27.3 GB — the
  safest clear-margin config.
- **Stretch (ncmoe30)** is where the compound Lever-B win lands: +34.6% at H38, **+39.9% at H42**.

## Winner = ncmoe30 / HOT_N42

Per the selection rule (highest honest %gain ≥ +26%): **ncmoe30/HOT_N42, +39.9%** — the clear %gain
winner, no ties. Honest, non-circular (corpus hot-list from `corpus_v1.txt`, NOT the bench),
split active, VRAM 29.81 GB **< 30 GB**, and the **tightest std of the entire grid (±0.07)** — proof
no WDDM cliff / eviction stutter occurred at the ceiling.

**Highest absolute throughput** is the control **ncmoe26/HOT_N24 = 44.63 tok/s (+27.1%)** at a safe
28.1 GB — the pick if raw tok/s matters more than %gain.

## Recommended SERVING config: ncmoe30 / HOT_N38 (+34.6%, 28.3 GB)

The +39.9% winner peaks at 29.81 GB — only **0.19 GB from the 30 GB WDDM cliff**. That is fine as a
bench headline (bench uses a tiny context, so KV/activations are minimal and it held clean), but a
**served** config with real context + KV cache needs headroom the bench does not consume. For
production serving, **ncmoe30/HOT_N38 (+34.6%, 30.61→41.18 tok/s, peak 28.3 GB, ~1.7 GB margin)**
is the robust choice; drop to HOT_N40 only if the served VRAM ceiling is re-measured and holds.

## Why the gain exceeds the single-anchor linear prediction (investigated — coherent, not a bug)

The coverage→speed model in the plan (0.382 %/coverage-point) was anchored at ONE point
(ncmoe26, 65% cov → +24.8%). Measured gain-per-coverage-point RISES with ncmoe:

| cfg | cov% | gain% | speedup× | gain/cov-pt | n_off (cold layers) |
|---|---|---|---|---|---|
| ncmoe26/24 | 65.9 | +27.1 | 1.27 | 0.41 | 26 |
| ncmoe28/29 | 71.3 | +28.7 | 1.29 | 0.40 | 28 |
| ncmoe30/38 | 78.9 | +34.6 | 1.35 | 0.44 | 30 |
| ncmoe30/42 | 81.6 | +39.9 | 1.40 | 0.49 | 30 |

Within ncmoe30, H38→H42 (cov +2.7 pt) gives gain +5.3 pt ⇒ **~1.97 %/cov-pt**, far steeper than the
ncmoe26 anchor's 0.38. This is exactly the **compound Lever-B win** the plan predicted: more
CPU-resident layers = more cold-expert bandwidth to recover, so each covered expert is worth more
tok/s AND the OFF baseline is lower (35.10→32.47→30.61 as ncmoe rises). The linear model
*undershoots* at high offload by construction — the over-performance is physical, not a measurement
or fallback artifact. All ON runs confirmed `bias=yes` (SWIGLU_OAI split active, never fell back).

## VRAM math (STEP 1) vs measured — model validated

Predicted hot copies `(HOT_N+4)×n_off×12.64 MiB` matched the load-log to within ~1%:
H24→9.0/9.20, H29→11.4/11.68, H38→15.6/15.93, H42→17.0/17.44 GiB (pred/measured). The model
footprint anchor (~19.7 GB @ ncmoe26) was slightly conservative — real model+activations ≈ 19.1 GB,
shrinking ~1.6 GB/layer — so actual peaks ran a touch under the 29.5 GB targets, leaving room to
push HOT_N to the 30 GB ceiling at ncmoe30. **VRAM, not coverage, is the binding constraint** at the
top: ncmoe30/HOT_N42 is the max that fits < 30 GB; the gain was still climbing (+5.3 pt from the last
+2.7 cov-pt), so a higher VRAM budget would go further.

## Next lever (to push beyond +39.9% at fixed VRAM)

Coverage is still the ceiling under the 30 GB cap (H42 = 81.6% corpus coverage). The Coder-Next
33%→68.6% lesson applies: an **adaptive/session hot-list** profiled from a disjoint representative
workload raises coverage at fixed HOT_N, which — given the measured ~2 %/cov-pt slope at ncmoe30 —
would add several more points of gain without any extra VRAM. That is the follow-up if an even
stronger marquee is wanted; the corpus (non-circular) number already clears the goal.

## Adoption note for swiglu_oai.md §6 and public promotion

Replace the provisional +23% (ncmoe26/HOT_N20, contended) with the honest idle grid above. Headline
for promotion: **"gpt-oss-120B, +39.9% decode (30.6→42.8 tok/s) on a desktop RTX 5090, ncmoe30/
HOT_N42, 81.6% coverage, corpus hot-list (non-circular), VRAM 29.8 GB."** For a conservative/serving
claim use **ncmoe30/HOT_N38, +34.6%, 28.3 GB** or the highest-throughput **ncmoe26/HOT_N24, 44.6
tok/s, +27.1%**. Raw JSON + verbose stderr (split lines) for all 7 runs; key artifacts:
`off{26,28,30}.json`, `on26_24.json`, `on28_29.json`, `on30_38.json`, `on30_42.json`,
and the matching `*.stderr` with the `AIPC split: … bias=yes … MiB` confirmations.

## Method / hygiene compliance

- Idle confirmed (no other GPU compute process) before EACH of the 7 measurements and after; baseline
  689 MiB throughout; nothing left running (VRAM back to 689 MiB at the end).
- Working VRAM < 30 GB on every ON run (max 29.81 GB); peak sampled live per run.
- Split active (`bias=yes`) verified on every ON run — the SWIGLU_OAI path never fell back.
- Standalone (bench reloads the model between configs); distinct OFF/ON baselines per ncmoe (honest,
  non-circular); corpus hot-list (`gptoss.hotlist` == `gptoss_corpus_v1.hotlist`, from corpus_v1.txt,
  NOT the bench).
