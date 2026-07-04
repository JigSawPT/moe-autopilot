# hot-experts — an adaptive hot/cold MoE expert split for llama.cpp

**A load-time hot/cold expert split on top of `--n-cpu-moe` that gives a repeatable
+26% decode band on consumer hardware, using per-session adaptive hot-lists and a
calibrated coverage→speed model.**

This is an opt-in, **decode-only**, **static** split (built once at model load — it is
**not** a dynamic VRAM cache with eviction). When you offload MoE expert weights to
system RAM to fit a large model on a small GPU, `--n-cpu-moe` re-reads *every* active
expert from RAM on *every* token — including the handful that are hit on almost every
token. This tool keeps a VRAM copy of the top-N most-activated experts per offloaded
layer and runs a dual FFN chain at decode time (hot experts from VRAM, cold ones from
RAM, merged once per layer). Prefill *routing* is unchanged (decode-only guard verified:
identical graph-split counts with the split armed), and the feature is **zero-overhead
when disabled** — but arming it is not free at prefill: the resident hot copies cost
**~10% prefill throughput** at HOT_N=96 on this config (measured; see the prefill
section below). The net end-to-end win depends on your prompt:output ratio.

It is a working, measured prototype of the mechanism requested in llama.cpp issue
[#20757](https://github.com/ggml-org/llama.cpp/issues/20757) (two-tier GPU+RAM expert
cache for MoE offload). It ships as (a) a small offline profiler that produces
hot-lists, and (b) a patched llama.cpp fork branch that consumes them.

> **Scope, stated up front (please read before quoting numbers):**
> - The numbers below are a **repeatable decode band**, not a production SLA.
> - The split applies to **decode only** (`n_tokens ≤ 8`); prefill *routing* uses the
>   stock path (verified), **but the resident hot copies cost ~10% prefill throughput**
>   at HOT_N=96 on this config. Net end-to-end win requires generated tokens to dominate
>   the prompt — break-even at **prompt:output ≈ 2.3** here (measured section below).
> - This is a **static load-time split, not a dynamic eviction cache** — the hot-list is
>   chosen offline and fixed until you restart with a different one.
> - All measurements are on **one machine** (RTX 5090 32 GB + Ryzen 9950X3D + 64 GB
>   DDR5-6000). Treat them as an existence proof on consumer hardware, not a universal
>   speedup. Recalibrate on your own hardware.

---

## Headline results

Primary model: **Qwen3-Coder-Next MXFP4** (80B-A3B, 48 layers, 10/512 experts),
`--n-cpu-moe 24`, `--no-mmap`, `--poll 100`, server decode throughput
(`timings.predicted_per_second`), temperature 0, identical unseen prompt, 2 reps.
Coverage = fraction of a live task's expert hits covered by the hot-list at top-96/512.

| Hot-list source | top-96/512 coverage | decode tok/s | vs baseline |
|---|---|---|---|
| none (baseline `--n-cpu-moe 24`) | 0% | **72.1** | — |
| generic corpus | 33.5% | 77.1 | +6.9% |
| **session history (3 prior tasks, task-disjoint)** | **68.6%** | **90.5–91.5 band** | **+26%** |
| same-task (circular, ceiling) | 80.4% | 92.3 (measured ceiling) | +28% |

- The **90.5–91.5 tok/s** figure is the repeatable band under pristine machine state
  (contemporaneous same-session control: best 90.93 / mean 90.53 tok/s, VRAM 31 680 MB).
  The raw 2-rep A/B that anchors the band read **85.8 / 86.2 tok/s** for the
  non-circular session hot-list; the 90.5–91.5 band is the same configuration measured
  under a controlled low VRAM baseline. Both are reported honestly below.
- Gain scales ~linearly with covered bytes; a bandwidth model (expert gather ≈ 42 GB/s
  on DDR5-6000 dual channel) predicts these within ~3–5%.

**Second architecture — Qwen3.6-35B-A3B (8/256 experts), forced `--n-cpu-moe 16`:**

| Config | coverage | decode tok/s | vs baseline |
|---|---|---|---|
| baseline | — | 115.8 / 117.6 | — |
| session split (HOT_N=48) | 64.2% | **128.3 / 128.4** | **+10.3%** |

The smaller relative gain on the faster model is consistent with a fixed per-layer
dispatch floor (~1–3 ms/token) of the cold chain that dominates once the cold byte
count shrinks.

**Zero overhead when disabled** — a build with the code present but the env vars unset
produces byte-for-byte the original graph. Verified on the 35B-A3B with all experts in
VRAM (`--n-cpu-moe 0`, `llama-bench`, 3 reps): **tg128 = 291.59 ± 0.71 tok/s**, matching
the vanilla reference (~288.9 prebuilt / 293.9 prior fork build) — the split code is
never entered.

**Quality at temperature 0:** the split output is a semantically-equivalent, correct
result versus the clean binary (verified on Coder-Next: same modulo-11 NIF-validator
algorithm, same signature/docstring). Small stylistic divergence is the expected
consequence of a slightly different floating-point path, not a defect.

---

## Prefill cost of the resident hot copies (measured)

The split is routing-decode-only, and we long *assumed* that meant "prefill unchanged."
The explicit A/B says otherwise, and the honest formulation is: **prefill routing is
unchanged (decode-only guard verified: identical graph-split counts, 74 in both arms),
but the resident hot copies cost ~10% prefill throughput at HOT_N=96 on this config.**

`llama-bench` on the primary model, both arms `-ngl 999 --n-cpu-moe 24 -b 2048 -ub 2048
--mmap 0 -fa on -t 16`, 2 reps per arm, the whole A/B run twice (independent model
loads):

| Arm | AIPC env | hot copies in VRAM | pp2048 (tok/s) | Δ vs control |
|---|---|---|---|---|
| A (control) | none | 0 | **3154.1** | — |
| C (isolation) | hotlist + `HOT_N=0` | 0 (split inactive) | 3154.30 ± 15.00 | **+0.006%** (identical) |
| B (split armed) | hotlist + `HOT_N=96` | **4452 MiB** | **2841.7** | **−9.9%** |

- **Reproducible, not noise:** run 1 −9.90%, run 2 −9.91%; per-run stddev ~0.4%.
- **The cause is isolated.** ARM C (hot-list parsed but no copies placed) is identical
  to control, and graph-split counts are identical in all arms — so it is **not**
  routing, parsing, or scheduling. The entire regression comes from the **4452 MiB of
  duplicate hot expert weights resident in VRAM** (72 hot tensors across 24 layers),
  which are not even used for compute during prefill. Most likely they reduce the
  large-`ub` prefill scratch/compute-buffer headroom or shift CUDA buffers to a less
  favourable region. VRAM stayed > 30 GB free in both arms, so this is *not* the WDDM
  paging cliff.
- **Decode in the same instrument gained +5.2%** (tg128 74.23 → 78.11). Note the
  instrument: the headline +22–26% decode numbers come from *server timings* under the
  serving config; this llama-bench A/B is a different instrument and batch config — its
  value here is that both arms are measured identically, so the −9.9%/+5.2% pair is
  internally consistent.

**What it means end-to-end** (TTFT ≈ prompt/pp2048; total ≈ TTFT + output/tg128, using
the A/B means above):

| Scenario | prompt | output | Δ total latency (split ON) | verdict |
|---|---|---|---|---|
| chat | 500 | 300 | **−183 ms** | split wins (decode dominates) |
| agent loop | 4 000 | 100 | +72 ms | split loses (prompt-heavy) |
| long context | 16 000 | 300 | +357 ms | split loses (prefill penalty swamps) |

**Break-even at prompt:output ≈ 2.3** on these numbers — the split stops being a
total-latency win once your prompts are ~2.3× longer than your outputs.

**Tuning guidance:**

- `HOT_N` is a three-way trade: **decode gain vs prefill cost vs VRAM.** More hot copies
  = more covered decode bytes but more resident VRAM working against prefill.
- **Prompt-heavy workloads (agents with big tool contexts, long-document QA): lower
  `HOT_N` or leave the split disabled.** Generation-heavy chat is where it pays.
- Whether the prefill penalty scales with hot-copy size (`HOT_N` ∈ {48, 96, 192}) has
  **not been swept yet** — treat the −9.9% as the measured point at HOT_N=96, not a
  universal constant. Prefill-aware hot-copy management is on the roadmap.

---

## How it works

Three pieces, all opt-in and decode-only.

**1. Offline profiler (`llama-aipc-moe-profile`).** A small eval-callback tool that
counts per-(layer, expert) selections from the `ffn_moe_topk` tensor while decoding a
corpus, then writes a hot-list: for each layer, expert ids sorted by hit count. Run it
offline over representative text — a generic corpus works, but a **short history of your
own recent sessions transfers far better** (see the ladder below). CPU-only; no changes
to ggml.

**2. Load-time hot copies + registry.** When `AIPC_MOE_HOT_LIST` and `AIPC_MOE_HOT_N`
are set, then for each RAM-resident MoE layer the top-N expert ids are validated and
deduped, and a contiguous VRAM tensor of shape `[ne0, ne1, n_hot + n_pad]` is allocated
per projection (up/gate/down). The hot rows are copied in; `n_pad` extra slots (one per
`n_expert_used` k-position) are explicitly zeroed. The split is registered in a
mutex-guarded global map keyed by the source expert tensor, and cleared on every model
load (stale entries would be a use-after-free).

**3. Dual-chain graph split.** At decode, `build_moe_ffn` builds two full FFN chains
(up/gate/act/down) per side and merges once per layer:

```
             ┌─────────────── hot experts (VRAM copy) ──┐
  router ──▶ │  up → gate → act → down                  │
   topk      └──────────────────────────────────────────┘
             ┌─────────────── cold experts (RAM) ───────┐   experts =
             │  up → gate → act → down                  │──▶ cold + (hot − cold) * mask
             └──────────────────────────────────────────┘   (one merge per layer)
```

Hot-side cold positions route to per-position zeroed dummy slots (so every expert id
stays unique per token — the CUDA `mul_mat_id` kernels require this — and the zeroed
slots avoid `0*Inf = NaN` in the merge); cold-side hot positions are clamped to expert 0
so the hot bytes are **not** re-read from RAM. Any unsupported shape returns `nullptr`
and the stock batched path runs untouched.

**Gating conditions (all must hold, else fall back):** decode / small batch only
(`n_tokens ≤ 8`); gated SwiGLU with no per-expert bias/scale (`SWIGLU_OAI` / gpt-oss
fall back); no active expert LoRAs; a split registered for up/gate/down.

---

## Quickstart

You need a CUDA build of the patched fork. The measured numbers used a Blackwell
(sm_120) build on Windows; a generic Linux/CUDA recipe is also given.

**Build (Linux / CUDA, mainline):**

```bash
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release \
  --target llama-bench llama-server llama-aipc-moe-profile -j
```

**Build (the exact Windows / Blackwell recipe these numbers were taken with):**

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T cuda=13.0 \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
cmake --build build --config Release \
  --target llama-bench llama-server llama-aipc-moe-profile -j 16
```

> The toolset **must** be `-T cuda=13.0`. Under nvcc 13.2 with MSVC 14.44 the CUDA
> stubs break with `C4003`. See [BUILD_WINDOWS.md](BUILD_WINDOWS.md) for the workaround.

**Profile → hot-list → serve (A/B):**

```bash
# 1) build a hot-list from representative text (a short session history transfers best)
llama-aipc-moe-profile -m Coder-Next.gguf -f session_history.txt -ngl 999 --n-cpu-moe 24
mv aipc_moe_profile.hotlist session.hotlist

# 2a) baseline (split OFF): env unset
llama-server -m Coder-Next.gguf -ngl 999 --n-cpu-moe 24 --no-mmap --poll 100 -c 8192 -t 16 --jinja

# 2b) split ON
AIPC_MOE_HOT_LIST=session.hotlist AIPC_MOE_HOT_N=96 \
llama-server -m Coder-Next.gguf -ngl 999 --n-cpu-moe 24 --no-mmap --poll 100 -c 8192 -t 16 --jinja
# metric: timings.predicted_per_second on a temp-0 request; compare 2a vs 2b
```

The Python `autopilot/` tooling wraps the profiler and computes coverage — see
[autopilot/README.md](autopilot/README.md). `AIPC_MOE_HOT_N` sets N (96 works well for
512-expert models; 48 for 256-expert). Full reproduction with exact evidence file
mappings is in [README_REPRO.md](README_REPRO.md).

**Two machine-state rules that matter (measured, not folklore):**
- **`--no-mmap` is mandatory** in measurements — a cold NVMe cache drops throughput ~5×.
- **`--poll 100` everywhere** — the default poll=50 costs 8–14% and drifts with machine
  state.

---

## The coverage → speed ladder

The whole mechanism rests on one relationship: **decode speed rises roughly linearly
with how much of a live task's expert traffic your hot-list covers.** On the primary
model at `--n-cpu-moe 24`:

| Hot-list source | top-96/512 coverage | decode tok/s |
|---|---|---|
| none | 0% | 72.1 |
| generic corpus | 33.5% | 77.1 |
| session history (task-disjoint) | 68.6% | 90.5–91.5 band |
| same-task (circular ceiling) | 80.4% | 92.3 |

Two things fall out of this:

1. **Routing skew is real but moderate**: ~80% of token-level expert hits land in
   ~28–31% of the experts on these models. That is enough for a large win but not a
   near-total one — the tail is always touched, so full eviction is never free.
2. **The win is adaptive, per-session hot-lists.** A generic corpus only covers ~33% of
   a live task's hits (→ +6.9%). A short history of your *own* recent sessions transfers
   at ~65–69% (→ +26%). The mechanism is only as good as the hot-list, and the hot-list
   is only good when it looks like what you are about to do.

A bandwidth model (expert gather ≈ 42 GB/s on this machine, plus a ~1 ms/token fixed
cost with the hot part in VRAM) predicts the measured points within ~3–5%.

---

## The async-overlap negative result (reported because it is useful)

The remaining gap after the static split is a fixed per-layer dispatch floor of the cold
chain. The obvious next move is to hide the per-layer CPU cold-chain dispatch behind the
GPU hot-chain work — on paper `max(cold, hot)` per layer instead of `cold + hot`. **It
does not pan out at scheduler granularity**, for a structural reason: splits of one
graph execute strictly sequentially in `ggml_backend_sched_compute_splits`, and the CPU
backend has no async compute path and no event support, so any overlap has to be
orchestrated at a coarser grain than the scheduler already provides. Two direct
measurements:

- **Forcing an extra split boundary costs +0.98 ms/token** (~41 µs per boundary-pair ×
  24 offloaded layers): a boundary between two CUDA regions requires inserting a
  non-CUDA node — exactly what carving out a landing spot for an async join costs,
  before any overlap benefit.
- **Async dispatch of the cold split at scheduler level is net −3.4%**, with **0% early
  completions** (`already_done = 0` of 9600 joins): the scheduler thread never found the
  cold chain already finished at the join point; there is no GPU work in flight to
  overlap against at that point in the graph structure.

Conclusion: overlap would need graph-partition granularity (attention of layer L+1 + the
hot chain as one cgraph running in parallel with the cold chain of layer L as a separate
cgraph), not scheduler-level tweaks. That is why the shipped feature is the static split
alone, and it is the design that the V3 roadmap item picks up (see below).

---

## Implementation gotchas (documented so you do not re-hit them)

1. **Merge once per layer, not once per projection.** Per-projection merging (hot/cold
   per matmul) creates CPU↔GPU ping-pong and measured **3.6× slower** than baseline.
   Full per-side chains with one merge per layer fix it.
2. **Clamp cold-side ids away from hot experts** (or you re-read the same RAM bytes and
   save nothing); the **hot-side clamp ids must stay unique per token** or the CUDA
   `mul_mat_id` batch kernels misbehave. Per-position zeroed dummy slots solve both at
   once.
3. **Zero the dummy slots.** Uninitialized hot slots leak NaN/Inf through the masked
   merge (`0×Inf = NaN`). Validate/dedupe the hot-list at load and zero the dummy slots
   explicitly.
4. **(Environment, not code)** On Windows, keep working VRAM + system baseline
   ≤ ~30 GB or **WDDM paging silently inverts your results.** This rule cost us two
   false "discoveries" before we pinned it down.

---

## Limitations

- **Decode-only** (`n_tokens ≤ 8`); prefill keeps the batched path for routing — but see
  the measured **~10% prefill throughput cost** of the resident hot copies above. The
  split is a net end-to-end win only when generated tokens dominate the prompt
  (break-even ≈ prompt:output 2.3 on this config).
- **Gated-SwiGLU only**, no per-expert bias/scale — `SWIGLU_OAI` / gpt-oss fall back
  silently. Extending the chain to those shapes is small and is the prerequisite for a
  second supported model family.
- **One model per process** (global registry, mutex + clear on load); concurrent
  multi-model in a single process is out of scope.
- **Hot-list swap requires a server restart** — no dynamic re-upload yet.
- Active expert LoRAs disable the split (consistency).
- Hot-lists are **model-specific** and **workload-dependent**.
- **CUDA-tested only.** The split composes existing ggml ops
  (`get_rows`/`cast`/`mul`/`add`/`mul_mat_id`) and should run on CPU, but it has only
  been measured on CUDA, and the unique-id invariant it relies on is a CUDA
  `mul_mat_id` property; the CPU path needs validation.

---

## Roadmap / planned work

- **CLI flags instead of env vars.** `AIPC_MOE_HOT_LIST` / `AIPC_MOE_HOT_N` should
  become proper `common_params` flags (e.g. `--moe-hot-list PATH`, `--moe-hot-n N`) with
  `arg.cpp` wiring. Env vars were the fastest way to measure the idea.
- **CPU-path validation + automated tests.** A targeted `test-backend-ops`-style
  split-vs-baseline output-equality test at temp 0 on a tiny MoE; validation of the
  merge/id-clamp logic on the CPU backend.
- **Prefill-aware hot-copy management.** The resident hot copies cost ~10% prefill
  throughput at HOT_N=96 (measured, cause isolated — see the prefill section). Planned
  work: sweep `HOT_N` ∈ {48, 96, 192} to quantify how the penalty scales with resident
  size, and investigate managing the copies around large-batch prefill (freeing or
  right-sizing them for prompt-heavy workloads) so arming the split stops being a
  prompt-shape gamble.
- **V3 — true async overlap of the cold expert chain: investigated end-to-end and
  closed as a measured negative on Windows.** The bespoke single-graph split executor
  described in [docs/15_v3_overlap_design.md](docs/15_v3_overlap_design.md) was built and
  spike-tested through three stages: the bare fork/join glue passed its own kill
  criterion with a large margin (~11 µs/layer vs a 20 µs bar), proving the earlier
  scheduler-marker tax was an artifact, not fundamental — but wiring it into one real
  offloaded layer broke CUDA-graph capture on the merge region, and a follow-up spike
  building two different capture-preserving wait gates found that **every in-graph wait
  node pays a ~23-29 µs WDDM submission-split tax that cancels the ~96 µs/layer overlap
  saving**, regardless of which of the three independent mechanisms carries it. Net
  effect: −3 tok/s on the incumbent, consistently, across all three. Full design +
  spike-by-spike evidence in docs/15. The mechanism is plausibly viable on native Linux
  (the same wait op is architecturally expected to be sub-µs there) — **this is an
  untested hypothesis**, not a claim; a Windows-hosted Linux compatibility layer would
  not qualify as a test, since its CUDA path still crosses the host WDDM driver.

---

## AI-assistance disclosure

This project was built with **heavy AI assistance** (Anthropic's Claude). The mechanism,
the code, the profiler, the experiment design, and this writeup were developed in close
collaboration with the model. What that assistance did **not** change:

- **Every measurement was taken on the author's own hardware** (RTX 5090 32 GB + Ryzen
  9950X3D + 64 GB DDR5-6000, Windows 11), by the author, under the machine-state
  discipline documented above.
- **All direction, decisions, and the standards the results are held to are the
  author's.** The method rules that saved the project (never promote a sweep number
  without a standalone confirmation at a known VRAM baseline; adversarial code review
  before promoting any C++ patch; the WDDM headroom rule) are enforced throughout, and
  the author stands behind every number and can explain every line.

This is disclosed openly and up front rather than buried, because honesty about how the
work was produced is part of the work.

---

## License

- **MIT** for the tooling in this repository (the profiler wrappers under `autopilot/`,
  the docs, and the build notes).
- The llama.cpp **fork branch** that carries the C++ split changes inherits **llama.cpp's
  MIT license** and its copyright notices — those changes live on top of upstream
  `ggml-org/llama.cpp` and are offered under the same terms.

---

*Author: JigSawPT. Built on top of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
Related upstream discussion: [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757).*
