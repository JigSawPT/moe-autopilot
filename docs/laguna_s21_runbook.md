# Laguna S 2.1 (118B MoE) on a single RTX 5090 — runbook & moe-autopilot framing

**What this is:** the modest-hardware recipe for **Laguna S 2.1** (118B total / ~8B
active MoE, 256 experts **top-10** + shared, **hybrid sliding-window attention**,
~75 GB at Q4_K_M) on **a single RTX 5090 (32 GB) + ~64 GB RAM** — this project's
reference-class box. It is built on the community gist (`poolside_118B_5090RTX.md`,
poolside's `laguna` llama.cpp fork), which is the *single-GPU / modest-hardware* path.
The separate 2× 5090 / 256 GB DFlash write-up is a heavier alternative, kept in §3.

> **Correction from this doc's first draft (honesty note):** I initially cut context to
> 16384 assuming a 256k KV cache wouldn't fit in 64 GB. **Wrong for this architecture.**
> Laguna uses **hybrid SWA** (36/48 layers are 512-token sliding-window, 8 KV heads), so
> **256k context is cheap on KV** and the gist runs `-c 262144` on 62 GB RAM. The RAM
> pressure here is the **~75 GB of resident model weights** under `--no-mmap`, relieved
> by **swap**, not the KV cache. The runbook below reflects that.

---

## The one idea that matters: pack full layers, don't exile all experts

The gist's core finding, and the reason this model belongs in *our* project:

- **Slow path — all experts on host** (`-ngl 999 --cpu-moe`): fits in ~18 GB VRAM but is
  host-bound. Measured **1.3 t/s prefill, 12.8 t/s warm decode**.
- **Fast path — pack full layers into VRAM** (`-ngl auto --fit on`): fills free VRAM with
  as many *whole* MoE layers (experts included) as fit, leaves the rest on host.
  Measured **~58 t/s prefill, ~18–19 t/s decode**, ~30 GB VRAM. **~40–45× prefill,
  ~1.5–2× decode.**

**`--fit` is a coarse hot/cold split.** It gives each layer VRAM residency *all-or-nothing*
— the first K layers 100% on-GPU, the rest 100% on host. `moe-autopilot` is the
*fine-grained* version of the same idea: keep, on **every** offloaded layer, its **top-N
hottest experts** in VRAM. For a fixed VRAM budget on concentrated routing, per-layer hot
coverage should beat whole-layer packing, because `--fit`'s exiled layers dump their *hot*
experts to host along with the cold ones. **That comparison — `--fit` layer-packing vs.
our per-layer hot split, same VRAM budget, same model — is the contribution this model
enables.** (See §4.)

---

## 0. Build (poolside `laguna` fork, CUDA / SM_120)

Stock mainline may not load `architecture = laguna` yet — use the fork.

```bash
git clone --branch laguna --single-branch https://github.com/poolsideai/llama.cpp.git
cd llama.cpp
# GCC 15 gotcha: if the build fails in common/speculative.cpp on std::isfinite,
# add  #include <cmath>  at the top of that file.
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-server
export LD_LIBRARY_PATH="$PWD/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

hf download poolside/Laguna-S-2.1-GGUF laguna-s-2.1-Q4_K_M.gguf --local-dir ./Laguna-S-2.1-GGUF
```

**Swap it before you run.** The gist's box is 62 GB RAM **+ 128 GB swap**; the ~75 GB of
`--no-mmap`-resident weights minus what `--fit` puts on the GPU (~30 GB) means ~45 GB
wants host residency. With 64 GB RAM you *will* touch swap (peak ~22 GB on the fast path).
Make sure a large swapfile exists on NVMe, or the first load thrashes.

---

## 1. Run — the fast path (use this)

```bash
llama-server -m laguna-s-2.1-Q4_K_M.gguf \
  -c 262144 \
  -ngl auto --fit on --fit-target 2048 \
  -fa on --jinja --no-mmap --poll 100 \
  -t "$(nproc)" -b 4096 -ub 4096 \
  -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-p 0.95 --top-k 20 \
  --host 127.0.0.1 --port 8095
```

- **`-ngl auto --fit on`** is the whole game. **Do NOT** set `-ngl 999` without
  `--cpu-moe` — that tries to put all ~75 GB on the GPU and OOMs.
- **`--fit-target 2048`** leaves ~2 GB VRAM for display/driver headroom (desktop safety).
- **`-c 262144`** is fine here (hybrid SWA). Lower it only if you want RAM/swap back.
- **`--no-mmap --poll 100`** — this project's non-negotiables (cold NVMe cache ~5× slower;
  `--poll 50` default costs 8–14%).
- **`-b/-ub 4096`** — prefill batching matters a lot for prompt throughput; keep it high.
- Only added over the gist: **`--poll 100`** (ours) and a loopback host.

**Windows/PowerShell:** same flags; set env with `$env:NAME="value"`. WDDM rule applies —
keep working VRAM + baseline ≤ ~30 GB or the driver pages silently and inverts results.

### Smoke test

```bash
curl -sS http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write merge_sorted(a,b) in Python, O(n+m). Code only."}],
       "temperature":0.2,"max_tokens":400,"chat_template_kwargs":{"enable_thinking":false}}' \
  | jq -r '.choices[0].message.content, .timings'

watch -n1 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv; free -h'
```

**Healthy:** ~30 GB VRAM, swap ~20 GB (not 60+), decode ~15–20 t/s. First load can take
~1 min (page-in) before `/v1/models` is ready even if `/health` answers early. Disable
thinking for latency tests (`enable_thinking:false`) — it burns hundreds–thousands of
hidden tokens first.

**Honest anchor for your box: ~18–19 tok/s warm decode, ~58 t/s prefill.** Below that on
decode → you're paging harder than expected (check swap, lower `-c`, free VRAM baseline).

### More modest cards (4090 / 3090, 24 GB) — the "even more modest" tier

Same method, fewer full layers land on the GPU: prefer `-ngl auto --fit` over `--cpu-moe`
whenever *anything* fits, expect results closer to the slow column but still better than
all-experts-host, and never force `-ngl 999` without `--cpu-moe`. NVFP4 (~71 GB) is a
multi-GPU / 128 GB-unified recipe — it does **not** fit a 32 GB card full-GPU; **GGUF
hybrid is the single-consumer-GPU path.**

---

## 2. Measured reference (gist's box: 5090 32 GB + 62 GB RAM + 128 GB swap)

| Mode | VRAM | Swap peak | Prefill (cold) | Decode (cold) | Decode (warm) |
|------|------|-----------|----------------|---------------|---------------|
| `--cpu-moe` + `-ngl 999` | ~18 GB | ~60 GB | 1.3 t/s | 8.2 t/s | 12.8 t/s |
| **`-ngl auto` + `--fit`** | **~30 GB** | **~22 GB** | **~58 t/s** | **~19 t/s** | **~18 t/s** |

Not our numbers — recalibrate on your machine. Agent UIs with 5–10k-token system prompts
still cost 1–3 min to first token cold at ~58 t/s prefill; that's page-in, not a hang.

---

## 3. DFlash speculative decode (optional, heavier — expect break-even here)

The 2× 5090 / 256 GB write-up used a *different* placement — experts spilled to CPU RAM
(`--cpu-moe`-style partial offload) plus a DFlash draft model. Its lesson: on fine-grained
MoE with partial offload, **verify cost scales with batch size** (a 16-token verify batch
touches up to 160 experts/layer, streaming the CPU-resident ones from RAM, ~6.8 ms per
extra verify token), so spec-decode barely wins. On the single-GPU `--fit` path above the
economics differ, but if you try it:

```bash
# quantize the drafter BF16 -> Q8_0 first (same acceptance, ~1 GB VRAM back):
build/bin/llama-quantize laguna-s-2.1-DFlash-BF16.gguf laguna-s-2.1-DFlash-Q8_0.gguf Q8_0
# then add to the §1 command:
  -md laguna-s-2.1-DFlash-Q8_0.gguf --spec-type draft-dflash \
  --spec-draft-n-max 7 --spec-draft-p-min 0.75
```

- **`--spec-draft-p-min 0.75`** is the knob nobody sets — the **0.00 default ships all N
  draft tokens regardless of confidence** and craters acceptance to ~10%. Draft only when
  confident. Single-GPU wants it stricter (0.75) than the dual-GPU run (0.6): pricier
  verify tokens → draft more selectively.
- Per Spec-Bench it wins on formulaic text (math +20–25%, translation +18–20%) and is
  **parity to −9% on RAG/QA/summarization** — "copyable ≠ draftable." **A/B on your own
  prompts; keep it only if it beats §1.** Default recommendation: run §1 without a drafter.

---

## 4. Why this model earns a place in the project — the experiment

`--fit` (§1) proves layer-packing beats expert-exile with a *coarse* all-or-nothing knob.
The project's thesis is that a *fine-grained* per-layer hot split does better at the same
VRAM budget when routing is concentrated. Laguna (256 experts, top-10, ~75 GB) is a clean
testbed. Ladder, cheapest → most work:

1. **Routing skew — offline, minutes, do this first.** Run `analyze_skew.py` on a Laguna
   imatrix: do a small slice of the 256 experts/layer cover ~80% of hits (concentrated,
   like Coder-Next → the split pays) or is it flat (like gpt-oss, +2.3% only → it won't)?
   ```bash
   python autopilot/analyze_skew.py <laguna-imatrix.gguf> \
     --model laguna-s-2.1-Q4_K_M.gguf --out laguna_skew.md
   ```
2. **Base anchor (§1)** — establish `--fit`'s ~18–19 tok/s on your box, and record how many
   full layers `--fit` actually placed (that is the coarse baseline to beat).
3. **Split-on-Laguna** — only if (1) says concentrated: port the hot/cold dual-chain onto
   the poolside `laguna` fork, give it the *same* VRAM budget `--fit` used, and compare
   decode. The delta at equal VRAM is the result.

**Porting caveats (state them plainly in any #20757 comment):** our split lives on
`aipc-hardening`; Laguna + its arch live on poolside's `laguna` fork — combining means
porting one onto the other. **Laguna's FFN activation and its shared-expert + hybrid-SWA
structure are unverified against our split** (we cover plain SwiGLU and `SWIGLU_OAI`; a
third variant falls back until threaded, as with gpt-oss). Confirm the activation and how
the shared expert interacts with the hot copies before promising a combined build.
