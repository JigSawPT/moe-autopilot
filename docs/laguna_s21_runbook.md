# Laguna S 2.1 (118B MoE) on a single RTX 5090 + 64 GB — runbook & moe-autopilot framing

**What this is:** a ready-to-run recipe for **Laguna S 2.1** (118B MoE, 8B active,
256 routed + 1 shared, **top-10** routing, ~71 GB at Q4_K_M) on **this project's
reference machine — a *single* RTX 5090 (32 GB) + Ryzen 9950X3D + 64 GB DDR5-6000**,
plus the reason this model is the sharpest test case we have for the hot/cold split.

It fuses two sources:

- the **launch recipe** from the community gist (`poolside_118B_5090RTX.md`,
  poolside's `laguna` llama.cpp fork), and
- the **DFlash speculative-decode tuning lessons** from the "23 → 64 tok/s" write-up
  (2× 5090 / 256 GB Xeon).

> **Hardware honesty up front — your box is not the write-up's box.** The DFlash
> write-up ran **2× 5090 (64 GB VRAM) + a Xeon w5-3423 with 256 GB of *8-channel*
> RAM**. You have **1× 5090 (32 GB VRAM) + 64 GB of *dual-channel* DDR5-6000**. Two
> differences dominate everything below:
> 1. **~41 GB of experts must live in system RAM** (71 GB model − ~30 GB usable VRAM),
>    vs. everything-in-VRAM on the dual-GPU rig.
> 2. **Your RAM bandwidth is the bottleneck.** Our own calibration puts effective
>    sparse expert-gather at **~42 GB/s** on this dual-channel DDR5-6000 (see
>    `autopilot/autopilot.py` constants). The Xeon's 8-channel bus is several times
>    wider. Decode here is **RAM-bandwidth-bound**, and no spec-decode knob changes that.
>
> **Realistic base anchor: ~19 tok/s decode** (the gist's own single-5090 + 50–64 GB
> figure), **not** the write-up's 30–33 tok/s (wider memory) and nowhere near its
> 62 tok/s (everything in VRAM). Recalibrate; do not quote these as ours.

---

## 0. Prereqs (~90 GB free, NVMe)

```bash
# poolside laguna fork (has the Laguna arch + DFlash spec path)
git clone -b laguna https://github.com/poolsideai/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=ON       # 120 = SM_120 / Blackwell / 5090
cmake --build llama.cpp/build --config Release -j

# weights (~71 GB) + optional drafter (~2.2 GB BF16)
hf download poolside/Laguna-S-2.1-GGUF laguna-s-2.1-Q4_K_M.gguf
hf download poolside/Laguna-S-2.1-GGUF laguna-s-2.1-DFlash-BF16.gguf   # only if you'll try spec-decode
```

---

## 1. Base path first (get this solid before any drafter)

The gist's command is written for a 256 GB / dual-GPU rig — its `-c 262144` context is
**impossible** in your 64 GB budget (after ~41 GB of experts + dense + compute buffers
you have ~15–20 GB of RAM left, and a 118B KV cache is not free). Start small and grow:

```bash
llama-server -m laguna-s-2.1-Q4_K_M.gguf \
  -ngl auto --fit on --fit-target 2048 \
  -c 16384 -ctk q8_0 -ctv q8_0 \
  -fa on --jinja --no-mmap --poll 100 \
  -t 16 -b 2048 -ub 2048 \
  --temp 0.7 --top-p 0.95 --top-k 20 \
  --host 127.0.0.1 --port 8095
```

Deltas from the gist, and why:

- **`-c 16384`** (was 262144) — fit the KV cache in what's left of 64 GB. Raise toward
  32768 only after you confirm headroom (watch RAM; if the OS starts paging, decode
  falls off a cliff exactly like the WDDM VRAM cliff we document elsewhere).
- **`--no-mmap --poll 100`** — this project's non-negotiables. A cold NVMe cache drops
  throughput ~5×; `--poll 50` (the default) costs 8–14%. See `autopilot/README.md`.
- **`-b/-ub 2048`** (was 4096) — smaller compute buffers leave more RAM for experts.
- **`--fit-target 2048`** — hold VRAM headroom open (matters once the drafter arrives).

**Windows/PowerShell:** same flags; set env vars with `$env:NAME="value"` instead of
`NAME=value`. The WDDM rule applies — keep **working VRAM + system baseline ≤ ~30 GB**
or the driver pages silently and *inverts* results (`autopilot.py` R3).

**Sanity target:** land near **~19 tok/s** decode. Below ~12 → you are paging (drop
`-c`, close VRAM-hungry apps, re-check the baseline). This base number is the honest
headline for your machine; treat everything below as "can we beat it."

---

## 2. DFlash speculative decode — expect *break-even at best* here

The write-up's hard-won lesson is that on **fine-grained MoE with partial CPU offload**,
spec-decode barely wins, because **verify cost scales with batch size**: one token
routes to 10 experts/layer, so a 16-token verify batch touches up to **160** experts/
layer and streams every CPU-resident one from RAM (**~6.8 ms per extra verify token**,
measured). That tax lands squarely on *your* configuration — you have ~41 GB of experts
in RAM. So: try it, but measure against §1, don't assume a win.

Two setup steps that the write-up proved matter:

```bash
# (a) quantize the drafter BF16 -> Q8_0: same acceptance, ~1 GB of VRAM back for experts
llama.cpp/build/bin/llama-quantize \
  laguna-s-2.1-DFlash-BF16.gguf laguna-s-2.1-DFlash-Q8_0.gguf Q8_0
```

```bash
# (b) launch with the SINGLE-GPU tuning (stricter p-min than the dual-GPU run)
llama-server -m laguna-s-2.1-Q4_K_M.gguf \
  -md laguna-s-2.1-DFlash-Q8_0.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 7 \
  --spec-draft-p-min 0.75 \
  -ngl auto --fit on --fit-target 2048 \
  -c 16384 -ctk q8_0 -ctv q8_0 \
  -fa on --jinja --no-mmap --poll 100 \
  -t 16 -b 2048 -ub 2048 \
  --host 127.0.0.1 --port 8095
```

The two knobs that are the whole ballgame (write-up's words, and they match our own
"residency tax" mechanics):

- **`--spec-draft-p-min 0.75`** — the default is **0.00**, which ships all N draft
  tokens every round regardless of confidence (acceptance cratered to ~10%). Drafting
  *only when confident* took acceptance to ~73% on the dual-GPU rig. On your slower,
  RAM-bound step you want to draft **even more selectively** (hence 0.75 > the 0.6 the
  write-up used on 2× GPU): every verify token you don't need is pure RAM-streaming cost.
- **`--spec-draft-n-max 7`** — draft short. Long drafts just inflate the verify batch,
  and on partial offload the verify batch is the expensive part.

**Per-workload reality (from the write-up's Spec-Bench run):** the drafter wins on
*formulaic* text (math +20–25%, translation +18–20%, boilerplate) and is **parity to
−9% on RAG / QA / summarization** — "copyable ≠ draftable." If your agentic workload is
retrieval/summary-heavy, DFlash likely costs you here. **Default recommendation for this
box: run §1 without the drafter, and only keep DFlash if your own A/B on your own prompts
shows a real gain.**

---

## 3. Why this model is *our* fight — the hot/cold split as the missing lever

This is the part worth taking to issue #20757. The write-up's central bottleneck —
**"the CPU-resident experts make every verification batch expensive"** — is a restatement
of the exact problem `moe-autopilot` exists to attack. Our hot/cold split keeps the
**top-N most-activated experts per offloaded layer resident in VRAM**. Applied to a verify
batch:

- A verify batch of *B* tokens touches up to *B×10* experts/layer.
- If a hot-list covers fraction *c* of the **weighted** expert traffic, only ~(1−*c*) of
  those touches still stream from RAM.
- So the split should cut the per-verify-token RAM cost by ~*c*. The write-up measured
  **~6.8 ms/extra-verify-token at c = 0** (plain `--n-cpu-moe`, no hot copies). At, say,
  ~60% weighted coverage you'd predict that toward **~2.7 ms** — which is precisely the
  lever that could flip DFlash on this model from "breaks even" to "actually wins."

And critically, **your dual-channel machine is where this matters *most*.** The write-up's
Xeon barely feels RAM streaming (8-channel); you feel it at ~42 GB/s. The more
bandwidth-starved the box, the more moving hot experts into VRAM buys — the split is a
bigger lever on *your* hardware than on the author's.

**But it only pays if Laguna's routing is concentrated, not flat** — the Coder-Next
(pays) vs gpt-oss (flat, +2.3% only) result from the top-level README. So the honest
first step is a **cheap, offline** check, no 71 GB run required:

```bash
# does Laguna concentrate its routing? (needs a GGUF imatrix carrying expert counts)
python autopilot/analyze_skew.py <laguna-imatrix.gguf> \
  --model laguna-s-2.1-Q4_K_M.gguf --out laguna_skew.md
# read: how many of the 256 experts/layer cover 80/90/95% of hits, and the VRAM cost
```

- **Concentrated** (a small slice of the 256 covers ~80%) → the split is worth porting
  onto the poolside `laguna` fork, and the DFlash verify-cost experiment above is a real
  paper-worthy result.
- **Flat** → report that honestly and move on; DFlash tuning (§2) is the only lever left.

**Porting caveat (state it plainly in any issue comment):** our split lives on the
`aipc-hardening` fork; DFlash + the Laguna arch live on poolside's `laguna` fork. Combining
them means porting the hot/cold dual-chain onto `laguna` (or Laguna support onto ours).
Also unverified: **Laguna's FFN activation** — our split covers plain SwiGLU and
`SWIGLU_OAI`; if Laguna uses a third variant it falls back until threaded, same as the
gpt-oss work. Confirm the activation before promising a combined build.

---

## 4. Experiment ladder (cheapest → most work)

1. **Routing skew** (`analyze_skew.py`) — offline, minutes. Decides if any of this is
   worth it. *Do this first.*
2. **Base path A/B** (§1) — establish the honest ~19 tok/s anchor on your box.
3. **DFlash A/B on your own prompts** (§2) — keep only if it beats §1 on *your* workload.
4. **Split-on-Laguna** — only if (1) says concentrated: port the dual-chain onto the
   `laguna` fork, then measure the per-verify-token RAM cost with the split armed vs. off.
   That delta is the contribution.
