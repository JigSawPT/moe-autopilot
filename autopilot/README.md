# autopilot — profiling & coverage tooling for the hot/cold MoE split

Two small tools that wrap the patched fork's profiler and turn expert-activation counts
into hot-lists and a coverage number. These are the offline half of the workflow; the
C++ split (the online half) lives in the fork branch.

- **`autopilot.py`** — build hot-lists from your own workload text, compute
  coverage of a hot-list against a live task, and (secondarily) plan/measure a static
  `--n-cpu-moe` placement.
- **`analyze_skew.py`** — read a llama.cpp GGUF imatrix and report per-layer expert
  utilization skew (how concentrated routing is, and the VRAM cost of the hot-set).

Requires `pip install gguf numpy`, and a built fork (see the top-level
[README](../README.md) and [BUILD_WINDOWS.md](../BUILD_WINDOWS.md)).

## Pointing the tools at your build

`autopilot.py` locates the fork binaries via **`AIPC_BIN_DIR`**:

```bash
export AIPC_BIN_DIR=/path/to/llama.cpp/build/bin                    # Linux
# or, PowerShell:
$env:AIPC_BIN_DIR = "C:\path\to\llama.cpp\build\bin\Release"        # Windows
```

It looks for `llama-server`, `llama-bench`, and `llama-aipc-moe-profile` there.

## Build a hot-list from your workload

```bash
python autopilot.py hotlist --model <model.gguf> \
  --from my_recent_session.txt [more.txt ...] --out session.hotlist
```

This runs the profiler over the concatenated text and writes `session.hotlist` (the
`"layer id id ..."` file the loader reads) plus `session.json` (per-layer counts). A
**short history of your own recent sessions transfers far better than a generic corpus** —
that is the whole point of the adaptive hot-list (see the coverage→speed ladder in the
top-level README). Then:

```bash
export AIPC_MOE_HOT_LIST=session.hotlist AIPC_MOE_HOT_N=96   # 96 for 512-expert; 48 for 256-expert
llama-server -m <model.gguf> -ngl 999 --n-cpu-moe 24 --no-mmap --poll 100 -c 8192 -t 16 --jinja
```

## Measure coverage (the x-axis of the ladder)

Coverage is the fraction of a live task's expert hits that fall within the top-N of a
hot-list. Profile the *live* task once to get its counts, then:

```bash
python autopilot.py coverage --hotlist session.hotlist --profile live.json --hot-n 96
# -> top-96 coverage: mean 68.6% | min 39.7% | max ... | layers 24
```

This is the single canonical coverage script — every "coverage" column in the results
tables comes from it.

## Static placement planner (secondary)

Independent of the hot/cold split, `plan` / `validate` compute a static
`--n-cpu-moe` placement for a VRAM budget using a bandwidth cost model, and `validate`
measures the plan and its neighbors with `llama-bench` to pick a winner with data:

```bash
python autopilot.py plan     --model <model.gguf> --vram-budget-gb 28
python autopilot.py validate --model <model.gguf> --vram-budget-gb 28
```

The cost-model constants (`RAM_BW ≈ 42 GB/s` effective expert gather, `VRAM_BW ≈
1.05 TB/s`, `~1 ms/token` fixed overhead) are calibrated on the author's machine and
**must be recalibrated** on other hardware. On the reference machine the planner's
predictions land within ~5% of measured.

## Routing skew

```bash
python analyze_skew.py <imatrix.gguf> --model <model.gguf> --out skew.md
```

Given a llama.cpp GGUF imatrix (which carries `blk.N.ffn_*_exps.weight.counts`), this
reports, per layer, how many experts account for 80/90/95% of activations, and — with
`--model` — the VRAM cost of each hot-set. On the primary model, ~80% of hits land in
~28% of experts per layer; skew varies by layer, which is why the hot-list budget is
per-layer.

## Machine-state rules (they dominate the numbers)

- **`--no-mmap`** is mandatory in measurements — a cold NVMe cache drops throughput ~5×.
- **`--poll 100`** everywhere — the default (50) costs 8–14% and drifts with machine
  state.
- **Working VRAM + system baseline ≤ ~30 GB** on Windows — above that WDDM pages
  silently and can *invert* results. `autopilot.py` reads the current baseline via
  `nvidia-smi` when planning.
