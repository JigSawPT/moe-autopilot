#!/usr/bin/env bash
# run_laguna_s21.sh — launch Laguna S 2.1 (118B MoE, ~75 GB Q4) on a SINGLE RTX 5090
# (32 GB) + ~64 GB RAM, via the gist's fast path: -ngl auto --fit (pack full MoE layers
# into VRAM, rest on host). NOT --cpu-moe (that exiles all experts to host: 1.3 t/s prefill).
# See docs/laguna_s21_runbook.md for reasoning and honest numbers (~18-19 tok/s warm decode).
#
# Laguna has hybrid sliding-window attention, so -c 262144 is cheap on KV. The RAM pressure
# is the ~75 GB of --no-mmap-resident weights minus what --fit puts on the GPU -> ensure a
# large NVMe SWAP file exists (the gist's box is 62 GB RAM + 128 GB swap; peak ~22 GB swap).
#
# Machine-state rules baked in: --no-mmap (cold NVMe cache ~5x slower); --poll 100
# (default 50 costs 8-14%). Windows/WDDM: keep working VRAM + baseline <= ~30 GB.
set -euo pipefail

MODEL="${MODEL:-laguna-s-2.1-Q4_K_M.gguf}"
DRAFT="${DRAFT:-laguna-s-2.1-DFlash-Q8_0.gguf}"   # optional; quantize BF16->Q8_0 first (runbook §3)
BIN="${AIPC_BIN_DIR:-./llama.cpp/build/bin}"
MODE="${MODE:-base}"                              # base | spec
CTX="${CTX:-262144}"                              # SWA makes 256k cheap; lower to reclaim RAM/swap
PORT="${PORT:-8095}"
SPEC_N_MAX="${SPEC_N_MAX:-7}"
SPEC_P_MIN="${SPEC_P_MIN:-0.75}"                  # footgun: default 0.00 ships all N regardless of confidence

common=(
  -m "$MODEL"
  -c "$CTX"
  -ngl auto --fit on --fit-target 2048           # DO NOT use -ngl 999 without --cpu-moe: OOMs (tries all 75 GB on GPU)
  -fa on --jinja --no-mmap --poll 100
  -t "$(nproc)" -b 4096 -ub 4096
  -ctk q8_0 -ctv q8_0
  --temp 0.7 --top-p 0.95 --top-k 20
  --host 127.0.0.1 --port "$PORT"
)

if [[ "$MODE" == "spec" ]]; then
  [[ -f "$DRAFT" ]] || { echo "drafter '$DRAFT' not found — quantize it first:"; \
    echo "  $BIN/llama-quantize laguna-s-2.1-DFlash-BF16.gguf $DRAFT Q8_0"; exit 1; }
  echo ">> Laguna S 2.1 + DFlash (n-max=$SPEC_N_MAX p-min=$SPEC_P_MIN) — A/B vs base; expect break-even at best"
  exec "$BIN/llama-server" "${common[@]}" \
    -md "$DRAFT" --spec-type draft-dflash \
    --spec-draft-n-max "$SPEC_N_MAX" --spec-draft-p-min "$SPEC_P_MIN"
else
  echo ">> Laguna S 2.1 fast path (-ngl auto --fit, ctx=$CTX) — expect ~18-19 tok/s warm, ~58 t/s prefill"
  exec "$BIN/llama-server" "${common[@]}"
fi
