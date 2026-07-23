#!/usr/bin/env bash
# run_laguna_s21.sh — launch Laguna S 2.1 (118B MoE, ~71 GB Q4) on a SINGLE RTX 5090
# (32 GB) + 64 GB DDR5. Base path by default; add DFlash speculative decode with MODE=spec.
# See docs/laguna_s21_runbook.md for the reasoning and honest expectations (~19 tok/s base;
# DFlash breaks even at best on this partial-offload, fine-grained MoE).
#
# Machine-state rules baked in (this project's non-negotiables):
#   --no-mmap  : a cold NVMe cache drops throughput ~5x
#   --poll 100 : the default (50) costs 8-14%
#   Windows/WDDM: keep working VRAM + system baseline <= ~30 GB or the driver pages silently.
set -euo pipefail

# --- config (override via env) -------------------------------------------------
MODEL="${MODEL:-laguna-s-2.1-Q4_K_M.gguf}"
DRAFT="${DRAFT:-laguna-s-2.1-DFlash-Q8_0.gguf}"   # quantize BF16->Q8_0 first; see runbook §2
BIN="${AIPC_BIN_DIR:-./llama.cpp/build/bin}"
MODE="${MODE:-base}"                              # base | spec
CTX="${CTX:-16384}"                               # 64 GB budget; raise toward 32768 iff RAM allows
THREADS="${THREADS:-16}"
PORT="${PORT:-8095}"
# DFlash knobs — single-GPU tuning (stricter than the dual-GPU write-up: verify tokens are pricier).
SPEC_N_MAX="${SPEC_N_MAX:-7}"
SPEC_P_MIN="${SPEC_P_MIN:-0.75}"                  # the footgun: default is 0.00 (ships all N regardless)
# ------------------------------------------------------------------------------

common=(
  -m "$MODEL"
  -ngl auto --fit on --fit-target 2048
  -c "$CTX" -ctk q8_0 -ctv q8_0
  -fa on --jinja --no-mmap --poll 100
  -t "$THREADS" -b 2048 -ub 2048
  --temp 0.7 --top-p 0.95 --top-k 20
  --host 127.0.0.1 --port "$PORT"
)

if [[ "$MODE" == "spec" ]]; then
  [[ -f "$DRAFT" ]] || { echo "drafter '$DRAFT' not found — quantize it first:"; \
    echo "  $BIN/llama-quantize laguna-s-2.1-DFlash-BF16.gguf $DRAFT Q8_0"; exit 1; }
  echo ">> Laguna S 2.1 + DFlash (n-max=$SPEC_N_MAX p-min=$SPEC_P_MIN) — A/B vs base before trusting it"
  exec "$BIN/llama-server" "${common[@]}" \
    -md "$DRAFT" --spec-type draft-dflash \
    --spec-draft-n-max "$SPEC_N_MAX" --spec-draft-p-min "$SPEC_P_MIN"
else
  echo ">> Laguna S 2.1 base path (ctx=$CTX) — expect ~19 tok/s decode; <12 means you're paging"
  exec "$BIN/llama-server" "${common[@]}"
fi
