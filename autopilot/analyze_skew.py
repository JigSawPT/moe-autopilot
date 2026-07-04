# Reads a llama.cpp GGUF imatrix and measures per-layer expert utilization skew.
# Usage: python analyze_skew.py <imatrix.gguf> [--model <model.gguf>] [--out <output.md>]
#   --model: derives per-layer expert bytes from the GGUF itself (hot-set in exact GB).
import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("Missing gguf lib: pip install gguf")


def expert_layer_bytes(model_path: str) -> dict:
    r = GGUFReader(model_path)
    out = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_(?:up|gate|down)_exps", t.name)
        if m:
            n_bytes = int(t.n_bytes) if hasattr(t, "n_bytes") else int(t.data.nbytes)
            out[int(m.group(1))] = out.get(int(m.group(1)), 0) + n_bytes
    return out


def top_n_for(arr, frac):
    s = np.sort(arr)[::-1]
    c = np.cumsum(s)
    if c[-1] <= 0:
        return 0
    return int(np.searchsorted(c, frac * c[-1]) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imatrix")
    ap.add_argument("--model")
    ap.add_argument("--out")
    args = ap.parse_args()
    out_md = Path(args.out) if args.out else Path(__file__).parent / "skew_report.md"

    r = GGUFReader(args.imatrix)
    counts = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_(?:up|gate|down)_exps\.weight\.counts", t.name)
        if m:
            counts.setdefault(int(m.group(1)), np.asarray(t.data, dtype=np.float64).flatten())
    if not counts:
        print("No expert '.counts' tensors found. Sample of names in the file:")
        for t in list(r.tensors)[:40]:
            print(" ", t.name, tuple(t.data.shape))
        sys.exit(1)

    layer_bytes = expert_layer_bytes(args.model) if args.model else {}

    lines = ["# Per-layer routing skew (imatrix)", ""]
    lines.append(f"Source: `{Path(args.imatrix).name}` | layers with data: {len(counts)}")
    lines.append("")
    lines.append("| layer | activations | experts used | top-N for 80% | for 90% | for 95% | max expert |")
    lines.append("|---|---|---|---|---|---|---|")

    tot80 = tot90 = tot95 = 0
    hot80_bytes = hot90_bytes = 0.0
    n_exp = None
    for l in sorted(counts):
        arr = counts[l]
        n_exp = len(arr)
        used = int((arr > 0).sum())
        n80, n90, n95 = (top_n_for(arr, f) for f in (0.80, 0.90, 0.95))
        tot80 += n80; tot90 += n90; tot95 += n95
        if l in layer_bytes:
            hot80_bytes += n80 / n_exp * layer_bytes[l]
            hot90_bytes += n90 / n_exp * layer_bytes[l]
        lines.append(f"| {l} | {int(arr.sum())} | {used}/{n_exp} | {n80} | {n90} | {n95} | "
                     f"{arr.max()/max(arr.sum(),1):.1%} |")

    L = len(counts)
    lines.append("")
    lines.append(f"**Global:** average experts for 80% of tokens: {tot80/L:.0f}/{n_exp} "
                 f"({tot80/L/n_exp:.1%}); for 90%: {tot90/L:.0f}; for 95%: {tot95/L:.0f}.")
    if layer_bytes:
        total_gb = sum(layer_bytes.values()) / 1e9
        lines.append(f"**Hot-set (V2 cache):** 80% ≈ {hot80_bytes/1e9:.1f} GB; 90% ≈ {hot90_bytes/1e9:.1f} GB "
                     f"(total experts: {total_gb:.1f} GB).")
    report = "\n".join(lines)
    out_md.write_text(report, encoding="utf-8")
    print(report[:1200])
    print(f"\n(full report at {out_md})")


if __name__ == "__main__":
    main()
