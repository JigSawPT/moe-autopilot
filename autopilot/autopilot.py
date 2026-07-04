# Autopilot V1 — MoE expert placement planner for llama.cpp on consumer GPUs.
# Usage:
#   python autopilot.py plan     --model <path.gguf> [options]
#   python autopilot.py validate --model <path.gguf> [options]   # plans, measures the plan and neighbors, picks a winner
# Requires: pip install gguf
#
# Rules measured on this machine (see autopilot/README.md and docs/05-06):
#   R1. The POSITION of offloaded layers is neutral (first-20 = last-20 = 80 tok/s post-reboot);
#       convention: offload from the end (equivalent to -ncmoe N on the first N).
#   R2. Guaranteed residency: --no-mmap (cold NVMe cache drops throughput 5x).
#   R3. Planned VRAM + system's CURRENT baseline <= total - reserve: the WDDM cliff is silent
#       and was the cause of the false "position discovery" (baseline varied 0.6 -> 3.8 GB between sessions).
#   R4. -t 16 spread out; never pin to CCD0/V-cache (8t on CCD1 ties with 16t: RAM-saturated).
#   R5. MTP (--spec-type draft-mtp) only when the GGUF has an MTP head AND there are experts on the CPU.
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("Missing gguf lib: pip install gguf")

# Constants calibrated by measurement (AI_PC, 2026-07-03, post-reboot; recalibrate on other machines):
# last-20: 80.0 tok/s; last-22: 76.1; last-24: 69.9 -> gather BW ~42 GB/s + ~1 ms fixed.
RAM_BW = 42e9        # effective B/s on sparse expert reads (gather; < 50 GB/s of copy)
VRAM_BW = 1.05e12    # effective B/s RTX 5090 (measured via dense 27B)
OVERHEAD_S = 0.001   # fixed cost per token (sync/kernels)
# Patched-fork binaries (needed for the hot/cold split; parity-validated against upstream).
# Point AIPC_BIN_DIR at the directory holding the built llama-* binaries, e.g.
#   export AIPC_BIN_DIR=/path/to/llama.cpp/build/bin        (Linux)
#   $env:AIPC_BIN_DIR = "C:\path\to\llama.cpp\build\bin\Release"   (Windows)
import os
_BIN = Path(os.environ.get("AIPC_BIN_DIR", "."))
_EXE = ".exe" if os.name == "nt" else ""
SRV_EXE = str(_BIN / f"llama-server{_EXE}")
BENCH_EXE = str(_BIN / f"llama-bench{_EXE}")
PROF_EXE = str(_BIN / f"llama-aipc-moe-profile{_EXE}")
PROFILES_DIR = Path(__file__).parent / "profiles"


def vram_baseline_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def read_model(path: str):
    r = GGUFReader(path)
    fields = {k: r.fields[k] for k in r.fields}
    arch = None
    for k in fields:
        if k.endswith(".block_count"):
            arch = k.split(".")[0]
    def fint(suffix, default=None):
        k = f"{arch}.{suffix}"
        if k in fields and fields[k].parts:
            return int(fields[k].parts[-1][0])
        return default
    layers = fint("block_count")
    n_exp = fint("expert_count", 0)
    n_used = fint("expert_used_count", 0)
    expert_bytes = {}
    other_bytes = 0
    has_mtp = False
    for t in r.tensors:
        n_bytes = int(t.n_bytes) if hasattr(t, "n_bytes") else int(t.data.nbytes)
        if "nextn" in t.name or "mtp" in t.name:
            has_mtp = True
        m = re.match(r"blk\.(\d+)\.ffn_(?:up|gate|down)_exps", t.name)
        if m:
            expert_bytes[int(m.group(1))] = expert_bytes.get(int(m.group(1)), 0) + n_bytes
        else:
            other_bytes += n_bytes
    return {
        "arch": arch, "layers": layers, "n_exp": n_exp, "n_used": n_used,
        "expert_bytes": expert_bytes, "other_bytes": other_bytes, "has_mtp": has_mtp,
    }


def predict_tps(info, n_cpu_layers: int) -> float:
    layers_sorted = sorted(info["expert_bytes"])
    cpu_layers = layers_sorted[len(layers_sorted) - n_cpu_layers:] if n_cpu_layers else []
    sparsity = (info["n_used"] / info["n_exp"]) if info["n_exp"] else 1.0
    cpu_bytes = sum(info["expert_bytes"][l] for l in cpu_layers)
    gpu_exp_bytes = sum(info["expert_bytes"].values()) - cpu_bytes
    t = (cpu_bytes * sparsity) / RAM_BW + (info["other_bytes"] + gpu_exp_bytes * sparsity) / VRAM_BW + OVERHEAD_S
    return 1.0 / t


def compute_plan(info, args):
    base_mb = vram_baseline_mb()
    usable = (args.vram_budget_gb - base_mb / 1024 - args.reserve_gb) * 1e9
    hot = info["other_bytes"]
    if hot > usable:
        sys.exit(f"Even the hot part doesn't fit: {hot/1e9:.1f} GB > {usable/1e9:.1f} GB usable")
    layers_sorted = sorted(info["expert_bytes"])
    gpu_layers, acc = [], hot
    for l in layers_sorted:
        if acc + info["expert_bytes"][l] <= usable:
            acc += info["expert_bytes"][l]
            gpu_layers.append(l)
        else:
            break
    cpu_layers = [l for l in layers_sorted if l not in gpu_layers]
    cpu_bytes = sum(info["expert_bytes"][l] for l in cpu_layers)
    return {
        "base_mb": base_mb, "usable": usable, "gpu_layers": gpu_layers, "cpu_layers": cpu_layers,
        "vram_bytes": acc, "cpu_bytes": cpu_bytes, "pred_tps": predict_tps(info, len(cpu_layers)),
    }


def build_cmd(args, info, n_cpu_layers: int) -> str:
    cmd = [SRV_EXE, "-m", args.model, "-ngl", "999"]
    if n_cpu_layers:
        cmd += ["--n-cpu-moe", str(n_cpu_layers)]
    # --poll 100: the default poll=50 costs 8-14% and drifts with machine state (docs/11)
    cmd += ["--no-mmap", "--poll", "100", "-c", str(args.ctx), "-t", "16", "--jinja", "--port", str(args.port)]
    if args.ub:
        cmd += ["-b", str(args.ub), "-ub", str(args.ub)]
    if info["has_mtp"] and n_cpu_layers:
        cmd += ["--spec-type", "draft-mtp"]
    line = " ".join(cmd)
    # V2.1: hot/cold cache — emit the env vars (PowerShell syntax) when there is a hot-list
    hotlist = getattr(args, "hotlist", None)
    if hotlist and n_cpu_layers:
        line = (f"$env:AIPC_MOE_HOT_LIST = \"{hotlist}\"; "
                f"$env:AIPC_MOE_HOT_N = \"{args.hot_n}\"; " + line)
    return line


def cmd_coverage(args):
    # CANONICAL coverage calculation: fraction of hits (counts from a profile) that
    # fall within the top-N of the given hot-list. Layers with no line in the hot-list count as 0.
    prof = json.loads(Path(args.profile).read_text(encoding="utf-8"))["layers"]
    hot = {}
    for line in Path(args.hotlist).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            hot[parts[0]] = [int(x) for x in parts[1:]]
    covs = []
    for il, d in sorted(prof.items(), key=lambda kv: int(kv[0])):
        top = set(hot.get(il, [])[: args.hot_n])
        c = d["counts"]
        total = sum(c)
        covs.append(sum(v for i, v in enumerate(c) if i in top) / total if total else 0.0)
    print(f"top-{args.hot_n} coverage: mean {100*sum(covs)/len(covs):.1f}% | "
          f"min {100*min(covs):.1f}% | max {100*max(covs):.1f}% | layers {len(covs)}")
    return sum(covs) / len(covs)


def cmd_hotlist(args):
    # builds a hot-list from real workload text(s) (V2.1 adaptive)
    text = "\n\n".join(Path(f).read_text(encoding="utf-8", errors="replace") for f in args.from_files)
    PROFILES_DIR.mkdir(exist_ok=True)
    corpus = PROFILES_DIR / f"_corpus_{Path(args.out).stem}.txt"
    corpus.write_text(text, encoding="utf-8")
    cmd = [PROF_EXE, "-m", args.model, "-f", str(corpus), "-ngl", "999",
           "--n-cpu-moe", str(args.ncmoe), "--no-mmap", "-b", str(args.ub), "-ub", str(args.ub), "-t", "16"]
    print(f"profiling {len(text)} chars...")
    subprocess.run(cmd, cwd=PROFILES_DIR, capture_output=True, text=True, timeout=1800)
    out = Path(args.out) if Path(args.out).is_absolute() else PROFILES_DIR / args.out
    (PROFILES_DIR / "aipc_moe_profile.hotlist").replace(out)
    (PROFILES_DIR / "aipc_moe_profile.json").replace(out.with_suffix(".json"))
    print(f"hot-list: {out}")
    print(f"use with: $env:AIPC_MOE_HOT_LIST = \"{out}\"; $env:AIPC_MOE_HOT_N = \"96\"")


def print_plan(info, plan, args):
    print(f"System's current VRAM baseline: {plan['base_mb']} MiB (deducted from the budget)")
    print(f"Model: {Path(args.model).name} | arch {info['arch']} | {info['layers']} layers | "
          f"experts {info['n_used']}/{info['n_exp']}")
    print(f"Hot: {info['other_bytes']/1e9:.1f} GB | experts: {sum(info['expert_bytes'].values())/1e9:.1f} GB")
    k = len(plan["cpu_layers"])
    print(f"Plan: {len(plan['gpu_layers'])} expert layers in VRAM, {k} on CPU (--n-cpu-moe {k})")
    print(f"Predicted VRAM: {plan['vram_bytes']/1e9:.1f} GB (+reserve {args.reserve_gb} GB) | "
          f"RAM: {plan['cpu_bytes']/1e9:.1f} GB | predicted decode: ~{plan['pred_tps']:.0f} tok/s")


def bench_config(args, n_cpu_layers: int):
    cmd = [BENCH_EXE, "-m", args.model, "-ngl", "999", "-ncmoe", str(n_cpu_layers),
           "-mmp", "0", "-t", "16", "-p", "0", "-n", "128", "-r", "3", "-fa", "1", "-o", "json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        rows = json.loads(out.stdout)
        return next((r["avg_ts"] for r in rows if r.get("n_gen")), None)
    except Exception as exc:
        print(f"  bench failed (ncmoe {n_cpu_layers}): {type(exc).__name__}")
        return None


def cmd_plan(args):
    info = read_model(args.model)
    plan = compute_plan(info, args)
    print_plan(info, plan, args)
    print("\nCommand:")
    print(build_cmd(args, info, len(plan["cpu_layers"])))


def cmd_validate(args):
    info = read_model(args.model)
    plan = compute_plan(info, args)
    print_plan(info, plan, args)
    k0 = len(plan["cpu_layers"])
    candidates = sorted({max(0, min(info["layers"], k0 + d)) for d in (-args.step, 0, args.step)})
    print(f"\nMeasuring candidates --n-cpu-moe {candidates} (standalone, --mmap 0, tg128 x3)...")
    results = {}
    for k in candidates:
        tps = bench_config(args, k)
        results[k] = tps
        pred = predict_tps(info, k)
        print(f"  ncmoe {k:3d}: measured {tps if tps else '-':>6} tok/s | predicted {pred:.0f}")
    ok = {k: v for k, v in results.items() if v}
    if not ok:
        sys.exit("No valid measurement.")
    best = max(ok, key=ok.get)
    print(f"\nWinner: --n-cpu-moe {best} @ {ok[best]:.1f} tok/s")
    print("Final command:")
    print(build_cmd(args, info, best))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--vram-budget-gb", type=float, default=32.6)
        p.add_argument("--reserve-gb", type=float, default=5.0)
        p.add_argument("--ram-budget-gb", type=float, default=45.0)
        p.add_argument("--ctx", type=int, default=16384)
        p.add_argument("--port", type=int, default=18201)
        p.add_argument("--ub", type=int, default=2048)
        p.add_argument("--step", type=int, default=2)
        p.add_argument("--hotlist", help="hot-list V2.1 (generated by the hotlist subcommand)")
        p.add_argument("--hot-n", type=int, default=96)
    ph = sub.add_parser("hotlist")
    ph.add_argument("--model", required=True)
    ph.add_argument("--from", dest="from_files", nargs="+", required=True, help="workload text file(s)")
    ph.add_argument("--out", required=True, help="output hot-list name (e.g.: session.hotlist)")
    ph.add_argument("--ncmoe", type=int, default=24)
    ph.add_argument("--ub", type=int, default=2048)
    pc = sub.add_parser("coverage")
    pc.add_argument("--hotlist", required=True)
    pc.add_argument("--profile", required=True, help="aipc-moe-profile counts json (workload to evaluate)")
    pc.add_argument("--hot-n", type=int, default=96)
    args = ap.parse_args()
    if args.cmd == "plan":
        cmd_plan(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "coverage":
        cmd_coverage(args)
    else:
        cmd_hotlist(args)


if __name__ == "__main__":
    main()
