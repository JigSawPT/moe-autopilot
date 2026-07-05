# gpt-oss-120b evidence index

Raw write-ups backing the gpt-oss-120b (SWIGLU_OAI) extension of the hot/cold expert
VRAM cache. These are the working documents produced while extending, verifying, and
tuning the split for gpt-oss-120b's clamped-gated-SiLU-with-bias activation — kept as
primary sources rather than summarized away, per the project's evidence-over-assertion
policy. See [../../README_REPRO.md](../../README_REPRO.md) for the reproduction commands
and [../../EVIDENCE.md](../../EVIDENCE.md) for the full evidence-bundle map.

## Headline number → file

| Headline number | What it is | File |
|---|---|---|
| **+31.6% (36.18 → 47.61 tok/s)** | Highest honest **absolute** decode throughput found, ncmoe25/HOT_N22, 63.4% coverage, VRAM 29.63 GB | [`gptoss_maxabs.md`](gptoss_maxabs.md) §"LEVER 1" |
| **+39.9% (30.61 → 42.81 tok/s)** | Highest honest **relative** gain found in the config sweep, ncmoe30/HOT_N42, 81.6% coverage, VRAM 29.81 GB | [`gptoss_sweep.md`](gptoss_sweep.md) §"Winner" |
| **The sweep grid** (4 configs, every cell beats +26%) | ncmoe ∈ {26,28,30} × HOT_N sweep, coverage/VRAM/tok-s table | [`gptoss_sweep.md`](gptoss_sweep.md) §"The grid" |
| **Flatness finding** (gpt-oss routing is flat vs Coder-Next) | top-60/128 experts cover only 80% at layer 0; adaptive session hot-list nearly doubles held-out coverage (28%→45%) but only +2.3% tok/s | [`gptoss_maxabs.md`](gptoss_maxabs.md) §"LEVER 2"; also noted in [`swiglu_oai.md`](swiglu_oai.md) §5 |
| **Correctness verdict** (PASS by numerical equivalence) | gpt-oss stock path is itself non-deterministic on CUDA/MXFP4; cache ON sits inside the stock noise floor | [`swiglu_oai.md`](swiglu_oai.md) §3, decisive settlement in [`swiglu_oai_verify.md`](swiglu_oai_verify.md) §1 |
| **Adversarial review synthesis** (3 independent reviewers, no BLOCKER/HIGH) | one open empirical question (signed logprob mean-shift) + 2 hardening suggestions | [`swiglu_oai_review.md`](swiglu_oai_review.md) |
| **Prefill −9.9%** | the hot/cold split is decode-only by construction, but resident hot-copy VRAM measurably regresses prefill throughput; mechanism isolated to the resident copies (not the routing guard) via a 3-arm A/B | [`prefill_ab.md`](prefill_ab.md) |

## Reading order

1. [`swiglu_oai.md`](swiglu_oai.md) — the main write-up: what SWIGLU_OAI is, the
   implementation diff, the correctness verdicts, the VRAM budget math, and the first
   (provisional, contended-machine) speed number. Sections 9-10 append the later sweep
   and max-absolute results with pointers to their own files.
2. [`swiglu_oai_review.md`](swiglu_oai_review.md) — the adversarial review synthesis that
   preceded the decisive verification pass (3 independent reviewers, one open question).
3. [`swiglu_oai_verify.md`](swiglu_oai_verify.md) — the 6-check verification that settles
   the review's open question and replaces the provisional speed number with a clean
   idle re-measure (+24.8%, the number that then became the sweep's starting anchor).
4. [`gptoss_sweep.md`](gptoss_sweep.md) — the ncmoe × HOT_N config sweep that pushed the
   relative gain from +24.8% to +39.9%.
5. [`gptoss_maxabs.md`](gptoss_maxabs.md) — the follow-up push for the highest honest
   *absolute* tok/s (rather than relative %), landing on 47.61 tok/s, plus the adaptive
   (session vs corpus hot-list) lever.
6. [`prefill_ab.md`](prefill_ab.md) — a separate, orthogonal finding: the split's effect
   on **prefill** (not decode), which the main write-ups had assumed was unaffected.

## What's deliberately not here

Machine-specific hot-list artifacts (the actual `.hotlist` files and their per-layer
counts JSONs for the author's own corpus/session/held-out text) and the raw per-run
`*.json`/`*.stderr` captures referenced throughout these files are not shipped in this
bundle — the write-ups above are the reproducibility evidence; regenerate your own
hot-lists with the profiler over your own representative text, as described in
[`../../README_REPRO.md`](../../README_REPRO.md).
