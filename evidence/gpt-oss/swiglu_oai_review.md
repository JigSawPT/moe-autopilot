# SWIGLU_OAI dual-chain — adversarial review synthesis (2026-07-05)

Branch `aipc-swiglu-oai` @ `850e1f947` vs published `aipc-v2-pr` @ `ab2fbb7a6`. Three independent code-only adversarial reviewers (Opus), distinct lenses. **No BLOCKER and no HIGH finding.** The patch is correct-by-construction within the shipped envelope (decode, n_tk≤8, gpt-oss config), with ONE open empirical question and two cheap hardening suggestions.

## Verdict per reviewer
- **Regression / edges:** GO. SILU path provably unchanged (a plain-SILU model cannot enter the OAI branch — gated by `type_op`, not tensor presence). All config extremes degrade safely (HOT_N=0 = stock bit-for-bit; HOT_N≥n_expert = masked-hot; partial-bias unreachable/fallback; LoRA bails before the new code; registry lifetime neutral). New risk surfaced: the split now reaches the CUDA fusion `mul_mat_id→add_id→mul_mat_id→add_id→GLU` for the first time.
- **Bias/dummy-slot crux:** no BLOCKER/HIGH; could NOT construct an OOB or NaN. Dummy-slot zeroing complete across all 3 bias tensors (rows = n_hot+n_pad, zeroed once at load, weight+bias share the same id slice). Bias indexing symmetry correct: cold clamp-to-0 picks expert-0's real bias but the add-then-mask merge `e_cold+(e_hot−e_cold)·mask` cancels it (mask is per-slot×token, broadcast over features). Op-order/constants (1.702, 7.0) match stock line-by-line. Residual: MED latent — CUDA `add_id` (`add-id.cu:21`) has NO bounds check (CPU asserts); safe today purely by graph-construction invariants (sel_hot ≤ n_hot+n_pad−1 < rows; sel_cold ≤ 127 < 128), silent-on-failure if a future change breaks the invariant.
- **Equivalence framing:** NEEDS-STRONGER-TEST. The gpt-oss stock path is genuinely run-to-run non-deterministic on CUDA/MXFP4 (reproduced on unmodified upstream b9826; bimodal greedy tie at token ~84), so literal byte-identity is undefined — verified true. BUT the author's "within noise floor" metrics (mean |Δlogprob|, NaN count) are **sign-blind**. Signed paired test shows a small offset: ON-vs-OFF mean signed Δlogprob +0.00198 (control OFF-vs-OFF ~0), exact-zero ties 70%→10%. **Suggestive, not conclusive** (bimodal noise, effective N≈2; honest per-position test only t=+1.27). Crucially: the deterministic anchor (Coder-Next) and the exercised new code (gpt-oss bias path: swiglu_oai + 3× add_id + hot_b dummy slots) are **DISJOINT** — Coder-Next proves the shared machinery, nothing about the bias-specific numerics.

## The one open question (promotion blocker for the gpt-oss number)
Is the +0.002 signed logprob mean-shift benign numerical difference (valid but different reduction/fusion order) or a small systematic bias (e.g. clamp-vs-bias order, mis-scaled bias, garbage dummy row)? Reviewer 3's construction proof says benign; reviewer 2's signed test says unproven. **Settle it deterministically, don't argue it.**

## Consolidated GPU verification checklist (all correctness checks are contention-IMMUNE in result; hold for idle only to avoid competing for VRAM + to get the clean speed number)
1. **DECISIVE — single-layer FFN tensor diff** (reviewer 2 centerpiece): fixed input activation → stock `build_moe_ffn` OAI vs `build_moe_ffn_split`, one gpt-oss layer, compare the FFN output tensor sign-sensitively. Run on **CPU backend first** (deterministic → proves the algebra; <1e-5 expected) then **CUDA** (structured residual = bug; random ~1e-3 = benign). Cold slot (mask=0): assert finite/zero. Hot slot: <1e-5 vs stock. This settles the mean-shift.
2. **Fusion on/off** (reviewer 1): gpt-oss split ON, HOT_N=20, default vs `GGML_CUDA_DISABLE_FUSION=1` — must match within noise.
3. **compute-sanitizer memcheck** (reviewer 3): one decode step — definitively prove CUDA `add_id` never reads OOB.
4. **Degenerate hot-list NaN stress** (reviewer 3): HOT_N=1 / rarely-selected experts → mostly-cold, every token hits dummy slots → assert finite + equivalent.
5. **Boundary HOT_N**: 0 (=stock bit-for-bit) and 128 (all-hot).
6. **Idle speed re-measure**: the +23% (36.6→45.0 tok/s) was contended/provisional — re-run swiglu_oai.md §6 clean.

## Cheap hardening (defer to post-verification pass, bundle with any fix)
- Add an explicit assertion tying `hot_b->ne[1]` (= n_hot+n_pad) to `max(sel_hot)+1`, making the load-bearing CUDA `add_id` bounds invariant explicit (turns reviewer-3's MED latent into a loud failure).
- Mirror the `zeros_b` vector out of the per-tensor loop (cosmetic, matches the weight-side `assign` pattern).

## Bottom line
Correctness: strong GO from code review; the gpt-oss path is correct-by-construction with one deterministic test outstanding. Public promotion of the gpt-oss speed number is CONDITIONAL on (1) the single-layer test passing and (2) the idle speed re-measure — both in one held GPU wave.
