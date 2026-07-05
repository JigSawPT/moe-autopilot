# F3.A Cheap-Gate — Dense-27B Student Raciocinio Regeneration (max_tokens fix)

**Date:** 2026-07-05
**Purpose:** Fix a truncation artifact in run 1's dense-27B student raciocinio answers. 4 of 30
items (idx 15, 19, 20, 24 — the 4 longest answers) were cut off mid-derivation before reaching
the required `FINAL=` line, because `cheapgate_client.py`'s generation call used `max_tokens=600`
(confirmed independently by `evidence/cheapgate2_summary.md`'s prior anomaly analysis of this
same run-1 file). This regenerates ONLY the student/raciocinio pair at `max_tokens=1500`, with
everything else held identical to run 1. **The original evidence file was never modified** —
confirmed byte-identical (MD5 `082f6548465df4b67e505ab27e7a2f8c`, 26069 bytes) before and after
this task.

---

## 1. Generation config: old (run 1) vs new (this regen)

| | Run 1 (original) | This regen (v2) |
|---|---|---|
| **Model** | `Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf` (dense 27B, ~16.33 GiB) | identical (same file, same path) |
| **Server binary** | `llama-server.exe` (build `llama.cpp-b9826-cuda13.3`) | identical |
| **Server flags** | `-ngl 999 --no-mmap -c 8192 -t 16 --jinja --port 18240` (reconstructed from `server_student.log.err`/`server_student2.log.err`: `n_parallel` auto→4, `n_ctx_slot=8192` per slot) | `-ngl 999 --no-mmap -c 8192 -t 16 --jinja --port 18241` (only the port differs, to avoid any ambiguity with leftover state; init log confirms identical `n_ctx_seq=8192`, `n_slots=4`, all 4 slots `n_ctx=8192`) |
| **Endpoint** | `POST /v1/chat/completions`, non-streaming (`"stream": false`) | identical |
| **Messages** | `[{"role": "user", "content": <prompt>}]` (no system prompt) | identical |
| **chat_template_kwargs** | `{"enable_thinking": false}` (student convention in `cheapgate_client.py::run_generation`) | identical |
| **temperature** | **0.7** (client's default for role=student/teacher when `--temperature` is not passed; no evidence anywhere of a CLI override for run 1) | **0.7** (unchanged, per protocol: "everything identical to run 1 EXCEPT max_tokens") |
| **max_tokens** | **600** | **1500** |
| **Prompts source** | `deltamoe/cheapgate_raciocinio_prompts.txt` (30 lines, read verbatim by `load_prompts()`) | identical file, identical order |
| **Client code path** | `cheapgate_client.py::run_generation` (writes `evidence/cheapgate_raciocinio_student.jsonl`) | same `load_prompts()`/`post_chat()` functions imported unchanged from `cheapgate_client.py`; only the output path and `max_tokens` were substituted (small wrapper script, since the CLI's output path is hardcoded and would have overwritten the original) |
| **Output file** | `deltamoe/evidence/cheapgate_raciocinio_student.jsonl` (untouched) | **NEW file:** `deltamoe/evidence/cheapgate_raciocinio_student_v2.jsonl` (30 lines, same `{"prompt","answer"}` schema) |
| **Server logs** | `evidence/server_student.log.err`, `evidence/server_student2.log.err` | `evidence/server_student_v2.log.err` / `.log.out` |

**Reproducibility note on the reconstructed server flags:** no run-1 doc records the exact CLI
invocation in prose; the flags above were reverse-engineered from the server's own startup log
(`n_parallel is set to auto, using n_parallel = 4`, `initializing slots, n_slots = 4`, `new slot,
n_ctx = 8192` for all 4 slots ⇒ `-c 8192` sets **per-slot** context in this llama-server build).
A first launch attempt at `-c 32768` (assuming `-c` was a *total* to be divided across slots)
produced `n_ctx = 32768` per slot instead — this was caught before any generation request was
sent, the misconfigured server was killed, and it was relaunched at `-c 8192`, which reproduced
run 1's slot state exactly. VRAM steady-state during generation: ~20.1 GB (well under the ~30 GB
free budget). `-b`/`-ub` (batch size) were left at binary defaults since run 1's logs never showed
a non-default batch size and these only affect throughput, not generated content.

---

## 2. Verification results

### 2a. Truncation fixed — YES, all 4

| idx | v1 (max_tokens=600) | v2 (max_tokens=1500) |
|---|---|---|
| 15 | truncated at 1457 chars, no `FINAL=` | **`FINAL=2.6667`** (1615 chars) |
| 19 | truncated at 1687 chars, no `FINAL=` | **`FINAL=Ana, Diogo, Carla, Bruno`** (2039 chars) |
| 20 | truncated at 1673 chars, no `FINAL=` | **`FINAL=Ines, Rui, Vasco, Sara, Tomas`** (2311 chars) |
| 24 | truncated at 1490 chars, no `FINAL=` | **`FINAL=180`** (1629 chars) |

None of the 4 regenerated answers hit the new 1500-token cap either (max length among them is
2311 chars ≈ 500-600 tokens of output — well inside budget), so the fix has headroom, not just a
narrower escape.

### 2b. Determinism / prefix-match — DOES NOT HOLD (temperature was 0.7, not 0 — flagged explicitly)

Per protocol: **run 1 used `temperature=0.7`** (client's default for generation roles; no CLI
override evidenced anywhere in the codebase for the original run). This is not temp=0, so
byte-identity and prefix-matching are **not expected or guaranteed**, and indeed did not hold:

- **4 previously-truncated items:** 0/4 are prefix-matches. All 4 diverge from the v1 truncated
  text within the first 60-125 characters — i.e. the model took a differently-worded (but still
  mathematically on-track) path through the same problem on this resample, not a continuation of
  the exact same token stream.
- **Other 26 items:** only 2/26 are byte-identical to v1; 24/26 differ (different phrasing/working,
  same underlying model). This is the expected signature of temp=0.7 resampling, not evidence of a
  parameter mismatch — every non-identical answer was individually checked against the same model
  path, same server flags, and the same `chat_template_kwargs`.

**This is an important correction to the task's working assumption.** The task brief anticipated
temp=0 determinism ("if run 1 used temperature 0... the 26 previously-completed items MUST be
byte-identical"). That branch does not apply here — `cheapgate_client.py`'s own source confirms
`temperature = args.temperature if args.temperature is not None else 0.7` for
role∈{student,teacher}, and no evidence of a `--temperature` override exists anywhere in the
repo for run 1. Per the protocol's explicit fallback ("If run 1 used temp>0, byte-identity won't
hold — in that case verify instead that all 26 still produce CORRECT FINAL= values... and flag the
temp explicitly"), section 2c below is the operative check.

### 2c. Correctness vs ground truth — 26/30 → 29/30 (not 30/30; one genuine regression, not a bug)

Ground truth = run-1 gpt-oss-120b teacher `FINAL=` values in `cheapgate_raciocinio_teacher.jsonl`
(wave-3 verified 30/30 correct; idx 4 and idx 16 have no literal `FINAL=` tag in the teacher file
but were hand-verified correct in `cheapgate2_summary.md`'s anchor: idx 4 = `11:05`, idx 16 =
`11:40` — used as the key here). Comparison used tolerant normalization: case/accent-insensitive,
whitespace-collapsed, unit-stripped (EUR/mg/litros/minutos/horas/segundos/dias/anos), comma→point
decimals, numeric equivalence within 1e-3, sim/não case-insensitive.

| idx | Ground truth | v1 (max_tokens=600) | v2 (max_tokens=1500) | Note |
|---|---|---|---|---|
| 0 | 36 EUR | 36 — OK | 36 — OK | |
| 1 | 6 | 6 — OK | 6 — OK | |
| 2 | 15 | 15 — OK | 15 — OK | |
| 3 | 36 | 36 — OK | 36 — OK | |
| 4 | 11:05 | 11:05 — OK | 11:05 — OK | |
| 5 | sim | sim — OK | sim — OK | |
| 6 | 25 minutos | 25 — OK | 25 — OK | |
| 7 | Beatriz | Beatriz — OK | Beatriz — OK | |
| 8 | sim | sim — OK | sim — OK | |
| **9** | **2** | **2 — OK** | **1 — MISS** | **New regression** (see below) |
| 10 | 12000 | 12000 — OK | 12000 — OK | |
| 11 | 5100 | 5100 — OK | 5100 — OK | |
| 12 | DBCA | DBCA — OK | "DBC A" — OK (whitespace-normalized) | |
| 13 | 7 | 7 — OK | 7 — OK | |
| 14 | 135 | 135 — OK | 135 — OK | |
| **15** | **2.6667** | **(truncated) — MISS** | **2.6667 — OK** | **Fixed** |
| 16 | 11:40 | 11:40 — OK | 11:40 — OK | |
| 17 | 16 | 16 — OK | 16 — OK | |
| 18 | 73.80 EUR | 73.80 EUR — OK | 73.80 — OK | |
| **19** | **Ana,Diogo,Carla,Bruno** | **(truncated) — MISS** | **Ana, Diogo, Carla, Bruno — OK** | **Fixed** |
| **20** | **Ines,Rui,Vasco,Sara,Tomas** | **(truncated) — MISS** | **Ines, Rui, Vasco, Sara, Tomas — OK** | **Fixed** |
| 21 | 1231.20 EUR | 1231.20 — OK | 1231.20 — OK | |
| 22 | 75 | 75 — OK | 75 — OK | |
| 23 | 6.6667 | 6.6667 — OK | 6.6667 — OK | |
| **24** | **180** | **(truncated) — MISS** | **180 — OK** | **Fixed** (as predicted: derives t=180 minutos) |
| 25 | 24 litros | 24 — OK | 24 — OK | |
| 26 | 1102.50 EUR | 1102.50 — OK | "1102,50" — OK (comma-decimal normalized) | |
| 27 | 120 | 120 — OK | 120 — OK | |
| 28 | nao | nao — OK | nao — OK | |
| 29 | 20 | 20 — OK | 20 — OK | |

**TOTAL: v1 = 26/30, v2 = 29/30.**

All 4 previously-truncated items are now correct, including idx 24 which derives `t = 3 horas =
180 minutos` exactly as predicted. **However, idx 9 is a new miss that did not exist in v1**, so
the net gain is +3 (not +4): the 4 truncation fixes are real, but temp=0.7 resampling cost one
previously-correct easy item.

**Idx 9 root-cause (manually verified, genuine reasoning slip, not a scoring artifact):** the
prompt asks how many times the digit 7 appears writing 1 to 25. The digit 7 appears twice: in the
standalone number "7" and in "17". v1's derivation explicitly enumerated both ("$1+1+0=2$ ... o
dígito 7 surge 2 vezes (no número 7 e no número 17)"). v2's resample enumerated the 1-9 range as
containing zero 7s ("nenhum contém") — incorrectly skipping the number 7 itself — then found only
the "17" occurrence, landing on `FINAL=1`. This is a genuine enumeration mistake in this specific
sample, independently confirmed by hand (1-25 written out: 7 and 17 both contain the digit 7).
Because temperature is 0.7, not 0, a different sample on an easy item can occasionally regress
even while harder items improve — this is expected variance under non-greedy decoding, not a
process error in this regeneration.

---

## 3. Recommendation: HOLD — do not promote v2 as a like-for-like drop-in replacement of v1

**v2 is not a strict superset improvement over v1** (29/30 vs 26/30, but via +4/-1, not +4/-0),
because run 1 used temperature 0.7, not 0. A straight swap changes the answer content of 26/30
items that were never the target of this fix, for no reason other than resampling noise, and
introduces one new wrong answer (idx 9) alongside the 4 intended fixes.

**Two honest paths forward, both defensible — this is the coordinator's call:**

- **Option A (surgical patch, recommended if the goal is purely "fix the 4 truncated items"):**
  keep the original `cheapgate_raciocinio_student.jsonl` as canonical, but splice in **only** the
  4 fixed answers (idx 15, 19, 20, 24) from `cheapgate_raciocinio_student_v2.jsonl`, leaving all 26
  other rows exactly as in v1. This achieves the stated goal (26/30→30/30 on the *intended* items)
  without introducing idx 9's regression, and is defensible because the 26 untouched rows were
  never in question. This is likely closer to the spirit of "fix a truncation artifact" than a
  full-file swap.
- **Option B (full promote, only if resampling variance is acceptable):** replace the whole file
  with v2 as originally proposed by the protocol. Net correctness still improves (26→29), and if
  the downstream use of this evidence is the F3.A kill/keep aggregate delta (not per-item audit),
  the aggregate direction is still positive. But this does re-roll 26 answers that had no defect,
  purely as a side effect of temp=0.7, which may not be desired for an evidence file whose whole
  purpose is a frozen, auditable snapshot of "what the student said in run 1."

**Command to execute Option A is NOT trivial to give as a one-liner** (it requires a per-line
splice, not a file copy) — flagging this rather than inventing a shell one-liner that risks
getting the line mapping wrong. If the coordinator picks Option A, a short script is warranted;
happy to write it in a follow-up, but per protocol this task does not execute either promotion.

**Command to execute Option B (full replace), if the coordinator decides resampling variance is
acceptable and this is the desired path:**

```
cp deltamoe/evidence/cheapgate_raciocinio_student.jsonl deltamoe/evidence/cheapgate_raciocinio_student.v1.bak
cp deltamoe/evidence/cheapgate_raciocinio_student_v2.jsonl deltamoe/evidence/cheapgate_raciocinio_student.jsonl
```

**This task did NOT execute either replacement** — both `cheapgate_raciocinio_student.jsonl`
(original, confirmed byte-identical throughout, MD5 `082f6548465df4b67e505ab27e7a2f8c`) and
`cheapgate_raciocinio_student_v2.jsonl` (new, 30/30 generated, no errors) exist side by side in
`deltamoe/evidence/`.

---

## 4. Housekeeping

- Server started on port 18241 (not 18240, to avoid any ambiguity with leftover state), flags
  mirrored from run 1's reconstructed config, health-checked before first request.
- One misconfigured launch attempt (`-c 32768`, wrongly assuming `-c` was a total split across
  slots) was caught from the startup log alone and killed **before any generation request was
  sent** — no wasted/contaminated generations, no VRAM left allocated.
- VRAM baseline before launch: 2257 MiB. Steady-state during generation: ~20.1 GB. After shutdown:
  2508 MiB. All within the idle-background range observed at session start (2508-2673 MiB) —
  confirmed no llama process left running (`tasklist` clean).
- Evidence files produced (new; run-1 evidence untouched): `evidence/cheapgate_raciocinio_student_v2.jsonl` (30 lines), `evidence/server_student_v2.log.err`, `evidence/server_student_v2.log.out`.

---

## Coordinator decision (2026-07-05): Option A — surgical splice, EXECUTED

Full-file replacement was rejected: run 1 generated at **temperature 0.7** (confirmed from cheapgate_client.py), so a whole-file swap re-rolls the 26 defect-free answers on pure sampling noise and — as the regen showed — introduced a fresh wrong answer (idx 9 → "1"). That violates the user's guardrail ("replace only once the other 26 are confirmed unchanged/still correct"), which the temp>0 resampling cannot satisfy.

Instead, spliced ONLY the 4 defective rows (idx 15, 19, 20, 24) from v2 into the original, keeping the other 26 exactly as-generated-and-judged in run 1.

Verification of the resulting canonical `cheapgate_raciocinio_student.jsonl`:
- 26 non-target rows **byte-identical** to the pre-splice original (backed up as `cheapgate_raciocinio_student.v1.bak`).
- 4 spliced rows == v2 rows, each with a proper `FINAL=` line, all 4 **correct** vs the gpt-oss teacher key (2.6667 / Ana,Diogo,Carla,Bruno / Ines,Rui,Vasco,Sara,Tomas / 180).
- Checkable-key correctness **27/27** (idx 4, 5, 16 have no teacher `FINAL=` to auto-check; kept from the original, previously hand-verified correct). **0 newly-wrong** items — the v2 idx-9 resample regression is excluded by construction.
- 30 lines; the truncation artifact (26/30 raw) is resolved without side effects.

Files: canonical = spliced (original 26 + 4 fixed); `cheapgate_raciocinio_student.v1.bak` = untouched run-1 original (md5 082f6548465df4b67e505ab27e7a2f8c); `cheapgate_raciocinio_student_v2.jsonl` = full 1500-token resample, kept for the record.

Method rule promoted: verify the generation temperature before assuming a regen is deterministic; when only a subset of items is defective under stochastic (temp>0) generation, splice the fixes — do not swap the whole file (it re-rolls good data and can regress).
