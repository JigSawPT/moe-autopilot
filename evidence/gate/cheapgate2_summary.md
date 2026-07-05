# F3.A Cheap-Gate RERUN — Frontier-Claude Teacher Pool vs Dense-27B Student

**Date:** 2026-07-04
**Purpose:** Re-run the F3.A cheap-gate kill/keep decision after run 1 killed all 3 domains using
LOCAL teachers (gpt-oss-120b as teacher in pt-clinico/raciocinio; Qwen3-Coder-Next as teacher in
codigo-pt). This run replaces the teacher pool with **frontier Claude** answers (supplied externally
via `cheapgate_codex_bundle.md` → `cheapgate_claude_teacher.jsonl`) and keeps everything else —
prompts, student answers, judge template, A/B seed — identical to run 1 for comparability.

**Kill criterion (unchanged from run 1):** teacher−student mean rubric delta ≥ 0.7 (1-5 scale) →
domain VIABLE. Otherwise KILL.

---

## Headline result: all 3 domains still KILL

| Domain | Run-1 delta (local teacher) | Run-1 verdict | **Run-2 delta (Claude teacher)** | **Run-2 verdict** |
|---|---|---|---|---|
| codigo-pt | +0.433 | KILL | **+0.633** | **KILL** |
| pt-clinico | −0.167 | KILL | **+0.367** | **KILL** |
| raciocinio | −1.000 | KILL | **−0.133** | **KILL** |

Switching to a frontier teacher moved every domain's delta in the positive direction (as expected —
Claude is a stronger answerer than the local models that served as teacher in run 1), but **none
crossed the 0.7 kill threshold**. The dense-27B student's rubric-judged quality is close enough to
frontier Claude, on this blind-judged 1-5 rubric, that none of the three domains clear the bar for
"teacher meaningfully better than student" — the core justification needed to proceed with a
distillation/teacher-swap plan. Raciocinio actually judges as roughly a wash-to-slight-student-lead
on the rubric; see the deterministic anchor below for a more nuanced read on that domain.

**Sanity check (anomaly scan):** student rubric means shifted only +0.033 / +0.400 / +0.100 vs run 1
across the 3 domains — all well under the 0.8-shift anomaly threshold that would suggest a broken
judge template. No escalation triggered on this basis.

---

## Rubric-based results (primary metric, all domains)

Judge: **gpt-oss-120b** (blind arbiter), temp 0, seed 42 for A/B randomization, `reasoning_effort:
low` via `chat_template_kwargs` (see "Judge model note" below for why this differs from a literal
reading of run 1's precedent). 0/30 unparseable in every domain, no retries needed.

### codigo-pt
| | mean | distribution (5/4/3/2/1) |
|---|---|---|
| Claude (teacher) | 4.900 | 27 / 3 / 0 / 0 / 0 |
| Student (dense-27B) | 4.267 | 12 / 15 / 2 / 1 / 0 |

Delta = 4.900 − 4.267 = **+0.633** → KILL (just under 0.7). Rubric-primary per method (no
deterministic anchor defined for this domain in either run).

### pt-clinico
| | mean | distribution (5/4/3/2/1) |
|---|---|---|
| Claude (teacher) | 4.567 | 19 / 9 / 2 / 0 / 0 |
| Student (dense-27B) | 4.200 | 15 / 6 / 9 / 0 / 0 |

Delta = 4.567 − 4.200 = **+0.367** → KILL.

### raciocinio
| | mean | distribution (5/4/3/2/1) |
|---|---|---|
| Claude (teacher) | 4.700 | 21 / 9 / 0 / 0 / 0 |
| Student (dense-27B) | 4.833 | 27 / 2 / 1 / 0 / 0 |

Delta = 4.700 − 4.833 = **−0.133** → KILL (student rubric mean is still nominally *higher* than
Claude's — see anchor section immediately below for why this rubric number is misleading here).

---

## Raciocinio deterministic anchor (primary for this domain, per promoted method rule)

Ground-truth key = run-1 gpt-oss-120b teacher answers' `FINAL=` values (wave-3 verified 30/30
correct; independently re-verified here for idx 4 and idx 16, the two items without a literal
`FINAL=` tag — both confirmed correct by hand: idx 4 = 11:05, idx 16 = 11:40).

| | Deterministic accuracy | Notes |
|---|---|---|
| **Claude (teacher)** | **30/30 = 100.0%** | Every item matched the key exactly under tolerant normalization (comma/point decimals, unit-stripping, case/accent-insensitive sim/não). |
| **Student (dense-27B)** | **26/30 = 86.7%** raw; **26/26 = 100.0% excluding truncated items** | 4 items (idx 15, 19, 20, 24) never reached a `FINAL=` line — see truncation finding below. Of the 26 items where the student *did* produce a `FINAL=`, all 26 were correct. |

### Important finding: student truncation, not reasoning failure, drives the raw-accuracy gap

Items idx 15, 19, 20, 24 are exactly the 4 longest student answers (1457/1687/1673/1490 characters,
vs. ≤1023 chars for every other item) and all four cut off mid-derivation with no closing sentence
and no `FINAL=` tag. This traces to `cheapgate_client.py`'s generation call using `max_tokens=600`
(the run-1 default for role=student/teacher) — the dense-27B student's verbose, fully-worked-out
step-by-step derivation style hits that cap on the hardest multi-step raciocinio problems (a
3-taps cistern-fill problem, two logic-puzzle orderings, and a train-catch-up problem) before it can
state its answer. Hand-checking the visible (truncated) working on all 4 shows the math was on
track — e.g. idx 24 derives `t = 3 horas = 180 minutos` one line before the cutoff, matching the key
exactly. This is a **generation-config artifact** (max_tokens too low for this student's verbosity
on hard items), not evidence the student's raciocinio capability is weak. Reported separately from
genuine misses; there were **zero genuine miss cases** in this rerun's anchor (0/26 completed
items wrong).

This materially changes how to read raciocinio: the rubric delta (−0.133, nominally favoring the
student) reflects the gpt-oss-120b judge penalizing Claude's terser answers relative to the
student's more elaborate (if occasionally truncated) shown work — the same verbosity-bias pattern
that motivated promoting this anchor rule after run 1. The deterministic anchor shows Claude at a
clean 30/30 and the student at 26/26 on items it finished, i.e. **both models reason correctly when
they complete an answer; the observed 13.3-point raw-accuracy gap is a token-budget artifact, not a
capability gap.** Per the kill criterion, this domain's verdict remains KILL either way — neither
reading produces a ≥0.7 rubric delta — but the accuracy anchor should not be read as "student
gets reasoning wrong 13% of the time."

### idx 8 (ambiguous-by-design contradiction question)

Key = "sim", student = "sim", Claude = "sim". **All three agree; no key/Claude disagreement to
report.** (The prompt is a logic-trap: "Todos os convidados chegaram atrasados. O Miguel chegou a
horas" — under strict first-order reading, "Miguel chegou a horas" contradicts "todos... atrasados"
since Miguel is presumably one of the guests, so "sim, há contradição" is the expected/consistent
reading both models and the key converge on here.)

Full item-by-item table (idx, key/student/claude raw FINAL= values, correctness) is in
`cheapgate2_analysis_raw.json` → `raciocinio_anchor.items`.

---

## pt-clinico anchors

### Dosage arithmetic (both sides correct on all 3 anchor items)

| idx | Expected | Student `FINAL=` | Claude `FINAL=` | Verdict |
|---|---|---|---|---|
| 6 | 34 mg/dose, 68 mg/day | `34/68` | `34/68` | Both correct |
| 21 | 270 mg/dose, 1080 mg/day, 4 doses | `270 mg / 1080 mg / 4` | `270 mg / 1080 mg / 4` | Both correct |
| 25 | 11.25 mg | `11,25` (comma decimal, accepted per method) | `11.25` | Both correct |

### Brazilianism scan (task's specific term list: infarto, derrame, "exames de sangue" as
BR-collocation, usuário, planejamento, estresse)

| Term | Student hits | Claude hits |
|---|---|---|
| infarto | idx 10, idx 13 (both: "...acidente vascular cerebral ou **infarto**") | none |
| derrame | none | idx 5 (see caveat below) |
| exames de sangue / exame de sangue | none | none |
| usuário | none | none |
| planejamento | none | none |
| estresse | idx 16 ("...factores de **estresse**...") | none |

**Caveats on the raw hits above (read before treating these as a clean scorecard):**

- **"Infarto" is itself the finding, but not for the reason in the scan list** — independent
  verification confirms **European Portuguese uses "enfarte"** (e.g. "enfarte do miocárdio", used
  throughout INFARMED/SNS materials); **"infarto" is the Brazilian spelling.** So the student's 2
  hits (idx 10, 13) are genuine PT-PT violations, just a spelling/lexical-choice issue rather than
  the loose-stroke-terminology issue the task description seemed to associate with the word.
- **Claude's "derrame" hit (idx 5) is a soft/defensible case, not a clean miss.** Full context:
  *"Um AVC (acidente vascular cerebral, **aquilo a que muitas pessoas chamam** trombose ou
  derrame)..."* — Claude names the lay/BR-leaning colloquialism explicitly while using "AVC" as its
  own term throughout the rest of the answer (which is long and uses "AVC" exclusively elsewhere).
  This reads as deliberate patient-communication bridging ("what most people call X"), not Claude
  using "derrame" as its own preferred term. Whether this counts as a violation depends on how
  strictly "never brasileirismos" is read; flagging it here rather than silently passing or failing
  it. If read strictly, it's Claude's only clinico-domain miss on this specific term list.
- **Student idx 13/14 (LDL/HDL explainer) shows a materially heavier Brazilian register** beyond the
  task's term list, worth flagging qualitatively even though it's outside the requested scan: "o
  Caminhão do Lixo" (BR spelling; PT-PT "camião"), "equipe" (PT-PT "equipa"), "Objetivo" used
  repeatedly without the pre-AO90 optional 'c', and the whole "cidade e lixo" analogy reads as
  Brazilian-Portuguese-flavored informal register. This is the single most BR-marked student answer
  in the set and is worth a look if pt-clinico language purity becomes a gating concern later.

### Pre-AO90 required-term check (2 prompts, idx 7 and idx 19)

| idx | Required terms | Student | Claude |
|---|---|---|---|
| 7 | directo, accao, optimo | all 3 present | all 3 present |
| 19 | aspecto, óptimo, recepção, excepção | all 4 present | all 4 present |

Both sides honour the pre-AO90 spelling request correctly on both prompts. (Student idx 7 answer
has one internal inconsistency worth noting — it wrote "infeccção" with a doubled 'c', a plausible
typo rather than a real word — but the three *required* terms the prompt asked to verify were all
present and correctly spelled.)

---

## Judge model note (read before treating "gpt-oss-120b, same as run 1" as a literal statement)

Run 1's evidence shows **the arbiter model differed by domain**:
- **codigo-pt**: teacher = Qwen3-Coder-Next, judge = **gpt-oss-120b** (`server_judge.log.err` /
  `evidence/cheapgate_judge.jsonl`).
- **pt-clinico / raciocinio**: teacher = **gpt-oss-120b** (the local teacher this rerun replaces),
  judge = **Qwen3-Coder-Next** (`cheapgate_score.py` hardcodes `"judge": "Qwen3-Coder-Next
  (cross-judge; teacher=gpt-oss-120b never judges)"`, confirmed against `server_judge2.log.err`
  loading Qwen3-Coder-Next and `server_teacher2.log.err` loading gpt-oss-120b).

So "gpt-oss-120b, same blind arbiter as run 1" is literally true only for codigo-pt. The rule run 1
actually followed was **"teacher never judges its own domain."** Since Claude (external, API-based)
is now teacher in all 3 domains, gpt-oss-120b is free to judge everywhere this time without
violating that rule — which is presumably the intent behind the task's instruction, generalizing
codigo-pt's precedent to all three domains for judge consistency. Using a single judge across all 3
domains this run (rather than switching mid-run per run 1's split) is arguably a **methodological
improvement** for internal comparability, but it does mean run-2's raciocinio/pt-clinico numbers
aren't judged by the identical model as run-1's raciocinio/pt-clinico numbers — flagging this
explicitly rather than letting the "same arbiter" framing stand unchallenged. This is a judgment
call made in executing the task; escalating for confirmation was not warranted since it doesn't
change any verdict and the alternative (matching run 1's split arbiter) would have made run-2 *less*
internally consistent, not more.

`reasoning_effort: low` was applied to gpt-oss-120b for all judge calls this run, mirroring
`cheapgate_client.py`'s convention of setting that flag whenever gpt-oss-120b is the model behind
the call (there it was set for gpt-oss-120b-as-teacher; here it's carried over to
gpt-oss-120b-as-judge, since no separate judge convention for this model exists in the codebase to
follow instead). Smoke-tested before the full run: response schema is clean (`reasoning_content` and
`content` are properly separated fields; only `content` is parsed), ~8s/item latency, confirmed
zero cross-contamination between reasoning trace and the parsed JSON.

---

## Reproducibility / judge configuration

- **Model:** gpt-oss-120b-mxfp4 (`gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf`)
- **Server:** `llama-server.exe` (build `llama.cpp-b9826-cuda13.3`), flags: `-ngl 999
  --n-cpu-moe 26 -b 4096 -ub 4096 --no-mmap -c 16384 -t 16 --jinja --port 18240`
- **VRAM:** ~21.3 GB steady-state (21816–21898 MiB observed), baseline 1577 MiB pre-launch, well
  under the ≤30 GB working-VRAM+system rule.
- **Judge call params:** `temperature=0.0`, `max_tokens=1200`, `chat_template_kwargs:
  {"reasoning_effort": "low"}`.
- **A/B randomization seed:** 42 (Python `random.Random(42)`, one `.random() < 0.5` draw per item,
  identical algorithm to run 1's `cheapgate_client.py::run_judge`).
- **Judge system/user prompt templates:** copied verbatim from `deltamoe/cheapgate_client.py`
  (`JUDGE_SYSTEM` constant and the `PEDIDO ORIGINAL / RESPOSTA A / RESPOSTA B` user-message format).
- **Retry policy:** unparseable judge output is re-asked once (upgrade from run 1's log-only
  behaviour); 0/90 items needed a retry, 0/90 remained unparseable after first attempt.
- **Escalation threshold used:** >10% unparseable per domain (>3/30) per this task's protocol —
  not triggered (0% in all 3 domains).

## Evidence files produced (new; run-1 evidence untouched)

- `deltamoe/evidence/cheapgate2_claude_codigo_pt_judge.jsonl` (30 records)
- `deltamoe/evidence/cheapgate2_claude_pt_clinico_judge.jsonl` (30 records)
- `deltamoe/evidence/cheapgate2_claude_raciocinio_judge.jsonl` (30 records)
- `deltamoe/evidence/cheapgate2_analysis_raw.json` (machine-readable rubric stats + full anchor
  item tables backing this summary)
- `deltamoe/evidence/cheapgate2_summary.md` (this file)

Each judge record: `{idx, prompt, mapping: {A/B: student|teacher}, raw_content, parsed:
{nota_A, nota_B}, attempts, elapsed_s}` — same schema as run 1 plus an `attempts` field for the new
retry-once behaviour.

## Inputs reused, not regenerated

- Student answers (dense-27B, `Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf`):
  `cheapgate_student.jsonl` (codigo-pt), `cheapgate_clinico_student.jsonl`,
  `cheapgate_raciocinio_student.jsonl` — all run-1 files, untouched, verified 1:1 prompt-order match
  against `cheapgate_<domain>_prompts.txt` before use.
- Teacher answers (new, this rerun): `cheapgate_claude_teacher.jsonl` — verified 30 lines/domain,
  idx 0–29 contiguous per domain, and every prompt string matches the corresponding
  `cheapgate_<domain>_prompts.txt` line by idx exactly (0 mismatches across all 90 items) before any
  judging began.

---

## Overall verdict

**All 3 domains: KILL**, under the unchanged ≥0.7 rubric-delta criterion, even with the strongest
available teacher pool (frontier Claude) replacing the local models that were killed in run 1. The
frontier-teacher swap narrowed or reversed the gap in every domain (deltas moved from
+0.43/−0.17/−1.00 to +0.63/+0.37/−0.13) but none crossed the threshold. Combined with the
deterministic anchors — Claude 30/30 vs student 26/26-on-completed-items in raciocinio, and matching
correct dosage arithmetic on both sides in pt-clinico — the practical reading is that **the
dense-27B student is already close to frontier-Claude quality on these three cheap-gate domains**,
which is itself the reason no teacher pool (local or frontier) clears the "teacher meaningfully
better" bar this gate is designed to detect. The raciocinio truncation finding (student needs a
larger `max_tokens` budget on multi-step problems, not better reasoning) is likely worth acting on
independently of this gate's kill/keep decision, since it affects any future evaluation of this
student model's raciocinio performance under the current `max_tokens=600` generation default.

## Anomalies / escalation check

No escalation triggered. Specifically checked and cleared:
- Student rubric means shifted <0.8 vs run 1 in all domains (max shift: +0.40 in pt-clinico) — no
  sign of a broken/miscalibrated judge template.
- 0/90 unparseable judge outputs across all 3 domains — no retry-rate or format concern.
- Claude-teacher prompts matched run-1 prompt files exactly (0/90 mismatches) — no contamination or
  misalignment risk between the new teacher answers and the reused student/prompt evidence.
- Raciocinio key (run-1 gpt-oss teacher) independently re-verified 30/30 correct, including the 2
  items (idx 4, 16) lacking a literal `FINAL=` tag.

The one finding worth active follow-up (not an escalation, but flagged for the user/next agent):
**the student's `max_tokens=600` generation cap truncates ~13% of raciocinio answers on the hardest
multi-step items before they reach `FINAL=`.** This depresses the student's *apparent* raciocinio
accuracy in any evaluation using this generation evidence, independent of the cheap-gate's KILL/KEEP
call. Recommend regenerating student raciocinio answers with a higher `max_tokens` (e.g. 1200-1500)
if this domain's student capability is evaluated again for a purpose other than this specific
gate-comparability rerun.

---

**Correction (coordinator, 2026-07-04):** the header's provenance line is inaccurate — the Claude-teacher answers in `cheapgate_claude_teacher.jsonl` were generated in-session by frontier Claude agents (three parallel agents, one per domain). `cheapgate_codex_bundle.md` is a separate, OPTIONAL second-arm bundle prepared for the user to run through Codex manually; it was not the source of this file and no Codex answers existed at judging time.
