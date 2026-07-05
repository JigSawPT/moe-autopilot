# F3.A.2 Phase 2b — Dense-27B student on the 120 NEW hard-set items (2026-07-04)

Second student pass of the failure-set gate, over the three NEW 40-item hard sets:
`hardset_codigo_ext.jsonl` (codigo-pt, idx 40–79), `hardset_compras.jsonl` (compras-online,
idx 0–39), `hardset_docslongos.jsonl` (docs-longos, idx 0–39). Same student and conventions as
Phase 2: dense 27B (Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P), temperature 0.0,
max_tokens 1500, bare user message, `chat_template_kwargs: {"enable_thinking": false}`.
Deterministic scoring by the CURRENT `deltamoe/hardset_score.py` (instrument v2, untouched).

- Answers: `evidence/hardset_student2.jsonl` (120/120, 0 request errors, 0 retries, 0 empty answers)
- Scores: `evidence/hardset_student2_scores.jsonl` (one record per item, all 120)
- Scorer self-test: **113/113** synthetic `FINAL=<expected>` cases pass (33 codigo-ext + 40
  compras + 40 docslongos; the other 7 codigo-ext items have `expected_final: null` —
  structure-only checks). Self-test ran before scoring in each of the 3 per-domain runs, as
  mandated. Scorer and hardsets were NOT modified.

## Per-domain results vs viability thresholds

| Domain | Pass | Fail | Threshold | Verdict |
|---|---|---|---|---|
| compras-online | 40 | **0** | ≥8/40 | **NOT viable** (0 < 8) |
| docs-longos | 38 | **2** | ≥8/40 | **NOT viable** (2 < 8) |
| codigo-pt (ext only) | 27 | **13** | — (feeds combined) | ext-only count reported per protocol |
| codigo-pt (COMBINED) | 62 | **18** | ≥16/80 (20%) | **VIABLE — 18/80 = 22.5% ≥ 20%** |

codigo-pt COMBINED = 5 base fails (instrument v2, `hardset_student_scores_v2.jsonl`: idx 1, 9,
12, 14, 15) + 13 ext fails (below) = 18/80.

Total Phase 2b: 105/120 pass (87.5%). The two web-workload domains produced almost no failure
signal at this difficulty; the codigo-pt extension produced a lot (13/40 = 32.5% on the ext set
alone, vs 12.5% on the base set).

## FAILURE SET (15 items)

### codigo-pt ext (13): idx 45, 51, 56, 58, 60, 61, 62, 65, 66, 68, 69, 75, 76

| idx | One-line reason | got_final |
|---|---|---|
| 45 | code-only answer (fix `int(v)` correct, `FINAL=35` correct); failed only the required `TypeError\|str.*int\|texto\|convert` — zero prose naming the error/conversion | `35` (correct) |
| 51 | answered with a code block ending `print(f"FINAL={comprimento}")` — never executed it, value 5 never stated anywhere | `{comprimento}")` |
| 56 | same pattern: `print(f"FINAL={resultado}")` in code, value 5 never stated | `{resultado}")` |
| 58 | same pattern: `print(f"FINAL={freq['the']}")`, value 3 never stated | `{freq['the']}")` |
| 60 | same pattern: pandas groupby code, `print(f"FINAL={maior_total}")`, value 210 never stated | `{maior_total}")` |
| 61 | same pattern: drop_duplicates code, `print(f"FINAL={total}")`, value 425 never stated | `{total}")` |
| 62 | correct hand-done UTC conversion, `FINAL=2` correct; prompt mandated `pandas.to_datetime(..., utc=True)` or datetime-with-tz — no mandated tool used or named (required `to_datetime\|utc\s*=\s*True\|astimezone\|timezone\|tz` missed) | `2` (correct) |
| 65 | same code-only pattern: `print(f"FINAL={soma_maximos}")`, value 20 never stated | `{soma_maximos}")` |
| 66 | same code-only pattern: `print(f"FINAL={total}")`, value 100 never stated | `{total}")` |
| 68 | correct regex and correct count, `FINAL=5` correct; the prompt-mandated `re.findall` never appears (required `re\.findall\|findall` missed) — counting done by hand | `5` (correct) |
| 69 | `FINAL=3` correct and the regex it built is the safe one; forbidden anti-backtracking pattern fired on the answer QUOTING the prompt's own counter-example `(\w+)+` while explaining why it is dangerous | `3` (correct) |
| 75 | correct bisect walkthrough, `FINAL=5` correct; described the binary search but never wrote the mandated command sequence (required `git\s+bisect\s+start` missed) | `5` (correct) |
| 76 | all 3 required checks PASSED (`git revert -m 1`, push, shared-history rationale); forbidden `push\s+--force` fired on the "Why NOT to use reset --hard + push --force" section that the prompt itself demanded | — |

### compras-online (0): —

No failures. All 40 unit-price traps, promo/threshold traps, multi-criteria filters,
review-rule items and import-cost calculations passed, including the inversion items
(idx 23 sim/nao threshold trap, idx 25 anchor-price history).

### docs-longos (2): idx 0, 29

| idx | One-line reason | got_final |
|---|---|---|
| 0 | dated the founding Christmas supper `1973-12-25` from world knowledge, overriding the prompt's explicit dating rule ("para os factos cuja âncora é a inauguração (ceia, escritura, inauguração) usa o dia 12 de Setembro do ano respectivo"); the other 6 of 7 dates exact, ordering correct | `1973-12-25, 1975-03-12, ...` (1 of 7 dates wrong) |
| 29 | the "a meio caminho" (midpoint) anchored milestone computed as +1 month (`2031-08-14`) instead of the true midpoint `2031-08-23`; other 4 of 5 dates exact | `..., 2031-08-14, ...` (1 of 5 dates wrong) |

## Truncation stats

- **0/120 answers hit max_tokens=1500.** No truncated_incomplete items in this wave (Phase 2 had 2).
- Largest answer: 1,415 tokens (codigo-pt). Total 40,068 completion tokens
  (mean 334/item; codigo-ext 393, compras 381, docs-longos 228).
- Terse-answer pattern (Phase-2 observation) recurs in docs-longos: 13/40 answers were bare
  FINAL-line responses of 6–14 tokens — but ALL 13 passed. With thinking disabled the model
  "answers from intuition" on long-document extraction and was right every time it did so;
  its 2 docs-longos failures came from longer answers.

## AUDIT — genuine capability miss vs scorer/check artifact

Per-failure judgment (quoted mismatches in the table above; classes below):

| Class | Items | Count |
|---|---|---|
| Genuine — answered-with-a-program (value never computed; prompt said "aplica ... e indica o valor", model returned an unexecuted script whose last line would print FINAL) | codigo 51, 56, 58, 60, 61, 65, 66 | 7 |
| Genuine — instruction compliance (content right, prompt-mandated tool/command never used or named) | codigo 62 (pandas/tz), 68 (re.findall), 75 (git bisect commands) | 3 |
| Genuine — reading/anchor-rule miss (explicit dating rule overridden; midpoint miscomputed) | docs-longos 0, 29 | 2 |
| **Artifact-looking (check-authoring narrowness)** | codigo 45, 69, 76 | **3** |

- **Artifact rate: 3/15 = 20% < 30% → no escalation.**
- The 3 artifact-looking failures, with the quoted evidence:
  - codigo 45: answer = fix + `FINAL=35`, both correct; the only miss is
    `required_miss:TypeError|str.*int|texto|convert`. The prompt asked to "corrige o defeito
    minimo ... e indica o valor" — both done. The check additionally expects the answer to name
    the error/conversion in prose, which the prompt never demanded. Check over-narrow for a
    fully compliant terse answer.
  - codigo 69: `forbidden_hit:\((?:\?:)?(?:\\?[dws.]|\[[^\]]*\])[+*]\)[+*]` matched the answer
    text "(como \`(\w+)+\`)" — the model QUOTING the prompt's own forbidden example while
    correctly explaining its own regex avoids it. The regex it actually built and tested is the
    safe `[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*` and FINAL=3 is correct. The authored check verified
    no false positive on the correct regex but did not anticipate the answer echoing the
    prompt's counter-example.
  - codigo 76: `forbidden_hit:push\s+--force` matched inside "### Por que NÃO usar
    \`git reset --hard\` + \`git push --force\`?" — the exact explanation the prompt demands
    ("Explica porque NAO se deve usar 'git reset --hard' seguido de 'git push --force'").
    Complying with the prompt in natural PT-PT virtually requires writing the literal string;
    all required checks passed. Clear check-authoring bug (forbidden intended to catch
    *recommendations* of force-push, catches the mandated warning too).
- Classification rationale for the 7 "answered-with-a-program" items: these are NOT capture
  artifacts even though got_final shows f-string placeholders. In all 7 the code block is the
  ENTIRE answer (zero prose), the expected value appears nowhere, and no literal FINAL line
  exists — the model delegated the computation instead of performing it. Under the same checks
  a teacher that traces the code and writes `FINAL=210` passes. This is a real,
  workload-relevant gap (user asked for the value, got a script) and the dominant failure
  cluster of the wave.
- Sensitivity (transparency, mirrors Phase 2): the codigo-pt COMBINED verdict is
  threshold-sensitive to the audit. Scored: 5 + 13 = 18/80 = 22.5% → VIABLE. If the 3
  artifact-looking ext failures are discounted: 5 + 10 = 15/80 = 18.75% < 20% → would flip to
  NOT viable. Per the Phase-2/Phase-3 precedent this resolves empirically: failure-set items
  the teacher ALSO fails under identical checks don't count toward viability, so the teacher
  pass on the 18-item combined failure set settles the verdict without editing any instrument.

## Domain-level reading (for the next phase)

- **codigo-pt is the live domain of this wave** — and the signal is coherent: 7 of 13 ext
  failures are one behaviour ("emit a script instead of executing the task"), concentrated in
  the data-wrangling/refactor items that ask to *apply* code to a given input. The remaining
  genuine misses are mandated-tool/command omissions (62, 68, 75) — the same
  instruction-compliance family as base-set idx 9/12/14. Combined failure map spans both
  flavours of the domain gap.
- **compras-online at 0/40 is a dead end at this difficulty**: the 27B handles unit-price
  normalization, promo arithmetic, threshold traps, multi-constraint filtering, weighted-review
  rules and import/IVA chains flawlessly at temp 0. Distillation has no signal here.
- **docs-longos at 2/40 is nearly dead**: long-document needle-work (date anchors, exclusion
  clauses, quantifier traps, two-Anselmo disambiguation) is almost entirely solved; the 2
  misses are real but isolated (one world-knowledge override of an explicit rule, one date
  midpoint slip). Below any useful failure mass.

## Session extras (same server session, for the F3.B.0 nano gate)

- **Tokenizer probe** (`evidence/tokenizer_probe.json`): POST /tokenize on llama.cpp b9826 for
  the 3 exact protocol strings; raw token-id arrays saved
  ({"engine","model","probes":[{"text","tokens"}...]} schema). 13/17/16 tokens respectively.
- **Nano self-output dataset** (`datasets/nano_selfoutput.jsonl`): 50 NEW invented benign PT-PT
  clinical-communication prompts (exam explanations, short advice letters, med-instruction
  rewordings — none copied from any existing prompt file), answered by the student at
  temperature 0.7 / max_tokens 500, stored in `train_lora.py::load_split`'s expected schema
  ({"messages":[user,assistant]} + provenance field `"teacher": "student-self@nano-plumbing-test"`).
  Schema verified against the loader's exact logic (split 41 train / 5 val / 4 test; 0 empty
  answers). MANDATORY gate check: `python make_gate_blacklist.py --check
  datasets/nano_selfoutput.jsonl` → **0/50 contaminated — OK, exit 0**.

## Reproducibility

- Model: `HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced/Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf`
- Server: `llama-server.exe` (build `llama.cpp-b9826-cuda13.3`) `-ngl 999 --no-mmap -c 16384 -t 16 --jinja --port 18240`
  (logs: `evidence/server_student2b.log.*`). Longest docs-longos prompt ≈ 1,400 words — well
  inside c=16384; no context warnings in the server log.
- VRAM: 19,903–19,927 MiB steady during generation; baseline 1,608 MiB before launch,
  **1,608 MiB after shutdown** (confirmed both ends; no llama processes left).
- Client: `deltamoe/hardset_student_gen.py --phase2b` (new flag added this wave; Phase-2 code
  paths and evidence untouched; log: `evidence/hardset_student2_gen.log`). Generation wall time
  ≈ 9.6 min for 120 items (answers file created 16:21:57, last write 16:31:32); 40,068
  completion tokens.
- Scoring: `python deltamoe/hardset_score.py --hardset <hardset_X.jsonl> --answers
  evidence/hardset_student2.jsonl --out <scores>`, run once per domain (self-test green each
  run), outputs concatenated (codigo-ext, compras, docslongos order) into
  `evidence/hardset_student2_scores.jsonl`. Hard-set inputs and scorer untouched.

## Next step (no GPU)

Frontier-teacher pass on the 18-item codigo-pt COMBINED failure set (base idx 1, 9, 12, 14, 15
+ ext idx 45, 51, 56, 58, 60, 61, 62, 65, 66, 68, 69, 75, 76), teacher answering from prompts
only, scored by the SAME checks. Viability arithmetic per protocol: 70% of 18 = 12.6 → teacher
must pass ≥13 of the 18 as scored. The 3 artifact-looking checks (45, 69, 76) will likely fail
the teacher identically and drop out of the count, which the sensitivity note above already
prices in.

---

## Instrument v3 addendum (coordinator, 2026-07-04)

The 3 audit-flagged artifacts were confirmed as intent-clear check-authoring bugs and fixed with provenance (pre-fix backup: `evidence/hardset_codigo_ext.pre_v3.bak`; v2 scores preserved in `hardset_student2_scores.jsonl`; v3 scores in `hardset_student2_scores_v3.jsonl`):
- idx 45: dropped the over-demanding prose-mention required (`TypeError|str.*int|texto|convert`) — the prompt demands the fix and the value, both still checked (`int(`, `FINAL=35`, forbidden `FINAL=1020`).
- idx 69: dropped the anti-backtracking forbidden — it fired on answers quoting the prompt's OWN counter-example; regex cannot separate quotation from proposal; the safe-pattern required + `FINAL=3` + forbidden `FINAL=4` still discriminate.
- idx 76: dropped both forbiddens that fired on the warning text the prompt itself demands ("explica porque NÃO usar push --force"); the requireds (revert -m 1, push, partilhad/reescrev) capture the correct recommendation. Precision over recall, documented.

Re-score flips EXACTLY the 3 flagged items; no other verdict moved.

**Official (v3) verdicts:** compras-online 0/40 NOT viable; docs-longos 2/40 NOT viable; codigo-pt ext 10/40 — **the ext pool alone meets the per-40 runbook criterion (≥8/40)**; combined dual accounting: 15/80 = 18.75%, below the 20% combined line set in this wave's brief. Both accountings stand as written; the domain verdict is resolved empirically by the Phase-3 teacher pass on all 15 genuine failures (bar ≥11/15 = 70%) — items the teacher also fails count against viability, so the resolution cannot be gamed by check narrowness.

codigo-pt genuine failure set (15): base idx 1, 9, 12, 14, 15 + ext idx 51, 56, 58, 60, 61, 62, 65, 66, 68, 75. Dominant cluster (7): answers with an UNEXECUTED program ending `print(f"FINAL={var}")` — the student writes code that would compute the answer but never states the value; second cluster (3): prompt-mandated tool/command never used (pandas tz, re.findall, git bisect). The distillable behavior is execution discipline / instruction compliance, not algorithmic knowledge.
