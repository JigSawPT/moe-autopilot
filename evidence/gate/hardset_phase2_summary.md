# F3.A.2 Phase 2 — Dense-27B student on the 120-prompt hard set (2026-07-04)

Student pass of the failure-set gate (docs/13_deltamoe_runbook.md, "F3.A.2"). The dense 27B
(Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P) answered all 120 hard-set prompts at
temperature 0.0, max_tokens 1500, bare user message, `chat_template_kwargs: {"enable_thinking": false}`
(same student convention as the cheap-gate runs). Deterministic scoring by `deltamoe/hardset_score.py`.

- Answers: `evidence/hardset_student.jsonl` (120/120, 0 request errors, no retries needed)
- Scores: `evidence/hardset_student_scores.jsonl`
- Scorer self-test: **87/87** synthetic `FINAL=<expected>` cases pass (37 codigo + 10 clinico +
  40 raciocinio; the other 33 clinico items have `expected_final: null` — structure-only checks).
  Self-test ran before scoring, as mandated.

## Per-domain results vs the ≥8-failures viability threshold

| Domain | Pass | Fail | ≥8 student failures? | Verdict |
|---|---|---|---|---|
| codigo-pt | 35 | **5** | no (5 < 8) | NOT viable at this difficulty |
| pt-clinico | 27 | **13** | **yes (13 ≥ 8)** | **VIABLE — proceed to teacher pass** |
| raciocinio | 34 | **6** | no (6 < 8) | NOT viable at this difficulty |

Total: 96/120 pass (80%). Only pt-clinico produces enough failure signal to train toward.
Phase 3 (frontier-teacher pass on the failure set, same deterministic checks) applies to the
24 failed items; the ≥8 + teacher-passes-≥70% viability criterion is only reachable by pt-clinico.

## FAILURE SET (24 items)

### codigo-pt (5): idx 1, 9, 12, 14, 15

| idx | One-line reason | got_final |
|---|---|---|
| 1 | truncated at 1500 tokens mid mod-97 digit-by-digit trace, before FINAL (`no_final`) | — |
| 9 | never wrote the required `csv.reader` code (prompt demanded it); traced the CSV by hand — parse result itself was right | `2:2` (correct value) |
| 12 | never wrote the required `sorted(`/`key=` call and emitted FINAL with spaces (`a, b, aa, bb, cc` vs exact `a,b,aa,bb,cc`) | `a, b, aa, bb, cc` (correct order) |
| 14 | FINAL emitted with spaces (`3, 1, 2, 4, 5`) vs the exact-format required regex `3,1,2,4,5` | `3, 1, 2, 4, 5` (correct order) |
| 15 | miscounted `split(' ')` elements: answered 3:12, expected 3:10 | `3:12` |

### pt-clinico (13): idx 0, 1, 5, 11, 13, 14, 15, 16, 17, 19, 25, 26, 39

| idx | One-line reason | got_final |
|---|---|---|
| 0 | wrapped all 4 mandated section headers in markdown bold (`**Motivo da referenciacao:**`), breaking the `^`-anchored exact-header checks | — |
| 1 | same markdown-bold header pattern on all 4 mandated headers (also wrote BR-accented "isquêmico", uncaught by any forbidden regex) | — |
| 5 | omitted the mandated age fact "47" from the discharge summary | — |
| 11 | wrote "diabetes mellitus tipo 2" — the required literal `diabetes tipo 2` never appears contiguously | — |
| 13 | pre-AO90 rewrite left "receção" unconverted (should be "recepção"); other 3 words converted | — |
| 14 | pre-AO90 rewrite returned the text essentially unchanged (atividade/direto/adoção/objetivo all still AO90) | — |
| 15 | pre-AO90 partial: converted infecção/acção/selecção/correcta but left "direto" and "ótima" | — |
| 16 | pre-AO90 partial: left "diretor" (should be "director"); converted the other 4 | — |
| 17 | pre-AO90 partial: left "atual" (should be "actual"); converted reacção/direcção/acção/correcta | — |
| 19 | pre-AO90 partial: converted aspecto/infecção but left "ação", "reação", "direta" | — |
| 25 | numerically correct Cockcroft-Gault (58.3) but FINAL carried the unit "mL/min", failing normalization | `58.3 mL/min` (correct value) |
| 26 | numerically correct (59,0 → 59.0) but FINAL carried "mL/min", failing normalization | `59,0 mL/min` (correct value) |
| 39 | wrote "dificuldade respiratória" — not matched by the required alternation `(dificuldade em respirar\|falta de ar\|dispneia)` | — |

### raciocinio (6): idx 0, 3, 14, 22, 23, 29

| idx | One-line reason | got_final |
|---|---|---|
| 0 | truncated at 1500 tokens enumerating the 24 setup permutations, before FINAL (`no_final`) | — |
| 3 | answered literally `FINAL=Rui` (4 tokens, zero visible work); constraint propagation gives Alfredo at 11:00 | `Rui` |
| 14 | real-value compound interest off by 0.74 EUR (5260,04 vs 5259.30) — intermediate rounding instead of full precision | `5260,04` |
| 22 | answered literally `FINAL=2`; 4 January is by ISO-8601 definition always week 1 | `2` |
| 23 | 4th Wednesday of Mar-2026 is the 25th (holiday) → 26/03; model answered 28/03 (a Saturday — not even a Wednesday+1) | `28/03` |
| 29 | ordered `reuniao,emails,...` violating the explicit "emails before reuniao" clue | `reuniao,emails,almoco,relatorio,revisao` |

## Truncation stats

- 2/120 answers hit max_tokens=1500: **codigo-pt idx 1** and **raciocinio idx 0**.
- Both failed ONLY on FINAL-related checks (the cut landed before the FINAL line; all their other
  required/forbidden checks pass) → both are marked **truncated_incomplete**. Per the temp-0/1500
  protocol they still count as failures, but they are generation-budget artifacts, not clean
  capability evidence — same failure family as the 4 max_tokens=600 truncations in the run-2
  raciocinio anchor check. Both models in Phase 3 face the same 1500-token discipline, so the
  comparison stays symmetric.
- No other answer came near the cap (next-largest n_tokens_out = 1244).

## AUDIT — genuine capability miss vs scorer/normalization artifact

Per-failure judgment (mismatches quoted above):

| Class | Items | Count |
|---|---|---|
| Genuine — knowledge/logic/arithmetic | codigo 15; clinico 13, 14, 15, 16, 17, 19; raciocinio 3, 14, 22, 23, 29 | 12 |
| Genuine — instruction compliance (content right, explicit exact-format/usage instruction violated) | codigo 9, 12, 14; clinico 0, 1, 5 | 6 |
| Truncated_incomplete (budget, not capability) | codigo 1; raciocinio 0 | 2 |
| **Artifact-looking (scorer/check narrowness)** | clinico 11, 25, 26, 39 | **4** |

- **Artifact rate: 4/24 = 16.7% < 30% → no escalation.**
- The 4 artifact-looking failures, with the quoted mismatch:
  - clinico 25/26: `final_mismatch: 58.3 ml/min ≠ 58.3` and `59.0 ml/min ≠ 59.0`. The values are
    correct; the unit-strip list in the scoring contract has `ml/h` and `mg/dl` but not `ml/min`,
    so stripping `ml` and `min` individually leaves a dangling `/` and blocks the numeric
    comparison. (The prompts did say `FINAL=<valor em mL/min>`, which invites the unit.)
  - clinico 11: `required_miss: diabetes tipo 2` against an answer saying "diabetes mellitus
    tipo 2" — clinically more precise, regex too literal.
  - clinico 39: `required_miss: (dificuldade em respirar|falta de ar|dispneia)` against
    "dificuldade respiratória" — synonymous PT-PT phrasing outside the alternation.
- Classification rationale for the 6 "instruction compliance" items: the scorer faithfully
  implements the authored checks, and every one of these prompts demanded the exact form
  ("EXACTAMENTE estes cabecalhos", "formato exacto FINAL=", "usa o módulo csv (csv.reader…)").
  Emitting `**Motivo…:**` when told "exactly these headers, each on its own line", or FINAL with
  spaces when shown the exact target string, is an instruction-following failure — a real,
  workload-relevant capability gap (and one the teacher can demonstrably beat under the same
  checks). They are therefore counted genuine, not artifact.
- Sensitivity (transparency): even if a reader reclassifies all 6 compliance failures as
  artifacts, the artifact rate would be 10/24 = 41.7% — above the escalation line. The
  classification above is defended item-by-item precisely because of this sensitivity; the
  borderline cases are all quoted in full so Phase 3 can re-derive either way. Note pt-clinico
  viability is robust to removing the 4 artifact-looking items (13 → 9 ≥ 8) but not to removing
  the 2 markdown-header items as well (→ 7 < 8); the teacher pass on all 13 will settle this
  empirically — items the teacher also fails don't count toward viability anyway.

## Domain-level reading (for Phase 3)

- **pt-clinico (13 fails) is the live domain.** The dominant cluster is pre-AO90 orthography:
  6/8 pre-AO90 rewrite items failed (partial conversions — the model reliably converts
  infecção/acção-type words but misses atual→actual, direto→directo, ótima→óptima,
  receção→recepção, diretor→director). This directly refutes the run-1/run-2 impression
  (single easy pre-AO90 item, aced) and is exactly the narrow, checkable, workload-flavoured
  gap the failure-set gate was designed to surface. The markdown-header compliance pair and
  the BR slips ("isquêmico") corroborate the run-2 register findings.
- codigo-pt (5) and raciocinio (6) sit below the ≥8 bar: the 27B handles trap-heavy code
  semantics and multi-step quantitative reasoning too well at realistic-hard difficulty for
  distillation to have enough signal. Consistent with the run-2 conclusion that the broad gap
  lives only where the student fails.
- Terse-answer pattern worth noting: with thinking disabled, several raciocinio answers were
  bare `FINAL=...` with zero visible derivation (n_tokens_out = 4–16). Two of those were wrong
  (idx 3, 22), one right (idx 36, `2296`). The failure mode is "answers from intuition instead
  of deliberating" — a plausible distillation target only if Phase 3 shows the teacher
  deliberates under the same convention.

## Reproducibility

- Model: `HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced/Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf`
- Server: `llama-server.exe` (build `llama.cpp-b9826-cuda13.3`) `-ngl 999 --no-mmap -c 16384 -t 16 --jinja --port 18240`
  (logs: `evidence/server_hardset_student.log.err`)
- VRAM: 19,905 MiB steady during generation; baseline 1,609 MiB before launch, 1,608 MiB after
  shutdown (confirmed both ends).
- Client: `deltamoe/hardset_student_gen.py` (mirrors `cheapgate_client.py` request shape; log:
  `evidence/hardset_student_gen.log`). Runtime ≈ 21 min for 120 items; 48,478 completion tokens
  total (mean 404/item; codigo 452, clinico 194, raciocinio 566).
- Scoring: `python deltamoe/hardset_score.py --hardset <hardset_X.jsonl> --answers evidence/hardset_student.jsonl --out <scores>`,
  run once per domain, outputs concatenated (codigo, clinico, raciocinio order) into
  `evidence/hardset_student_scores.jsonl`. Hard-set input files untouched.

## Next step (Phase 3, no GPU)

Frontier-teacher (Claude) answers the 24 failure-set items only; scored by the SAME
`hardset_score.py` checks. Viability per domain = ≥8 student failures AND teacher passes ≥70%
of them. Only pt-clinico can reach it (needs teacher pass on ≥10 of its 13; if the 4
artifact-looking checks fail the teacher identically, they simply don't count toward viability,
leaving 9 live items — still enough only if the teacher passes ≥7... arithmetic per protocol:
70% of 13 = 9.1 → teacher must pass ≥10 of the 13 as scored).

---

## Instrument v2 addendum (coordinator, 2026-07-04)

The 4 audit-flagged artifact failures were confirmed as instrument bugs and fixed with provenance:
- `hardset_clinico.jsonl` idx 11: required literal `diabetes tipo 2` → `diabetes(\s+mellitus)?\s+tipo\s+2` (the student wrote the clinically fuller form).
- `hardset_clinico.jsonl` idx 39: symptom alternation extended with `dificuldade\s+respirat[óo]ria`.
- `hardset_score.py`: UNIT_TOKENS extended with compound clinical/commerce units (ml/min, mmol/l, mg/ml, mg/kg, mg/dia, mcg/kg/min, gotas/min, l/100km, kwh, g, l, dl, ui, ug) and `strip_units` now drops slashes orphaned by unit removal (a slash survives only between two digits, preserving 120/80-style multipart values).
- Provenance: pre-fix hardset backed up at `evidence/hardset_clinico.v1.bak`; v1 scores preserved untouched (`hardset_student_scores.jsonl`); v2 scores written to `hardset_student_scores_v2.jsonl`. Protocol note honoured: scorer bugs are fixed in the scorer; the two hardset edits are intent-clear check-authoring bugs, not tuning (both loosen toward clinically-equivalent phrasings).
- Post-fix self-test: 87/87. Regression probes green (58.3 ml/min≡58.3; 120/80 preserved; sim/nao≠sim). Re-score flips EXACTLY the 4 flagged items (pt-clinico 11, 25, 26, 39 FAIL→PASS); no other verdict moved.

**Official (v2) failure map: codigo-pt 5/40; pt-clinico 9/40 (VIABLE — ≥8 threshold met); raciocinio 6/40 (2 of the 6 are max_tokens truncation artifacts).**
pt-clinico failure set (v2): idx 0, 1, 5, 13, 14, 15, 16, 17, 19. Phase-3 teacher bar: ≥7/9 (70%) under the SAME checks, teacher answering from prompts only (checks never shown to the teacher).
