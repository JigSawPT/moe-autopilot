# DeltaMoE gate evidence (staged, not published)

**Status: staged for the user's explicit publish decision. Not committed to any git repository,
not pushed anywhere.** This folder exists so the "available on request" claim in write-up 2 can
become a concrete file set if and when the user decides to publish it.

## Honest framing (read this before anything else)

DeltaMoE is a **frozen research program**, not a shipped result. The idea was: distill a small
set of LoRA adapters from a stronger "teacher" model into a clean dense 27B "student," targeted at
domains where the student demonstrably underperforms. The methodology that survived two failed
gate designs is a **failure-set gate**: instead of asking an LLM judge to rate answer quality on a
1-5 rubric (which saturates once the student is already good — see `cheapgate2_summary.md` and the
runbook's "rubric saturation" diagnosis), the gate authors a hard, realistic, machine-checkable
prompt set per domain, runs the student on it at temperature 0, and looks at what the student
*actually gets wrong*. A domain is declared **viable** only if the student fails enough of the hard
set (≥8/40) to have a trainable signal, AND a frontier teacher (Claude) passes most of those
specific failures (≥70%) under the identical deterministic checks. Two prior domains (raciocinio,
reasoning) and two more (compras-online, docs-longos) were **not viable** — the student was already
too good at realistic difficulty for a distillable gap to exist. **Exactly two domains cleared the
bar: pt-clinico (9/9 teacher pass) and codigo-pt (15/15 teacher pass)** — see
`13_deltamoe_runbook.md` §F3.A.2 for the full table. The program is frozen at that point: the
methodology is validated end-to-end (cheap gate → failure-set gate → teacher-pass viability → nano
MTP×LoRA training-pipeline gate, which also passed), but the next step (bulk teacher-generation +
full QLoRA training for the two viable domains) was paused by user decision, not by a negative
result. Nothing here claims a finished distillation — it claims a validated, two-domain-viable
*gate methodology*, with the raw evidence to back that specific claim.

## Index

### Methodology
- **`13_deltamoe_runbook.md`** — the executable runbook (F3.0 through F3.D) with the outcome
  tables for every gate stage: the cheap rubric gate (killed 3/3 domains, both with a local and a
  frontier teacher pool), the rubric-saturation diagnosis that motivated the redesign, the
  failure-set gate design (F3.A.2), its per-domain outcome table (pt-clinico and codigo-pt VIABLE;
  raciocinio/compras-online/docs-longos not viable), and the F3.B.0 nano MTP×LoRA training-pipeline
  gate (PASS, 118 tok/s serving). This is the file to read first for the "what is this and why"
  narrative — everything else here is the evidence backing its claims.

### Hard-sets (the failure-set gate's test data)
- **`hardset_codigo.jsonl`** (40 items, idx 0-39) — codigo-pt base hard-set: Python/SQL/PowerShell
  traps (mod-97 IBAN, banker's rounding, DST boundaries, off-by-one slicing, SQL NULL semantics,
  window functions, PowerShell array-unrolling, etc.), each with a deterministic `checks` block
  (`required`/`forbidden` regex and/or an executed `expected_final` value).
- **`hardset_codigo_ext.jsonl`** (40 items, idx 40-79) — codigo-pt extension hard-set: debugging
  from a traceback, refactoring under an anti-pattern constraint, pandas/CSV data-wrangling,
  regex authoring, git recovery scenarios, and Dockerfile/compose reasoning.
- **`hardset_clinico.jsonl`** (40 items) — pt-clinico hard-set: structured clinical documents
  (referral letters, discharge notes, SOAP notes, informed-consent explanations) under strict
  European-Portuguese terminology and mandatory-section-header constraints, pre-AO90 orthography
  rewrites, EN→PT-PT clinical translation, dosage/Cockcroft-Gault arithmetic, and emergency-triage
  scripts (112 number, no definitive diagnosis).
- **`hardset_raciocinio.jsonl`** (40 items) — reasoning hard-set: multi-step arithmetic, logic
  puzzles, scheduling, compound-interest, ISO-8601 week/DST edge cases. **Not viable** (only 6/40
  student failures, below the ≥8 threshold; 2 of those 6 are token-budget truncations, not
  reasoning errors).
- **`hardset_compras.jsonl`** (40 items) — shopping-math hard-set (unit-price traps, promo/threshold
  logic, multi-criteria filtering). **Not viable** (0/40 student failures — the student aces this
  domain at temperature 0).
- **`hardset_docslongos.jsonl`** (40 items) — long-document reading hard-set (date-anchor traps,
  exclusion clauses, quantifier traps in 650-1300+ word passages). **Not viable** (2/40 failures,
  below threshold).

### Scorer
- **`hardset_score.py`** — the deterministic, domain-agnostic scorer used for every hard-set above
  and for the teacher pass. Contract: every `required` regex must match, no `forbidden` regex may
  match, and the last `FINAL=...` occurrence (when `expected_final` is non-null) must match the
  expected value under tolerant normalization (accents, case, comma/point decimals, unit-stripping,
  multi-part comparison). Runs a mandatory self-test (every hard-set item's own `expected_final`
  must round-trip through the comparison logic) before scoring anything real — this is what caught
  and fixed the four/three check-authoring bugs documented in the phase-2/2b summaries below.

### Gate-outcome summaries (the evidence backing the runbook's tables)
- **`cheapgate2_summary.md`** — the rubric-gate rerun with a frontier-Claude teacher pool (after
  the first, local-teacher rubric gate killed all 3 domains). All 3 domains KILL again, but the
  deterministic raciocinio anchor (Claude 30/30, student 26/26-on-completed-items) and the pt-clinico
  dosage/pre-AO90 anchors reveal the rubric was saturated, not that no gap exists — this is the file
  that motivated abandoning the rubric-delta gate for the failure-set gate.
- **`hardset_phase2_summary.md`** — the failure-set gate's first student pass (120 items across
  codigo-pt/pt-clinico/raciocinio base hard-sets). pt-clinico clears the ≥8-failure viability bar
  (13/40, later 9/40 after fixing 4 audited check-authoring bugs with full provenance); the other
  two domains do not.
- **`hardset_phase2b_summary.md`** — the second student pass on 3 NEW 40-item hard-sets
  (codigo-pt extension, compras-online, docs-longos). codigo-pt's extension pool produces enough
  failure signal to make the *combined* codigo-pt domain viable (18/80, later 15/80 after fixing 3
  audited check bugs); compras-online and docs-longos are dead ends at this difficulty.
- **`cheapgate_raciocinio_regen.md`** — a narrow, honest correction task: regenerating the
  raciocinio student's cheap-gate answers at a higher `max_tokens` to check whether 4 truncated
  answers were genuine reasoning failures or a generation-budget artifact (confirmed: artifact, all
  4 fixed; one unrelated new miss surfaced from temperature-0.7 resampling, handled by a surgical
  splice rather than a full-file replace so untouched data isn't re-rolled on sampling noise).

## What is deliberately NOT staged here

- The raw teacher/student/judge answer JSONLs (`cheapgate_*_student.jsonl`,
  `cheapgate_*_teacher.jsonl`, `hardset_student*.jsonl`, judge `*.jsonl` files, etc.) and the raw
  server logs, tokenizer probes, and training logs referenced throughout the summaries above. Those
  are large, numerous, and mostly mechanical (model completions + deterministic scores) — the
  summaries above already quote every specific claim (exact deltas, exact failure items, exact
  quoted mismatches) needed to audit the gate's conclusions. They can be staged on request if finer
  provenance is wanted.
- `docs/12_deltamoe_blueprint.md` (the pivot-rationale design doc the runbook points back to) —
  narrower in scope than the runbook's own outcome sections; omitted to keep this bundle focused on
  gate *evidence* rather than program design history.
- Anything from the F3.B.0 nano MTP×LoRA gate's raw evidence (`evidence/f3b0_*`, 19 files) beyond
  the one-paragraph outcome already quoted in the runbook — that gate is a training-*pipeline*
  smoke test (does LoRA+MTP work at all), not part of the failure-set gate's domain-viability
  evidence this bundle is centered on.

## Sanitization note

One hard-set item (`hardset_codigo.jsonl` idx 37, a PowerShell file-move prompt) originally
referenced a literal local folder path as its test fixture; both the prompt text and its
`checks.required` regex were changed to a generic placeholder path consistently, with no change to
the item's difficulty, intent, or pass/fail semantics. No other content changes were made to the
hard-sets or the scorer. The methodology summaries had a handful of absolute local build/model
paths genericized to relative binary/model names; no data or findings were altered.
