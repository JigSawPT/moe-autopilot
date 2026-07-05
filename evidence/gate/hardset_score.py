# DeltaMoE F3.A.2 hard-set deterministic scorer.
#
# Reusable for both the student pass and the (later) teacher pass on the failure set —
# the contract below is domain-agnostic and driven entirely by each item's "checks" block.
#
# CLI:
#   python hardset_score.py --hardset <hardset.jsonl> --answers <answers.jsonl> --out <scores.jsonl>
#
# Contract (exact, per docs/13_deltamoe_runbook.md F3.A.2):
#   required:  every regex (compiled re.I|re.M|re.S) must match the answer;
#              else FAIL "required_miss:<pattern>".
#   forbidden: no regex may match; else FAIL "forbidden_hit:<pattern>".
#   expected_final (when non-null): take the LAST occurrence of `FINAL\s*=\s*(.+)` in the
#              answer (captured group runs to end of line); if absent -> FAIL "no_final".
#              Normalize both sides (see normalize_value / values_match below); mismatch ->
#              FAIL "final_mismatch:<got_norm>≠<exp_norm>".
#   PASS only if all applicable checks pass (an item with no required/forbidden/expected_final
#   trivially PASSes -- none of the hard-set items are like that, but the contract is general).
#
# Output: one JSON record per item:
#   {"domain","idx","pass":bool,"reasons":[...],"got_final":"<raw>"}
# "reasons" is empty on PASS. "got_final" is the raw (non-normalized) captured text, or ""
# if no FINAL= line was found (independent of whether expected_final is null for this item).
#
# Mandatory self-test (run automatically before scoring; see run_self_test()): for every
# hardset item with a non-null expected_final, the synthetic answer "FINAL=" + expected_final
# must PASS the final-comparison logic. This catches normalization bugs before they're blamed
# on the model. Reports "<n_pass>/<n_with_final> self-test" and aborts scoring (exit 1) if any
# synthetic case fails -- fix the scorer, never the hardset, per protocol.
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

FINAL_RE = re.compile(r"FINAL\s*=\s*(.+)", re.I | re.M)

# Unit/currency tokens stripped before the "purely numeric" check. Matched case-insensitively
# as whole tokens (word-boundary), not as substrings, so e.g. "mgabc" is not affected.
UNIT_TOKENS = [
    "mcg/kg/min", "ml/min", "mg/dl", "mmol/l", "mg/ml", "mg/kg", "mg/dia", "ml/h", "l/100km",
    "gotas/min", "tok/s", "kwh", "mg", "ml", "mcg", "ug", "kg", "km", "g", "l", "dl", "ui",
    "eur", "euros", "h", "min", "s", "%", "€",
]
# Longer tokens first so e.g. "mg/dl" is stripped whole before "mg" would otherwise leave a
# dangling "/dl". Sorted once at import time.
UNIT_TOKENS = sorted(UNIT_TOKENS, key=len, reverse=True)


def compile_checks(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.I | re.M | re.S) for p in patterns]


def strip_accents(s: str) -> str:
    # Unicode NFKD accent-strip: decompose then drop combining marks.
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_string(s: str) -> str:
    """Shared textual normalization applied to both sides before any comparison."""
    s = strip_accents(s)
    s = s.lower().strip()
    # decimal comma -> point, only between digits (avoid touching thousands separators/lists)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    # drop trailing punctuation: . ; : ! (possibly repeated, possibly with trailing whitespace)
    s = re.sub(r"[.;:!\s]+$", "", s)
    # collapse whitespace runs
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_units(s: str) -> str:
    """Remove known currency/unit tokens (whole-word) so pure numeric comparison is possible."""
    out = s
    for tok in UNIT_TOKENS:
        # word-boundary-ish strip; tokens like "%" and "€" have no \w boundary, handle directly
        if tok in ("%", "€"):
            out = out.replace(tok, "")
        else:
            out = re.sub(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", "", out, flags=re.I)
    # drop slashes left orphaned by unit removal ("58.3 /" from "58.3 ml/min");
    # a slash survives only between two digits (multi-part values like 120/80)
    out = re.sub(r"/(?!\d)|(?<!\d)/", " ", out)
    return out


def is_purely_numeric(s: str) -> bool:
    stripped = strip_units(s).strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", stripped))


def to_float(s: str) -> float:
    return float(strip_units(s).strip())


def split_multipart(s: str) -> list[str]:
    """Split on / or , for multi-part values (e.g. blood pressure 120/80, or list answers)."""
    parts = re.split(r"[/,]", s)
    return [p.strip() for p in parts if p.strip() != ""]


def values_match(got_raw: str, exp_raw: str) -> bool:
    """Compare a captured FINAL value against the expected value per the normalization contract."""
    got_norm = normalize_string(got_raw)
    exp_norm = normalize_string(exp_raw)

    got_parts = split_multipart(got_norm)
    exp_parts = split_multipart(exp_norm)

    # If splitting yields a different number of parts, multi-part comparison isn't meaningful;
    # fall back to whole-string comparison (numeric if both sides qualify, else string).
    if len(got_parts) != len(exp_parts) or len(exp_parts) <= 1:
        return _scalar_match(got_norm, exp_norm)

    return all(_scalar_match(g, e) for g, e in zip(got_parts, exp_parts))


def _scalar_match(got_norm: str, exp_norm: str) -> bool:
    if is_purely_numeric(got_norm) and is_purely_numeric(exp_norm):
        try:
            return abs(to_float(got_norm) - to_float(exp_norm)) <= 1e-9 * max(1.0, abs(to_float(exp_norm)))
        except ValueError:
            return got_norm == exp_norm
    return got_norm == exp_norm


def extract_final(answer: str) -> Optional[str]:
    """Return the raw captured text of the LAST FINAL=... occurrence, or None if absent."""
    matches = list(FINAL_RE.finditer(answer))
    if not matches:
        return None
    return matches[-1].group(1)


def score_item(item: dict, answer: str) -> dict:
    reasons: list[str] = []
    checks = item.get("checks", {}) or {}

    required = checks.get("required") or []
    for pattern in required:
        if not re.search(pattern, answer, re.I | re.M | re.S):
            reasons.append(f"required_miss:{pattern}")

    forbidden = checks.get("forbidden") or []
    for pattern in forbidden:
        if re.search(pattern, answer, re.I | re.M | re.S):
            reasons.append(f"forbidden_hit:{pattern}")

    expected_final = checks.get("expected_final")
    got_final_raw = extract_final(answer)
    if expected_final is not None:
        if got_final_raw is None:
            reasons.append("no_final")
        else:
            if not values_match(got_final_raw, str(expected_final)):
                got_norm = normalize_string(got_final_raw)
                exp_norm = normalize_string(str(expected_final))
                reasons.append(f"final_mismatch:{got_norm}≠{exp_norm}")

    return {
        "domain": item["domain"],
        "idx": item["idx"],
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "got_final": got_final_raw if got_final_raw is not None else "",
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_self_test(hardset: list[dict]) -> None:
    """Mandatory pre-flight: synthetic 'FINAL=' + expected_final must PASS the final check
    for every item with a non-null expected_final. Aborts (exit 1) on any failure -- per
    protocol the scorer gets fixed, never the hardset."""
    with_final = [it for it in hardset if it.get("checks", {}).get("expected_final") is not None]
    failures = []
    for item in with_final:
        exp = str(item["checks"]["expected_final"])
        synthetic_answer = f"FINAL={exp}"
        got_final_raw = extract_final(synthetic_answer)
        ok = got_final_raw is not None and values_match(got_final_raw, exp)
        if not ok:
            failures.append((item["domain"], item["idx"], exp, got_final_raw))

    n_total = len(with_final)
    n_pass = n_total - len(failures)
    print(f"SELF-TEST: {n_pass}/{n_total} synthetic FINAL= cases pass the final comparison.")
    if failures:
        print(f"SELF-TEST FAILURES ({len(failures)}):")
        for domain, idx, exp, got in failures:
            print(f"  domain={domain} idx={idx} expected_final={exp!r} extracted={got!r}")
        raise SystemExit(
            "Scorer self-test failed -- fix hardset_score.py normalization logic before "
            "scoring real answers. Do NOT edit the hardset."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score hard-set answers deterministically (F3.A.2).")
    ap.add_argument("--hardset", required=True, type=Path, help="Path to hardset_<domain>.jsonl")
    ap.add_argument("--answers", required=True, type=Path, help="Path to answers JSONL "
                     "({domain, idx, answer, ...} per line)")
    ap.add_argument("--out", required=True, type=Path, help="Path to write per-item scores JSONL")
    ap.add_argument("--skip-self-test", action="store_true",
                     help="Skip the mandatory self-test (debugging only -- not for real runs)")
    args = ap.parse_args()

    hardset_rows = load_jsonl(args.hardset)
    if not args.skip_self_test:
        run_self_test(hardset_rows)

    answers_rows = load_jsonl(args.answers)
    answers_by_key = {(row["domain"], row["idx"]): row["answer"] for row in answers_rows}

    results = []
    n_pass = 0
    missing = []
    for item in hardset_rows:
        key = (item["domain"], item["idx"])
        if key not in answers_by_key:
            missing.append(key)
            result = {
                "domain": item["domain"],
                "idx": item["idx"],
                "pass": False,
                "reasons": ["missing_answer"],
                "got_final": "",
            }
        else:
            result = score_item(item, answers_by_key[key])
        results.append(result)
        if result["pass"]:
            n_pass += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if missing:
        print(f"WARNING: {len(missing)} hardset items had no matching answer: {missing}")
    print(f"SCORED {len(results)} items: {n_pass} PASS, {len(results) - n_pass} FAIL -> {args.out}")


if __name__ == "__main__":
    main()
