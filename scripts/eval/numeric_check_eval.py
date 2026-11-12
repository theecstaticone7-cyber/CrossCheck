#!/usr/bin/env python3
"""Offline eval of the deterministic numeric checker + override policy.

Reads saved golden period reports (no API calls, no retrieval), runs the
numeric checker on each claim's saved passages, applies the override policy
to the saved LLM label/confidence, and reports before/after metrics.

Anti-overfitting split:

- **tune**     AAPL, AMZN, GOOGL — the override threshold is chosen here only.
- **held-out** META, NVDA — reported separately at the tuned threshold.

The checker receives only the claim text, the passage texts and the fiscal
quarter. Expected labels / ground-truth references are read only for scoring.

Outputs (``--out-dir``, default ``data/runs/golden/numeric_check/``)::

    summary.md     before/after tables, threshold grid, checker diagnostics
    metrics.json   same numbers, machine-readable
    per_claim.csv  one row per claim (verdict, labels, override)

Examples::

    python scripts/eval/numeric_check_eval.py
    python scripts/eval/numeric_check_eval.py --reports-dir path/to/reports/golden
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck.analysis.numeric_check import (  # noqa: E402
    NumericCheckResult,
    apply_override,
    check_claim,
)

LABELS = ("Consistent", "Contradictory", "Unverifiable")
TUNE_TICKERS = ("AAPL", "AMZN", "GOOGL")
HELDOUT_TICKERS = ("META", "NVDA")
# 0.0 = never override (baseline); 1.01 = override whenever decisive.
THRESHOLD_GRID = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01)

DEFAULT_REPORTS_DIR = ROOT / "data" / "reports" / "golden"
DEFAULT_OUT_DIR = ROOT / "data" / "runs" / "golden" / "numeric_check"


@dataclass
class EvalRow:
    claim_id: str
    ticker: str
    fiscal_quarter: str
    claim: str
    llm_label: str
    llm_confidence: float
    expected: str  # scoring only - never passed to the checker
    retrieval_hit: bool | None
    bucket_before: str
    check: NumericCheckResult
    hinted: bool  # report was produced with the hint already in the prompt


def _bucket(expected: str, predicted: str, retrieval_hit: bool | None) -> str:
    if expected == "Unverifiable":
        return "unverifiable_nli"
    ok = expected == predicted
    if retrieval_hit:
        return "both_ok" if ok else "nli_hallucination"
    return "lucky_nli" if ok else "both_fail"


def load_rows(reports_dir: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for path in sorted(reports_dir.glob("**/*_golden_eval.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        quarter = str(data.get("fiscal_quarter", ""))
        ticker = str(data.get("ticker", "")).upper()
        for c in data.get("claims", []):
            passages = [r.get("text") or "" for r in c.get("retrieved", [])]
            # Checker input: claim text + passages + quarter. Nothing else.
            check = check_claim(c["claim"], passages, fiscal_quarter=quarter)
            hinted = "numeric_check" in c
            llm_label = c.get("llm_predicted_nli_label") or c["predicted_nli_label"]
            expected = c["expected_nli_label"]
            hit = c.get("retrieval_hit")
            rows.append(
                EvalRow(
                    claim_id=c["claim_id"],
                    ticker=ticker,
                    fiscal_quarter=quarter,
                    claim=c["claim"],
                    llm_label=llm_label,
                    llm_confidence=float(c.get("nli_confidence") or 0.0),
                    expected=expected,
                    retrieval_hit=hit,
                    bucket_before=_bucket(expected, llm_label, hit),
                    check=check,
                    hinted=hinted,
                )
            )
    return rows


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    m = {e: {p: 0 for p in LABELS} for e in LABELS}
    for exp, pred in pairs:
        m[exp][pred] += 1
    return m


def _acc(pairs: list[tuple[str, str]]) -> tuple[int, int]:
    return sum(1 for e, p in pairs if e == p), len(pairs)


def evaluate(rows: list[EvalRow], threshold: float) -> dict[str, Any]:
    before: list[tuple[str, str]] = []
    after: list[tuple[str, str]] = []
    overrides = override_correct = fixed = broken = halluc_n = halluc_fixed = 0
    buckets_after: Counter[str] = Counter()
    for r in rows:
        new, overridden = apply_override(
            r.llm_label, r.llm_confidence, r.check, threshold=threshold
        )
        before.append((r.expected, r.llm_label))
        after.append((r.expected, new))
        buckets_after[_bucket(r.expected, new, r.retrieval_hit)] += 1
        if overridden:
            overrides += 1
            override_correct += int(new == r.expected)
        if r.llm_label != r.expected and new == r.expected:
            fixed += 1
        if r.llm_label == r.expected and new != r.expected:
            broken += 1
        if r.bucket_before == "nli_hallucination":
            halluc_n += 1
            halluc_fixed += int(new == r.expected)

    cx_before = [p for p in before if p[0] != "Unverifiable"]
    cx_after = [p for p in after if p[0] != "Unverifiable"]
    a_b, n = _acc(before)
    a_a, _ = _acc(after)
    cxb, ncx = _acc(cx_before)
    cxa, _ = _acc(cx_after)
    return {
        "threshold": threshold,
        "n": n,
        "acc_all_before": [a_b, n],
        "acc_all_after": [a_a, n],
        "acc_cx_before": [cxb, ncx],
        "acc_cx_after": [cxa, ncx],
        "confusion_before": _confusion(before),
        "confusion_after": _confusion(after),
        "buckets_before": dict(Counter(r.bucket_before for r in rows)),
        "buckets_after": dict(buckets_after),
        "nli_hallucination_n": halluc_n,
        "nli_hallucination_fixed": halluc_fixed,
        "fixed": fixed,
        "broken": broken,
        "overrides": overrides,
        "override_correct": override_correct,
        "override_precision": (override_correct / overrides) if overrides else None,
    }


def checker_diagnostics(rows: list[EvalRow]) -> dict[str, Any]:
    """Verdict x expected label, independent of any threshold."""
    table = {v: {e: 0 for e in LABELS} for v in ("MATCH", "MISMATCH", "NO_EVIDENCE")}
    for r in rows:
        table[r.check.verdict][r.expected] += 1
    m, x = table["MATCH"], table["MISMATCH"]
    n_match, n_mis = sum(m.values()), sum(x.values())
    return {
        "verdict_by_expected": table,
        "match_precision": (m["Consistent"] / n_match) if n_match else None,
        "mismatch_precision": (x["Contradictory"] / n_mis) if n_mis else None,
        "coverage": (n_match + n_mis) / len(rows) if rows else 0.0,
    }


def choose_threshold(tune_rows: list[EvalRow]) -> tuple[float, list[dict[str, Any]]]:
    """Best tune-split accuracy; ties go to the lower threshold (fewer overrides)."""
    grid = [evaluate(tune_rows, t) for t in THRESHOLD_GRID]
    best = max(grid, key=lambda g: (g["acc_all_after"][0], -g["threshold"]))
    return best["threshold"], grid


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _pct(pair: list[int]) -> str:
    k, n = pair
    return f"{(k / n * 100 if n else 0):.1f}% ({k}/{n})"


def _prec(m: dict[str, Any]) -> str:
    if m["override_precision"] is None:
        return "n/a"
    return f"{m['override_precision'] * 100:.0f}% ({m['override_correct']}/{m['overrides']})"


def before_after_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Split | Claims | NLI acc (all) before -> after | NLI acc (C/X) before -> after "
        "| nli_hallucination fixed | Correct -> broken | Overrides | Override precision |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['n']} | {_pct(m['acc_all_before'])} -> {_pct(m['acc_all_after'])} "
            f"| {_pct(m['acc_cx_before'])} -> {_pct(m['acc_cx_after'])} "
            f"| {m['nli_hallucination_fixed']}/{m['nli_hallucination_n']} "
            f"| {m['broken']} | {m['overrides']} | {_prec(m)} |"
        )
    return "\n".join(lines)


def confusion_table(m: dict[str, dict[str, int]]) -> str:
    lines = [
        "| Expected \\ predicted | Consistent | Contradictory | Unverifiable |",
        "| --- | ---: | ---: | ---: |",
    ]
    for e in LABELS:
        lines.append(f"| {e} | " + " | ".join(str(m[e][p]) for p in LABELS) + " |")
    return "\n".join(lines)


def grid_table(grid: list[dict[str, Any]]) -> str:
    lines = [
        "| Threshold | NLI acc (all) | Fixed | Broken | Overrides | Override precision |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for g in grid:
        t = "never" if g["threshold"] == 0.0 else (
            "always" if g["threshold"] > 1 else f"< {g['threshold']:.2f}"
        )
        lines.append(
            f"| {t} | {_pct(g['acc_all_after'])} | {g['fixed']} | {g['broken']} "
            f"| {g['overrides']} | {_prec(g)} |"
        )
    return "\n".join(lines)


def diag_table(d: dict[str, Any]) -> str:
    lines = [
        "| Checker verdict | Expected Consistent | Expected Contradictory | Expected Unverifiable |",
        "| --- | ---: | ---: | ---: |",
    ]
    for v, row in d["verdict_by_expected"].items():
        lines.append(f"| {v} | " + " | ".join(str(row[e]) for e in LABELS) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = load_rows(args.reports_dir)
    if not rows:
        print(
            f"No *_golden_eval.json reports under {args.reports_dir}.\n"
            "Run scripts/eval/run_golden_eval.py --full first, or pass --reports-dir."
        )
        sys.exit(1)

    tune = [r for r in rows if r.ticker in TUNE_TICKERS]
    held = [r for r in rows if r.ticker in HELDOUT_TICKERS]
    other = sorted({r.ticker for r in rows} - set(TUNE_TICKERS) - set(HELDOUT_TICKERS))
    if not tune:
        print("No tuning-split reports (AAPL/AMZN/GOOGL); cannot choose a threshold.")
        sys.exit(1)

    threshold, grid = choose_threshold(tune)
    results = {
        "tune (AAPL, AMZN, GOOGL)": evaluate(tune, threshold),
        "held-out (META, NVDA)": evaluate(held, threshold) if held else None,
        "all": evaluate(rows, threshold),
    }
    results = {k: v for k, v in results.items() if v is not None}
    diags = {
        "tune": checker_diagnostics(tune),
        "held-out": checker_diagnostics(held) if held else None,
        "all": checker_diagnostics(rows),
    }
    n_hinted = sum(r.hinted for r in rows)

    md: list[str] = [
        "# Numeric checker - offline golden eval",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Reports: `{args.reports_dir}` ({len(rows)} claims). No API calls.",
        "",
        "Policy: a decisive checker verdict (MATCH -> Consistent, MISMATCH -> "
        "Contradictory) replaces the saved Gemini label only when Gemini's "
        f"confidence is below the threshold. Threshold chosen on the tuning split "
        f"only: **{'never override' if threshold == 0 else ('always' if threshold > 1 else f'< {threshold:.2f}')}**.",
        "",
        "Offline this measures the **override** only. The prompt-hint half of the "
        "policy needs a live Gemini run (`run_golden_eval.py --full --numeric-check`).",
    ]
    if n_hinted:
        md += ["", f"Note: {n_hinted} claims come from reports that already had the "
               "hint in the prompt; their saved LLM label is used as 'before'."]
    if other:
        md += ["", f"Tickers outside both splits (counted only in 'all'): {', '.join(other)}."]
    md += ["", "## Before / after", "", before_after_table(results), ""]
    md += ["## Threshold grid (tuning split only)", "", grid_table(grid), ""]
    for name, m in results.items():
        md += [
            f"## Confusion matrix - {name}",
            "",
            "Before:",
            "",
            confusion_table(m["confusion_before"]),
            "",
            "After:",
            "",
            confusion_table(m["confusion_after"]),
            "",
        ]
    md += ["## Checker verdicts vs expected label (no threshold)", ""]
    for name, d in diags.items():
        if d is None:
            continue
        mp = "n/a" if d["match_precision"] is None else f"{d['match_precision'] * 100:.0f}%"
        xp = "n/a" if d["mismatch_precision"] is None else f"{d['mismatch_precision'] * 100:.0f}%"
        md += [
            f"**{name}** - coverage {d['coverage'] * 100:.0f}% decisive, "
            f"MATCH precision {mp}, MISMATCH precision {xp}",
            "",
            diag_table(d),
            "",
        ]
    summary = "\n".join(md)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    payload = {
        "reports_dir": str(args.reports_dir),
        "n_claims": len(rows),
        "tune_tickers": TUNE_TICKERS,
        "heldout_tickers": HELDOUT_TICKERS,
        "chosen_threshold": threshold,
        "threshold_grid_tune": grid,
        "results": results,
        "checker_diagnostics": diags,
        "claims_with_hint_in_saved_report": n_hinted,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "per_claim.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "claim_id", "ticker", "quarter", "split", "expected", "llm_label",
            "llm_confidence", "verdict", "after", "overridden", "passage_index",
            "claim_figure", "passage_figure", "passage_label", "claim",
        ])
        for r in rows:
            after, ov = apply_override(r.llm_label, r.llm_confidence, r.check, threshold=threshold)
            split = "tune" if r.ticker in TUNE_TICKERS else (
                "held-out" if r.ticker in HELDOUT_TICKERS else "other"
            )
            w.writerow([
                r.claim_id, r.ticker, r.fiscal_quarter, split, r.expected, r.llm_label,
                f"{r.llm_confidence:.2f}", r.check.verdict, after, ov,
                r.check.passage_index, r.check.claim_figure, r.check.passage_figure,
                r.check.passage_label, r.claim,
            ])

    print(summary)
    print()
    print(f"wrote {args.out_dir / 'summary.md'}")
    print(f"wrote {args.out_dir / 'metrics.json'}")
    print(f"wrote {args.out_dir / 'per_claim.csv'}")


if __name__ == "__main__":
    main()
