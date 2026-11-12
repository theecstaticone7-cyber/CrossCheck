"""Tests for the deterministic numeric checker."""

from __future__ import annotations

import pytest

from crosscheck.analysis.numeric_check import (
    NumericCheckResult,
    apply_override,
    check_claim,
    extract_claim_figures,
    extract_passage_figures,
    parse_cell,
)

INCOME_STATEMENT = """\
CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except per-share amounts)
| | Three Months Ended | |
| --- | --- | --- |
| | December 28, 2024 | December 30, 2023 |
| Net sales: | | |
| Products | $ | 69,138 | $ | 66,893 |
| Services | | 26,340 | | 23,117 |
| Total net sales | | 94,930 | | 85,777 |
| Other income/(expense), net | | (248) | | (50) |
| Earnings per share: | | |
| Basic | $ | 2.41 | $ | 2.19 |
| Diluted | $ | 2.40 | $ | 2.18 |
"""

MARGIN_TABLE = """\
| | Q1 2025 | Q1 2024 |
| --- | --- | --- |
| Gross margin percentage | 46.9 | % | 45.9 | % |
"""


# --- scale and rounding -----------------------------------------------------


def test_claim_scale_and_precision() -> None:
    (fig,) = extract_claim_figures("Total revenue was $94.9 billion.")
    assert fig.kind == "dollar"
    assert fig.value == pytest.approx(94.9e9)
    assert fig.tol == pytest.approx(0.05e9)


def test_rounded_billions_match_table_millions() -> None:
    r = check_claim("Total revenue was $94.9 billion.", [INCOME_STATEMENT])
    assert r.verdict == "MATCH"
    assert r.passage_index == 1
    assert r.passage_figure == "94,930"


def test_coarser_rounding_still_matches() -> None:
    assert check_claim("Revenue was $95 billion.", [INCOME_STATEMENT]).verdict == "MATCH"


def test_outside_stated_precision_is_mismatch() -> None:
    r = check_claim("Revenue was $97.0 billion.", [INCOME_STATEMENT])
    assert r.verdict == "MISMATCH"
    assert r.passage_label == "Total net sales"


def test_per_share_row_is_not_scaled_to_millions() -> None:
    r = check_claim("Diluted EPS was $2.40.", [INCOME_STATEMENT])
    assert r.verdict == "MATCH"
    assert r.passage_figure == "2.40"


# --- table parsing -----------------------------------------------------------


def test_table_row_labels_and_parent_context() -> None:
    figs = extract_passage_figures(INCOME_STATEMENT, passage_index=3)
    services = [f for f in figs if f.label == "Services" and not f.computed]
    assert [f.value for f in services] == [26_340e6, 23_117e6]
    assert all(f.passage_index == 3 for f in services)
    assert "revenue" in services[0].label_tokens  # parent "Net sales:"


def test_year_header_row_is_skipped() -> None:
    figs = extract_passage_figures("| | 2025 | 2024 |\n| --- | --- | --- |\n")
    assert figs == []


def test_derived_yoy_change_from_columns() -> None:
    r = check_claim("Services revenue grew 14% year over year.", [INCOME_STATEMENT])
    assert r.verdict == "MATCH"
    assert "+13.9%" in r.passage_figure


def test_segment_row_needs_its_own_keyword() -> None:
    # "Services" figure must not be compared against "Total net sales".
    r = check_claim("Services revenue was $94.9 billion.", [INCOME_STATEMENT])
    assert r.verdict != "MATCH"


# --- parentheses negatives ----------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("(248)", (-248.0, 0, False)),
        ("$ (1,234.5)", (-1234.5, 1, False)),
        ("(6)%", (-6.0, 0, True)),
        ("12 %", (12.0, 0, True)),
        ("94,930", (94930.0, 0, False)),
        ("—", None),
        ("Net sales", None),
    ],
)
def test_parse_cell(cell: str, expected: tuple[float, int, bool] | None) -> None:
    assert parse_cell(cell) == expected


def test_split_parenthesis_cells_are_negative() -> None:
    figs = extract_passage_figures(
        "(In millions)\n| Operating loss | $ | (1,234 | ) | $ | 500 |\n"
    )
    raw = [f for f in figs if not f.computed]
    assert raw[0].value == pytest.approx(-1234e6)


def test_signed_percent_conflicts_with_direction() -> None:
    passage = "| | Change |\n| --- | --- |\n| Services revenue | (6)% |\n"
    assert check_claim("Services revenue grew 6% year over year.", [passage]).verdict == "MISMATCH"
    assert check_claim("Services revenue declined 6% year over year.", [passage]).verdict == "MATCH"


# --- percent vs dollar --------------------------------------------------------


def test_percent_claim_ignores_dollar_cells() -> None:
    (fig,) = extract_claim_figures("Gross margin was 46.9%.")
    assert fig.kind == "percent"
    assert check_claim("Gross margin was 46.9%.", [MARGIN_TABLE]).verdict == "MATCH"
    # A dollar-only table cannot verify a percent level claim.
    assert check_claim("Gross margin was 46.9%.", [INCOME_STATEMENT]).verdict == "NO_EVIDENCE"


def test_dollar_claim_ignores_percent_cells() -> None:
    passage = "| | Q1 |\n| --- | --- |\n| Services | 47 | % |\n"
    assert check_claim("Services revenue was $47 billion.", [passage]).verdict == "NO_EVIDENCE"


def test_level_percent_not_matched_by_derived_change() -> None:
    # 13.9% is a derived YoY change, not a stated level.
    r = check_claim("Services margin was 13.9%.", [INCOME_STATEMENT])
    assert r.verdict == "NO_EVIDENCE"


# --- conservative NO_EVIDENCE --------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "Paid clicks grew 4% year over year.",  # metric absent
        "We delivered strong results this quarter.",  # no figures
        "Revenue was about $97 billion.",  # hedged: never MISMATCH
        "Adjusted operating income was $30.1 billion.",  # non-GAAP
        "We expect revenue of $98 billion next quarter.",  # forward-looking
        "Gross margin expanded 120 basis points.",  # bps unsupported
        "We shipped $94.9 billion worth in December.",  # no metric keyword
    ],
)
def test_conservative_no_evidence(claim: str) -> None:
    assert check_claim(claim, [INCOME_STATEMENT]).verdict == "NO_EVIDENCE"


def test_q4_never_mismatches() -> None:
    r = check_claim("Revenue was $124.3 billion.", [INCOME_STATEMENT], fiscal_quarter="Q4")
    assert r.verdict == "NO_EVIDENCE"


def test_unknown_table_scale_never_mismatches() -> None:
    passage = "| Total net sales | 94,930 | 85,777 |\n"
    assert check_claim("Revenue was $97.0 billion.", [passage]).verdict == "NO_EVIDENCE"


def test_partial_match_is_not_match() -> None:
    claim = "Revenue was $94.9 billion and paid clicks grew 4%."
    assert check_claim(claim, [INCOME_STATEMENT]).verdict == "NO_EVIDENCE"


def test_no_passages() -> None:
    assert check_claim("Revenue was $94.9 billion.", []).verdict == "NO_EVIDENCE"


# --- override policy ---------------------------------------------------------


def test_override_only_below_threshold() -> None:
    match = NumericCheckResult("MATCH", passage_index=1)
    assert apply_override("Unverifiable", 0.6, match, threshold=0.7) == ("Consistent", True)
    assert apply_override("Unverifiable", 0.9, match, threshold=0.7) == ("Unverifiable", False)
    assert apply_override("Consistent", 0.1, match, threshold=0.7) == ("Consistent", False)
    none = NumericCheckResult("NO_EVIDENCE")
    assert apply_override("Contradictory", 0.1, none, threshold=0.7) == ("Contradictory", False)
