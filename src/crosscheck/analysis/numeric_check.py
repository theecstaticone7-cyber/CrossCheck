"""Deterministic numeric checker: claim figures vs. retrieved filing passages.

Runs before (or alongside) the LLM NLI step. It extracts dollar / percent /
count figures from a transcript claim, extracts figures from filing passages
(Markdown tables from 10-Q / 10-K chunks and prose sentences), and compares
them with rounding-aware tolerance.

Verdicts:

- ``MATCH``        every checkable claim figure was found, within the claim's
                   stated precision, next to the same metric keyword.
- ``MISMATCH``     at least one claim figure conflicts with a figure for the
                   *same metric* (row label / nearby words) in the same passage,
                   and nothing anywhere matched it.
- ``NO_EVIDENCE``  everything else. This is the safe default.

The checker only ever sees the claim text, the passage texts and the fiscal
quarter. It never reads expected labels or ground-truth references.

Pure standard library so it can be unit-tested and run offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

Verdict = Literal["MATCH", "MISMATCH", "NO_EVIDENCE"]

#: LLM confidence below which a decisive checker verdict may override the label.
#: Tune with ``scripts/eval/numeric_check_eval.py`` (tuning split only).
DEFAULT_OVERRIDE_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_SCALES: dict[str, float] = {
    "trillion": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mm": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}

_UP_WORDS = {
    "up", "increase", "increased", "increasing", "increases", "grew", "grow",
    "grows", "growth", "growing", "rose", "rise", "rising", "higher", "gain",
    "gained", "expanded", "expansion", "jumped", "climbed", "improved",
    "improvement",
}
_DOWN_WORDS = {
    "down", "decrease", "decreased", "decreasing", "decreases", "decline",
    "declined", "declining", "declines", "fell", "fall", "falling", "drop",
    "dropped", "lower", "contracted", "contraction", "shrank", "reduction",
    "reduced",
}

# Words that never identify a metric.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "by",
    "with", "from", "as", "was", "were", "is", "are", "be", "been", "being",
    "our", "we", "its", "it", "this", "that", "these", "those", "their", "they",
    "us", "s", "has", "had", "have", "which", "while", "also", "both", "each",
    "quarter", "quarterly", "quarters", "year", "years", "annual", "yoy",
    "year-over-year", "over", "compared", "versus", "vs", "prior", "period",
    "record", "all-time", "alltime", "new", "high", "reached", "came", "come",
    "delivered", "reported", "report", "generated", "totaled", "totalled",
    "achieved", "posted", "saw", "see", "billion", "million", "thousand",
    "trillion", "percent", "percentage", "point", "points", "dollar",
    "approximately", "about", "nearly", "roughly", "around", "almost",
    "fiscal", "first", "second", "third", "fourth", "full", "q1", "q2", "q3",
    "q4", "fy", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "company", "business", "strong", "solid", "robust", "continued",
    "segment", "result", "results", "total-company", "level", "basis",
    "apple", "amazon", "google", "alphabet", "meta", "nvidia", "inc",
    "than", "more", "less", "just", "under", "above", "below", "least",
    "exceeded", "surpassed", "topped", "hit", "landed", "ended", "ending",
    "three", "month", "months", "ended", "same", "last", "this", "time",
    "mark", "marking", "driven", "led", "thanks", "due", "primarily",
} | _UP_WORDS | _DOWN_WORDS

# Metric words that may appear in a row label without being in the claim.
_LABEL_GENERIC = {"total", "net", "consolidated"}
# Metric words that do not, on their own, pin down which metric is meant.
_CLAIM_GENERIC = {"total", "net", "consolidated", "revenue", "gaap"}

# Synonyms applied after singularisation.
_SYNONYMS = {
    "sale": "revenue",
    "revenues": "revenue",
    "turnover": "revenue",
    "earning": "earnings",
    "eps": "earnings",
}

_APPROX_LOWER = ("over", "more than", "exceeded", "surpassed", "at least",
                 "north of", "upwards of", "above", "topped")
_APPROX_UPPER = ("under", "less than", "below", "just under", "up to")
_APPROX_NEAR = ("nearly", "almost", "close to", "approaching")
_APPROX_ABOUT = ("about", "approximately", "around", "roughly", "~", "some",
                 "circa", "approx")

# Claim context that makes a filing comparison unreliable: never MISMATCH.
_NO_MISMATCH_RE = re.compile(
    r"non-gaap|adjusted|constant[\s-]currency|excluding|ex-|fx[\s-]neutral|"
    r"organic|pro\s+forma|free\s+cash\s+flow|run[\s-]rate|annuali[sz]ed|"
    r"sequential|quarter[\s-]over[\s-]quarter|q/q|backlog|guidance",
    re.I,
)
# Forward-looking claims: never decide anything.
_FORWARD_RE = re.compile(
    r"\b(expect|expects|expected|guidance|outlook|forecast|anticipate|"
    r"will be|next quarter|going forward|we see)\b",
    re.I,
)
_SEQUENTIAL_RE = re.compile(r"sequential|quarter[\s-]over[\s-]quarter|q/q", re.I)
_CHANGE_RE = re.compile(
    r"year[\s-]over[\s-]year|\byoy\b|y/y|compared|versus|\bvs\.?\b|"
    r"from a year ago|prior[\s-]year|last year",
    re.I,
)

_TABLE_SCALE_RE = re.compile(
    r"in\s+(thousands|millions|billions)\b", re.I
)
_PER_SHARE_RE = re.compile(
    r"per\s+(?:diluted\s+|basic\s+|common\s+)?share|\beps\b|^\s*(?:basic|diluted)\b",
    re.I,
)
_SHARE_COUNT_RE = re.compile(r"shares\s+used|weighted[\s-]average|number\s+of\s+shares", re.I)

_FIG_RE = re.compile(
    r"(?P<dollar>\$\s*)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?P<scale>(?i:trillion|billion|million|thousand|bn|mm|mn)\b|[BMK]\b))?"
    r"(?:\s*(?P<pct>%|(?i:percent(?:age\s+points?)?|pct)\b))?"
    r"(?:\s*(?P<bps>(?i:basis\s+points?|bps)\b))?"
)
_WORD_RE = re.compile(r"[a-z][a-z0-9&'\-]*")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ClaimFigure:
    """One figure stated in a claim."""

    raw: str
    value: float  # dollars / percent points / count, unsigned
    kind: Literal["dollar", "percent", "count", "bps"]
    tol: float  # half a unit in the last stated digit, in ``value`` units
    direction: int = 0  # +1 up, -1 down, 0 none stated (percents only)
    is_change: bool = False  # percent describes a change (YoY etc.)
    approx: str | None = None  # "lower" / "upper" / "near" / "about"
    keywords: frozenset[str] = frozenset()

    @property
    def signed(self) -> float:
        return self.value * (self.direction or 1)


@dataclass
class PassageFigure:
    """One figure found in a passage (table cell or prose)."""

    raw: str
    value: float
    kind: Literal["dollar", "amount", "percent", "count"]
    tol: float
    passage_index: int  # 1-based
    label: str  # row label or sentence snippet
    label_tokens: frozenset[str]  # tokens that define the metric for this figure
    signed: bool = True  # percent sign is meaningful
    scale_known: bool = True
    scale: float = 1.0
    computed: bool = False  # derived YoY change from two cells
    is_row: bool = True


@dataclass
class FigureCheck:
    claim_figure: str
    status: Verdict
    passage_index: int = 0
    passage_figure: str = ""
    passage_label: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_figure": self.claim_figure,
            "status": self.status,
            "passage_index": self.passage_index,
            "passage_figure": self.passage_figure,
            "passage_label": self.passage_label,
            "note": self.note,
        }


@dataclass
class NumericCheckResult:
    verdict: Verdict
    passage_index: int = 0  # 1-based passage that decided it; 0 = none
    claim_figure: str = ""
    passage_figure: str = ""
    passage_label: str = ""
    figures: list[FigureCheck] = field(default_factory=list)
    note: str = ""

    @property
    def decisive(self) -> bool:
        return self.verdict in ("MATCH", "MISMATCH")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passage_index": self.passage_index,
            "claim_figure": self.claim_figure,
            "passage_figure": self.passage_figure,
            "passage_label": self.passage_label,
            "note": self.note,
            "figures": [f.to_dict() for f in self.figures],
        }

    def prompt_hint(self) -> str:
        """Short block appended to the NLI user message."""
        lines = [
            "[Deterministic numeric pre-check — rule-based hint, may be wrong; "
            "verify against the passages yourself]",
            f"Verdict: {self.verdict}",
        ]
        for f in self.figures:
            if f.passage_index:
                lines.append(
                    f'- claim "{f.claim_figure}" vs Passage {f.passage_index} '
                    f'"{f.passage_label}" {f.passage_figure}: {f.status}'
                )
            else:
                lines.append(f'- claim "{f.claim_figure}": {f.status} ({f.note})')
        if not self.figures:
            lines.append(f"- {self.note or 'no checkable figures in claim'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------


def _norm_word(word: str) -> str:
    word = word.strip("'-").lower()
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    return _SYNONYMS.get(word, word)


def _tokens(text: str) -> list[str]:
    return [_norm_word(w) for w in _WORD_RE.findall(text.lower())]


_STOP_NORM = frozenset(_STOPWORDS | {_norm_word(w) for w in _STOPWORDS})


def content_tokens(text: str) -> frozenset[str]:
    """Metric-identifying tokens (normalised, stopwords removed)."""
    low = text.lower()
    low = re.sub(r"net\s+sales", "net revenue", low)
    low = re.sub(r"\beps\b", "earnings per share", low)
    low = low.replace("per-share", "per share")
    return frozenset(t for t in _tokens(low) if t and t not in _STOP_NORM)


def _decimals(num: str) -> int:
    return len(num.split(".", 1)[1]) if "." in num else 0


def _to_float(num: str) -> float:
    return float(num.replace(",", ""))


def _skip_match(text: str, m: re.Match[str]) -> bool:
    """Drop identifiers like Q1, FY2025, 10-Q, Item 2, COVID-19."""
    start, end = m.start("num"), m.end("num")
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    if before.isalpha() or before == "_":
        return True
    if before in "-–" and start >= 2 and text[start - 2].isalpha():
        return True
    if after in "-–" and end + 1 < len(text) and text[end + 1].isalpha():
        return True
    if after.isalpha() and not m.group("scale") and not m.group("pct"):
        return True  # 3rd, 2x
    prev_word = text[max(0, start - 8):start].lower().strip()
    if prev_word.endswith(("item", "note", "part", "page")):
        return True
    return False


# ---------------------------------------------------------------------------
# Claim figures
# ---------------------------------------------------------------------------


def _approx_kind(prefix: str) -> str | None:
    p = " " + " ".join(prefix.lower().split()) + " "
    for words, kind in (
        (_APPROX_UPPER, "upper"),  # before "lower" so "just under" wins over "under"
        (_APPROX_LOWER, "lower"),
        (_APPROX_NEAR, "near"),
        (_APPROX_ABOUT, "about"),
    ):
        for w in words:
            if p.endswith(" " + w + " ") or (w == "~" and p.rstrip().endswith("~")):
                return kind
    return None


def _direction(words: list[str]) -> int:
    up = any(w in _UP_WORDS for w in words)
    down = any(w in _DOWN_WORDS for w in words)
    if up and not down:
        return 1
    if down and not up:
        return -1
    return 0


def extract_claim_figures(claim: str) -> list[ClaimFigure]:
    """Extract dollar / percent / scaled-count figures from claim text.

    Bare unscaled numbers (dates, years, small counts) are ignored.
    """
    figures: list[ClaimFigure] = []
    prev_end = 0
    prev_keywords: frozenset[str] = frozenset()
    matches = [m for m in _FIG_RE.finditer(claim) if not _skip_match(claim, m)]
    for i, m in enumerate(matches):
        num = m.group("num")
        has_dollar = bool(m.group("dollar"))
        scale_word = (m.group("scale") or "").lower()
        has_pct = bool(m.group("pct"))
        has_bps = bool(m.group("bps"))
        tail = claim[m.end(): m.end() + 10].lower()
        if not has_dollar and tail.lstrip().startswith("dollar"):
            has_dollar = True

        if has_bps:
            kind = "bps"
        elif has_pct:
            kind = "percent"
        elif has_dollar:
            kind = "dollar"
        elif scale_word:
            kind = "count"
        else:
            continue  # bare number

        scale = _SCALES.get(scale_word, 1.0) if kind in ("dollar", "count") else 1.0
        value = _to_float(num) * scale
        tol = 0.5 * (10 ** -_decimals(num)) * scale

        before = claim[prev_end: m.start()]
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(claim)
        after = claim[m.end(): next_start]
        after_clause = re.split(r"[,;.]|\band\b|\bwhile\b", after, maxsplit=1)[0]

        keywords = content_tokens(before)
        if not keywords:
            keywords = content_tokens(after_clause)
        if not keywords:
            keywords = prev_keywords

        window_words = _tokens(before)[-6:] + _tokens(after_clause)[:4]
        direction = _direction(window_words) if kind == "percent" else 0
        is_change = kind == "percent" and (
            direction != 0
            or bool(_CHANGE_RE.search(before + " " + after_clause))
        )
        figures.append(
            ClaimFigure(
                raw=m.group(0).strip(),
                value=abs(value),
                kind=kind,  # type: ignore[arg-type]
                tol=tol,
                direction=direction,
                is_change=is_change,
                approx=_approx_kind(claim[max(0, m.start() - 20): m.start()]),
                keywords=keywords,
            )
        )
        prev_end = m.end()
        prev_keywords = keywords
    return figures


# ---------------------------------------------------------------------------
# Passage figures
# ---------------------------------------------------------------------------


def _table_scale(text: str) -> float | None:
    m = _TABLE_SCALE_RE.search(text)
    if not m:
        return None
    return {"thousands": 1e3, "millions": 1e6, "billions": 1e9}[m.group(1).lower()]


def parse_cell(cell: str) -> tuple[float, int, bool] | None:
    """Parse one table cell → ``(value, decimals, is_percent)`` or None.

    Handles ``$``, comma thousands, ``(1,234)`` negatives, ``−``/``-`` signs,
    and ``%`` suffixes. Returns None for text, dashes and empty cells.
    """
    s = cell.strip()
    if not s:
        return None
    is_pct = "%" in s
    s = s.replace("%", "").replace("$", "").replace(" ", "").replace(" ", "")
    neg = False
    if s.startswith("(") or s.endswith(")"):
        neg = s.startswith("(")
        s = s.strip("()")
    if s[:1] in "-−–" and len(s) > 1:
        neg = True
        s = s[1:]
    if not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+", s):
        return None
    value = _to_float(s)
    return (-value if neg else value, _decimals(s), is_pct)


def _is_year(value: float, decimals: int, raw: str) -> bool:
    return decimals == 0 and "," not in raw and 1990 <= value <= 2100


def _split_row(line: str) -> list[str]:
    line = line.strip().replace("\\|", "/")
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _parse_table_row(
    cells: list[str],
    *,
    passage_index: int,
    parent: str,
    table_scale: float | None,
) -> tuple[str, list[PassageFigure]] | None:
    """Return ``(label, figures)`` for a data row; None for header/text rows."""
    label_parts: list[str] = []
    parsed: list[tuple[float, int, bool, str]] = []
    for cell in cells:
        stripped = cell.strip()
        if stripped in ("", "$", "—", "–", "-", "*"):
            continue
        if stripped in ("%", ")%", "%)") and parsed:
            v, d, _, raw = parsed[-1]
            parsed[-1] = (v, d, True, raw + "%")
            continue
        if stripped == ")" and parsed:
            v, d, p, raw = parsed[-1]
            parsed[-1] = (-abs(v), d, p, raw + ")")
            continue
        got = parse_cell(stripped)
        if got is None:
            if not parsed:
                label_parts.append(stripped)
            continue
        v, d, p = got
        parsed.append((v, d, p, stripped))

    label = " ".join(label_parts).strip()
    if not parsed:
        return None
    if all(_is_year(abs(v), d, raw) and not p for v, d, p, raw in parsed):
        return None  # header row of years
    if not label:
        return None

    context = f"{parent} {label}".strip()
    label_tokens = content_tokens(context)
    per_share = bool(_PER_SHARE_RE.search(label)) or (
        bool(_PER_SHARE_RE.search(parent)) and not _SHARE_COUNT_RE.search(label)
    )
    share_count = bool(_SHARE_COUNT_RE.search(context))

    figures: list[PassageFigure] = []
    amounts: list[tuple[float, PassageFigure]] = []
    for v, d, p, raw in parsed:
        if p:
            figures.append(
                PassageFigure(
                    raw=raw, value=v, kind="percent", tol=0.5 * 10 ** -d,
                    passage_index=passage_index, label=label,
                    label_tokens=label_tokens,
                )
            )
            continue
        if per_share and not share_count:
            kind, scale, known = "dollar", 1.0, True
        elif share_count:
            kind, scale, known = "count", table_scale or 1.0, table_scale is not None
        else:
            kind, scale, known = "amount", table_scale or 1.0, table_scale is not None
        fig = PassageFigure(
            raw=raw, value=v * scale, kind=kind,  # type: ignore[arg-type]
            tol=0.5 * 10 ** -d * scale, passage_index=passage_index,
            label=label, label_tokens=label_tokens,
            scale_known=known, scale=scale,
        )
        figures.append(fig)
        amounts.append((v, fig))

    # Derived YoY changes: (current, prior) column pairs 0-1, 2-3, …
    for j in range(0, len(amounts) - 1, 2):
        cur, prior = amounts[j][0], amounts[j + 1][0]
        if prior == 0:
            continue
        change = (cur - prior) / abs(prior) * 100.0
        figures.append(
            PassageFigure(
                raw=f"({amounts[j][1].raw} vs {amounts[j + 1][1].raw}) -> {change:+.1f}%",
                value=change, kind="percent", tol=0.05,
                passage_index=passage_index, label=label,
                label_tokens=label_tokens, computed=True,
            )
        )
    return label, figures


_SENT_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(\$])|\n+")


def _prose_figures(text: str, passage_index: int) -> list[PassageFigure]:
    out: list[PassageFigure] = []
    for sentence in _SENT_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        snippet = " ".join(sentence.split())[:80]
        for m in _FIG_RE.finditer(sentence):
            if _skip_match(sentence, m):
                continue
            num = m.group("num")
            scale_word = (m.group("scale") or "").lower()
            if m.group("bps"):
                continue
            if m.group("pct"):
                kind = "percent"
            elif m.group("dollar"):
                kind = "dollar"
            elif scale_word:
                kind = "count"
            else:
                continue
            scale = _SCALES.get(scale_word, 1.0) if kind != "percent" else 1.0
            value = _to_float(num) * scale
            before_words = _tokens(sentence[: m.start()])
            near_before = sentence[: m.start()]
            # Metric words: last ~12 tokens before the figure + 3 after.
            words_before = _WORD_RE.findall(near_before.lower())[-12:]
            words_after = _WORD_RE.findall(sentence[m.end():].lower())[:3]
            near = content_tokens(" ".join(words_before + words_after))
            signed = True
            if kind == "percent":
                d = _direction(before_words[-5:])
                if d == 0:
                    signed = False
                else:
                    value *= d
            out.append(
                PassageFigure(
                    raw=m.group(0).strip(), value=value, kind=kind,  # type: ignore[arg-type]
                    tol=0.5 * 10 ** -_decimals(num) * scale,
                    passage_index=passage_index, label=snippet,
                    label_tokens=near, signed=signed, is_row=False,
                )
            )
    return out


def extract_passage_figures(text: str, passage_index: int = 1) -> list[PassageFigure]:
    """Extract figures from one passage: Markdown table rows + prose sentences."""
    scale = _table_scale(text)
    figures: list[PassageFigure] = []
    prose_lines: list[str] = []
    parent = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            prose_lines.append(line)
            parent = ""
            continue
        cells = _split_row(stripped)
        if all(re.fullmatch(r":?-{3,}:?", c) or not c for c in cells):
            continue  # separator
        parsed = _parse_table_row(
            cells, passage_index=passage_index, parent=parent, table_scale=scale
        )
        if parsed is None:
            # Section label ("Net sales:") = text in the first cell only.
            # Column headers ("| | Change |") have an empty first cell.
            first, rest = (cells[0], cells[1:]) if cells else ("", [])
            if first and parse_cell(first) is None and all(
                c in ("", "$", "%") for c in rest
            ):
                parent = first
            continue
        figures.extend(parsed[1])
        if parsed[0].lower().startswith("total"):
            parent = ""  # a "Total …" row closes its group
    figures.extend(_prose_figures("\n".join(prose_lines), passage_index))
    return figures


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _specific(keywords: frozenset[str]) -> frozenset[str]:
    return frozenset(k for k in keywords if k not in _CLAIM_GENERIC)


def _metric_related(cf: ClaimFigure, pf: PassageFigure) -> bool:
    """Weak link: the figure sits next to at least one identifying claim keyword."""
    spec = _specific(cf.keywords)
    if spec:
        return bool(spec & pf.label_tokens)
    return bool(cf.keywords & pf.label_tokens)


def _same_metric(cf: ClaimFigure, pf: PassageFigure) -> bool:
    """Strong link required for MISMATCH: unambiguously the claim's metric."""
    k = cf.keywords
    if not k:
        return False
    lt = pf.label_tokens
    if not lt or not (k & lt):
        return False
    if not _specific(k) <= lt:
        return False
    if pf.is_row:
        # Row label must not name a metric the claim didn't mention.
        return lt <= (k | _LABEL_GENERIC)
    return k <= lt


def _compatible(cf: ClaimFigure, pf: PassageFigure) -> bool:
    if cf.kind == "percent":
        return pf.kind == "percent" and (cf.is_change or not pf.computed)
    if cf.kind == "dollar":
        return pf.kind in ("dollar", "amount")
    if cf.kind == "count":
        return pf.kind in ("count", "amount")
    return False


def _values_match(cf: ClaimFigure, pf: PassageFigure) -> bool:
    if cf.kind == "percent":
        if cf.direction and pf.signed:
            c, v = cf.signed, pf.value
        else:
            c, v = cf.value, abs(pf.value)
        scales = [1.0]
    else:
        c, v = cf.value, abs(pf.value)
        scales = [1.0] if pf.scale_known else [1.0, 1e3, 1e6, 1e9]

    for s in scales:
        vv = v * s
        tol = max(cf.tol, pf.tol * s) + 1e-9 * max(abs(c), 1.0)
        if cf.approx == "lower":
            if c <= vv <= c * 1.25 + tol:
                return True
        elif cf.approx == "upper":
            if c * 0.8 - tol <= vv <= c:
                return True
        elif cf.approx == "near":
            if c * 0.95 - tol <= vv <= c + tol:
                return True
        elif cf.approx == "about":
            if abs(c - vv) <= max(tol, 0.02 * abs(c)):
                return True
        elif abs(c - vv) <= tol:
            return True
    return False


def _conflicts(cf: ClaimFigure, pf: PassageFigure) -> bool:
    """Comparable but different: same order of magnitude, outside tolerance."""
    if cf.kind == "percent":
        if cf.direction and pf.signed:
            c, v = cf.signed, pf.value
        else:
            c, v = cf.value, abs(pf.value)
        if abs(c) < 1.0 or abs(v) < 1.0:
            return abs(c - v) <= 5.0
    else:
        if not pf.scale_known:
            return False
        c, v = cf.value, abs(pf.value)
    if c == 0 or v == 0:
        return False
    ratio = abs(v) / abs(c)
    return 0.5 <= ratio <= 2.0 or (cf.kind == "percent" and (c > 0) != (v > 0))


def _q4_differences(cf: ClaimFigure, figs: list[PassageFigure]) -> list[PassageFigure]:
    """Q4 = 12-month (10-K) − 9-month (Q3 10-Q) for same-metric rows across passages."""
    rows = [
        f for f in figs
        if f.is_row and f.kind in ("dollar", "amount") and f.scale_known
        and _same_metric(cf, f)
    ]
    out: list[PassageFigure] = []
    for a in rows:
        for b in rows:
            if a.passage_index == b.passage_index or b.value == 0:
                continue
            ratio = abs(a.value) / abs(b.value)
            if not 1.15 <= ratio <= 1.6:
                continue
            diff = abs(a.value) - abs(b.value)
            out.append(
                PassageFigure(
                    raw=f"{a.raw} (P{a.passage_index}) - {b.raw} (P{b.passage_index})",
                    value=diff, kind=a.kind, tol=a.tol + b.tol,
                    passage_index=a.passage_index, label=a.label,
                    label_tokens=a.label_tokens, computed=True,
                )
            )
    return out


def check_claim(
    claim: str,
    passages: list[str],
    *,
    fiscal_quarter: str | None = None,
) -> NumericCheckResult:
    """Compare a claim's figures against passages. Conservative by design.

    ``passages`` are the retrieved filing chunk texts in retrieval order;
    passage indices in the result are 1-based into that list.
    """
    if _FORWARD_RE.search(claim):
        return NumericCheckResult("NO_EVIDENCE", note="forward-looking claim")

    claim_figs = [f for f in extract_claim_figures(claim) if f.kind != "bps"]
    if not claim_figs:
        return NumericCheckResult("NO_EVIDENCE", note="no checkable figures in claim")

    pass_figs: list[PassageFigure] = []
    for i, text in enumerate(passages, start=1):
        pass_figs.extend(extract_passage_figures(text or "", i))

    is_q4 = (fiscal_quarter or "").upper() in ("Q4", "4")
    sequential = bool(_SEQUENTIAL_RE.search(claim))
    mismatch_allowed_claim = not _NO_MISMATCH_RE.search(claim) and not is_q4

    checks: list[FigureCheck] = []
    for cf in claim_figs:
        if not cf.keywords:
            checks.append(FigureCheck(cf.raw, "NO_EVIDENCE", note="no metric keyword"))
            continue
        candidates = [
            pf for pf in pass_figs
            if _compatible(cf, pf) and not (sequential and pf.computed)
        ]
        if is_q4 and cf.kind in ("dollar", "count"):
            candidates += _q4_differences(cf, pass_figs)

        matches = [
            pf for pf in candidates
            if _metric_related(cf, pf) and _values_match(cf, pf)
        ]
        if matches:
            best = sorted(
                matches,
                key=lambda pf: (not _same_metric(cf, pf), pf.computed, pf.passage_index),
            )[0]
            checks.append(
                FigureCheck(cf.raw, "MATCH", best.passage_index, best.raw, best.label)
            )
            continue

        can_mismatch = (
            mismatch_allowed_claim
            and cf.approx is None
            and cf.kind in ("dollar", "percent")
        )
        conflicts = (
            [pf for pf in candidates if _same_metric(cf, pf) and _conflicts(cf, pf)]
            if can_mismatch
            else []
        )
        if conflicts:
            def _dist(pf: PassageFigure) -> float:
                v = pf.value if (cf.kind == "percent" and cf.direction and pf.signed) else abs(pf.value)
                c = cf.signed if (cf.kind == "percent" and cf.direction and pf.signed) else cf.value
                return abs(v - c)

            best = sorted(conflicts, key=lambda pf: (pf.computed, _dist(pf)))[0]
            checks.append(
                FigureCheck(cf.raw, "MISMATCH", best.passage_index, best.raw, best.label)
            )
            continue
        checks.append(FigureCheck(cf.raw, "NO_EVIDENCE", note="no same-metric figure"))

    mismatches = [c for c in checks if c.status == "MISMATCH"]
    if mismatches:
        d = mismatches[0]
        return NumericCheckResult(
            "MISMATCH", d.passage_index, d.claim_figure, d.passage_figure,
            d.passage_label, checks,
        )
    if all(c.status == "MATCH" for c in checks):
        d = checks[0]
        return NumericCheckResult(
            "MATCH", d.passage_index, d.claim_figure, d.passage_figure,
            d.passage_label, checks,
        )
    return NumericCheckResult("NO_EVIDENCE", figures=checks, note="not all figures matched")


# ---------------------------------------------------------------------------
# Override policy
# ---------------------------------------------------------------------------

_VERDICT_LABEL = {"MATCH": "Consistent", "MISMATCH": "Contradictory"}


def apply_override(
    llm_label: str,
    llm_confidence: float,
    result: NumericCheckResult,
    *,
    threshold: float = DEFAULT_OVERRIDE_THRESHOLD,
) -> tuple[str, bool]:
    """Return ``(final_label, overridden)``.

    A decisive checker verdict replaces the LLM label only when the LLM's
    confidence is strictly below ``threshold`` and the labels disagree.
    """
    target = _VERDICT_LABEL.get(result.verdict)
    if target is None or target == llm_label:
        return llm_label, False
    if llm_confidence < threshold:
        return target, True
    return llm_label, False
