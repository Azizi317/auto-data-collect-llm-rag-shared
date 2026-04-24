from __future__ import annotations

from dataclasses import dataclass
import re

from .metadata import KPIItem, accepts_unit, matches_excluded_term
from .retrieval import RankedChunk


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUMBER_RE = re.compile(r"(?<![\w/])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
PERCENT_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%")
UNIT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(?:tCO2e|CO2e|CO2|MWh|GWh|kWh|TJ|GJ|MJ|kg|tonnes?|tons?|ton|m3|kl|hours?|count|rate|t|L)(?![A-Za-z])|%",
    re.I,
)

TARGET_WORDS = (
    "target",
    "goal",
    "plan",
    "roadmap",
    "vision",
    "re100",
    "net zero",
    "net-zero",
    "carbon neutral",
)

WEAK_VALUE_WORDS = (
    "ratio",
    "share",
    "goal",
    "policy",
    "initiative",
    "campaign",
    "satisfaction",
)

INDEX_WORDS = (
    "gri",
    "index",
    "content index",
    "appendix",
    "reference",
    "references",
    "sasb",
    "ifrs",
    "tcfd",
)

HEADER_HINTS = ("category", "item", "unit", "scope", "year")

SCOPE_PATTERNS = {
    "scope1": ("scope 1", "scope1"),
    "scope2": ("scope 2", "scope2"),
    "scope3": ("scope 3", "scope3"),
    "scope12": ("scope 1+2", "scope1+2", "scope 1 and 2", "scope 1·2", "scope 1/2"),
}


@dataclass(frozen=True)
class TableExpression:
    value: str
    unit: str | None
    year: str | None
    cell_text: str
    column_index: int = 0
    score: float = 0.0


@dataclass(frozen=True)
class TableCandidate:
    row_label: str
    expression: TableExpression
    evidence: str
    source_page: int
    matched_terms: tuple[str, ...]
    confidence: float


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_match(text: str) -> str:
    return re.sub(r"\s+", " ", _normalize(text).lower())


def _split_rows(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in table_text.splitlines():
        line = _normalize(raw_line)
        if not line:
            continue
        cells = [_normalize(cell) for cell in line.split("|")] if "|" in line else [line]
        if cells:
            rows.append(cells)
    return rows


def _is_target_like(text: str) -> bool:
    low = _normalize_match(text)
    return any(word in low for word in TARGET_WORDS)


def _contains_weak_value_context(text: str) -> bool:
    low = _normalize_match(text)
    return any(word in low for word in WEAK_VALUE_WORDS)


def _is_year(value: str) -> bool:
    compact = value.replace(",", "").replace(".", "").replace("%", "")
    if not compact.isdigit():
        return False
    number = int(compact)
    return 1900 <= number <= 2100


def _extract_unit(text: str, fallback: str | None = None) -> str | None:
    match = UNIT_TOKEN_RE.search(text)
    if match:
        return match.group(0)
    if fallback:
        fallback_match = UNIT_TOKEN_RE.search(fallback)
        if fallback_match:
            return fallback_match.group(0)
    return None


def _extract_year(cell: str) -> str | None:
    match = YEAR_RE.search(cell)
    return match.group(0) if match else None


def _extract_years(cell: str) -> list[str]:
    return [match.group(0) for match in YEAR_RE.finditer(cell)]


def _numbers_from_cell(cell: str) -> list[str]:
    percent = PERCENT_RE.search(cell)
    if percent:
        return [percent.group(0).replace(" ", "")]
    return [match.group(0) for match in NUMBER_RE.finditer(cell)]


def _looks_like_header_row(cells: list[str]) -> bool:
    header_hits = 0
    numeric_cells = 0
    year_cells = 0
    for cell in cells:
        low = _normalize_match(cell)
        if _extract_year(cell):
            header_hits += 1
            year_cells += 1
        elif any(token in low for token in HEADER_HINTS):
            header_hits += 1
        elif _numbers_from_cell(cell):
            numeric_cells += 1
    if year_cells >= 1 and header_hits >= 2:
        return True
    return header_hits >= max(2, len(cells) // 2 + 1) and numeric_cells == 0


def _latest_header_year(header_rows: list[list[str]]) -> str | None:
    years: list[int] = []
    for row in header_rows:
        for cell in row:
            for year in _extract_years(cell):
                years.append(int(year))
    return str(max(years)) if years else None


def _header_column_cells(header_rows: list[list[str]], index: int) -> list[str]:
    cells: list[str] = []
    for row in header_rows:
        if index < len(row):
            cells.append(row[index])
            continue
        if row:
            cells.append(row[-1])
    return cells


def _scope_tag(text: str) -> str | None:
    low = _normalize_match(text)
    for tag, patterns in SCOPE_PATTERNS.items():
        if any(pattern in low for pattern in patterns):
            return tag
    return None


def _target_scope_tag(kpi: KPIItem) -> str | None:
    return _scope_tag(" ".join((kpi.kpi_ko, kpi.kpi_en, kpi.topic_ko, kpi.topic_en)))


def _scope_conflict_penalty(kpi: KPIItem, text: str) -> float:
    target = _target_scope_tag(kpi)
    actual = _scope_tag(text)
    if not target or not actual or target == actual:
        return 0.0
    if {target, actual} == {"scope1", "scope12"}:
        return 3.0
    if {target, actual} == {"scope2", "scope12"}:
        return 3.0
    return 2.2


def _is_index_like_row(row_text: str, header_rows: list[list[str]]) -> bool:
    row_low = _normalize_match(row_text)
    return any(token in row_low for token in INDEX_WORDS)


def _match_terms(kpi: KPIItem, text: str) -> tuple[str, ...]:
    low = _normalize_match(text)
    matched: list[str] = []
    for term in kpi.all_terms():
        normalized = _normalize_match(term)
        if not normalized:
            continue
        if len(normalized) == 1 and normalized != "%":
            continue
        if normalized in {"scope 1", "scope1"} and re.search(r"scope\s*1(?:\+|/| and 2|·2)", low):
            continue
        if normalized in {"scope 2", "scope2"} and re.search(r"scope\s*1(?:\+|/| and 2|·2)", low):
            continue
        if normalized in low:
            matched.append(term)
    return tuple(dict.fromkeys(matched))


def _row_match_score(kpi: KPIItem, row_label: str, row_text: str) -> tuple[float, tuple[str, ...]]:
    matched = _match_terms(kpi, f"{row_label} {row_text}")
    score = 0.0
    label_low = _normalize_match(row_label)
    row_low = _normalize_match(row_text)

    for term in matched:
        term_low = _normalize_match(term)
        if term_low and term_low in label_low:
            score += 2.4
        elif term_low and term_low in row_low:
            score += 1.0

    if row_label and _contains_weak_value_context(row_label) and "energy" in _normalize_match(kpi.kpi_en):
        score -= 2.5
    if _is_target_like(row_text):
        score -= 1.8
    scope_penalty = _scope_conflict_penalty(kpi, f"{row_label} {row_text}")
    if scope_penalty >= 3.0:
        return -5.0, ()
    score -= scope_penalty
    target_scope = _target_scope_tag(kpi)
    actual_scope = _scope_tag(f"{row_label} {row_text}")
    if target_scope and actual_scope == target_scope:
        score += 0.8
    return score, matched


def _expression_from_row(
    cells: list[str],
    header_rows: list[list[str]],
    kpi: KPIItem,
) -> TableExpression | None:
    header_units = " ".join(" ".join(row) for row in header_rows[-2:])
    latest_year = _latest_header_year(header_rows)
    row_label = " ".join(cells[:1])
    best: TableExpression | None = None
    best_score = float("-inf")

    for index, cell in enumerate(cells[1:], start=1):
        numbers = _numbers_from_cell(cell)
        if not numbers or matches_excluded_term(kpi.expression_template, cell):
            continue

        value = next((number for number in numbers if not _is_year(number)), None)
        if value is None:
            continue

        year = _extract_year(cell)
        if year is None:
            for header_cell in reversed(_header_column_cells(header_rows, index)):
                year = _extract_year(header_cell)
                if year is not None:
                    break

        column_header_text = " ".join(_header_column_cells(header_rows, index))
        unit = _extract_unit(cell, f"{column_header_text} {header_units}")
        if unit is None and kpi.unit:
            unit = _extract_unit(kpi.unit)

        if _is_target_like(cell) and kpi.disclosure_type == "numeric":
            continue
        if _contains_weak_value_context(cell) and kpi.unit and "%" not in kpi.unit and unit == "%":
            continue
        if not accepts_unit(kpi.expression_template, unit):
            continue

        score = 0.0
        if unit:
            score += 1.1
        if year:
            score += 0.8
        if latest_year and year == latest_year:
            score += 1.8
        elif latest_year and year:
            score -= max(0.0, (int(latest_year) - int(year)) * 0.2)
        if len(value.replace(",", "")) >= 3:
            score += 0.25
        score -= _scope_conflict_penalty(kpi, f"{row_label} {cell}")

        candidate = TableExpression(value=value, unit=unit, year=year, cell_text=cell, column_index=index, score=score)
        if score > best_score:
            best = candidate
            best_score = score

    return best


def extract_table_candidate(kpi: KPIItem, ranked_chunks: list[RankedChunk]) -> TableCandidate | None:
    best: TableCandidate | None = None

    for ranked in ranked_chunks:
        if ranked.chunk.kind != "table":
            continue

        rows = _split_rows(ranked.chunk.text)
        if len(rows) < 2:
            continue

        header_rows: list[list[str]] = []
        for row in rows[:4]:
            if _looks_like_header_row(row):
                header_rows.append(row)
                continue
            first_cell = _normalize_match(row[0]) if row else ""
            if header_rows and (not first_cell or first_cell in {"category", "item", "metric"}) and _scope_tag(" ".join(row)):
                header_rows.append(row)
        latest_year = _latest_header_year(header_rows)
        for row in rows:
            if row in header_rows:
                continue
            row_label = row[0] if row else ""
            row_text = " | ".join(row)
            if _is_index_like_row(row_text, header_rows):
                continue

            expression = _expression_from_row(row, header_rows, kpi)
            score, matched = _row_match_score(kpi, row_label, row_text)
            if score <= 0 and expression is not None:
                target_scope = _target_scope_tag(kpi)
                header_scope = _scope_tag(" ".join(_header_column_cells(header_rows, expression.column_index)))
                if target_scope and header_scope == target_scope:
                    score = 1.2
            if score <= 0:
                continue
            if expression is None and kpi.disclosure_type == "numeric":
                continue
            if expression and expression.unit == "%" and kpi.unit and "%" not in kpi.unit:
                continue

            total_score = score + (expression.score if expression else 0.0)
            confidence = min(0.99, 0.42 + (0.08 * len(matched)) + (0.12 if expression else 0.0) + max(0.0, total_score) * 0.02)
            evidence = row_text
            if expression and expression.year:
                evidence = f"{row_text} [year: {expression.year}]"

            candidate = TableCandidate(
                row_label=row_label,
                expression=expression or TableExpression(value="", unit=None, year=None, cell_text=row_text),
                evidence=evidence,
                source_page=ranked.chunk.page_start,
                matched_terms=matched,
                confidence=round(confidence, 2),
            )

            if best is None:
                best = candidate
                continue
            if candidate.confidence > best.confidence:
                best = candidate
                continue
            if candidate.expression.year and not best.expression.year:
                best = candidate

    return best
