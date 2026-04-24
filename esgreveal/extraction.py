from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .metadata import KPIItem, matches_excluded_term
from .retrieval import RankedChunk
from .table_semantics import extract_table_candidate


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
NUMBER_RE = re.compile(r"(?<![\w/])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
PERCENT_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%")
RATIO_RE = re.compile(r"(?<!\w)(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>x|X|times?)\b")
UNIT_RE = re.compile(
    r"\b(?:tCO2e|CO2e|CO2|MWh|kWh|GJ|TJ|kg|tonnes?|tons?|ton|m3|cubic meters?|liters?|litres?|hours?|days?|employees?|headcount|ratio|rate|시간|명|건|회|%)\b|(?:\bt\b)",
    re.I,
)
KOREAN_UNIT_HINTS = (
    "배출",
    "사용",
    "발생",
    "명",
    "시간",
    "이직",
    "재해",
    "취수",
    "폐수",
    "폐기물",
    "교육",
    "비율",
    "강도",
    "집약",
    "단위",
    "원",
    "건",
    "회",
)
IGNORE_CONTEXT = (
    "표",
    "그림",
    "참고문헌",
    "연도",
    "페이지",
    "isbn",
    "issn",
    "gri",
    "sasb",
    "ifrs",
    "index",
    "appendix",
)
TARGET_WORDS = (
    "목표",
    "계획",
    "비전",
    "로드맵",
    "추진",
    "예정",
    "re100",
    "net zero",
    "net-zero",
)

INDEX_WORDS = ("gri", "index", "appendix", "reference", "references", "sasb", "ifrs", "tcfd")

SCOPE_PATTERNS = {
    "scope1": ("scope 1", "scope1"),
    "scope2": ("scope 2", "scope2"),
    "scope3": ("scope 3", "scope3"),
    "scope12": ("scope 1+2", "scope1+2", "scope 1 and 2", "scope 1·2", "scope 1/2"),
}


@dataclass(frozen=True)
class HeuristicExtraction:
    disclosed: bool | None
    value: str | None
    unit: str | None
    target: str | None
    action: str | None
    evidence: str | None
    confidence: float | None
    source_page: int | None
    matched_terms: list[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _score_sentence(sentence: str, terms: list[str]) -> tuple[int, list[str]]:
    low = sentence.lower()
    matches = [term for term in terms if term and term.lower() in low]
    return len(matches), matches


def _scope_tag(text: str) -> str | None:
    low = text.lower()
    for tag, patterns in SCOPE_PATTERNS.items():
        if any(pattern in low for pattern in patterns):
            return tag
    return None


def _target_scope_tag(kpi: KPIItem) -> str | None:
    return _scope_tag(" ".join((kpi.kpi_ko, kpi.kpi_en, kpi.topic_ko, kpi.topic_en)).lower())


def _candidate_lines(chunk_text: str) -> list[str]:
    lines = [_normalize(line) for line in re.split(r"[\r\n]+", chunk_text) if _normalize(line)]
    if lines:
        return lines
    return [_normalize(sentence) for sentence in SENTENCE_RE.split(chunk_text) if _normalize(sentence)]


def _pick_sentence(chunk_text: str, terms: list[str]) -> tuple[str | None, list[str]]:
    candidates = _candidate_lines(chunk_text)
    if not candidates:
        return None, []

    scored: list[tuple[float, int, str, list[str]]] = []
    for sentence in candidates:
        score, matches = _score_sentence(sentence, terms)
        if not score:
            continue
        score += 0.3 * len(NUMBER_RE.findall(sentence))
        if PERCENT_RE.search(sentence) or UNIT_RE.search(sentence):
            score += 2.0
        if any(hint in sentence for hint in KOREAN_UNIT_HINTS):
            score += 1.2
        if any(word in sentence.lower() for word in TARGET_WORDS):
            score -= 1.5
        if any(word in sentence.lower() for word in INDEX_WORDS):
            score -= 2.2
        scored.append((score, len(sentence), sentence, matches))

    if not scored:
        return candidates[0], []

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored[0][2], scored[0][3]


def _is_year_like(token: str) -> bool:
    try:
        year = int(token.replace(",", "").replace(".", "").replace("%", ""))
    except ValueError:
        return False
    return 1900 <= year <= 2100


def _has_ignore_context(sentence: str, candidate: str) -> bool:
    idx = sentence.find(candidate)
    if idx < 0:
        return False
    before = sentence[max(0, idx - 10) : idx].lower()
    after = sentence[idx + len(candidate) : idx + len(candidate) + 12].lower()
    context = f"{before}{after}"
    if any(marker in context for marker in IGNORE_CONTEXT):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", candidate):
        return True
    return False


def _extract_value_and_unit(sentence: str) -> tuple[str | None, str | None]:
    percent = PERCENT_RE.search(sentence)
    if percent and not any(word in sentence.lower() for word in TARGET_WORDS):
        return percent.group(0).replace(" ", ""), "%"

    ratio = RATIO_RE.search(sentence)
    if ratio:
        return ratio.group("value"), ratio.group("unit")

    candidates = [match.group(0) for match in NUMBER_RE.finditer(sentence)]
    if not candidates:
        return None, None

    filtered: list[str] = []
    for candidate in candidates:
        compact = candidate.replace(",", "").replace(".", "")
        if len(compact) == 1:
            continue
        if _is_year_like(candidate):
            continue
        if _has_ignore_context(sentence, candidate):
            continue
        idx = sentence.find(candidate)
        before = sentence[max(0, idx - 10) : idx]
        after = sentence[idx + len(candidate) : idx + len(candidate) + 30]
        if re.search(r"\b\d+-\d+\b", after) or re.search(r"\b[A-Z]\d+\b", before):
            continue
        filtered.append(candidate)

    if not filtered:
        return None, None

    picked = filtered[0]
    idx = sentence.find(picked)
    tail = sentence[idx + len(picked) : idx + len(picked) + 80] if idx >= 0 else sentence
    unit_match = UNIT_RE.search(tail)
    unit = unit_match.group(0) if unit_match else None

    if len(filtered) > 1 and unit:
        return " / ".join(filtered[:4]), unit
    if not unit and not any(hint in sentence for hint in KOREAN_UNIT_HINTS):
        return None, None
    return picked, unit


def _scope_conflict(kpi: KPIItem, sentence: str) -> bool:
    target = _target_scope_tag(kpi)
    actual = _scope_tag(sentence.lower())
    if not target or not actual or target == actual:
        return False
    if {target, actual} == {"scope1", "scope12"}:
        return True
    if {target, actual} == {"scope2", "scope12"}:
        return True
    return True


def heuristic_extract(kpi: KPIItem, ranked_chunks: list[RankedChunk]) -> HeuristicExtraction:
    if kpi.disclosure_type == "numeric":
        table_candidate = extract_table_candidate(kpi, ranked_chunks)
        if table_candidate is not None and table_candidate.expression.value:
            return HeuristicExtraction(
                disclosed=True,
                value=table_candidate.expression.value,
                unit=table_candidate.expression.unit or (kpi.unit if kpi.unit else None),
                target=None,
                action=None,
                evidence=table_candidate.evidence,
                confidence=table_candidate.confidence,
                source_page=table_candidate.source_page,
                matched_terms=list(table_candidate.matched_terms),
            )

    search_terms = list(kpi.all_terms())
    for ranked in ranked_chunks:
        sentence, matched_terms = _pick_sentence(ranked.chunk.text, search_terms)
        if not sentence:
            continue
        if kpi.disclosure_type == "numeric" and matches_excluded_term(kpi.expression_template, sentence):
            continue
        if kpi.disclosure_type == "numeric" and _scope_conflict(kpi, sentence):
            continue

        if kpi.disclosure_type == "text":
            confidence = min(0.9, 0.35 + 0.1 * len(matched_terms))
            return HeuristicExtraction(
                disclosed=True if matched_terms else None,
                value=None,
                unit=None,
                target=None,
                action=None,
                evidence=sentence,
                confidence=round(confidence, 2),
                source_page=ranked.chunk.page_start,
                matched_terms=matched_terms,
            )

        value, unit = _extract_value_and_unit(sentence)
        if value:
            digits = value.replace(",", "").replace(".", "").replace("%", "")
            if len(digits) < 2:
                value = None
        if value and not unit and kpi.unit:
            unit = kpi.unit
        if value or matched_terms:
            confidence = min(0.92, 0.22 + 0.11 * len(matched_terms))
            if value:
                confidence = min(0.96, confidence + 0.25)
            if unit:
                confidence = min(0.98, confidence + 0.05)
            return HeuristicExtraction(
                disclosed=True if matched_terms or value else None,
                value=value,
                unit=unit,
                target=None,
                action=None,
                evidence=sentence,
                confidence=round(confidence, 2),
                source_page=ranked.chunk.page_start,
                matched_terms=matched_terms,
            )

    fallback_sentence = ranked_chunks[0].chunk.text if ranked_chunks else None
    return HeuristicExtraction(
        disclosed=None,
        value=None,
        unit=None,
        target=None,
        action=None,
        evidence=fallback_sentence,
        confidence=None,
        source_page=ranked_chunks[0].chunk.page_start if ranked_chunks else None,
        matched_terms=[],
    )


def extraction_to_output(item: HeuristicExtraction) -> dict[str, Any]:
    return {
        "disclosed": item.disclosed,
        "value": item.value,
        "unit": item.unit,
        "target": item.target,
        "action": item.action,
        "evidence": item.evidence,
        "confidence": item.confidence,
        "source_page": item.source_page,
        "matched_terms": item.matched_terms,
    }
