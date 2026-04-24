from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpressionTemplate:
    name: str
    output_fields: tuple[str, ...]
    value_mode: str
    preferred_units: tuple[str, ...] = field(default_factory=tuple)
    exclude_terms: tuple[str, ...] = field(default_factory=tuple)
    evidence_priority: tuple[str, ...] = field(default_factory=tuple)
    prompt_rules: tuple[str, ...] = field(default_factory=tuple)


def expression_templates() -> dict[str, ExpressionTemplate]:
    common_numeric_exclude = (
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
    return {
        "numeric_absolute": ExpressionTemplate(
            name="numeric_absolute",
            output_fields=("disclosed", "value", "unit", "evidence", "confidence", "source_page", "matched_terms"),
            value_mode="numeric",
            preferred_units=("tCO2e", "MWh", "GJ", "m3", "t", "명", "건", "원"),
            exclude_terms=common_numeric_exclude,
            evidence_priority=("table", "year_header", "row_label"),
            prompt_rules=("정량 수치만 추출", "전략·목표 문구 제외", "가장 최근 연도 우선"),
        ),
        "numeric_ratio": ExpressionTemplate(
            name="numeric_ratio",
            output_fields=("disclosed", "value", "unit", "evidence", "confidence", "source_page", "matched_terms"),
            value_mode="numeric",
            preferred_units=("%", "rate"),
            exclude_terms=("목표", "계획", "향후", "예정"),
            evidence_priority=("table", "row_label"),
            prompt_rules=("실측 비율만 추출", "목표 비율 제외"),
        ),
        "numeric_intensity": ExpressionTemplate(
            name="numeric_intensity",
            output_fields=("disclosed", "value", "unit", "evidence", "confidence", "source_page", "matched_terms"),
            value_mode="numeric",
            preferred_units=("원단위", "단위당", "tCO2e/매출", "MWh/매출", "t/생산량"),
            exclude_terms=("목표", "계획"),
            evidence_priority=("table", "row_label", "section"),
            prompt_rules=("집약도 수치만 추출", "절대값과 구분"),
        ),
        "text_actions": ExpressionTemplate(
            name="text_actions",
            output_fields=("disclosed", "action", "evidence", "confidence", "source_page", "matched_terms"),
            value_mode="text",
            evidence_priority=("paragraph", "table_note"),
            prompt_rules=("핵심 조치만 요약", "근거 문장 기반으로 작성"),
        ),
        "text_policy": ExpressionTemplate(
            name="text_policy",
            output_fields=("disclosed", "action", "evidence", "confidence", "source_page", "matched_terms"),
            value_mode="text",
            evidence_priority=("paragraph", "section"),
            prompt_rules=("정책·프로세스 중심으로 요약"),
        ),
        "text_risk": ExpressionTemplate(
            name="text_risk",
            output_fields=("disclosed", "action", "evidence", "confidence", "source_page", "matched_terms"),
            value_mode="text",
            evidence_priority=("paragraph", "section"),
            prompt_rules=("리스크 유형과 대응을 함께 정리"),
        ),
    }


def build_expression_template(raw: dict[str, Any] | None) -> ExpressionTemplate:
    raw = raw or {}
    return ExpressionTemplate(
        name=str(raw.get("name", "generic_numeric")).strip() or "generic_numeric",
        output_fields=tuple(raw.get("output_fields") or ("disclosed", "value", "unit", "evidence", "confidence", "source_page", "matched_terms")),
        value_mode=str(raw.get("value_mode", "numeric")).strip() or "numeric",
        preferred_units=tuple(raw.get("preferred_units") or ()),
        exclude_terms=tuple(raw.get("exclude_terms") or ()),
        evidence_priority=tuple(raw.get("evidence_priority") or ()),
        prompt_rules=tuple(raw.get("prompt_rules") or ()),
    )


def matches_excluded_term(template: ExpressionTemplate, text: str) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in template.exclude_terms)


def accepts_unit(template: ExpressionTemplate, unit: str | None) -> bool:
    if not template.preferred_units or not unit:
        return True
    unit_low = unit.lower()
    preferred = tuple(token.lower() for token in template.preferred_units)
    if unit_low in preferred:
        return True
    if "%" in unit_low and "%" in preferred:
        return True
    return False
