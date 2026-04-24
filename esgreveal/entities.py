from __future__ import annotations

from dataclasses import dataclass, field

from .expressions import ExpressionTemplate


@dataclass(frozen=True)
class KPIItem:
    id: str
    category: str
    standard_family: str
    standard_refs: tuple[str, ...]
    aspect_ko: str
    aspect_en: str
    kpi_ko: str
    kpi_en: str
    topic_ko: str
    topic_en: str
    quantity_ko: str
    quantity_en: str
    unit: str
    disclosure_type: str = "numeric"
    obligation: str = "recommended"
    k_esg_guideline: str = ""
    k_esg_codes: tuple[str, ...] = field(default_factory=tuple)
    k_esg_labels: tuple[str, ...] = field(default_factory=tuple)
    k_esg_alignment: str = "related"
    industry_scope: tuple[str, ...] = field(default_factory=tuple)
    search_terms: tuple[str, ...] = field(default_factory=tuple)
    knowledge: tuple[str, ...] = field(default_factory=tuple)
    expression_template: ExpressionTemplate = field(
        default_factory=lambda: ExpressionTemplate(
            name="generic_numeric",
            output_fields=("disclosed", "value", "unit", "evidence", "confidence", "source_page", "matched_terms"),
            value_mode="numeric",
        )
    )

    @property
    def domain_ko(self) -> str:
        return self.aspect_ko

    @property
    def domain_en(self) -> str:
        return self.aspect_en

    @property
    def value_format(self) -> str:
        return self.quantity_ko

    @property
    def output_fields(self) -> tuple[str, ...]:
        return self.expression_template.output_fields

    @property
    def quantity(self) -> str:
        return self.quantity_en

    @property
    def aspect(self) -> str:
        return self.aspect_en

    @property
    def kpi(self) -> str:
        return self.kpi_en

    @property
    def topic(self) -> str:
        return self.topic_en

    def all_terms(self) -> list[str]:
        return [
            term
            for term in (
                self.aspect_ko,
                self.aspect_en,
                self.kpi_ko,
                self.kpi_en,
                self.topic_ko,
                self.topic_en,
                self.quantity_ko,
                self.quantity_en,
                self.unit,
                self.standard_family,
                *self.standard_refs,
                *self.k_esg_codes,
                *self.k_esg_labels,
                *self.industry_scope,
                *self.search_terms,
                *self.knowledge,
                *self.expression_template.preferred_units,
            )
            if str(term).strip()
        ]
