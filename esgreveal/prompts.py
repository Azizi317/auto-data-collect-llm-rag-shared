from __future__ import annotations

from .metadata import KPIItem
from .retrieval import RankedChunk


def _few_shot_examples(disclosure_type: str) -> str:
    quantitative = """[Example: Quantitative Indicator]
{
  "Analysis": "A table row explicitly lists Scope 1 emissions for 2024 with a numerical value and unit.",
  "disclosure": 1,
  "data": {
    "KPI": "Scope 1 온실가스 배출량",
    "Topic": "직접 배출",
    "Value": "342",
    "Unit": "천 톤CO2e",
    "Target": null,
    "Action": null,
    "Evidence": "Scope 1 emissions | 342 | 천 톤CO2e",
    "SourcePage": 24,
    "MatchedTerms": ["Scope 1", "온실가스 배출량"]
  }
}"""

    textual = """[Example: Textual Indicator]
{
  "Analysis": "The paragraph describes concrete reduction actions rather than only targets.",
  "disclosure": 1,
  "data": {
    "KPI": "온실가스 감축 활동",
    "Topic": "배출 저감 조치",
    "Value": null,
    "Unit": null,
    "Target": null,
    "Action": ["고효율 설비 교체", "재생에너지 전환 확대"],
    "Evidence": "The company replaced low-efficiency equipment and expanded renewable electricity procurement.",
    "SourcePage": 31,
    "MatchedTerms": ["감축", "재생에너지", "설비 교체"]
  }
}"""

    return quantitative if disclosure_type == "numeric" else textual


def format_context(chunks: list[RankedChunk]) -> str:
    if not chunks:
        return "검색된 근거가 없습니다."

    lines: list[str] = []
    for index, item in enumerate(chunks, start=1):
        section = item.chunk.section or "미확인 섹션"
        outline = " > ".join(item.chunk.outline_path) if item.chunk.outline_path else section
        matched = ", ".join(item.matched_terms[:8]) if item.matched_terms else "없음"
        lines.append(
            f"[{index}] page={item.chunk.page_start}-{item.chunk.page_end} | "
            f"kind={item.chunk.kind} | section={section} | outline={outline} | "
            f"matched={matched}\n"
            f"{item.chunk.text}"
        )
    return "\n\n".join(lines)


def build_question(kpi: KPIItem) -> str:
    if kpi.disclosure_type == "numeric":
        return (
            f"{kpi.aspect_ko}의 '{kpi.kpi_ko}' 지표에 대해 "
            f"{kpi.topic_ko} 관련 공시가 있는지 판단하고, "
            f"{kpi.quantity_ko}에 해당하는 실제 수치와 단위를 추출하라. "
            "목표, 계획, 비전, 로드맵만 말하는 문장은 정량값으로 취급하지 말라."
        )
    return (
        f"{kpi.aspect_ko}의 '{kpi.kpi_ko}' 지표에 대해 "
        f"{kpi.topic_ko} 관련 공시가 있는지 판단하고, "
        "직접 관련된 목표 또는 실행 조치를 추출하라."
    )


def build_answer_format(kpi: KPIItem) -> str:
    fields = ", ".join(kpi.expression_template.output_fields)
    if kpi.disclosure_type == "numeric":
        value_hint = '"Value": string | null,\n    "Unit": string | null,\n    "UnitInferred": true | false,'
        action_hint = '"Target": string | null,\n    "Action": string | null,'
    else:
        value_hint = '"Value": null,\n    "Unit": null,\n    "UnitInferred": false,'
        action_hint = '"Target": string | null,\n    "Action": string | list | null,'

    return (
        "다음 단일 JSON 객체만 반환하라.\n"
        "{\n"
        '  "Analysis": string | null,\n'
        '  "disclosure": 1 | 0 | null,\n'
        '  "data": {\n'
        f'    "KPI": "{kpi.kpi_ko}",\n'
        f'    "Topic": "{kpi.topic_ko}",\n'
        f"    {value_hint}\n"
        f"    {action_hint}\n"
        '    "Evidence": string | null,\n'
        '    "SourcePage": integer | null,\n'
        '    "MatchedTerms": array\n'
        "  }\n"
        "}\n"
        f"data는 <Disclosure KPI, Topic, Value, Unit, Target, Action>에 대응하며 참고 필드는 {fields}이다."
    )


def build_prompt(kpi: KPIItem, chunks: list[RankedChunk]) -> str:
    context = format_context(chunks)
    standard_refs = ", ".join(kpi.standard_refs) if kpi.standard_refs else "없음"
    industry_scope = ", ".join(kpi.industry_scope) if kpi.industry_scope else "전 산업"
    k_esg_refs = ", ".join(
        f"{code} {label}".strip()
        for code, label in zip(kpi.k_esg_codes, kpi.k_esg_labels, strict=False)
    ) or "직접 대응 없음"
    knowledge = "\n".join(f"- {item}" for item in kpi.knowledge) if kpi.knowledge else "- 없음"
    search_terms = "\n".join(f"- {item}" for item in kpi.search_terms) if kpi.search_terms else "- 없음"
    prompt_rules = (
        "\n".join(f"- {item}" for item in kpi.expression_template.prompt_rules)
        if kpi.expression_template.prompt_rules
        else "- 없음"
    )
    preferred_units = ", ".join(kpi.expression_template.preferred_units) if kpi.expression_template.preferred_units else "없음"
    exclude_terms = ", ".join(kpi.expression_template.exclude_terms) if kpi.expression_template.exclude_terms else "없음"

    return f"""[Preset Information]
You are an expert in ESG information extraction.
Use only the retrieved evidence and the supplied expert knowledge.
Do not infer missing facts. If the evidence is insufficient, return null values.
Exception — Unit inference: if a numeric Value is found but the Unit is not explicitly stated in the evidence,
you MAY infer the unit from (1) the table column header context in the evidence, (2) Preferred Units listed below,
or (3) the KPI domain convention. If you infer the unit, set "UnitInferred": true so the reader knows to verify.
Return exactly one JSON object and no extra text.

[Reference Content]
{context}

[Expert Knowledge]
- Aspect: {kpi.aspect_ko} / {kpi.aspect_en}
- KPI: {kpi.kpi_ko} / {kpi.kpi_en}
- Topic: {kpi.topic_ko} / {kpi.topic_en}
- Quantity: {kpi.quantity_ko} / {kpi.quantity_en}
- Standard Family: {kpi.standard_family}
- Standard Refs: {standard_refs}
- K-ESG Crosscheck: {k_esg_refs}
- Industry Scope: {industry_scope}
- Obligation: {kpi.obligation}
- Preferred Units: {preferred_units}
- Exclude Terms: {exclude_terms}
- Search Terms:
{search_terms}
- Expert Knowledge Notes:
{knowledge}
- Prompt Rules:
{prompt_rules}

[Examples]
{_few_shot_examples(kpi.disclosure_type)}

[Question]
{build_question(kpi)}

[Answer Format]
{build_answer_format(kpi)}
"""
