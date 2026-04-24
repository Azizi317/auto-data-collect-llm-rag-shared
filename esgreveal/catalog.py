from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


K_ESG_GUIDELINE_URL = "https://k-esg.org/guide/guideline02"


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


@dataclass(frozen=True)
class ExpressionTemplate:
    name: str
    output_fields: tuple[str, ...]
    value_mode: str
    preferred_units: tuple[str, ...] = field(default_factory=tuple)
    exclude_terms: tuple[str, ...] = field(default_factory=tuple)
    evidence_priority: tuple[str, ...] = field(default_factory=tuple)
    prompt_rules: tuple[str, ...] = field(default_factory=tuple)


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
    k_esg_guideline: str = K_ESG_GUIDELINE_URL
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


def _expression_templates() -> dict[str, ExpressionTemplate]:
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


def _crosswalks() -> dict[str, tuple[tuple[str, ...], tuple[str, ...], str]]:
    return {
        "K-ENV-N01": (("E-3-1",), ("온실가스 배출량 Scope 1·2",), "direct"),
        "K-ENV-N02": (("E-3-1",), ("온실가스 배출량 Scope 1·2",), "direct"),
        "K-ENV-N03": (("E-3-2",), ("온실가스 배출량 Scope 3",), "direct"),
        "K-ENV-N04": (("E-4-1",), ("에너지 사용량",), "direct"),
        "K-ENV-N05": (("E-4-2",), ("재생에너지 사용 비율",), "direct"),
        "K-ENV-N07": (("E-5-1",), ("용수 취수량",), "direct"),
        "K-ENV-N08": (("E-5-2",), ("폐수 배출량",), "direct"),
        "K-ENV-N09": (("E-6-1",), ("폐기물 발생량",), "direct"),
        "K-SOC-N04": (("S-2-4",), ("교육훈련",), "related"),
        "K-SOC-N05": (("S-4-2",), ("산업재해",), "direct"),
        "K-SOC-N09": (("S-3-1",), ("다양성",), "direct"),
        "K-GOV-N01": (("G-1-3",), ("사외이사 비율",), "direct"),
        "K-GOV-N03": (("G-1-5",), ("윤리경영 교육",), "related"),
    }


def _make_item(
    *,
    id: str,
    category: str,
    standard_family: str,
    standard_refs: tuple[str, ...],
    aspect_ko: str,
    aspect_en: str,
    kpi_ko: str,
    kpi_en: str,
    topic_ko: str,
    topic_en: str,
    quantity_ko: str,
    quantity_en: str,
    unit: str,
    disclosure_type: str,
    template: str,
    obligation: str = "recommended",
    industry_scope: tuple[str, ...] = ("전 산업",),
    search_terms: tuple[str, ...] = (),
    knowledge: tuple[str, ...] = (),
) -> KPIItem:
    codes, labels, alignment = _crosswalks().get(id, ((), (), "none"))
    return KPIItem(
        id=id,
        category=category,
        standard_family=standard_family,
        standard_refs=standard_refs,
        aspect_ko=aspect_ko,
        aspect_en=aspect_en,
        kpi_ko=kpi_ko,
        kpi_en=kpi_en,
        topic_ko=topic_ko,
        topic_en=topic_en,
        quantity_ko=quantity_ko,
        quantity_en=quantity_en,
        unit=unit,
        disclosure_type=disclosure_type,
        obligation=obligation,
        k_esg_guideline=K_ESG_GUIDELINE_URL,
        k_esg_codes=codes,
        k_esg_labels=labels,
        k_esg_alignment=alignment,
        industry_scope=industry_scope,
        search_terms=search_terms,
        knowledge=knowledge,
        expression_template=_expression_templates()[template],
    )


def _environment_numerical() -> list[KPIItem]:
    a_ko = "환경"
    a_en = "Environment"
    specs = [
        ("K-ENV-N01", "ISSB S2", ("ISSB S2 29(a)", "GRI 305-1"), "Scope 1 온실가스 배출량", "Scope 1 greenhouse gas emissions", "직접 배출", "Direct emissions", "절대값", "Absolute value", "tCO2e", "numeric", "numeric_absolute", ("Scope 1", "스코프 1", "직접 배출", "온실가스 배출량"), ("연간 직접 온실가스 배출 총량을 우선 수집한다.",)),
        ("K-ENV-N02", "ISSB S2", ("ISSB S2 29(a)", "GRI 305-2"), "Scope 2 온실가스 배출량", "Scope 2 greenhouse gas emissions", "간접 배출", "Indirect emissions", "절대값", "Absolute value", "tCO2e", "numeric", "numeric_absolute", ("Scope 2", "스코프 2", "간접 배출", "전력 사용"), ("location-based와 market-based를 구분해 표기하는 경우가 많다.",)),
        ("K-ENV-N03", "ISSB S2", ("ISSB S2 29(a)", "GRI 305-3"), "Scope 3 온실가스 배출량", "Scope 3 greenhouse gas emissions", "가치사슬 배출", "Value chain emissions", "절대값", "Absolute value", "tCO2e", "numeric", "numeric_absolute", ("Scope 3", "스코프 3", "가치사슬", "공급망 배출"), ("카테고리별 배출량이 별도 표로 공시되기도 한다.",)),
        ("K-ENV-N04", "ISSB S2 / GRI", ("GRI 302-1", "SASB RT-CH-130a.1"), "에너지 사용량", "Energy consumption", "직·간접 에너지 소비", "Direct and indirect energy use", "절대값", "Absolute value", "MWh / GJ / kWh", "numeric", "numeric_absolute", ("에너지 사용량", "전력 사용량", "연료 사용량", "MWh", "GJ"), ("재생에너지 목표와 총 에너지 사용량을 구분해야 한다.",)),
        ("K-ENV-N05", "ISSB S2 / GRI", ("GRI 302-1", "RE100"), "재생에너지 사용 비율", "Renewable energy use ratio", "재생에너지 조달 및 사용", "Renewable energy procurement and use", "비율", "Ratio", "%", "numeric", "numeric_ratio", ("재생에너지 비율", "재생전력", "renewable electricity", "REC"), ("RE100 목표와 실제 사용 비율이 함께 제시되는 경우가 많다.",)),
        ("K-ENV-N06", "ISSB S2 / GRI", ("GRI 302-3",), "에너지 집약도", "Energy intensity", "매출 또는 생산량 대비 에너지 강도", "Energy intensity per revenue or production", "집약도", "Intensity", "원단위 / 단위당", "numeric", "numeric_intensity", ("에너지 집약도", "원단위", "단위당 에너지", "energy intensity"), ("절대 사용량과 집약도를 혼동하지 않도록 단위당 표현을 확인한다.",)),
        ("K-ENV-N07", "GRI", ("GRI 303-3",), "물 취수량", "Water withdrawal", "용수 취수", "Water withdrawal", "절대값", "Absolute value", "m3", "numeric", "numeric_absolute", ("취수량", "용수 사용량", "water withdrawal", "용수"), ("지하수, 상수도, 재이용수 등 원천별로 분리 공시될 수 있다.",)),
        ("K-ENV-N08", "GRI", ("GRI 303-4",), "폐수 배출량", "Water discharge", "폐수 배출", "Water discharge", "절대값", "Absolute value", "m3", "numeric", "numeric_absolute", ("폐수 배출량", "방류량", "water discharge", "폐수"), ("방류 수질 지표와 함께 제시되는 표를 우선 확인한다.",)),
        ("K-ENV-N09", "GRI", ("GRI 306-3",), "폐기물 발생량", "Waste generated", "유해 및 비유해 폐기물", "Hazardous and non-hazardous waste", "절대값", "Absolute value", "t / ton", "numeric", "numeric_absolute", ("폐기물 발생량", "유해 폐기물", "비유해 폐기물", "waste generated"), ("총량과 세부 유형을 분리해 읽는다.",)),
        ("K-ENV-N10", "GRI", ("GRI 306-4",), "폐기물 재활용률", "Waste recycling ratio", "재활용 및 재사용", "Recycling and reuse", "비율", "Ratio", "%", "numeric", "numeric_ratio", ("재활용률", "재활용 비율", "resource circulation", "recycling rate"), ("재활용 목표가 아니라 실적 비율을 우선 수집한다.",)),
        ("K-ENV-N11", "GRI", ("GRI 305-7",), "대기오염물질 배출량", "Air pollutant emissions", "NOx, SOx, VOC, 먼지", "NOx, SOx, VOC, particulate matter", "절대값", "Absolute value", "t / kg", "numeric", "numeric_absolute", ("NOx", "SOx", "VOC", "먼지", "대기오염물질"), ("오염물질별 단위가 다를 수 있으니 행별 단위를 확인한다.",)),
        ("K-ENV-N12", "GRI", ("GRI 306-3", "GRI 306-5"), "환경 사고 및 누출 건수", "Environmental incidents and spills", "누출 및 환경 사고", "Spills and environmental incidents", "건수", "Count", "건", "numeric", "numeric_absolute", ("환경 사고", "누출", "spill", "incident"), ("중대 사고와 경미한 사고를 구분해 표기하기도 한다.",)),
    ]
    return [
        _make_item(
            id=id,
            category="E",
            standard_family=sf,
            standard_refs=refs,
            aspect_ko=a_ko,
            aspect_en=a_en,
            kpi_ko=kko,
            kpi_en=ken,
            topic_ko=tko,
            topic_en=ten,
            quantity_ko=qko,
            quantity_en=qen,
            unit=unit,
            disclosure_type=dtype,
            template=template,
            obligation="required" if id in {"K-ENV-N01", "K-ENV-N02", "K-ENV-N03"} else "recommended",
            search_terms=terms,
            knowledge=knowledge,
        )
        for id, sf, refs, kko, ken, tko, ten, qko, qen, unit, dtype, template, terms, knowledge in specs
    ]


def _environment_textual() -> list[KPIItem]:
    a_ko = "환경"
    a_en = "Environment"
    specs = [
        ("K-ENV-T01", "ISSB S2", ("ISSB S2 33",), "온실가스 감축 목표 및 이행조치", "Emissions targets and actions", "배출 목표와 달성 경로", "Emissions targets and actions", "주요 조치", "Key actions", "문장", "text", "text_actions", ("감축 목표", "탄소중립", "배출 감축", "이행조치"), ("수치 목표보다 실행 조치와 관리체계를 함께 읽는다.",)),
        ("K-ENV-T02", "ISSB S2", ("ISSB S2 9", "ISSB S2 13"), "기후 전환 계획", "Climate transition plan", "전환 전략", "Transition strategy", "핵심 서술", "Key narrative", "문장", "text", "text_policy", ("전환 계획", "transition plan", "Net Zero roadmap", "감축 로드맵"), ("전환 투자, 기술 전환, 재생에너지 확대 계획이 함께 언급된다.",)),
        ("K-ENV-T03", "ISSB S2", ("ISSB S2 29(g)",), "내부 탄소가격 운영", "Internal carbon pricing", "탄소 가격 적용", "Carbon pricing approach", "핵심 서술", "Key narrative", "문장", "text", "text_policy", ("내부 탄소가격", "carbon price", "shadow price"), ("실제 적용 여부와 의사결정 활용 방식을 구분한다.",)),
        ("K-ENV-T04", "ISSB S2", ("ISSB S2 22",), "기후 시나리오 분석", "Climate scenario analysis", "시나리오 기반 리스크 평가", "Scenario-based risk analysis", "핵심 서술", "Key narrative", "문장", "text", "text_risk", ("시나리오 분석", "1.5도", "2도", "TCFD"), ("시나리오 이름, 가정, 영향요인을 함께 확인한다.",)),
        ("K-ENV-T05", "GRI", ("GRI 304-3",), "생물다양성 영향 및 복원", "Biodiversity impacts and restoration", "생태계 영향", "Biodiversity impacts", "주요 조치", "Key actions", "문장", "text", "text_actions", ("생물다양성", "복원", "서식지", "ecosystem"), ("보호지역 인접 사업장과 복원 활동이 함께 제시되는 경우가 많다.",)),
        ("K-ENV-T06", "GRI", ("GRI 303-1",), "수자원 관리 정책", "Water stewardship policy", "수자원 관리", "Water stewardship", "핵심 서술", "Key narrative", "문장", "text", "text_policy", ("수자원", "water stewardship", "용수 관리", "물 리스크"), ("고위험 수계, 재이용, 지역사회 영향 평가를 함께 본다.",)),
        ("K-ENV-T07", "GRI", ("GRI 306-2",), "자원순환 및 폐기물 감축 정책", "Waste reduction and circularity policy", "자원순환", "Circularity policy", "핵심 서술", "Key narrative", "문장", "text", "text_policy", ("자원순환", "폐기물 감축", "circular economy", "재활용"), ("재사용, 재생원료, 감량 활동이 묶여 설명되는 경우가 많다.",)),
        ("K-ENV-T08", "GRI", ("GRI 305-7",), "오염 방지 관리체계", "Pollution prevention management", "오염 저감", "Pollution prevention", "주요 조치", "Key actions", "문장", "text", "text_actions", ("오염 방지", "오염물질 관리", "scrubber", "집진"), ("시설 투자와 공정 개선 조치를 함께 읽는다.",)),
        ("K-ENV-T09", "ISSB S2 / RE100", ("ISSB S2", "RE100"), "재생에너지 조달 전략", "Renewable energy procurement strategy", "PPA, REC, 자가발전", "PPA, REC, self-generation", "주요 조치", "Key actions", "문장", "text", "text_actions", ("PPA", "REC", "재생에너지 조달", "자가발전"), ("실제 실적이 아닌 전략 설명 문단이므로 정량값과 구분한다.",)),
        ("K-ENV-T10", "GRI", ("GRI 307-1",), "환경 법규 위반 및 대응", "Environmental compliance cases", "법규 위반, 벌금, 시정조치", "Environmental compliance and remediation", "핵심 서술", "Key narrative", "문장", "text", "text_policy", ("환경 위반", "벌금", "시정조치", "환경 법규"), ("위반 사실이 없을 경우 '없음' 공시가 중요하다.",)),
        ("K-ENV-T11", "ISSB S2 / EU Taxonomy", ("ISSB S2",), "환경 투자 및 녹색 CapEx 계획", "Environmental investment and green CapEx plan", "환경 설비 투자", "Environmental capex", "주요 조치", "Key actions", "문장", "text", "text_actions", ("환경 투자", "CapEx", "녹색 투자", "설비 개선"), ("설비투자 목적과 예상 효과를 함께 확인한다.",)),
        ("K-ENV-T12", "ISSB S2", ("ISSB S2 6",), "환경·기후 거버넌스", "Climate governance oversight", "이사회 및 위원회 감독", "Board and committee oversight", "핵심 서술", "Key narrative", "문장", "text", "text_policy", ("기후 거버넌스", "위원회", "이사회 감독"), ("책임 주체와 보고 체계를 요약한다.",)),
        ("K-ENV-T13", "GRI", ("GRI 308-1",), "공급망 환경 요구사항", "Supplier environmental requirements", "협력사 환경 기준", "Supplier environmental standards", "핵심 서술", "Key narrative", "문장", "text", "text_policy", ("협력사 환경", "공급망 기준", "supplier code"), ("평가 기준, 실사, 개선 요구를 같이 본다.",)),
        ("K-ENV-T14", "GRI", ("GRI 301-3", "GRI 416-1"), "제품 생애주기 및 친환경 설계", "Product lifecycle and eco-design", "제품 환경성", "Product lifecycle impacts", "주요 조치", "Key actions", "문장", "text", "text_actions", ("생애주기", "LCA", "에코디자인", "친환경 제품"), ("제품 사용 단계의 환경성 개선 조치를 확인한다.",)),
    ]
    return [
        _make_item(
            id=id,
            category="E",
            standard_family=sf,
            standard_refs=refs,
            aspect_ko=a_ko,
            aspect_en=a_en,
            kpi_ko=kko,
            kpi_en=ken,
            topic_ko=tko,
            topic_en=ten,
            quantity_ko=qko,
            quantity_en=qen,
            unit=unit,
            disclosure_type=dtype,
            template=template,
            search_terms=terms,
            knowledge=knowledge,
        )
        for id, sf, refs, kko, ken, tko, ten, qko, qen, unit, dtype, template, terms, knowledge in specs
    ]


def _social_numerical() -> list[KPIItem]:
    a_ko = "사회"
    a_en = "Social"
    specs = [
        ("K-SOC-N01", ("GRI 2-7",), "임직원 수", "Total workforce", "고용 규모", "Workforce size", "절대값", "Absolute value", "명", "numeric_absolute", ("임직원 수", "headcount", "고용 규모"), ("국내/해외, 정규/비정규 구분 표를 함께 본다.",)),
        ("K-SOC-N02", ("GRI 401-1",), "신규 채용 인원", "New hires", "채용", "Hiring", "절대값", "Absolute value", "명", "numeric_absolute", ("신규 채용", "채용 인원", "new hires"), ("전체 인원과 채용률이 함께 제시될 수 있다.",)),
        ("K-SOC-N03", ("GRI 401-1",), "이직률", "Turnover rate", "이직 및 유지", "Turnover and retention", "비율", "Ratio", "%", "numeric_ratio", ("이직률", "퇴직률", "turnover"), ("연도별 추이를 우선 확인한다.",)),
        ("K-SOC-N04", ("GRI 404-1",), "평균 교육시간", "Average training hours", "교육 및 훈련", "Training and development", "시간", "Hours", "시간", "numeric_absolute", ("평균 교육시간", "training hours", "교육훈련"), ("1인당 평균시간인지 총시간인지 구분한다.",)),
        ("K-SOC-N05", ("GRI 403-9",), "산업재해율", "Lost time injury frequency rate", "안전 및 보건", "Occupational safety", "비율", "Ratio", "rate", "numeric_ratio", ("산업재해율", "LTIFR", "LTI", "재해율"), ("단위 백만시간당 비율인지 확인한다.",)),
        ("K-SOC-N06", ("GRI 403-9",), "사망 재해 건수", "Fatalities", "중대 재해", "Fatal occupational injuries", "건수", "Count", "건", "numeric_absolute", ("사망 재해", "fatality", "중대재해"), ("협력사 포함 여부를 확인한다.",)),
        ("K-SOC-N07", ("GRI 403-9",), "재해 건수", "Occupational injury count", "재해 건수", "Occupational injuries", "건수", "Count", "건", "numeric_absolute", ("재해 건수", "injury count", "occupational injury"), ("건수와 재해율을 구분한다.",)),
        ("K-SOC-N08", ("GRI 405-1",), "여성 임직원 비율", "Female employee ratio", "성별 다양성", "Gender diversity", "비율", "Ratio", "%", "numeric_ratio", ("여성 비율", "female employees", "gender diversity"), ("전체 임직원 대비 비율을 우선 읽는다.",)),
        ("K-SOC-N09", ("GRI 405-1",), "여성 관리자 비율", "Female manager ratio", "관리자 다양성", "Management diversity", "비율", "Ratio", "%", "numeric_ratio", ("여성 관리자", "women in management", "관리자 비율"), ("임원 비율과 구분한다.",)),
        ("K-SOC-N10", ("GRI 405-1",), "여성 이사회 비율", "Women on board", "이사회 다양성", "Board diversity", "비율", "Ratio", "%", "numeric_ratio", ("여성 이사회", "board diversity", "women directors"), ("사외/사내 구분이 있을 수 있다.",)),
        ("K-SOC-N11", ("GRI 405-1",), "장애인 고용 비율", "Employees with disabilities ratio", "포용 고용", "Inclusive employment", "비율", "Ratio", "%", "numeric_ratio", ("장애인 고용", "disability employment", "inclusive hiring"), ("법정 의무고용률과 실제 고용률을 구분한다.",)),
        ("K-SOC-N12", ("GRI 2-30",), "노조 가입률", "Union membership rate", "노사관계", "Labour relations", "비율", "Ratio", "%", "numeric_ratio", ("노조 가입률", "union membership", "단체교섭"), ("교섭 대상 인원과 노조 가입률이 같이 제시될 수 있다.",)),
        ("K-SOC-N13", ("GRI 2-29",), "임직원 참여도 점수", "Employee engagement score", "직원 몰입", "Employee engagement", "절대값", "Absolute value", "점", "numeric_absolute", ("참여도", "engagement score", "직원 만족"), ("설문 지표는 지수 또는 점수로 제시된다.",)),
        ("K-SOC-N14", ("GRI 403-6",), "결근율", "Absenteeism rate", "근태 관리", "Attendance and absence", "비율", "Ratio", "%", "numeric_ratio", ("결근율", "absenteeism", "결근"), ("질병 결근과 일반 결근을 구분할 수 있다.",)),
        ("K-SOC-N15", ("GRI 414-1",), "협력사 사회 평가 비율", "Suppliers assessed for social impacts", "협력사 평가", "Supplier social assessment", "비율", "Ratio", "%", "numeric_ratio", ("협력사 사회 평가", "supplier assessment", "social audit"), ("전체 협력사 대비 평가 완료 비율을 우선 본다.",)),
        ("K-SOC-N16", ("GRI 418-1",), "고객 개인정보 침해 건수", "Customer privacy incidents", "개인정보 보호", "Privacy incidents", "건수", "Count", "건", "numeric_absolute", ("개인정보 침해", "privacy incident", "data breach"), ("영 건수 공시도 중요하다.",)),
        ("K-SOC-N17", ("GRI 413-1",), "지역사회 투자 금액", "Community investment amount", "사회공헌", "Community investment", "금액", "Amount", "원", "numeric_absolute", ("사회공헌", "community investment", "기부금"), ("기부금과 사회공헌 총액을 구분한다.",)),
        ("K-SOC-N18", ("GRI 406-1",), "인권 침해 신고 건수", "Human rights grievances", "인권", "Human rights grievances", "건수", "Count", "건", "numeric_absolute", ("인권 침해", "grievance", "human rights incident"), ("신고, 접수, 처리 완료를 분리해 읽는다.",)),
    ]
    return [
        _make_item(
            id=id,
            category="S",
            standard_family="ISSB S1 / GRI",
            standard_refs=refs,
            aspect_ko=a_ko,
            aspect_en=a_en,
            kpi_ko=kko,
            kpi_en=ken,
            topic_ko=tko,
            topic_en=ten,
            quantity_ko=qko,
            quantity_en=qen,
            unit=unit,
            disclosure_type="numeric",
            template=template,
            search_terms=terms,
            knowledge=knowledge,
        )
        for id, refs, kko, ken, tko, ten, qko, qen, unit, template, terms, knowledge in specs
    ]


def _social_textual() -> list[KPIItem]:
    a_ko = "사회"
    a_en = "Social"
    specs = [
        ("K-SOC-T01", ("GRI 405-1",), "다양성·형평성·포용 정책", "DEI policy", "다양성 정책", "Diversity, equity and inclusion policy", "핵심 서술", "Key narrative", "문장", "text_policy", ("DEI", "다양성 정책", "형평성", "포용"), ("지표보다 제도와 운영 범위를 요약한다.",)),
        ("K-SOC-T02", ("GRI 412-1",), "인권 실사 체계", "Human rights due diligence", "인권 정책 및 실사", "Human rights due diligence", "핵심 서술", "Key narrative", "문장", "text_policy", ("인권 실사", "human rights due diligence", "인권 정책"), ("실사 범위와 시정조치 체계를 함께 읽는다.",)),
        ("K-SOC-T03", ("GRI 2-30",), "노동 관행 및 단체교섭", "Labour practices and collective bargaining", "노사관계", "Labour relations", "핵심 서술", "Key narrative", "문장", "text_policy", ("단체교섭", "노사관계", "collective bargaining"), ("협의체, 고충처리, 교섭 대상 범위를 확인한다.",)),
        ("K-SOC-T04", ("GRI 403-6",), "임직원 복지 및 웰빙 프로그램", "Employee well-being programs", "복지", "Well-being", "주요 조치", "Key actions", "문장", "text_actions", ("웰빙", "복지 프로그램", "employee wellness"), ("건강검진, 상담, 유연근무 조치를 함께 본다.",)),
        ("K-SOC-T05", ("GRI 401-1",), "인재 확보 및 유지 전략", "Talent attraction and retention strategy", "채용 및 유지", "Talent retention", "주요 조치", "Key actions", "문장", "text_actions", ("인재 확보", "retention", "채용 전략"), ("핵심인재, 유지율, 승계계획을 함께 읽는다.",)),
        ("K-SOC-T06", ("GRI 404-2",), "리스킬링 및 역량 전환 프로그램", "Reskilling and upskilling program", "역량 개발", "Reskilling", "주요 조치", "Key actions", "문장", "text_actions", ("리스킬링", "upskilling", "직무 전환"), ("디지털 전환이나 AI 교육과 함께 등장할 수 있다.",)),
        ("K-SOC-T07", ("GRI 403-1",), "안전보건 거버넌스", "Occupational safety governance", "안전보건 관리체계", "Safety governance", "핵심 서술", "Key narrative", "문장", "text_policy", ("안전보건", "중대재해", "EHS", "안전 거버넌스"), ("위원회, 최고책임자, 공정 안전관리 체계를 요약한다.",)),
        ("K-SOC-T08", ("GRI 414-1",), "공급망 노동 실사 프로세스", "Supply chain labour audit process", "협력사 노동 기준", "Supply chain labour process", "핵심 서술", "Key narrative", "문장", "text_policy", ("공급망 노동", "social audit", "협력사 실사"), ("감사, 시정조치, 재평가 흐름을 본다.",)),
        ("K-SOC-T09", ("GRI 418-1",), "고객 정보보호 정책", "Customer data protection policy", "개인정보 및 보안", "Customer data protection", "핵심 서술", "Key narrative", "문장", "text_policy", ("정보보호", "개인정보", "data protection"), ("기술적 통제와 조직적 통제를 함께 요약한다.",)),
        ("K-SOC-T10", ("GRI 413-1",), "지역사회 참여 전략", "Community engagement strategy", "지역사회 이해관계자 참여", "Community engagement", "주요 조치", "Key actions", "문장", "text_actions", ("지역사회 참여", "community engagement", "사회공헌 전략"), ("참여 채널과 중점 프로그램을 함께 본다.",)),
        ("K-SOC-T11", ("GRI 2-25",), "고충처리 메커니즘", "Grievance mechanism", "신고 및 고충 채널", "Grievance channels", "핵심 서술", "Key narrative", "문장", "text_policy", ("고충처리", "grievance", "신고 채널"), ("내부 임직원과 외부 이해관계자 채널을 구분한다.",)),
        ("K-SOC-T12", ("GRI 416-1",), "접근성 및 제품책임 정책", "Accessibility and product responsibility policy", "제품 접근성 및 안전", "Product accessibility", "핵심 서술", "Key narrative", "문장", "text_policy", ("접근성", "제품 책임", "product safety"), ("품질, 안전, 접근성 정책을 통합적으로 요약한다.",)),
        ("K-SOC-T13", ("GRI 404-2",), "인재개발 프레임워크", "Employee development framework", "승계 및 역량 프레임워크", "Talent development", "핵심 서술", "Key narrative", "문장", "text_policy", ("승계계획", "인재개발", "역량 프레임워크"), ("리더십 파이프라인과 평가 제도를 함께 본다.",)),
        ("K-SOC-T14", ("GRI 408-1", "GRI 409-1"), "현대판 노예제·강제노동 대응", "Modern slavery response", "강제노동 및 아동노동", "Modern slavery controls", "핵심 서술", "Key narrative", "문장", "text_policy", ("강제노동", "아동노동", "modern slavery"), ("위험 평가와 금지 정책을 함께 본다.",)),
        ("K-SOC-T15", ("GRI 417-1",), "책임 있는 마케팅·제품 품질 관리", "Responsible marketing and product quality", "제품 정보 및 품질", "Product quality and marketing", "주요 조치", "Key actions", "문장", "text_actions", ("제품 품질", "responsible marketing", "품질 관리"), ("표시·광고 관리와 리콜 대응을 함께 읽는다.",)),
    ]
    return [
        _make_item(
            id=id,
            category="S",
            standard_family="ISSB S1 / GRI",
            standard_refs=refs,
            aspect_ko=a_ko,
            aspect_en=a_en,
            kpi_ko=kko,
            kpi_en=ken,
            topic_ko=tko,
            topic_en=ten,
            quantity_ko=qko,
            quantity_en=qen,
            unit=unit,
            disclosure_type="text",
            template=template,
            search_terms=terms,
            knowledge=knowledge,
        )
        for id, refs, kko, ken, tko, ten, qko, qen, unit, template, terms, knowledge in specs
    ]


def _governance_numerical() -> list[KPIItem]:
    a_ko = "지배구조"
    a_en = "Governance"
    specs = [
        ("K-GOV-N01", ("GRI 2-9",), "사외이사 비율", "Independent director ratio", "이사회 독립성", "Board independence", "비율", "Ratio", "%", "numeric_ratio", ("사외이사", "독립성", "board independence"), ("독립성 비율은 전체 이사 대비 사외이사 비율로 공시되는 경우가 많다.",)),
        ("K-GOV-N02", ("GRI 2-12",), "이사회 출석률", "Board attendance rate", "이사회 운영", "Board operations", "비율", "Ratio", "%", "numeric_ratio", ("출석률", "board attendance", "위원회 출석"), ("이사회와 위원회 출석률이 별도 공시될 수 있다.",)),
        ("K-GOV-N03", ("GRI 205-2",), "윤리·반부패 교육 이수율", "Anti-corruption training completion", "윤리 교육", "Anti-corruption training", "비율", "Ratio", "%", "numeric_ratio", ("반부패 교육", "윤리 교육", "training completion"), ("임직원 전체와 고위험 조직 대상 이수율을 구분한다.",)),
        ("K-GOV-N04", ("GRI 205-3", "GRI 307-1"), "법규 위반 및 제재 건수", "Compliance violations and sanctions", "위반 및 벌금", "Compliance breaches", "건수", "Count", "건", "numeric_absolute", ("법규 위반", "벌금", "제재", "sanction"), ("금액과 건수가 함께 나오면 건수를 우선 포착한다.",)),
    ]
    return [
        _make_item(
            id=id,
            category="G",
            standard_family="ISSB S1 / GRI",
            standard_refs=refs,
            aspect_ko=a_ko,
            aspect_en=a_en,
            kpi_ko=kko,
            kpi_en=ken,
            topic_ko=tko,
            topic_en=ten,
            quantity_ko=qko,
            quantity_en=qen,
            unit=unit,
            disclosure_type="numeric",
            template=template,
            search_terms=terms,
            knowledge=knowledge,
        )
        for id, refs, kko, ken, tko, ten, qko, qen, unit, template, terms, knowledge in specs
    ]


def _governance_textual() -> list[KPIItem]:
    a_ko = "지배구조"
    a_en = "Governance"
    specs = [
        ("K-GOV-T01", ("GRI 2-9",), "이사회 다양성 정책 및 구성", "Board diversity policy and composition", "이사회 구성", "Board composition", "핵심 서술", "Key narrative", "문장", "text_policy", ("이사회 다양성", "board diversity", "위원회 구성"), ("성별, 전문성, 독립성 구성을 함께 정리한다.",)),
        ("K-GOV-T02", ("ISSB S1 5",), "ESG 거버넌스 구조", "ESG governance structure", "ESG 의사결정 구조", "ESG governance structure", "핵심 서술", "Key narrative", "문장", "text_policy", ("ESG 거버넌스", "위원회", "책임 체계"), ("이사회-경영진-실무조직의 연결 구조를 요약한다.",)),
        ("K-GOV-T03", ("GRI 2-19",), "경영진 보상과 ESG 연계", "Executive compensation linked to ESG", "보상 연계", "ESG-linked remuneration", "핵심 서술", "Key narrative", "문장", "text_policy", ("보상 연계", "ESG KPI", "incentive"), ("단순 언급과 실제 보상 반영을 구분한다.",)),
        ("K-GOV-T04", ("ISSB S1 41",), "리스크 관리 및 내부통제", "Risk management and internal control", "리스크 관리", "Risk management", "핵심 서술", "Key narrative", "문장", "text_risk", ("내부통제", "리스크 관리", "internal control"), ("식별-평가-대응-모니터링 흐름을 요약한다.",)),
        ("K-GOV-T05", ("GRI 207-1",), "조세 투명성 접근", "Tax transparency approach", "세무 정책", "Tax transparency", "핵심 서술", "Key narrative", "문장", "text_policy", ("조세 투명성", "세무 정책", "tax transparency"), ("세무 거버넌스와 조세원칙을 함께 본다.",)),
        ("K-GOV-T06", ("GRI 2-26",), "제보 및 내부신고 메커니즘", "Whistleblowing mechanism", "내부 제보", "Whistleblowing", "핵심 서술", "Key narrative", "문장", "text_policy", ("whistleblowing", "내부신고", "제보 채널"), ("익명성, 보호조치, 처리절차를 함께 요약한다.",)),
        ("K-GOV-T07", ("GRI 418-1",), "사이버보안 거버넌스", "Cybersecurity governance", "정보보호 거버넌스", "Cybersecurity governance", "핵심 서술", "Key narrative", "문장", "text_policy", ("사이버보안", "정보보호 위원회", "cyber governance"), ("거버넌스, 모니터링, 대응훈련 체계를 함께 본다.",)),
    ]
    return [
        _make_item(
            id=id,
            category="G",
            standard_family="ISSB S1 / GRI",
            standard_refs=refs,
            aspect_ko=a_ko,
            aspect_en=a_en,
            kpi_ko=kko,
            kpi_en=ken,
            topic_ko=tko,
            topic_en=ten,
            quantity_ko=qko,
            quantity_en=qen,
            unit=unit,
            disclosure_type="text",
            template=template,
            search_terms=terms,
            knowledge=knowledge,
        )
        for id, refs, kko, ken, tko, ten, qko, qen, unit, template, terms, knowledge in specs
    ]


def default_metadata() -> list[KPIItem]:
    return (
        _environment_numerical()
        + _environment_textual()
        + _social_numerical()
        + _social_textual()
        + _governance_numerical()
        + _governance_textual()
    )


def _build_expression_template(raw: dict[str, Any] | None) -> ExpressionTemplate:
    raw = raw or {}
    return ExpressionTemplate(
        name=str(raw.get("name", "generic_numeric")).strip() or "generic_numeric",
        output_fields=_as_tuple(raw.get("output_fields"))
        or ("disclosed", "value", "unit", "evidence", "confidence", "source_page", "matched_terms"),
        value_mode=str(raw.get("value_mode", "numeric")).strip() or "numeric",
        preferred_units=_as_tuple(raw.get("preferred_units")),
        exclude_terms=_as_tuple(raw.get("exclude_terms")),
        evidence_priority=_as_tuple(raw.get("evidence_priority")),
        prompt_rules=_as_tuple(raw.get("prompt_rules")),
    )


def _build_item(spec: dict[str, Any]) -> KPIItem:
    expression_template = _build_expression_template(spec.get("expression_template"))
    codes, labels, alignment = _crosswalks().get(str(spec.get("id", "")), ((), (), "none"))
    return KPIItem(
        id=str(spec["id"]),
        category=str(spec.get("category", "E")).strip() or "E",
        standard_family=str(spec.get("standard_family", "ISSB/KSSB")).strip(),
        standard_refs=_as_tuple(spec.get("standard_refs")),
        aspect_ko=str(spec.get("aspect_ko", spec.get("domain_ko", spec.get("aspect", "")))).strip(),
        aspect_en=str(spec.get("aspect_en", spec.get("domain_en", spec.get("aspect", "")))).strip(),
        kpi_ko=str(spec.get("kpi_ko", spec.get("kpi", ""))).strip(),
        kpi_en=str(spec.get("kpi_en", spec.get("kpi", ""))).strip(),
        topic_ko=str(spec.get("topic_ko", spec.get("topic", ""))).strip(),
        topic_en=str(spec.get("topic_en", spec.get("topic", ""))).strip(),
        quantity_ko=str(spec.get("quantity_ko", spec.get("value_format", spec.get("quantity", "")))).strip(),
        quantity_en=str(spec.get("quantity_en", spec.get("quantity", spec.get("value_format", "")))).strip(),
        unit=str(spec.get("unit", "")).strip(),
        disclosure_type=str(spec.get("disclosure_type", "numeric")).strip() or "numeric",
        obligation=str(spec.get("obligation", "recommended")).strip() or "recommended",
        k_esg_guideline=str(spec.get("k_esg_guideline", K_ESG_GUIDELINE_URL)).strip() or K_ESG_GUIDELINE_URL,
        k_esg_codes=_as_tuple(spec.get("k_esg_codes")) or codes,
        k_esg_labels=_as_tuple(spec.get("k_esg_labels")) or labels,
        k_esg_alignment=str(spec.get("k_esg_alignment", alignment)).strip() or alignment,
        industry_scope=_as_tuple(spec.get("industry_scope")) or ("전 산업",),
        search_terms=_as_tuple(spec.get("search_terms")),
        knowledge=_as_tuple(spec.get("knowledge")),
        expression_template=expression_template,
    )


def load_metadata(path: str | Path) -> list[KPIItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("KPI metadata must be a JSON array.")
    return [_build_item(item) for item in raw]


def to_jsonable(items: list[KPIItem]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items:
        data = asdict(item)
        data["aspect"] = item.aspect_en
        data["topic"] = item.topic_en
        data["quantity"] = item.quantity_en
        data["domain_ko"] = item.aspect_ko
        data["domain_en"] = item.aspect_en
        data["value_format"] = item.quantity_ko
        payload.append(data)
    return payload


from .entities import KPIItem as KPIItem
from .expressions import (
    ExpressionTemplate as ExpressionTemplate,
    accepts_unit as accepts_unit,
    build_expression_template as _build_expression_template,
    expression_templates as _expression_templates,
    matches_excluded_term as matches_excluded_term,
)
from .extensions import (
    K_ESG_GUIDELINE_URL as K_ESG_GUIDELINE_URL,
    as_tuple as _as_tuple,
    crosswalks as _crosswalks,
    resolve_crosswalk as resolve_crosswalk,
)
