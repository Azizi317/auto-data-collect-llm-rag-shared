from __future__ import annotations

from typing import Any


K_ESG_GUIDELINE_URL = "https://k-esg.org/guide/guideline02"


def as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def crosswalks() -> dict[str, tuple[tuple[str, ...], tuple[str, ...], str]]:
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


def resolve_crosswalk(item_id: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    return crosswalks().get(item_id, ((), (), "none"))
