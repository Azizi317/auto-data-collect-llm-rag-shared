from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable
import importlib
import os
import sys


@dataclass(frozen=True)
class GeoLayoutLMBlock:
    bbox: tuple[float, float, float, float]
    kind: str
    score: float | None = None


def _add_repo_path() -> None:
    repo_path = os.environ.get("ESGREVEAL_GEOLAYOUTLM_PATH", "").strip()
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _import_callable(spec: str) -> Callable[..., Any] | None:
    module_name, _, attr_name = spec.replace(":", ".").rpartition(".")
    if not module_name or not attr_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    candidate = getattr(module, attr_name, None)
    return candidate if callable(candidate) else None


def _normalize_kind(label: str) -> str:
    normalized = str(label or "").strip().lower()
    if any(token in normalized for token in ("title", "header", "heading", "section")):
        return "heading"
    if "table" in normalized:
        return "table"
    return "paragraph"


def _normalize_block(raw: Any) -> GeoLayoutLMBlock | None:
    if isinstance(raw, dict):
        bbox = raw.get("bbox")
        kind = raw.get("kind") or raw.get("type") or raw.get("label")
        score = raw.get("score")
    else:
        bbox = getattr(raw, "bbox", None)
        kind = getattr(raw, "kind", None) or getattr(raw, "type", None) or getattr(raw, "label", None)
        score = getattr(raw, "score", None)

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        normalized_bbox = tuple(float(value) for value in bbox)
    except Exception:
        return None

    return GeoLayoutLMBlock(
        bbox=normalized_bbox,
        kind=_normalize_kind(str(kind or "")),
        score=float(score) if score is not None else None,
    )


@lru_cache(maxsize=1)
def _predictor() -> Callable[..., Any] | None:
    _add_repo_path()
    spec = os.environ.get("ESGREVEAL_GEOLAYOUTLM_PREDICTOR", "").strip()
    if not spec:
        return None
    return _import_callable(spec)


def predict_layout(*, image: Any, page_num: int) -> list[GeoLayoutLMBlock]:
    predictor = _predictor()
    if predictor is None:
        return []

    try:
        result = predictor(image=image, page_num=page_num)
    except TypeError:
        try:
            result = predictor(image)
        except Exception:
            return []
    except Exception:
        return []

    if not result:
        return []

    blocks: list[GeoLayoutLMBlock] = []
    for raw in cast_iterable(result):
        block = _normalize_block(raw)
        if block is not None:
            blocks.append(block)
    return blocks


def cast_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if hasattr(value, "__iter__"):
        return value
    return ()
