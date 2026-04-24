from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any, Callable, Iterable
import importlib
import os
import sys
import threading


@dataclass(frozen=True)
class LayoutRegion:
    page_num: int
    order: int
    kind: str
    bbox: tuple[float, float, float, float]
    score: float | None = None
    source: str = "layout_model"
    label: str | None = None


def _backend_order() -> list[str]:
    configured = os.environ.get("ESGREVEAL_LAYOUT_BACKEND", "auto").strip().lower()
    if configured in {"", "auto"}:
        return ["geolayoutlm", "layoutlmv3", "layoutparser"]
    return [item.strip() for item in configured.split(",") if item.strip()]


def _page_image(page: Any):
    from PIL import Image  # type: ignore

    pix = page.get_pixmap(alpha=False)
    return Image.open(BytesIO(pix.tobytes("png")))


def _kind_from_type(block_type: str) -> str:
    normalized = str(block_type or "").lower()
    if any(token in normalized for token in ("title", "header", "heading", "section")):
        return "heading"
    if "table" in normalized:
        return "table"
    return "paragraph"


def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    except Exception:
        return None


def _regions_from_iterable(
    page_num: int,
    blocks: Iterable[Any],
    source: str,
) -> list[LayoutRegion]:
    regions: list[LayoutRegion] = []
    for order, block in enumerate(blocks, start=1):
        bbox = _normalize_bbox(getattr(block, "bbox", None) if not isinstance(block, dict) else block.get("bbox"))
        if bbox is None:
            continue
        label = getattr(block, "kind", None) if not isinstance(block, dict) else block.get("kind")
        if not label:
            label = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        score = getattr(block, "score", None) if not isinstance(block, dict) else block.get("score")
        regions.append(
            LayoutRegion(
                page_num=page_num,
                order=order,
                kind=_kind_from_type(str(label or "")),
                bbox=bbox,
                score=float(score) if score is not None else None,
                source=source,
                label=str(label or "") or None,
            )
        )
    return regions


def _import_from_spec(spec: str) -> Callable[..., Any] | None:
    module_name, _, attr_name = spec.replace(":", ".").rpartition(".")
    if not module_name or not attr_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    target = getattr(module, attr_name, None)
    return target if callable(target) else None


def _add_optional_repo_path(env_name: str) -> None:
    repo_path = os.environ.get(env_name, "").strip()
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _analyze_with_geolayoutlm(page: Any, page_num: int) -> list[LayoutRegion]:
    _add_optional_repo_path("ESGREVEAL_GEOLAYOUTLM_PATH")
    adapter_spec = os.environ.get(
        "ESGREVEAL_GEOLAYOUTLM_ADAPTER",
        "esgreveal.adapters.geolayoutlm_workspace:predict_layout",
    ).strip()
    if not adapter_spec:
        return []

    adapter = _import_from_spec(adapter_spec)
    if adapter is None:
        return []

    try:
        image = _page_image(page)
        blocks = adapter(image=image, page_num=page_num, page=page)
    except Exception:
        return []
    if not blocks:
        return []
    return _regions_from_iterable(page_num, blocks, source="geolayoutlm")


def _ocr_words_and_boxes(image) -> tuple[list[str], list[list[int]], list[tuple[float, float, float, float]], list[tuple[int, int, int]]]:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return [], [], [], []

    data = pytesseract.image_to_data(image, lang="kor+eng", output_type=pytesseract.Output.DICT)
    width, height = image.size
    words: list[str] = []
    boxes: list[list[int]] = []
    pixel_boxes: list[tuple[float, float, float, float]] = []
    line_keys: list[tuple[int, int, int]] = []
    for index, text in enumerate(data.get("text", [])):
        token = str(text or "").strip()
        if not token:
            continue
        left = int(data["left"][index])
        top = int(data["top"][index])
        box_width = int(data["width"][index])
        box_height = int(data["height"][index])
        if box_width <= 0 or box_height <= 0:
            continue
        x0 = max(0, min(1000, int((left / max(width, 1)) * 1000)))
        y0 = max(0, min(1000, int((top / max(height, 1)) * 1000)))
        x1 = max(0, min(1000, int(((left + box_width) / max(width, 1)) * 1000)))
        y1 = max(0, min(1000, int(((top + box_height) / max(height, 1)) * 1000)))
        words.append(token)
        boxes.append([x0, y0, x1, y1])
        pixel_boxes.append((float(left), float(top), float(left + box_width), float(top + box_height)))
        line_keys.append(
            (
                int(data.get("block_num", [0])[index]),
                int(data.get("par_num", [0])[index]),
                int(data.get("line_num", [0])[index]),
            )
        )
    return words, boxes, pixel_boxes, line_keys


_layoutlmv3_lock = threading.Lock()
_layoutlmv3_cache: tuple[Any, Any] | None | bool = False  # False = not loaded yet


def _layoutlmv3_bundle() -> tuple[Any, Any] | None:
    global _layoutlmv3_cache
    if _layoutlmv3_cache is not False:
        return _layoutlmv3_cache  # type: ignore[return-value]
    with _layoutlmv3_lock:
        if _layoutlmv3_cache is not False:
            return _layoutlmv3_cache  # type: ignore[return-value]
        model_name = os.environ.get("ESGREVEAL_LAYOUTLMV3_MODEL", "").strip()
        if not model_name:
            _layoutlmv3_cache = None
            return None
        try:
            from transformers import AutoModelForTokenClassification, AutoProcessor  # type: ignore
        except Exception:
            _layoutlmv3_cache = None
            return None
        try:
            processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            _layoutlmv3_cache = (processor, model)
        except Exception:
            _layoutlmv3_cache = None
        return _layoutlmv3_cache  # type: ignore[return-value]


def _merge_bbox(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _aggregate_layoutlmv3_predictions(
    *,
    page_num: int,
    token_labels: list[str],
    token_scores: list[float],
    line_keys: list[tuple[int, int, int]],
    pixel_boxes: list[tuple[float, float, float, float]],
) -> list[LayoutRegion]:
    per_line: dict[tuple[int, int, int], list[tuple[str, float, tuple[float, float, float, float]]]] = {}
    line_order: list[tuple[int, int, int]] = []
    for label, score, line_key, bbox in zip(token_labels, token_scores, line_keys, pixel_boxes, strict=False):
        if line_key not in per_line:
            line_order.append(line_key)
        per_line.setdefault(line_key, []).append((label, score, bbox))

    regions: list[LayoutRegion] = []
    for order, line_key in enumerate(line_order, start=1):
        items = per_line[line_key]
        if not items:
            continue
        best_label, best_score, first_bbox = items[0]
        merged_bbox = first_bbox
        for label, score, bbox in items[1:]:
            if score > best_score:
                best_label, best_score = label, score
            merged_bbox = _merge_bbox(merged_bbox, bbox)
        regions.append(
            LayoutRegion(
                page_num=page_num,
                order=order,
                kind=_kind_from_type(best_label),
                bbox=merged_bbox,
                score=float(best_score),
                source="layoutlmv3",
                label=best_label,
            )
        )
    return regions


def _analyze_with_layoutlmv3(page: Any, page_num: int) -> list[LayoutRegion]:
    bundle = _layoutlmv3_bundle()
    if bundle is None:
        return []

    processor, model = bundle
    try:
        import torch  # type: ignore
    except Exception:
        return []

    try:
        image = _page_image(page)
        words, norm_boxes, pixel_boxes, line_keys = _ocr_words_and_boxes(image)
        if not words or not norm_boxes:
            return []
        encoding = processor(
            image,
            words,
            boxes=norm_boxes,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
        word_ids = encoding.word_ids(batch_index=0)
        labels: list[str] = []
        scores: list[float] = []
        boxes: list[tuple[float, float, float, float]] = []
        grouped_line_keys: list[tuple[int, int, int]] = []
        seen_word_ids: set[int] = set()
        probabilities = outputs.logits.softmax(-1).squeeze(0)
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids or word_id >= len(norm_boxes):
                continue
            seen_word_ids.add(word_id)
            label_id = predictions[token_index]
            label = str(model.config.id2label.get(label_id, "paragraph"))
            score = float(probabilities[token_index, label_id].item())
            boxes.append(pixel_boxes[word_id])
            labels.append(label)
            scores.append(score)
            grouped_line_keys.append(line_keys[word_id])
    except Exception:
        return []

    return _aggregate_layoutlmv3_predictions(
        page_num=page_num,
        token_labels=labels,
        token_scores=scores,
        line_keys=grouped_line_keys,
        pixel_boxes=boxes,
    )


@lru_cache(maxsize=1)
def _layoutparser_model() -> Any:
    try:
        import layoutparser as lp  # type: ignore
    except Exception:
        return None

    config = os.environ.get(
        "ESGREVEAL_LAYOUTPARSER_CONFIG",
        "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    )
    label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", float(os.environ.get("ESGREVEAL_LAYOUTPARSER_THRESHOLD", "0.72"))]
    try:
        return lp.Detectron2LayoutModel(config, extra_config=extra_config, label_map=label_map)
    except Exception:
        return None


def _analyze_with_layoutparser(page: Any, page_num: int) -> list[LayoutRegion]:
    model = _layoutparser_model()
    if model is None:
        return []
    try:
        image = _page_image(page)
        layout = model.detect(image)
    except Exception:
        return []

    regions: list[LayoutRegion] = []
    for order, block in enumerate(layout, start=1):
        rect = getattr(block, "block", None)
        if rect is None:
            continue
        regions.append(
            LayoutRegion(
                page_num=page_num,
                order=order,
                kind=_kind_from_type(getattr(block, "type", "")),
                bbox=(float(rect.x_1), float(rect.y_1), float(rect.x_2), float(rect.y_2)),
                score=float(getattr(block, "score", 0.0) or 0.0),
                source="layoutparser",
            )
        )
    return regions


def analyze_page_layout(page: Any, page_num: int) -> list[LayoutRegion]:
    analyzers: dict[str, Callable[[Any, int], list[LayoutRegion]]] = {
        "geolayoutlm": _analyze_with_geolayoutlm,
        "layoutlmv3": _analyze_with_layoutlmv3,
        "layoutparser": _analyze_with_layoutparser,
    }
    for backend in _backend_order():
        analyzer = analyzers.get(backend)
        if analyzer is None:
            continue
        regions = analyzer(page, page_num)
        if regions:
            return regions
    return []
