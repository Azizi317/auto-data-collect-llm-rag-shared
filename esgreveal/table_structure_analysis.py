from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from itertools import chain
from typing import Any, Callable, Iterable
import importlib
import os
import re
import sys


@dataclass(frozen=True)
class TableStructureCell:
    row_nums: tuple[int, ...]
    column_nums: tuple[int, ...]
    bbox: tuple[float, float, float, float]
    text: str = ""
    is_header: bool = False


@dataclass(frozen=True)
class TableStructureResult:
    bbox: tuple[float, float, float, float]
    rows: tuple[tuple[float, float, float, float], ...] = ()
    columns: tuple[tuple[float, float, float, float], ...] = ()
    cells: tuple[TableStructureCell, ...] = ()
    source: str = "fitz"


def _backend_order() -> list[str]:
    configured = os.environ.get("ESGREVEAL_TABLE_BACKEND", "auto").strip().lower()
    if configured in {"", "auto"}:
        return ["table_transformer", "lore_tsr", "fitz"]
    return [item.strip() for item in configured.split(",") if item.strip()]


def _normalize_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return tuple(float(value) for value in raw)  # type: ignore[return-value]
    except Exception:
        return None


def _page_image(page: Any):
    from PIL import Image  # type: ignore

    pix = page.get_pixmap(alpha=False)
    return Image.open(BytesIO(pix.tobytes("png")))


def _clip_words(page: Any, bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    try:
        raw_words = page.get_text("words", clip=bbox) or []
    except Exception:
        raw_words = []
    for item in raw_words:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        text = str(item[4] or "").strip()
        if not text:
            continue
        words.append(
            {
                "bbox": [float(item[0]), float(item[1]), float(item[2]), float(item[3])],
                "text": text,
                "block_num": int(item[5]) if len(item) > 5 else 0,
                "line_num": int(item[6]) if len(item) > 6 else 0,
                "span_num": int(item[7]) if len(item) > 7 else len(words),
                "flags": 0,
            }
        )
    return words


def _normalize_cells(raw_cells: Iterable[Any]) -> tuple[TableStructureCell, ...]:
    cells: list[TableStructureCell] = []
    for raw in raw_cells:
        if isinstance(raw, dict):
            bbox = raw.get("bbox")
            row_nums = raw.get("row_nums") or raw.get("rows") or ()
            column_nums = raw.get("column_nums") or raw.get("columns") or ()
            text = str(raw.get("cell text") or raw.get("text") or "").strip()
            is_header = bool(raw.get("column header") or raw.get("header"))
        else:
            bbox = getattr(raw, "bbox", None)
            row_nums = getattr(raw, "row_nums", ())
            column_nums = getattr(raw, "column_nums", ())
            text = str(getattr(raw, "text", "") or "").strip()
            is_header = bool(getattr(raw, "is_header", False))
        normalized_bbox = _normalize_bbox(bbox)
        if normalized_bbox is None:
            continue
        try:
            row_tuple = tuple(int(value) for value in row_nums)
            col_tuple = tuple(int(value) for value in column_nums)
        except Exception:
            row_tuple = ()
            col_tuple = ()
        cells.append(
            TableStructureCell(
                row_nums=row_tuple,
                column_nums=col_tuple,
                bbox=normalized_bbox,
                text=text,
                is_header=is_header,
            )
        )
    return tuple(cells)


def _normalize_box_items(raw_items: Iterable[Any]) -> tuple[tuple[float, float, float, float], ...]:
    boxes: list[tuple[float, float, float, float]] = []
    for item in raw_items:
        if isinstance(item, dict):
            bbox = item.get("bbox", item)
        else:
            bbox = getattr(item, "bbox", item)
        normalized = _normalize_bbox(bbox)
        if normalized is not None:
            boxes.append(normalized)
    return tuple(boxes)


def _normalize_structure(raw: Any, bbox: tuple[float, float, float, float], source: str) -> TableStructureResult | None:
    if raw is None:
        return None
    if isinstance(raw, TableStructureResult):
        return raw
    if isinstance(raw, dict):
        rows = _normalize_box_items(raw.get("rows", ()))
        columns = _normalize_box_items(raw.get("columns", ()))
        cells = _normalize_cells(raw.get("cells", ()))
        return TableStructureResult(bbox=bbox, rows=rows, columns=columns, cells=cells, source=source)
    return None


def _table_to_grid_text(result: TableStructureResult) -> str:
    if not result.cells:
        return ""
    cell_map: dict[tuple[int, int], str] = {}
    max_row = -1
    max_col = -1
    for cell in result.cells:
        if not cell.row_nums or not cell.column_nums:
            continue
        row = min(cell.row_nums)
        col = min(cell.column_nums)
        max_row = max(max_row, row)
        max_col = max(max_col, col)
        text = re.sub(r"\s+", " ", cell.text or "").strip()
        cell_map[(row, col)] = text
    if max_row < 0 or max_col < 0:
        return ""
    lines: list[str] = []
    for row in range(max_row + 1):
        values = [cell_map.get((row, col), "") for col in range(max_col + 1)]
        if any(value for value in values):
            lines.append(" | ".join(values))
    return "\n".join(lines)


def reconstruct_table_text(page: Any, bbox: tuple[float, float, float, float], fitz_text: str = "") -> tuple[str, TableStructureResult | None]:
    for backend in _backend_order():
        if backend == "table_transformer":
            result = _analyze_with_table_transformer(page, bbox)
        elif backend == "lore_tsr":
            result = _analyze_with_lore_tsr(page, bbox)
        elif backend == "fitz":
            result = None
        else:
            result = None
        if result is None:
            continue
        structured = _table_to_grid_text(result)
        if structured.strip():
            return structured.strip(), result
    return fitz_text.strip(), None


def _add_repo_path(env_name: str) -> None:
    repo_path = os.environ.get(env_name, "").strip()
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


@lru_cache(maxsize=1)
def _table_transformer_bundle() -> dict[str, Any] | None:
    model_name = os.environ.get("ESGREVEAL_TABLE_TRANSFORMER_MODEL", "").strip()
    if not model_name:
        return None
    try:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection  # type: ignore
    except Exception:
        return None
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        model = TableTransformerForObjectDetection.from_pretrained(model_name)
    except Exception:
        return None
    device = os.environ.get("ESGREVEAL_TABLE_TRANSFORMER_DEVICE", "").strip().lower() or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return {"torch": torch, "processor": processor, "model": model, "device": device}


def _crop_image(page: Any, bbox: tuple[float, float, float, float]):
    from PIL import Image  # type: ignore

    image = _page_image(page)
    x0, y0, x1, y1 = [int(value) for value in bbox]
    return image.crop((x0, y0, x1, y1))


def _shift_words(words: list[dict[str, Any]], bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    x0, y0, _, _ = bbox
    shifted: list[dict[str, Any]] = []
    for word in words:
        wx0, wy0, wx1, wy1 = word["bbox"]
        copied = dict(word)
        copied["bbox"] = [wx0 - x0, wy0 - y0, wx1 - x0, wy1 - y0]
        shifted.append(copied)
    return shifted


@lru_cache(maxsize=1)
def _table_transformer_repo() -> dict[str, Any] | None:
    repo_path = os.path.join(PathLike.root(), "external", "table-transformer")
    if not os.path.isdir(repo_path):
        return None
    src_path = os.path.join(repo_path, "src")
    detr_path = os.path.join(repo_path, "detr")
    added_paths: list[str] = []
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        added_paths.append(src_path)
    if detr_path not in sys.path:
        sys.path.append(detr_path)
        added_paths.append(detr_path)
    try:
        inference = importlib.import_module("inference")
    except Exception:
        return None
    finally:
        for added in added_paths:
            while added in sys.path:
                sys.path.remove(added)
        _cleanup_shadowed_table_transformer_modules(repo_path)
    return {"inference": inference}


class PathLike:
    @staticmethod
    def root() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _cleanup_shadowed_table_transformer_modules(repo_path: str) -> None:
    normalized_repo = os.path.normcase(os.path.abspath(repo_path))
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        normalized_file = os.path.normcase(os.path.abspath(module_file))
        if normalized_repo not in normalized_file:
            continue
        if name == "datasets" or name.startswith("datasets."):
            sys.modules.pop(name, None)


def _analyze_with_table_transformer(page: Any, bbox: tuple[float, float, float, float]) -> TableStructureResult | None:
    bundle = _table_transformer_bundle()
    repo = _table_transformer_repo()
    if bundle is None or repo is None:
        return None

    crop = _crop_image(page, bbox)
    words = _shift_words(_clip_words(page, bbox), bbox)
    if not words:
        return None

    torch = bundle["torch"]
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]
    inference = repo["inference"]
    try:
        encoding = processor(
            images=crop,
            size={"shortest_edge": 800, "longest_edge": 1000},
            return_tensors="pt",
        )
        encoding = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
        target_sizes = torch.tensor([crop.size[::-1]], device=device)
        processed = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        id2label = model.config.id2label
        objects: list[dict[str, Any]] = []
        for score, label, box in zip(processed["scores"], processed["labels"], processed["boxes"], strict=False):
            label_name = str(id2label.get(int(label.item()), ""))
            if label_name.lower() == "no object":
                continue
            x0, y0, x1, y1 = [float(value) for value in box.tolist()]
            objects.append({"label": label_name, "score": float(score.item()), "bbox": [x0, y0, x1, y1]})
        if not objects:
            return None
        structures = inference.objects_to_structures(objects, words, inference.structure_class_thresholds)
        if not structures:
            return None
        structure = structures[0]
        cell_result = inference.structure_to_cells(structure, words)
        if isinstance(cell_result, tuple):
            cells = cell_result[0]
        else:
            cells = cell_result
    except Exception:
        return None

    rows = tuple(
        _normalize_bbox(item.get("bbox"))
        for item in structure.get("rows", ())
        if _normalize_bbox(item.get("bbox")) is not None
    )
    columns = tuple(
        _normalize_bbox(item.get("bbox"))
        for item in structure.get("columns", ())
        if _normalize_bbox(item.get("bbox")) is not None
    )
    normalized_cells: list[TableStructureCell] = []
    x0, y0, _, _ = bbox
    for cell in cells:
        cell_bbox = _normalize_bbox(cell.get("bbox"))
        if cell_bbox is None:
            continue
        normalized_cells.append(
            TableStructureCell(
                row_nums=tuple(int(value) for value in cell.get("row_nums", ())),
                column_nums=tuple(int(value) for value in cell.get("column_nums", ())),
                bbox=(cell_bbox[0] + x0, cell_bbox[1] + y0, cell_bbox[2] + x0, cell_bbox[3] + y0),
                text=str(cell.get("cell text") or "").strip(),
                is_header=bool(cell.get("column header")),
            )
        )
    shifted_rows = tuple((row[0] + x0, row[1] + y0, row[2] + x0, row[3] + y0) for row in rows)
    shifted_cols = tuple((col[0] + x0, col[1] + y0, col[2] + x0, col[3] + y0) for col in columns)
    return TableStructureResult(bbox=bbox, rows=shifted_rows, columns=shifted_cols, cells=tuple(normalized_cells), source="table_transformer")


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


def _analyze_with_lore_tsr(page: Any, bbox: tuple[float, float, float, float]) -> TableStructureResult | None:
    _add_repo_path("ESGREVEAL_LORE_PATH")
    adapter_spec = os.environ.get("ESGREVEAL_LORE_PREDICTOR", "").strip()
    if adapter_spec:
        predictor = _import_from_spec(adapter_spec)
        if predictor is None:
            return None
        try:
            crop = _crop_image(page, bbox)
            words = _shift_words(_clip_words(page, bbox), bbox)
            raw = predictor(image=crop, bbox=bbox, words=words)
        except Exception:
            return None
        return _normalize_structure(raw, bbox, source="lore_tsr")

    model_name = os.environ.get("ESGREVEAL_LORE_MODEL", "").strip()
    if not model_name:
        return None
    bundle = _lore_pipeline_bundle(model_name)
    if bundle is None:
        return None
    try:
        crop = _crop_image(page, bbox)
        words = _shift_words(_clip_words(page, bbox), bbox)
        raw = bundle["pipeline"](crop)
    except Exception:
        return None
    return _normalize_lore_output(raw, bbox, words)


@lru_cache(maxsize=2)
def _lore_pipeline_bundle(model_name: str) -> dict[str, Any] | None:
    try:
        import torch  # type: ignore
        from modelscope.pipelines import pipeline  # type: ignore
        from modelscope.utils.constant import Tasks  # type: ignore
    except Exception:
        return None

    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[assignment]
        torch.nn.Module.cuda = lambda self, *args, **kwargs: self  # type: ignore[assignment]

    try:
        recognizer = pipeline(Tasks.lineless_table_recognition, model=model_name)
    except Exception:
        return None
    return {"pipeline": recognizer}


def _word_overlap_ratio(word_bbox: tuple[float, float, float, float], cell_bbox: tuple[float, float, float, float]) -> float:
    wx0, wy0, wx1, wy1 = word_bbox
    cx0, cy0, cx1, cy1 = cell_bbox
    x_overlap = max(0.0, min(wx1, cx1) - max(wx0, cx0))
    y_overlap = max(0.0, min(wy1, cy1) - max(wy0, cy0))
    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0
    word_area = max((wx1 - wx0) * (wy1 - wy0), 1e-6)
    return (x_overlap * y_overlap) / word_area


def _text_for_cell(words: list[dict[str, Any]], bbox: tuple[float, float, float, float]) -> str:
    matched = [
        word
        for word in words
        if _word_overlap_ratio(tuple(word["bbox"]), bbox) >= 0.4
    ]
    if not matched:
        return ""
    matched.sort(key=lambda item: (item.get("block_num", 0), item.get("line_num", 0), item.get("span_num", 0)))
    return re.sub(r"\s+", " ", " ".join(str(item.get("text", "")).strip() for item in matched if str(item.get("text", "")).strip())).strip()


def _normalize_lore_output(
    raw: Any,
    table_bbox: tuple[float, float, float, float],
    words: list[dict[str, Any]],
) -> TableStructureResult | None:
    if not isinstance(raw, dict):
        return None
    polygons = raw.get("polygons")
    logical_boxes = raw.get("boxes")
    if polygons is None or logical_boxes is None:
        return None

    cells: list[TableStructureCell] = []
    x0, y0, _, _ = table_bbox
    row_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    col_boxes: dict[int, list[tuple[float, float, float, float]]] = {}

    for polygon, logical in zip(polygons, logical_boxes, strict=False):
        try:
            points = [float(value) for value in polygon]
            logical_values = [int(round(float(value))) for value in logical]
        except Exception:
            continue
        if len(points) != 8 or len(logical_values) != 4:
            continue
        local_bbox = (
            min(points[0], points[6]),
            min(points[1], points[3]),
            max(points[2], points[4]),
            max(points[5], points[7]),
        )
        row_start, row_end, col_start, col_end = logical_values
        row_nums = tuple(range(min(row_start, row_end), max(row_start, row_end) + 1))
        col_nums = tuple(range(min(col_start, col_end), max(col_start, col_end) + 1))
        text = _text_for_cell(words, local_bbox)
        global_bbox = (local_bbox[0] + x0, local_bbox[1] + y0, local_bbox[2] + x0, local_bbox[3] + y0)
        cells.append(
            TableStructureCell(
                row_nums=row_nums,
                column_nums=col_nums,
                bbox=global_bbox,
                text=text,
                is_header=min(row_nums, default=1) == 0,
            )
        )
        for row_num in row_nums:
            row_boxes.setdefault(row_num, []).append(global_bbox)
        for col_num in col_nums:
            col_boxes.setdefault(col_num, []).append(global_bbox)

    if not cells:
        return None

    def merge_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    rows = tuple(merge_boxes(row_boxes[index]) for index in sorted(row_boxes))
    columns = tuple(merge_boxes(col_boxes[index]) for index in sorted(col_boxes))
    return TableStructureResult(
        bbox=table_bbox,
        rows=rows,
        columns=columns,
        cells=tuple(cells),
        source="lore_tsr",
    )
