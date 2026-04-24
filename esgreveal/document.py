from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable
import re
import threading

# PyMuPDF find_tables()는 내부 C 코드가 non-reentrant이므로 멀티스레드 동시 호출 시 데드락 발생.
# 전역 lock으로 직렬화한다.
_fitz_tables_lock = threading.Lock()

from .layout_analysis import analyze_page_layout
from .table_structure_analysis import reconstruct_table_text


_WHITESPACE_RE = re.compile(r"\s+")
_SECTION_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[A-Z]\d+(?:\.\d+)*|[IVX]+)\b")
_ALL_CAPS_RE = re.compile(r"^[A-Z0-9 ,.&()/-]{8,}$")


@dataclass(frozen=True)
class DocumentBlock:
    page_num: int
    order: int
    kind: str
    text: str
    section: str | None = None
    section_level: int | None = None
    outline_path: tuple[str, ...] = ()
    font_size: float | None = None
    bold: bool = False
    source: str = "pymupdf"
    table_index: int | None = None


@dataclass(frozen=True)
class DocumentSummary:
    page_count: int
    block_count: int
    heading_count: int
    table_count: int
    section_count: int
    sections: tuple[str, ...]
    outline_count: int
    document_kind: str
    ocr_page_count: int = 0
    extraction_mode: str = "text"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def looks_like_heading(text: str, font_size: float | None = None, bold: bool = False) -> bool:
    candidate = normalize_text(text)
    if not candidate or len(candidate) > 140:
        return False
    if len(candidate) < 4 and candidate.isdigit():
        return False
    if any(mark in candidate for mark in ("=", "{", "}", "||")):
        return False
    if _SECTION_RE.match(candidate):
        return True
    if font_size is not None and font_size >= 12.5 and len(candidate) <= 100:
        return True
    if bold and len(candidate) <= 90 and candidate.count(" ") <= 12:
        return True
    if _ALL_CAPS_RE.match(candidate) and len(candidate.split()) <= 12:
        return True
    if re.fullmatch(r"[가-힣0-9·\-\s]{2,60}", candidate) and len(candidate) <= 30:
        return True
    return False


def looks_like_table(text: str) -> bool:
    candidate = normalize_text(text)
    if not candidate:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        number_lines = sum(1 for line in lines if re.search(r"\d", line))
        if number_lines >= max(2, len(lines) - 1):
            return True
    if candidate.count("  ") >= 3:
        return True
    if "|" in candidate or "\t" in text:
        return True
    numeric_tokens = sum(1 for token in candidate.split() if re.fullmatch(r"[\d,.()%/-]+", token or ""))
    if numeric_tokens >= 4:
        return True
    if re.search(r"\b(?:tCO2e|MWh|GJ|kWh|m3|㎥|%|명|시간|건|원)\b", candidate, re.I) and len(lines) >= 2:
        return True
    return False


def _bbox_tuple(raw_bbox: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        return tuple(float(value) for value in raw_bbox)  # type: ignore[return-value]
    except Exception:
        return None


def _bbox_intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    x_overlap = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    y_overlap = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    if x_overlap <= 0 or y_overlap <= 0:
        return False
    left_area = max((lx1 - lx0) * (ly1 - ly0), 1e-6)
    right_area = max((rx1 - rx0) * (ry1 - ry0), 1e-6)
    overlap = x_overlap * y_overlap
    return overlap / min(left_area, right_area) >= 0.18


def _bbox_contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ox0 <= ix0 and oy0 <= iy0 and ox1 >= ix1 and oy1 >= iy1


def _bbox_overlap_ratio(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    x_overlap = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    y_overlap = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0
    overlap = x_overlap * y_overlap
    left_area = max((lx1 - lx0) * (ly1 - ly0), 1e-6)
    return overlap / left_area


def _best_layout_region(
    bbox: tuple[float, float, float, float] | None,
    regions: list[object],
    *,
    kind: str | None = None,
    min_ratio: float = 0.28,
) -> object | None:
    if bbox is None:
        return None
    best_region = None
    best_score = -1.0
    for region in regions:
        if kind and getattr(region, "kind", None) != kind:
            continue
        region_bbox = getattr(region, "bbox", None)
        if not isinstance(region_bbox, tuple) or len(region_bbox) != 4:
            continue
        ratio = _bbox_overlap_ratio(bbox, region_bbox)
        if ratio < min_ratio:
            continue
        score = float(getattr(region, "score", 0.0) or 0.0)
        weighted = ratio + (score * 0.15)
        if weighted > best_score:
            best_score = weighted
            best_region = region
    return best_region


def _numbering_level(text: str) -> int | None:
    candidate = normalize_text(text)
    if not candidate:
        return None
    match = re.match(r"^\s*([A-Z]?\d+(?:\.\d+)*)\b", candidate)
    if match:
        return min(match.group(1).count(".") + 1, 6)
    match = re.match(r"^\s*([IVX]+)\b", candidate)
    if match:
        return 1
    return None


def _font_level_map(blocks: Iterable[DocumentBlock]) -> dict[float, int]:
    sizes = sorted(
        {
            round(float(block.font_size), 1)
            for block in blocks
            if block.kind == "heading" and block.font_size is not None
        },
        reverse=True,
    )
    return {size: min(index + 1, 6) for index, size in enumerate(sizes)}


def _infer_heading_level(block: DocumentBlock, font_levels: dict[float, int]) -> int:
    numbered = _numbering_level(block.text)
    if numbered is not None:
        return numbered
    if block.font_size is not None:
        level = font_levels.get(round(float(block.font_size), 1))
        if level is not None:
            return level
    if block.bold:
        return 2
    return 3


def _apply_outline_structure(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    if not blocks:
        return []

    font_levels = _font_level_map(blocks)
    outlined: list[DocumentBlock] = []
    outline_stack: list[tuple[int, str]] = []

    for block in blocks:
        if block.kind == "heading":
            level = _infer_heading_level(block, font_levels)
            while outline_stack and outline_stack[-1][0] >= level:
                outline_stack.pop()
            outline_stack.append((level, block.text))
            outline_path = tuple(title for _, title in outline_stack)
            outlined.append(
                DocumentBlock(
                    page_num=block.page_num,
                    order=block.order,
                    kind=block.kind,
                    text=block.text,
                    section=block.text,
                    section_level=level,
                    outline_path=outline_path,
                    font_size=block.font_size,
                    bold=block.bold,
                    source=block.source,
                    table_index=block.table_index,
                )
            )
            continue

        current_section = outline_stack[-1][1] if outline_stack else block.section
        current_level = outline_stack[-1][0] if outline_stack else block.section_level
        outline_path = tuple(title for _, title in outline_stack)
        outlined.append(
            DocumentBlock(
                page_num=block.page_num,
                order=block.order,
                kind=block.kind,
                text=block.text,
                section=current_section,
                section_level=current_level,
                outline_path=outline_path,
                font_size=block.font_size,
                bold=block.bold,
                source=block.source,
                table_index=block.table_index,
            )
        )

    return outlined


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def _find_tables(page) -> list[tuple[int, object]]:
    try:
        with _fitz_tables_lock, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            finder = page.find_tables()
    except Exception:
        return []
    tables = list(getattr(finder, "tables", []) or [])
    return list(enumerate(tables, start=1))


def _words_to_rows(page, bbox: tuple[float, float, float, float]) -> str:
    """Extract text from a bbox by grouping words into rows (y-proximity) and cells (x-gap)."""
    try:
        import fitz  # type: ignore
        rect = fitz.Rect(*bbox)
        raw_words = page.get_text("words", clip=rect) or []
    except Exception:
        return ""
    if not raw_words:
        return ""

    # word tuple: (x0, y0, x1, y1, "text", block_no, line_no, word_no)
    words = [
        (float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]).strip())
        for w in raw_words
        if str(w[4]).strip()
    ]
    if not words:
        return ""

    ROW_TOL = 5.0   # points — words within this vertical distance share a row
    COL_GAP = 8.0   # points — x-gap larger than this → new cell

    # Group words into rows keyed by first-seen y_mid (stable anchor)
    rows: list[tuple[float, list[tuple[float, float, str]]]] = []  # (anchor_y, [(x0, x1, text)])

    for x0, y0, x1, y1, text in sorted(words, key=lambda w: (w[1], w[0])):
        y_mid = (y0 + y1) / 2
        placed = False
        for i, (anchor_y, row_words) in enumerate(rows):
            if abs(anchor_y - y_mid) <= ROW_TOL:
                row_words.append((x0, x1, text))
                placed = True
                break
        if not placed:
            rows.append((y_mid, [(x0, x1, text)]))

    lines: list[str] = []
    for anchor_y, row_words in sorted(rows, key=lambda r: r[0]):
        sorted_words = sorted(row_words, key=lambda w: w[0])
        if not sorted_words:
            continue
        # Merge words into cells by detecting x-gaps larger than COL_GAP
        cells: list[list[str]] = [[sorted_words[0][2]]]
        prev_x1 = sorted_words[0][1]
        for x0, x1, text in sorted_words[1:]:
            if x0 - prev_x1 >= COL_GAP:
                cells.append([text])
            else:
                cells[-1].append(text)
            prev_x1 = x1
        lines.append(" | ".join(" ".join(cell) for cell in cells))

    return "\n".join(lines)


def _table_to_text(page, table) -> str:
    try:
        data = table.extract()
    except Exception:
        data = []

    lines: list[str] = []
    for row in data or []:
        cells = []
        for cell in row:
            text = normalize_text("" if cell is None else str(cell))
            if text:
                cells.append(text)
        if cells:
            lines.append(" | ".join(cells))
    fitz_text = normalize_text("\n".join(lines))
    bbox = _bbox_tuple(getattr(table, "bbox", None))
    if bbox is None:
        return fitz_text

    # If table.extract() captured only the header row (≤1 non-empty row), the table detector
    # found the bbox correctly but failed to parse data rows. Fall back to word-level extraction
    # within the exact bbox, which preserves all visible text including numeric data rows.
    non_empty_rows = sum(
        1 for row in (data or [])
        if any(normalize_text("" if cell is None else str(cell)) for cell in row)
    )
    if non_empty_rows <= 1:
        word_text = _words_to_rows(page, bbox)
        if word_text and len(word_text) > len(fitz_text):
            fitz_text = word_text

    structured_text, _ = reconstruct_table_text(page, bbox, fitz_text=fitz_text)
    return normalize_text(structured_text)


def _extract_text_in_bbox(page, bbox: tuple[float, float, float, float]) -> str:
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        rect = fitz.Rect(*bbox)
        return normalize_text(page.get_text("text", clip=rect))
    except Exception:
        return ""


def _ocr_page(page, page_index: int) -> list[DocumentBlock]:
    if not _ocr_available():
        return []

    try:
        import fitz  # type: ignore
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(image, lang="kor+eng")
        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        order = 0
        for line in [part.strip() for part in text.splitlines() if part.strip()]:
            cleaned = normalize_text(line)
            if not cleaned:
                continue
            kind = "heading" if looks_like_heading(cleaned) else "table" if looks_like_table(cleaned) else "paragraph"
            if kind == "heading":
                current_section = cleaned
            blocks.append(
                DocumentBlock(
                    page_num=page_index,
                    order=order,
                    kind=kind,
                    text=cleaned,
                    section=current_section,
                    source="ocr",
                )
            )
            order += 1
        return blocks
    except Exception:
        return []


def _extract_with_pymupdf(pdf_path: str | Path) -> list[DocumentBlock]:
    import fitz  # type: ignore

    doc = fitz.open(str(pdf_path))
    blocks: list[DocumentBlock] = []
    current_section: str | None = None

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")
        layout_regions = analyze_page_layout(page, page_index + 1)
        page_tables = _find_tables(page)
        table_bboxes: list[tuple[float, float, float, float]] = []
        table_items: list[tuple[tuple[float, float, int], DocumentBlock]] = []
        text_items: list[tuple[tuple[float, float, int], DocumentBlock]] = []
        page_text_length = 0
        layout_heading_regions = [region for region in layout_regions if region.kind == "heading"]
        layout_table_regions = [region for region in layout_regions if region.kind == "table"]

        layout_bboxes: list[tuple[float, float, float, float]] = [region.bbox for region in layout_regions]

        for table_index, table in page_tables:
            bbox = _bbox_tuple(getattr(table, "bbox", None))
            if bbox is None:
                continue
            table_bboxes.append(bbox)
            text = _table_to_text(page, table)
            if not text:
                continue
            item = DocumentBlock(
                page_num=page_index + 1,
                order=table_index,
                kind="table",
                text=text,
                section=None,
                source="table",
                table_index=table_index,
            )
            table_items.append(((bbox[1], bbox[0], 1000 + table_index), item))

        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                continue

            bbox = _bbox_tuple(raw_block.get("bbox"))
            if bbox and any(_bbox_intersects(bbox, table_bbox) for table_bbox in table_bboxes):
                continue

            line_texts: list[str] = []
            max_font_size: float | None = None
            bold = False
            for line in raw_block.get("lines", []):
                span_texts: list[str] = []
                for span in line.get("spans", []):
                    span_text = normalize_text(span.get("text", ""))
                    if not span_text:
                        continue
                    span_texts.append(span_text)
                    size = span.get("size")
                    if isinstance(size, (int, float)):
                        max_font_size = size if max_font_size is None else max(max_font_size, float(size))
                    font_name = str(span.get("font", "")).lower()
                    bold = bold or "bold" in font_name
                if span_texts:
                    line_texts.append(" ".join(span_texts))

            text = normalize_text("\n".join(line_texts))
            if not text:
                continue

            page_text_length += len(text)
            kind = "paragraph"
            heading_region = _best_layout_region(bbox, layout_heading_regions, kind="heading", min_ratio=0.35)
            table_region = _best_layout_region(bbox, layout_table_regions, kind="table", min_ratio=0.4)
            if table_region is not None and not looks_like_heading(text, font_size=max_font_size, bold=bold):
                continue
            if heading_region is not None:
                kind = "heading"
            elif looks_like_heading(text, font_size=max_font_size, bold=bold):
                kind = "heading"
            elif looks_like_table(text):
                kind = "table"

            source = "pymupdf"
            if heading_region is not None:
                source = f"layout_hint:{getattr(heading_region, 'source', 'layout')}"

            item = DocumentBlock(
                page_num=page_index + 1,
                order=len(text_items),
                kind=kind,
                text=text,
                section=None,
                font_size=max_font_size,
                bold=bold,
                source=source,
            )
            sort_key = ((bbox[1], bbox[0], len(text_items)) if bbox else (0.0, 0.0, len(text_items)))
            text_items.append((sort_key, item))

        if not text_items and layout_regions:
            for region in layout_regions:
                text = _extract_text_in_bbox(page, region.bbox)
                if not text:
                    continue
                item = DocumentBlock(
                    page_num=page_index + 1,
                    order=region.order,
                    kind=region.kind,
                    text=text,
                    section=None,
                    source=region.source,
                )
                text_items.append(((region.bbox[1], region.bbox[0], region.order), item))

        page_items = [item for _, item in sorted(text_items + table_items, key=lambda pair: pair[0])]

        if not page_items or (page_text_length < 80 and not table_items):
            ocr_blocks = _ocr_page(page, page_index + 1)
            if ocr_blocks:
                if current_section and ocr_blocks and ocr_blocks[0].section is None:
                    ocr_blocks = [
                        DocumentBlock(
                            page_num=block.page_num,
                            order=block.order,
                            kind=block.kind,
                            text=block.text,
                            section=current_section if block.kind != "heading" else block.text,
                            font_size=block.font_size,
                            bold=block.bold,
                            source=block.source,
                            table_index=block.table_index,
                        )
                        for block in ocr_blocks
                    ]
                page_items = ocr_blocks

        order = 0
        for block in page_items:
            if block.kind == "heading":
                current_section = block.text
                block = DocumentBlock(
                    page_num=block.page_num,
                    order=order,
                    kind=block.kind,
                    text=block.text,
                    section=current_section,
                    font_size=block.font_size,
                    bold=block.bold,
                    source=block.source,
                    table_index=block.table_index,
                )
            else:
                block = DocumentBlock(
                    page_num=block.page_num,
                    order=order,
                    kind=block.kind,
                    text=block.text,
                    section=block.section or current_section,
                    font_size=block.font_size,
                    bold=block.bold,
                    source=block.source,
                    table_index=block.table_index,
                )
            blocks.append(block)
            order += 1

    return blocks


def _extract_fallback(pdf_path: str | Path) -> list[DocumentBlock]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    blocks: list[DocumentBlock] = []
    current_section: str | None = None

    for page_index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        order = 0
        for line in [part.strip() for part in raw.splitlines() if part.strip()]:
            text = normalize_text(line)
            if not text:
                continue
            kind = "heading" if looks_like_heading(text) else "table" if looks_like_table(text) else "paragraph"
            if kind == "heading":
                current_section = text
            blocks.append(
                DocumentBlock(
                    page_num=page_index,
                    order=order,
                    kind=kind,
                    text=text,
                    section=current_section,
                    source="pypdf",
                )
            )
            order += 1
    return blocks


def extract_document_blocks(pdf_path: str | Path) -> list[DocumentBlock]:
    try:
        return _apply_outline_structure(_extract_with_pymupdf(pdf_path))
    except Exception:
        return _apply_outline_structure(_extract_fallback(pdf_path))


def summarize_blocks(blocks: Iterable[DocumentBlock], page_count: int) -> DocumentSummary:
    block_list = list(blocks)
    sections: list[str] = []
    outlines: list[tuple[str, ...]] = []
    ocr_pages: set[int] = set()
    sources: set[str] = set()
    for block in block_list:
        sources.add(block.source)
        if block.source == "ocr":
            ocr_pages.add(block.page_num)
        if block.kind == "heading" and block.text not in sections:
            sections.append(block.text)
        if block.outline_path and block.outline_path not in outlines:
            outlines.append(block.outline_path)

    heading_text = " ".join(section.lower() for section in sections)
    body_text = " ".join(block.text.lower() for block in block_list[:120] if block.kind != "table")

    academic_markers = {
        "abstract",
        "introduction",
        "method",
        "methods",
        "results",
        "discussion",
        "literature review",
        "methodology",
        "conclusion",
        "서론",
        "초록",
        "방법",
        "결과",
        "고찰",
        "참고문헌",
    }
    academic_core_markers = {
        "abstract",
        "초록",
        "this paper",
        "we propose",
        "our study",
        "본 연구",
        "실험",
    }
    esg_markers = {
        "esg",
        "sustainability",
        "sustainable",
        "environment",
        "social",
        "governance",
        "scope 1",
        "scope 2",
        "scope 3",
        "greenhouse gas",
        "carbon",
        "emission",
        "energy",
        "water",
        "waste",
        "human rights",
        "ethics",
        "supply chain",
        "safety",
        "diversity",
        "지속가능",
        "환경",
        "사회",
        "지배구조",
        "온실가스",
        "배출",
        "에너지",
        "용수",
        "폐기물",
        "안전",
        "윤리",
    }
    report_markers = {
        "sustainability report",
        "esg report",
        "annual report",
        "integrated report",
        "gri index",
        "gri content index",
        "sasb index",
        "tcfd index",
        "지속가능경영보고서",
        "지속가능성보고서",
        "통합보고서",
        "esg 데이터",
    }

    academic_score = sum(1.8 for marker in academic_markers if marker in heading_text)
    academic_core_score = sum(2.0 for marker in academic_core_markers if marker in heading_text)
    academic_core_score += sum(1.2 for marker in academic_core_markers if marker in body_text)
    esg_score = sum(1.8 for marker in esg_markers if marker in heading_text)
    esg_score += sum(0.3 for marker in esg_markers if marker in body_text)
    esg_score += sum(2.4 for marker in report_markers if marker in heading_text)
    esg_score += sum(0.9 for marker in report_markers if marker in body_text)
    esg_score += sum(0.02 for block in block_list if block.kind == "table")
    document_kind = (
        "academic_paper"
        if academic_score >= 4.0 and academic_core_score >= 1.2 and academic_score > esg_score + 1.2
        else "esg_report"
    )
    extraction_mode = "ocr" if "ocr" in sources and len(sources) == 1 else "mixed" if len(sources) > 1 else "text"
    return DocumentSummary(
        page_count=page_count,
        block_count=len(block_list),
        heading_count=sum(1 for block in block_list if block.kind == "heading"),
        table_count=sum(1 for block in block_list if block.kind == "table"),
        section_count=len(sections),
        sections=tuple(sections[:20]),
        outline_count=len(outlines),
        document_kind=document_kind,
        ocr_page_count=len(ocr_pages),
        extraction_mode=extraction_mode,
    )
