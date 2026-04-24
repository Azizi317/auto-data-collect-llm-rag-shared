from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .document import DocumentBlock, extract_document_blocks, normalize_text


@dataclass(frozen=True)
class DocumentChunk:
    page_start: int
    page_end: int
    section: str | None
    kind: str
    text: str
    section_level: int | None = None
    outline_path: tuple[str, ...] = ()
    source_blocks: int = 0


def _split_at_sentence_boundary(text: str, max_chars: int) -> tuple[str, str] | None:
    if len(text) <= max_chars:
        return None
    window_start = max(0, max_chars - 400)
    window = text[window_start : max_chars + 1]
    last_boundary = max(window.rfind(token) for token in (". ", "! ", "? ", ".", "!", "?", "\n"))
    if last_boundary < 0:
        return None
    split_at = window_start + last_boundary + 1
    head = normalize_text(text[:split_at])
    tail = normalize_text(text[split_at:])
    if not head or not tail:
        return None
    return head, tail


def _append_chunk(
    chunks: list[DocumentChunk],
    buffer_pages: list[int],
    buffer_section: str | None,
    buffer_section_level: int | None,
    buffer_outline_path: tuple[str, ...],
    buffer_kind: str,
    source_blocks: int,
    text: str,
) -> None:
    chunks.append(
        DocumentChunk(
            page_start=min(buffer_pages),
            page_end=max(buffer_pages),
            section=buffer_section,
            section_level=buffer_section_level,
            outline_path=buffer_outline_path,
            kind=buffer_kind,
            text=text,
            source_blocks=source_blocks,
        )
    )


def _flush_chunk(
    chunks: list[DocumentChunk],
    buffer: list[str],
    buffer_pages: list[int],
    buffer_section: str | None,
    buffer_section_level: int | None,
    buffer_outline_path: tuple[str, ...],
    buffer_kind: str,
    source_blocks: int,
    max_chars: int,
) -> None:
    text = normalize_text(" ".join(buffer))
    if not text:
        return
    split = _split_at_sentence_boundary(text, max_chars=max_chars)
    if split is None:
        _append_chunk(
            chunks,
            buffer_pages,
            buffer_section,
            buffer_section_level,
            buffer_outline_path,
            buffer_kind,
            source_blocks,
            text,
        )
        return

    head, tail = split
    _append_chunk(
        chunks,
        buffer_pages,
        buffer_section,
        buffer_section_level,
        buffer_outline_path,
        buffer_kind,
        source_blocks,
        head,
    )
    _append_chunk(
        chunks,
        buffer_pages,
        buffer_section,
        buffer_section_level,
        buffer_outline_path,
        buffer_kind,
        source_blocks,
        tail,
    )


def build_chunks(
    blocks: Iterable[DocumentBlock],
    max_chars: int = 1800,
    min_chars: int = 260,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []
    buffer_section: str | None = None
    buffer_section_level: int | None = None
    buffer_outline_path: tuple[str, ...] = ()
    buffer_kind = "mixed"
    source_blocks = 0

    def flush() -> None:
        nonlocal buffer, buffer_pages, buffer_section, buffer_section_level, buffer_outline_path, buffer_kind, source_blocks
        if buffer:
            _flush_chunk(
                chunks,
                buffer,
                buffer_pages,
                buffer_section,
                buffer_section_level,
                buffer_outline_path,
                buffer_kind,
                source_blocks,
                max_chars,
            )
        buffer = []
        buffer_pages = []
        buffer_section = None
        buffer_section_level = None
        buffer_outline_path = ()
        buffer_kind = "mixed"
        source_blocks = 0

    for block in blocks:
        if block.kind == "heading":
            flush()
            buffer_section = block.text
            buffer_section_level = block.section_level
            buffer_outline_path = block.outline_path
            continue

        if block.kind == "table" and buffer and buffer_kind == "table":
            flush()

        if buffer_section is None and block.section:
            buffer_section = block.section
        if buffer_section_level is None:
            buffer_section_level = block.section_level
        if not buffer_outline_path and block.outline_path:
            buffer_outline_path = block.outline_path
        if buffer_kind == "mixed":
            buffer_kind = block.kind
        elif buffer_kind != block.kind:
            buffer_kind = "mixed"

        should_flush = False
        current_length = len(" ".join(buffer))
        if buffer and block.kind == "table" and buffer_kind != "table":
            should_flush = True
        elif buffer and block.section and buffer_section and block.section != buffer_section:
            should_flush = True
        elif buffer and current_length + len(block.text) + 1 > max_chars:
            should_flush = True

        if should_flush:
            flush()
            buffer_section = block.section or buffer_section
            buffer_section_level = block.section_level
            buffer_outline_path = block.outline_path or buffer_outline_path
            buffer_kind = block.kind

        buffer.append(block.text)
        buffer_pages.append(block.page_num)
        source_blocks += 1

        if len(" ".join(buffer)) >= max_chars and len(" ".join(buffer)) >= min_chars:
            flush()

    if buffer:
        flush()

    return chunks


def load_pdf_chunks(pdf_path: str | Path, max_chars: int = 1800) -> list[DocumentChunk]:
    blocks = extract_document_blocks(pdf_path)
    return build_chunks(blocks, max_chars=max_chars)
