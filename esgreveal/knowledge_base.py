from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import re

import numpy as np

from .chunking import build_chunks
from .document import DocumentBlock, extract_document_blocks, normalize_text
from .metadata import KPIItem, default_metadata


DEFAULT_SUMMARY_MODEL = os.getenv("ESGREVEAL_SUMMARY_MODEL", "lcw99/t5-base-korean-text-summary")
DEFAULT_EMBEDDING_MODEL = os.getenv("ESGREVEAL_EMBEDDING_MODEL", "dragonkue/BGE-m3-ko")

_TABLE_HEADER_TOKENS = {
    "구분",
    "항목",
    "지표",
    "내용",
    "단위",
    "비고",
    "metric",
    "metrics",
    "item",
    "items",
    "category",
    "unit",
    "value",
    "values",
    "note",
    "notes",
}

_TABLE_NOISE_TERMS = {
    "합계",
    "총계",
    "subtotal",
    "total",
    "remark",
    "remarks",
    "note",
    "notes",
    "reference",
    "references",
}

_STRATEGY_TERMS = {
    "목표",
    "계획",
    "전략",
    "비전",
    "로드맵",
    "이행",
    "추진",
    "commitment",
    "target",
    "targets",
    "plan",
    "plans",
    "strategy",
    "strategies",
    "vision",
    "roadmap",
    "initiative",
    "initiatives",
    "re100",
    "net zero",
    "net-zero",
    "carbon neutral",
    "carbon neutrality",
}


@dataclass(frozen=True)
class KnowledgeBaseRecord:
    record_id: str
    component: str
    source_path: str
    page_start: int
    page_end: int
    section: str | None
    section_level: int | None
    outline_path: tuple[str, ...]
    text: str
    summary: str | None = None
    vector_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeBaseBuild:
    input_path: str
    output_dir: str
    source_count: int
    record_count: int
    component_counts: dict[str, int]
    summary_model: str
    embedding_model: str
    faiss_index_path: str | None
    milvus_uri: str | None
    milvus_collection: str | None
    records_path: str
    textual_records_path: str
    outline_records_path: str
    table_records_path: str
    checkpoint_dir: str | None = None
    resume_checkpoints: bool = True
    resumed_source_count: int = 0
    pending_source_count: int = 0
    summary_mode: str = "mt5"


def _iter_pdf_paths(input_path: str | Path | list[str] | list[Path] | tuple[str | Path, ...], recursive: bool) -> list[Path]:
    if isinstance(input_path, (list, tuple)):
        return sorted(Path(path) for path in input_path if Path(path).is_file())
    source = Path(input_path)
    if source.is_file():
        return [source]
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in source.glob(pattern) if path.is_file())


def _hash_record_id(*parts: object) -> str:
    joined = "||".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _sanitize_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    if not cleaned:
        return "esgreveal_knowledge_base"
    if cleaned[0].isdigit():
        cleaned = f"knowledge_base_{cleaned}"
    return cleaned[:200]


def _pdf_signature(pdf_path: Path) -> dict[str, Any]:
    stat = pdf_path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _checkpoint_root(output_dir: Path) -> Path:
    return output_dir / "_checkpoints"


def _checkpoint_name(pdf_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("._-") or "source"
    stable = _hash_record_id(str(pdf_path.resolve()))[:12]
    return f"{safe_stem}-{stable}.json"


def _checkpoint_path(output_dir: Path, pdf_path: Path) -> Path:
    return _checkpoint_root(output_dir) / _checkpoint_name(pdf_path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _record_from_payload(payload: dict[str, Any]) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(
        record_id=str(payload.get("record_id", "")),
        component=str(payload.get("component", "")),
        source_path=str(payload.get("source_path", "")),
        page_start=int(payload.get("page_start", 0)),
        page_end=int(payload.get("page_end", 0)),
        section=payload.get("section"),
        section_level=payload.get("section_level"),
        outline_path=tuple(payload.get("outline_path") or ()),
        text=str(payload.get("text", "")),
        summary=payload.get("summary"),
        vector_text=payload.get("vector_text"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _load_pdf_checkpoint(
    checkpoint_path: Path,
    expected_source_path: str,
    expected_signature: dict[str, Any],
) -> tuple[list[KnowledgeBaseRecord], Counter[str]] | None:
    if not checkpoint_path.exists():
        return None
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("source_path") != expected_source_path:
        return None
    if payload.get("signature") != expected_signature:
        return None
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return None
    try:
        records = [_record_from_payload(item) for item in raw_records if isinstance(item, dict)]
    except Exception:
        return None
    counts = Counter(payload.get("component_counts") or {})
    return records, counts


def _save_pdf_checkpoint(
    checkpoint_path: Path,
    pdf_path: Path,
    signature: dict[str, Any],
    records: list[KnowledgeBaseRecord],
    counts: Counter[str],
    stage: str,
) -> None:
    payload = {
        "source_path": str(pdf_path),
        "signature": signature,
        "stage": stage,
        "record_count": len(records),
        "component_counts": dict(counts),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "records": [asdict(record) for record in records],
    }
    _atomic_write_text(checkpoint_path, json.dumps(payload, ensure_ascii=False, indent=2))


def _extractive_summary(text: str, limit: int = 280) -> str:
    candidate = normalize_text(text)
    if len(candidate) <= limit:
        return candidate
    sentences = re.split(r"(?<=[.!?。！？])\s+", candidate)
    summary = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(summary) + len(sentence) + 1 > limit and summary:
            break
        summary = f"{summary} {sentence}".strip()
        if len(summary) >= limit * 0.8:
            break
    if summary:
        return summary[:limit].strip()
    return candidate[:limit].strip()


def _truncate(text: str, limit: int = 1800) -> str:
    candidate = normalize_text(text)
    if len(candidate) <= limit:
        return candidate
    return candidate[: limit - 3].rstrip() + "..."


def _normalize_keyword(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip(" |:-")


def _normalize_match(text: str) -> str:
    return re.sub(r"\s+", " ", _normalize_keyword(text).lower())


def _is_noise_keyword(text: str) -> bool:
    candidate = _normalize_keyword(text)
    if not candidate:
        return True
    low = candidate.lower()
    if low in _TABLE_HEADER_TOKENS:
        return True
    if low in _TABLE_NOISE_TERMS:
        return True
    if re.fullmatch(r"(?:19|20)\d{2}", candidate):
        return True
    if re.fullmatch(r"[\d,.()%/\-]+", candidate):
        return True
    if len(candidate) <= 1:
        return True
    return False


def _match_tokens(text: str) -> set[str]:
    normalized = _normalize_match(text)
    if not normalized:
        return set()
    pieces = re.split(r"[^0-9a-z가-힣]+", normalized)
    return {piece for piece in pieces if len(piece) >= 2}


def _is_strategy_like(text: str) -> bool:
    low = _normalize_match(text)
    if not low:
        return False
    return any(term in low for term in _STRATEGY_TERMS)


def _should_exclude_table_keyword(keyword: str, row_label: str, row_text: str) -> bool:
    joined = " ".join(part for part in (keyword, row_label, row_text) if part)
    if _is_strategy_like(joined):
        return True
    return False


def _build_kpi_term_index(kpis: list[KPIItem]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for kpi in kpis:
        terms = []
        term_tokens: set[str] = set()
        for term in kpi.all_terms():
            normalized = _normalize_match(term)
            if not normalized:
                continue
            terms.append(normalized)
            term_tokens.update(_match_tokens(normalized))
        index.append(
            {
                "id": kpi.id,
                "category": kpi.category,
                "aspect_ko": _normalize_match(kpi.aspect_ko),
                "aspect_en": _normalize_match(kpi.aspect_en),
                "kpi_ko": _normalize_match(kpi.kpi_ko),
                "kpi_en": _normalize_match(kpi.kpi_en),
                "terms": tuple(dict.fromkeys(terms)),
                "tokens": term_tokens,
            }
        )
    return index


def _rank_keyword_against_kpis(
    keyword: str,
    row_label: str,
    row_text: str,
    section: str | None,
    outline_path: tuple[str, ...],
    kpi_index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keyword_low = _normalize_match(keyword)
    row_label_low = _normalize_match(row_label)
    row_text_low = _normalize_match(row_text)
    context_low = " ".join(part for part in (_normalize_match(section or ""), _normalize_match(" ".join(outline_path))) if part)
    query_tokens = _match_tokens(" ".join(part for part in (keyword, row_label, row_text, section or "", " ".join(outline_path)) if part))

    ranked: list[dict[str, Any]] = []
    for item in kpi_index:
        score = 0.0
        matched_terms: list[str] = []

        for term in item["terms"]:
            if not term:
                continue
            if term == keyword_low:
                score += 3.0
                matched_terms.append(term)
            elif term and term in row_label_low:
                score += 2.2
                matched_terms.append(term)
            elif term and term in row_text_low:
                score += 1.4
                matched_terms.append(term)
            elif context_low and term in context_low:
                score += 0.7
                matched_terms.append(term)

        token_overlap = len(query_tokens & item["tokens"])
        score += min(token_overlap * 0.25, 1.5)

        if item["aspect_ko"] and item["aspect_ko"] in context_low:
            score += 0.35
        if item["aspect_en"] and item["aspect_en"] in context_low:
            score += 0.35

        if score <= 0:
            continue

        ranked.append(
            {
                "id": item["id"],
                "score": round(score, 3),
                "matched_terms": tuple(dict.fromkeys(matched_terms))[:8],
            }
        )

    ranked.sort(key=lambda value: (-value["score"], value["id"]))
    return ranked[:5]


def _keyword_priority(
    keyword: str,
    row_label: str,
    row_text: str,
    section: str | None,
    outline_path: tuple[str, ...],
    ontology_matches: list[dict[str, Any]],
) -> float:
    score = 0.0
    keyword_low = _normalize_match(keyword)
    row_label_low = _normalize_match(row_label)
    row_text_low = _normalize_match(row_text)
    context_low = _normalize_match(" ".join(outline_path) + " " + (section or ""))

    if keyword_low == row_label_low and keyword_low:
        score += 1.8
    if keyword_low and keyword_low in row_label_low:
        score += 1.2
    if keyword_low and keyword_low in row_text_low:
        score += 0.8
    if keyword_low and keyword_low in context_low:
        score += 0.4
    if re.search(r"[a-z가-힣]", keyword_low):
        score += 0.2
    if ontology_matches:
        score += min(ontology_matches[0]["score"], 4.0)
    return round(score, 3)


def _split_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in str(text or "").splitlines():
        line = _normalize_keyword(raw_line)
        if not line:
            continue
        cells = [_normalize_keyword(part) for part in line.split("|")] if "|" in line else [line]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    return rows


def _looks_like_header_row(cells: list[str]) -> bool:
    if not cells:
        return False
    header_hits = 0
    for cell in cells:
        low = _normalize_keyword(cell).lower()
        if low in _TABLE_HEADER_TOKENS:
            header_hits += 1
        elif re.fullmatch(r"(?:19|20)\d{2}", cell):
            header_hits += 1
    return header_hits >= max(1, len(cells) // 2)


def _table_row_keywords(row: list[str], header_rows: list[list[str]]) -> list[str]:
    if not row:
        return []
    row_label = _normalize_keyword(row[0])
    keywords: list[str] = []
    if not _is_noise_keyword(row_label):
        keywords.append(row_label)

    for index, _ in enumerate(row[1:], start=1):
        header_value = ""
        for header_row in reversed(header_rows):
            if index < len(header_row):
                header_value = _normalize_keyword(header_row[index])
                if header_value:
                    break
        if _is_noise_keyword(header_value):
            continue
        if not _is_noise_keyword(row_label):
            keywords.append(_normalize_keyword(f"{row_label} {header_value}"))
        else:
            keywords.append(header_value)

    ordered: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = keyword.lower()
        if normalized in seen or _is_noise_keyword(keyword):
            continue
        seen.add(normalized)
        ordered.append(keyword)
    return ordered[:8]


class T5Summarizer:
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 4,
        max_new_tokens: int = 96,
        max_input_chars: int = 1800,
    ) -> None:
        # model_name 이 None 이거나 빈 문자열이면 추출적 요약만 사용
        self.model_name = model_name or None
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_input_chars = max_input_chars
        self._pipeline = None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "text2text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            device=device,
        )
        return self._pipeline

    def summarize_many(self, texts: list[str], progress_callback: Any = None) -> list[str]:
        if not texts:
            return []

        # 모델이 지정되지 않으면 추출적 요약만 사용 (빠르고 한국어 품질 안정적)
        if self.model_name is None:
            if progress_callback is not None:
                progress_callback("추출적 요약 사용", len(texts), len(texts))
            return [_extractive_summary(text) for text in texts]

        prompts = [f"summarize: {_truncate(text, self.max_input_chars)}" for text in texts]
        try:
            if progress_callback is not None:
                progress_callback(f"요약 모델 로딩 중 ({self.model_name})", 0, len(texts))
            generator = self._load()
            summaries: list[str] = []
            for index in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[index : index + self.batch_size]
                batch_texts = texts[index : index + self.batch_size]
                if progress_callback is not None:
                    progress_callback("mt5 요약 생성 중", index, len(texts))
                results = generator(
                    batch_prompts,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    truncation=True,
                )
                for raw, original in zip(results, batch_texts, strict=False):
                    summary = normalize_text(str(raw.get("generated_text", "")))
                    summaries.append(summary or _extractive_summary(original))
                if progress_callback is not None:
                    progress_callback("mt5 요약 생성 중", min(index + len(batch_texts), len(texts)), len(texts))
            return summaries
        except Exception:
            if progress_callback is not None:
                progress_callback("요약 모델 실패, 추출적 요약으로 대체", len(texts), len(texts))
            return [_extractive_summary(text) for text in texts]


class M3EEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ONNX 백엔드: CPU 추론 2~3배 가속. GPU 사용 시에는 기본 백엔드 유지.
        backend = "onnx" if device == "cpu" else "torch"
        try:
            self._model = SentenceTransformer(self.model_name, device=device, backend=backend)
        except Exception:
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def encode(self, texts: list[str], progress_callback: Any = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        if progress_callback is None:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        # 배치 단위로 직접 인코딩해서 진행률 보고
        results: list[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_vectors = model.encode(
                batch,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            results.append(np.asarray(batch_vectors, dtype=np.float32))
            progress_callback(start + len(batch), total)
        return np.concatenate(results, axis=0)


def _make_textual_records(pdf_path: Path, chunks: list[Any]) -> list[KnowledgeBaseRecord]:
    records: list[KnowledgeBaseRecord] = []
    for index, chunk in enumerate(chunks, start=1):
        text = normalize_text(chunk.text)
        if not text or chunk.kind == "table":
            continue
        record_id = _hash_record_id(pdf_path, "textual_contents", chunk.page_start, chunk.page_end, index, text[:120])
        records.append(
            KnowledgeBaseRecord(
                record_id=record_id,
                component="textual_contents",
                source_path=str(pdf_path),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                section_level=chunk.section_level,
                outline_path=chunk.outline_path,
                text=text,
                metadata={
                    "kind": chunk.kind,
                    "source_blocks": chunk.source_blocks,
                },
            )
        )
    return records


def _make_outline_records(pdf_path: Path, blocks: list[DocumentBlock]) -> list[KnowledgeBaseRecord]:
    records: list[KnowledgeBaseRecord] = []
    seen: set[tuple[int, tuple[str, ...], str]] = set()
    for index, block in enumerate(blocks, start=1):
        if block.kind != "heading":
            continue
        outline_path = block.outline_path or (block.text,)
        key = (block.page_num, outline_path, block.text)
        if key in seen:
            continue
        seen.add(key)
        outline_text = " > ".join(outline_path) if outline_path else block.text
        record_id = _hash_record_id(pdf_path, "document_outlines", block.page_num, index, outline_text[:120])
        records.append(
            KnowledgeBaseRecord(
                record_id=record_id,
                component="document_outlines",
                source_path=str(pdf_path),
                page_start=block.page_num,
                page_end=block.page_num,
                section=block.text,
                section_level=block.section_level,
                outline_path=outline_path,
                text=normalize_text(outline_text),
                metadata={
                    "heading": block.text,
                    "font_size": block.font_size,
                    "bold": block.bold,
                    "source": block.source,
                },
            )
        )
    return records


def _make_table_base_record(pdf_path: Path, block: DocumentBlock, table_number: int) -> KnowledgeBaseRecord:
    text = normalize_text(block.text)
    record_id = _hash_record_id(pdf_path, "table_contents", "table", block.page_num, table_number, text[:120])
    return KnowledgeBaseRecord(
        record_id=record_id,
        component="table_contents",
        source_path=str(pdf_path),
        page_start=block.page_num,
        page_end=block.page_num,
        section=block.section,
        section_level=block.section_level,
        outline_path=block.outline_path,
        text=text,
        metadata={
            "table_index": block.table_index,
            "source": block.source,
            "mapping_type": "table_base",
        },
    )


def _make_table_records(
    pdf_path: Path,
    blocks: list[DocumentBlock],
    kpi_index: list[dict[str, Any]],
) -> list[KnowledgeBaseRecord]:
    records: list[KnowledgeBaseRecord] = []
    for index, block in enumerate((item for item in blocks if item.kind == "table"), start=1):
        raw_text = str(block.text or "")
        text = normalize_text(raw_text)
        if not text:
            continue
        base_record = _make_table_base_record(pdf_path, block, index)
        records.append(base_record)
        child_ids: list[str] = []

        rows = _split_table_rows(raw_text)
        if len(rows) < 2:
            continue

        header_rows = [row for row in rows[:2] if _looks_like_header_row(row)]
        for row_index, row in enumerate(rows, start=1):
            if _looks_like_header_row(row):
                continue
            row_text = " | ".join(row)
            row_label = _normalize_keyword(row[0]) if row else ""
            keywords = _table_row_keywords(row, header_rows)
            if not keywords:
                continue

            ranked_keywords: list[tuple[str, list[dict[str, Any]], float]] = []
            for keyword in keywords:
                if _should_exclude_table_keyword(keyword, row_label, row_text):
                    continue
                ontology_matches = _rank_keyword_against_kpis(
                    keyword=keyword,
                    row_label=row_label,
                    row_text=row_text,
                    section=block.section,
                    outline_path=block.outline_path,
                    kpi_index=kpi_index,
                )
                if not ontology_matches:
                    continue
                priority = _keyword_priority(
                    keyword=keyword,
                    row_label=row_label,
                    row_text=row_text,
                    section=block.section,
                    outline_path=block.outline_path,
                    ontology_matches=ontology_matches,
                )
                ranked_keywords.append((keyword, ontology_matches, priority))

            ranked_keywords.sort(key=lambda item: (-item[2], item[0]))
            ranked_keywords = ranked_keywords[:4]
            if not ranked_keywords:
                continue

            for keyword_index, (keyword, ontology_matches, priority) in enumerate(ranked_keywords, start=1):
                keyword_record_id = _hash_record_id(
                    pdf_path,
                    "table_contents",
                    "keyword",
                    base_record.record_id,
                    row_index,
                    keyword_index,
                    keyword,
                )

                # 헤더에서 단위 힌트 추출 (tCO2e, MWh, t 등)
                from .table_semantics import UNIT_TOKEN_RE as _UNIT_RE
                _header_text = " ".join(" ".join(r) for r in header_rows)
                _unit_matches = _UNIT_RE.findall(_header_text)
                # row_text에서도 단위 추출 (셀 안에 직접 있는 경우)
                _unit_matches += _UNIT_RE.findall(row_text)
                unit_hint = " ".join(dict.fromkeys(m for m in _unit_matches if m)) if _unit_matches else ""

                # row_text에서 숫자 힌트 추출 (최대 4개)
                _nums = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", row_text)
                value_hint = " ".join(_nums[:4]) if _nums else ""

                vector_text = " | ".join(
                    part
                    for part in (
                        keyword,
                        row_label if row_label and row_label.lower() != keyword.lower() else "",
                        unit_hint,
                        value_hint,
                        block.section or "",
                        " > ".join(block.outline_path),
                    )
                    if part
                )

                # LLM 가독성: 헤더 행을 text 앞에 붙여 열 구조(연도·단위) 파악 가능하게 함
                header_prefix = "\n".join(" | ".join(r) for r in header_rows) if header_rows else ""
                enriched_text = f"{header_prefix}\n{row_text}" if header_prefix else row_text
                records.append(
                    KnowledgeBaseRecord(
                        record_id=keyword_record_id,
                        component="table_contents",
                        source_path=str(pdf_path),
                        page_start=block.page_num,
                        page_end=block.page_num,
                        section=block.section,
                        section_level=block.section_level,
                        outline_path=block.outline_path,
                        text=normalize_text(enriched_text),
                        vector_text=normalize_text(vector_text),
                        metadata={
                            "table_index": block.table_index,
                            "source": block.source,
                            "mapping_type": "table_keyword",
                            "parent_table_id": base_record.record_id,
                            "row_index": row_index,
                            "row_label": row_label,
                            "keyword": keyword,
                            "keyword_rank": keyword_index,
                            "keyword_priority": priority,
                            "ontology_kpi_ids": [item["id"] for item in ontology_matches],
                            "ontology_scores": [item["score"] for item in ontology_matches],
                            "ontology_matched_terms": [list(item["matched_terms"]) for item in ontology_matches],
                        },
                    )
                )
                child_ids.append(keyword_record_id)
        if child_ids:
            records[records.index(base_record)] = replace(
                base_record,
                metadata={
                    **base_record.metadata,
                    "child_record_ids": child_ids,
                    "child_count": len(child_ids),
                },
            )
    return records


def _link_outline_records(records: list[KnowledgeBaseRecord]) -> dict[str, dict[str, Any]]:
    linked_by_outline: dict[tuple[str, tuple[str, ...]], list[KnowledgeBaseRecord]] = {}
    for record in records:
        if record.component == "document_outlines":
            continue
        for depth in range(1, len(record.outline_path) + 1):
            key = (record.source_path, record.outline_path[:depth])
            linked_by_outline.setdefault(key, []).append(record)

    linked_payloads: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.component != "document_outlines":
            continue
        linked = linked_by_outline.get((record.source_path, record.outline_path), [])
        linked_keywords: list[str] = []
        linked_sections: list[str] = []
        for item in linked:
            keyword = str(item.metadata.get("keyword", "")).strip()
            if keyword and keyword not in linked_keywords:
                linked_keywords.append(keyword)
            if item.section and item.section not in linked_sections:
                linked_sections.append(item.section)
        linked_payloads[record.record_id] = {
            "linked_record_ids": [item.record_id for item in linked[:24]],
            "linked_count": len(linked),
            "linked_keywords": linked_keywords[:12],
            "linked_sections": linked_sections[:8],
        }
    return linked_payloads


def _enrich_records_with_summaries(
    records: list[KnowledgeBaseRecord],
    summarizer: T5Summarizer,
    summary_mode: str = "mt5",
    progress_callback: Any = None,
) -> list[KnowledgeBaseRecord]:
    textual = [record for record in records if record.component == "textual_contents"]
    if textual and progress_callback is not None:
        progress_callback("요약 대상 정리 중...", 0, len(textual))
    summaries = (
        summarizer.summarize_many([record.text for record in textual], progress_callback=progress_callback)
        if textual
        else []
    )
    if textual and progress_callback is not None:
        progress_callback("요약 생성 완료", len(textual), len(textual))
    summary_by_id = {record.record_id: summary for record, summary in zip(textual, summaries, strict=False)}
    outline_links = _link_outline_records(records)

    enriched: list[KnowledgeBaseRecord] = []
    total_records = len(records)
    for index, record in enumerate(records, start=1):
        summary = summary_by_id.get(record.record_id)
        vector_text = summary or record.text
        metadata = dict(record.metadata)
        if record.component == "textual_contents" and summary_mode != "mt5":
            vector_text = record.text

        if record.component == "document_outlines":
            link_info = outline_links.get(record.record_id, {})
            metadata.update(link_info)
            anchor = " > ".join(record.outline_path) or record.text
            related = " | ".join(link_info.get("linked_keywords", [])[:8] or link_info.get("linked_sections", [])[:4])
            vector_text = normalize_text(" | ".join(part for part in (anchor, related) if part))
        elif record.component == "table_contents":
            mapping_type = metadata.get("mapping_type")
            if mapping_type == "table_base":
                # 단일 행 테이블(헤더만 파싱된 경우): section·outline을 더 강하게 반영
                _rows = [l for l in record.text.splitlines() if l.strip()]
                _is_header_only = len(_rows) <= 1
                if _is_header_only:
                    # section/outline이 KPI 이름을 담고 있으므로 반복 강조
                    section_str = record.section or ""
                    outline_str = " > ".join(record.outline_path)
                    vector_text = normalize_text(
                        " | ".join(
                            part
                            for part in (
                                section_str,
                                outline_str,
                                section_str,   # 가중치를 위해 반복
                                record.text,
                            )
                            if part
                        )
                    )
                else:
                    vector_text = normalize_text(
                        " | ".join(
                            part
                            for part in (
                                record.section or "",
                                " > ".join(record.outline_path),
                                record.text[:240],
                            )
                            if part
                        )
                    )
            elif mapping_type == "table_keyword":
                vector_text = record.vector_text or normalize_text(
                    " | ".join(
                        part
                        for part in (
                            str(metadata.get("keyword", "")),
                            str(metadata.get("row_label", "")),
                            record.section or "",
                            " > ".join(record.outline_path),
                        )
                        if part
                    )
                )

        enriched.append(replace(record, summary=summary, vector_text=normalize_text(vector_text), metadata=metadata))
        if progress_callback is not None and (index == total_records or index % 500 == 0):
            progress_callback("요약/벡터용 레코드 정리 중...", index, total_records)
    return enriched


def _records_to_jsonl(records: list[KnowledgeBaseRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_faiss_index(vectors: np.ndarray, output_dir: Path) -> str | None:
    if vectors.size == 0:
        return None
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    target = output_dir / "text_vectors.faiss"
    faiss.write_index(index, str(target))
    return str(target)


def _write_milvus(
    records: list[KnowledgeBaseRecord],
    vectors: np.ndarray,
    uri: str,
    collection_name: str | None,
) -> str | None:
    if not uri or vectors.size == 0:
        return None

    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri)
    base_name = _sanitize_collection_name(collection_name or "esgreveal_knowledge_base")
    actual_name = base_name
    if client.has_collection(actual_name):
        actual_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    client.create_collection(
        collection_name=actual_name,
        dimension=int(vectors.shape[1]),
        primary_field_name="id",
        id_type="string",
        vector_field_name="vector",
        metric_type="COSINE",
        auto_id=False,
    )

    payload: list[dict[str, Any]] = []
    for record, vector in zip(records, vectors, strict=False):
        payload.append({"id": record.record_id, "vector": vector.tolist()})
    client.insert(collection_name=actual_name, data=payload)
    client.close()
    return actual_name


def build_knowledge_base(
    input_path: str | Path,
    output_dir: str | Path,
    recursive: bool = False,
    summary_model: str | None = DEFAULT_SUMMARY_MODEL,
    embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
    summary_mode: str = "mt5",
    use_faiss: bool = True,
    milvus_uri: str | None = None,
    milvus_collection: str | None = None,
    resume_checkpoints: bool = True,
    worker: Any = None,
) -> KnowledgeBaseBuild:
    def _report(value: int, total: int, message: str) -> None:
        if worker is not None:
            worker.progress(value, total, message)

    summary_mode = (summary_mode or "mt5").strip().lower()
    summary_model = summary_model or DEFAULT_SUMMARY_MODEL or None
    if summary_mode in {"extractive", "off"}:
        summary_model = None
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    milvus_uri = milvus_uri or os.getenv("ESGREVEAL_MILVUS_URI")
    pdf_paths = _iter_pdf_paths(input_path, recursive=recursive)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _checkpoint_root(output_path)
    kpi_index = _build_kpi_term_index(default_metadata())

    n_pdfs = max(len(pdf_paths), 1)
    # 단계별 진행 범위: PDF 파싱 0-30, 요약 30-60, 임베딩 60-90, 저장 90-100
    TOTAL = 100

    def _parse_pdf(
        pdf_path: Path,
        signature: dict[str, Any],
        checkpoint_path: Path,
    ) -> tuple[Path, list[KnowledgeBaseRecord], Counter[str]]:
        import time as _time
        _t0 = _time.monotonic()
        _report(None, TOTAL, f"  [{pdf_path.name}] 블록 추출 시작...")
        blocks = extract_document_blocks(pdf_path)
        _report(None, TOTAL, f"  [{pdf_path.name}] 블록 {len(blocks)}개 추출 완료 ({_time.monotonic()-_t0:.1f}s)")
        chunks = build_chunks(blocks)
        t_records = _make_textual_records(pdf_path, chunks)
        o_records = _make_outline_records(pdf_path, blocks)
        tbl_records = _make_table_records(pdf_path, blocks, kpi_index)
        counts: Counter[str] = Counter({
            "textual_contents": len(t_records),
            "document_outlines": len(o_records),
            "table_contents": len(tbl_records),
        })
        if resume_checkpoints:
            try:
                _save_pdf_checkpoint(
                    checkpoint_path=checkpoint_path,
                    pdf_path=pdf_path,
                    signature=signature,
                    records=t_records + o_records + tbl_records,
                    counts=counts,
                    stage="parsed",
                )
            except Exception as exc:
                _report(None, TOTAL, f"  [{pdf_path.name}] 체크포인트 저장 실패: {exc}")
        return pdf_path, t_records + o_records + tbl_records, counts

    records: list[KnowledgeBaseRecord] = []
    component_counts: Counter[str] = Counter()
    completed = 0
    _report(0, TOTAL, f"PDF 파싱 중 (0/{n_pdfs})...")
    # layout 모델을 병렬 파싱 전에 메인 스레드에서 미리 로딩해 thread-safe 초기화 보장
    from .layout_analysis import _layoutlmv3_bundle, _layoutparser_model
    _layoutlmv3_bundle()
    _layoutparser_model()
    pending_jobs: list[tuple[Path, dict[str, Any], Path]] = []
    for pdf_path in pdf_paths:
        signature = _pdf_signature(pdf_path)
        checkpoint_path = _checkpoint_path(output_path, pdf_path)
        if resume_checkpoints:
            checkpoint = _load_pdf_checkpoint(checkpoint_path, str(pdf_path), signature)
            if checkpoint is not None:
                pdf_records, pdf_counts = checkpoint
                records.extend(pdf_records)
                component_counts.update(pdf_counts)
                completed += 1
                _report(int(completed * 30 / n_pdfs), TOTAL, f"PDF 체크포인트 재개 ({completed}/{n_pdfs}): {pdf_path.name}")
                continue
        pending_jobs.append((pdf_path, signature, checkpoint_path))

    if pending_jobs:
        with ThreadPoolExecutor(max_workers=min(len(pending_jobs), 4)) as executor:
            futures = {
                executor.submit(_parse_pdf, pdf_path, signature, checkpoint_path): pdf_path
                for pdf_path, signature, checkpoint_path in pending_jobs
            }
            for future in as_completed(futures):
                pdf_path, pdf_records, pdf_counts = future.result()
                records.extend(pdf_records)
                component_counts.update(pdf_counts)
                completed += 1
                _report(int(completed * 30 / n_pdfs), TOTAL, f"PDF 파싱 중 ({completed}/{n_pdfs}): {pdf_path.name}")
    else:
        _report(30, TOTAL, f"PDF 파싱은 체크포인트로 모두 재개됨 ({completed}/{n_pdfs})")
    if summary_mode == "mt5" and summary_model:
        _report(30, TOTAL, f"요약 모델 로딩 중 ({summary_model})...")
    elif summary_mode == "off":
        _report(30, TOTAL, "요약 생략 모드...")
    else:
        _report(30, TOTAL, "정확도 우선 빠른 모드: 추출적 요약 + 원문 벡터 사용...")

    summarizer = T5Summarizer(model_name=summary_model if summary_mode == "mt5" else None)
    _report(35, TOTAL, f"요약 생성 중 ({len(records)}개 레코드)...")

    def _summary_progress(message: str, done: int, total_items: int) -> None:
        if total_items <= 0:
            _report(40, TOTAL, message)
            return
        pct = 35 + int(done * 25 / total_items)
        _report(pct, TOTAL, f"{message} ({done}/{total_items})")

    enriched_records = _enrich_records_with_summaries(
        records,
        summarizer,
        summary_mode=summary_mode,
        progress_callback=_summary_progress,
    )
    _report(60, TOTAL, "임베딩 모델 로딩 중...")

    embedder = M3EEmbedder(model_name=embedding_model)
    n_vectors = len(enriched_records)
    _report(65, TOTAL, f"임베딩 인코딩 중 (0/{n_vectors})...")
    vector_texts = [record.vector_text or record.text for record in enriched_records]

    def _embed_progress(done: int, total_vecs: int) -> None:
        # 65~90 구간에 선형 매핑
        pct = 65 + int(done * 25 / total_vecs) if total_vecs else 90
        _report(pct, TOTAL, f"임베딩 인코딩 중 ({done}/{total_vecs})...")

    vectors = embedder.encode(vector_texts, progress_callback=_embed_progress)
    _report(90, TOTAL, "결과 저장 중...")

    textual_path = output_path / "textual_contents.jsonl"
    outline_path = output_path / "document_outlines.jsonl"
    table_path = output_path / "table_contents.jsonl"
    records_path = output_path / "records.jsonl"
    vector_path = output_path / "text_vectors.npy"
    manifest_path = output_path / "manifest.json"

    _report(91, TOTAL, "textual_contents 저장 중...")
    _records_to_jsonl([record for record in enriched_records if record.component == "textual_contents"], textual_path)
    _report(92, TOTAL, "document_outlines 저장 중...")
    _records_to_jsonl([record for record in enriched_records if record.component == "document_outlines"], outline_path)
    _report(93, TOTAL, "table_contents 저장 중...")
    _records_to_jsonl([record for record in enriched_records if record.component == "table_contents"], table_path)
    _report(94, TOTAL, "records.jsonl 저장 중...")
    _records_to_jsonl(enriched_records, records_path)
    _report(95, TOTAL, "text_vectors.npy 저장 중...")
    np.save(vector_path, vectors)

    _report(96, TOTAL, "FAISS 인덱스 생성 중...")
    faiss_path = _write_faiss_index(vectors, output_path) if use_faiss else None
    _report(97, TOTAL, "Milvus/manifest 정리 중...")
    actual_milvus_collection = _write_milvus(
        enriched_records,
        vectors,
        uri=milvus_uri or "",
        collection_name=milvus_collection,
    )

    manifest = {
        "input_path": str(input_path),
        "source_count": len(pdf_paths),
        "record_count": len(enriched_records),
        "component_counts": dict(component_counts),
        "resumed_source_count": completed - len(pending_jobs),
        "pending_source_count": len(pending_jobs),
        "summary_mode": summary_mode,
        "summary_model": summary_model,
        "embedding_model": embedding_model,
        "records_path": str(records_path),
        "textual_records_path": str(textual_path),
        "outline_records_path": str(outline_path),
        "table_records_path": str(table_path),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_checkpoints": resume_checkpoints,
        "vector_path": str(vector_path),
        "faiss_index_path": faiss_path,
        "milvus_uri": milvus_uri,
        "milvus_collection": actual_milvus_collection,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(100, TOTAL, f"완료 ({len(enriched_records)}개 레코드)")

    return KnowledgeBaseBuild(
        input_path=str(input_path),
        output_dir=str(output_path),
        source_count=len(pdf_paths),
        record_count=len(enriched_records),
        component_counts=dict(component_counts),
        summary_model=summary_model,
        embedding_model=embedding_model,
        faiss_index_path=faiss_path,
        milvus_uri=milvus_uri,
        milvus_collection=actual_milvus_collection,
        records_path=str(records_path),
        textual_records_path=str(textual_path),
        outline_records_path=str(outline_path),
        table_records_path=str(table_path),
        checkpoint_dir=str(checkpoint_dir),
        resume_checkpoints=resume_checkpoints,
        resumed_source_count=completed - len(pending_jobs),
        pending_source_count=len(pending_jobs),
        summary_mode=summary_mode,
    )


def knowledge_base_to_json(build: KnowledgeBaseBuild) -> str:
    return json.dumps(asdict(build), ensure_ascii=False, indent=2)
