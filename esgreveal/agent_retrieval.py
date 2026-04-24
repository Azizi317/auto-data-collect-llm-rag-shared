from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import os
import re
import threading

import numpy as np

from .chunking import DocumentChunk
from .knowledge_base import DEFAULT_EMBEDDING_MODEL, KnowledgeBaseRecord, M3EEmbedder
from .metadata import KPIItem
from .retrieval import RankedChunk, normalize_for_match


@dataclass(frozen=True)
class KnowledgeVectorBase:
    root_dir: str
    records: tuple[KnowledgeBaseRecord, ...]
    vectors: np.ndarray


@dataclass(frozen=True)
class KnowledgeHit:
    record: KnowledgeBaseRecord
    cosine_score: float
    rerank_score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class _RerankerBundle:
    kind: str
    model: Any


def _record_from_dict(payload: dict[str, Any]) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(
        record_id=str(payload["record_id"]),
        component=str(payload["component"]),
        source_path=str(payload["source_path"]),
        page_start=int(payload["page_start"]),
        page_end=int(payload["page_end"]),
        section=payload.get("section"),
        section_level=payload.get("section_level"),
        outline_path=tuple(payload.get("outline_path", ()) or ()),
        text=str(payload.get("text", "")),
        summary=payload.get("summary"),
        vector_text=payload.get("vector_text"),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


@lru_cache(maxsize=4)
def load_knowledge_vector_base(root_dir: str | Path) -> KnowledgeVectorBase:
    root = Path(root_dir)
    records_path = root / "records.jsonl"
    vectors_path = root / "text_vectors.npy"
    records = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_dict(json.loads(line)))
    vectors = np.load(vectors_path)
    return KnowledgeVectorBase(root_dir=str(root), records=tuple(records), vectors=vectors)


def build_query_text(kpi: KPIItem) -> str:
    search_terms = ", ".join(kpi.search_terms)
    knowledge = " ".join(kpi.knowledge)
    refs = ", ".join(kpi.standard_refs)
    return normalize_for_match(
        " ".join(
            part
            for part in (
                kpi.aspect_ko,
                kpi.aspect_en,
                kpi.kpi_ko,
                kpi.kpi_en,
                kpi.topic_ko,
                kpi.topic_en,
                kpi.quantity_ko,
                kpi.quantity_en,
                kpi.unit,
                search_terms,
                knowledge,
                refs,
            )
            if part
        )
    )


@lru_cache(maxsize=1)
def _query_embedder() -> M3EEmbedder:
    return M3EEmbedder(model_name=os.getenv("ESGREVEAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))


@lru_cache(maxsize=256)
def _encode_query(query_text: str) -> np.ndarray:
    """동일 query_text에 대해 BGE 추론을 1회만 수행하고 캐싱한다."""
    return _query_embedder().encode([query_text])[0]


_haystack_cache: dict[str, str] = {}


def _record_haystack(record: KnowledgeBaseRecord) -> str:
    """record별 검색 대상 문자열을 1회만 조립·정규화해 캐싱한다."""
    key = record.record_id
    if key not in _haystack_cache:
        _haystack_cache[key] = normalize_for_match(
            " ".join(
                part
                for part in (
                    record.text,
                    record.summary or "",
                    record.vector_text or "",
                    " ".join(record.outline_path),
                    str(record.metadata.get("keyword", "")),
                    str(record.metadata.get("row_label", "")),
                )
                if part
            )
        )
    return _haystack_cache[key]


def _match_terms(kpi: KPIItem, record: KnowledgeBaseRecord) -> tuple[str, ...]:
    haystack = _record_haystack(record)
    matched = []
    for term in kpi.all_terms():
        normalized = normalize_for_match(term)
        if normalized and normalized in haystack:
            matched.append(term)
    return tuple(dict.fromkeys(matched))[:10]


@lru_cache(maxsize=8)
def _build_source_index(root_dir: str) -> dict[str, np.ndarray]:
    """bundle root_dir별로 source_path → indices 매핑을 1회만 계산해 캐싱."""
    bundle = load_knowledge_vector_base(root_dir)
    mapping: dict[str, list[int]] = {}
    for index, record in enumerate(bundle.records):
        try:
            key = str(Path(record.source_path).resolve(strict=False))
        except Exception:
            key = str(Path(record.source_path))
        mapping.setdefault(key, []).append(index)
    return {k: np.asarray(v, dtype=np.int64) for k, v in mapping.items()}


def _source_filtered_indices(bundle: KnowledgeVectorBase, source_path: str | Path | None) -> np.ndarray:
    if source_path is None:
        return np.arange(len(bundle.records))
    try:
        target = str(Path(source_path).resolve(strict=False))
    except Exception:
        target = str(Path(source_path))
    index_map = _build_source_index(bundle.root_dir)
    if target in index_map:
        return index_map[target]
    return np.arange(len(bundle.records))


def _component_bonus(record: KnowledgeBaseRecord, kpi: KPIItem) -> float:
    score = 0.0
    mapping_type = str(record.metadata.get("mapping_type", ""))
    if record.component == "table_contents":
        if mapping_type == "table_keyword":
            score += 0.9 if kpi.disclosure_type == "numeric" else 0.45
        elif mapping_type == "table_base":
            score += 0.25
    elif record.component == "document_outlines":
        score += 0.2
    elif record.component == "textual_contents":
        score += 0.35 if kpi.disclosure_type == "text" else 0.15
    if record.metadata.get("ontology_kpi_ids") and kpi.id in record.metadata.get("ontology_kpi_ids", []):
        score += 1.1
    if any(token in normalize_for_match(record.text) for token in ("gri", "index", "appendix", "references")):
        score -= 0.9
    return score


def _quality_bonus(record: KnowledgeBaseRecord, kpi: KPIItem) -> float:
    score = 0.0
    text_low = normalize_for_match(record.text)
    if kpi.disclosure_type == "numeric":
        if re.search(r"\b(19|20)\d{2}\b", record.text):
            score += 0.2
        if re.search(r"\b(?:tco2e|mwh|gj|kwh|m3|%)\b", text_low):
            score += 0.4
        if any(token in text_low for token in ("scope 1+2", "scope1+2", "scope 1 and 2")) and any(
            token in normalize_for_match(" ".join((kpi.kpi_ko, kpi.kpi_en, kpi.topic_ko, kpi.topic_en)))
            for token in ("scope 1", "scope 2")
        ):
            score -= 0.8
    return score


def _rule_rerank_score(kpi: KPIItem, record: KnowledgeBaseRecord, cosine_score: float, matched_terms: tuple[str, ...]) -> float:
    text_low = normalize_for_match(record.text)
    outline_low = normalize_for_match(" ".join(record.outline_path))
    query_low = build_query_text(kpi)
    phrase_hits = 0.0

    for phrase in (kpi.kpi_ko, kpi.kpi_en, kpi.topic_ko, kpi.topic_en):
        normalized = normalize_for_match(phrase)
        if normalized and normalized in text_low:
            phrase_hits += 0.55
        if normalized and normalized in outline_low:
            phrase_hits += 0.35

    if record.summary:
        summary_low = normalize_for_match(record.summary)
        for phrase in (kpi.kpi_ko, kpi.kpi_en):
            normalized = normalize_for_match(phrase)
            if normalized and normalized in summary_low:
                phrase_hits += 0.25

    lexical = 0.1 * len(matched_terms)
    query_tokens = {token for token in re.split(r"[^0-9a-z가-힣]+", query_low) if len(token) >= 2}
    record_tokens = {token for token in re.split(r"[^0-9a-z가-힣]+", normalize_for_match(record.vector_text or "")) if len(token) >= 2}
    overlap = len(query_tokens & record_tokens)
    coverage = min(overlap * 0.03, 0.18)
    return round(cosine_score * 0.6 + phrase_hits + lexical + coverage + _component_bonus(record, kpi) + _quality_bonus(record, kpi), 6)


def _should_use_modelscope_reranker(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized.startswith(("iic/", "damo/", "modelscope:"))


@lru_cache(maxsize=4)
def _corom_model(model_name: str | None = None):
    model_name = (model_name or os.getenv("ESGREVEAL_COROM_MODEL", "").strip() or os.getenv("ESGREVEAL_RERANKER_MODEL", "").strip()).strip()
    if not model_name:
        return None
    if _should_use_modelscope_reranker(model_name):
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except Exception:
            return None
        try:
            return _RerankerBundle(kind="modelscope", model=pipeline(Tasks.text_ranking, model=model_name))
        except Exception:
            return None

    try:
        import torch
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        return _RerankerBundle(
            kind="cross_encoder",
            model=CrossEncoder(model_name, default_activation_function=torch.nn.Sigmoid()),
        )
    except Exception:
        return None


def _corom_rerank(query_text: str, hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
    model = _corom_model()
    if model is None or not hits:
        return hits
    try:
        candidate_texts = [(hit.record.vector_text or hit.record.text) for hit in hits]
        if model.kind == "modelscope":
            scores = []
            for candidate_text in candidate_texts:
                result = model.model(
                    {
                        "source_sentence": [query_text],
                        "sentences_to_compare": [candidate_text],
                    }
                )
                scores.append(_extract_corom_score(result))
        else:
            pairs = [(query_text, candidate_text) for candidate_text in candidate_texts]
            raw_scores = model.model.predict(pairs, convert_to_numpy=True)
            flattened = np.asarray(raw_scores, dtype=np.float32).reshape(-1).tolist()
            scores = [float(score) for score in flattened[: len(hits)]]
    except Exception:
        return hits
    reranked = []
    for hit, score in zip(hits, scores, strict=False):
        calibrated = _calibrate_reranker_score(float(score))
        final_score = (calibrated * 0.75) + (hit.rerank_score * 0.25)
        reranked.append(
            KnowledgeHit(
                record=hit.record,
                cosine_score=hit.cosine_score,
                rerank_score=float(final_score),
                matched_terms=hit.matched_terms,
            )
        )
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return reranked


def _calibrate_reranker_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + np.exp(-score))


def _extract_corom_score(result: Any) -> float:
    if result is None:
        return 0.0
    if isinstance(result, dict):
        for key in ("scores", "score", "similarity", "similarities", "logits"):
            if key in result:
                return _extract_corom_score(result[key])
    if isinstance(result, (list, tuple)):
        if not result:
            return 0.0
        first = result[0]
        if isinstance(first, dict):
            for key in ("scores", "score", "similarity", "similarities", "logits"):
                if key in first:
                    return _extract_corom_score(first[key])
            for key in ("output", "result"):
                if key in first:
                    return _extract_corom_score(first[key])
        if isinstance(first, (list, tuple, np.ndarray)):
            return _extract_corom_score(first)
        try:
            return float(first)
        except Exception:
            return 0.0
    if isinstance(result, np.ndarray):
        if result.size == 0:
            return 0.0
        return _extract_corom_score(result.reshape(-1).tolist())
    try:
        return float(result)
    except Exception:
        return 0.0


def _record_to_ranked_chunk(hit: KnowledgeHit) -> RankedChunk:
    mapping_type = str(hit.record.metadata.get("mapping_type", ""))
    if hit.record.component == "table_contents":
        kind = "table"
    elif hit.record.component == "document_outlines":
        kind = "heading"
    else:
        kind = "paragraph"
    if mapping_type == "table_keyword":
        kind = "table"

    chunk = DocumentChunk(
        page_start=hit.record.page_start,
        page_end=hit.record.page_end,
        section=hit.record.section,
        section_level=hit.record.section_level,
        outline_path=hit.record.outline_path,
        kind=kind,
        text=hit.record.text,
        source_blocks=1,
    )
    return RankedChunk(chunk=chunk, score=hit.rerank_score, matched_terms=hit.matched_terms)


def rank_knowledge_base(
    kb_dir: str | Path,
    kpi: KPIItem,
    top_k: int = 3,
    initial_k: int = 20,
    source_path: str | Path | None = None,
) -> list[RankedChunk]:
    bundle = load_knowledge_vector_base(kb_dir)
    if bundle.vectors.size == 0 or not bundle.records:
        return []

    query_text = build_query_text(kpi)
    query_vector = _encode_query(query_text)
    indices = _source_filtered_indices(bundle, source_path)
    candidate_vectors = bundle.vectors[indices]
    cosine_scores = candidate_vectors @ query_vector
    order = np.argsort(cosine_scores)[::-1][: max(initial_k, top_k)]

    hits: list[KnowledgeHit] = []
    for local_index in order.tolist():
        global_index = int(indices[local_index])
        record = bundle.records[global_index]
        matched_terms = _match_terms(kpi, record)
        rerank_score = _rule_rerank_score(kpi, record, float(cosine_scores[local_index]), matched_terms)
        hits.append(
            KnowledgeHit(
                record=record,
                cosine_score=float(cosine_scores[local_index]),
                rerank_score=rerank_score,
                matched_terms=matched_terms,
            )
        )

    reranked = _corom_rerank(query_text, hits)
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return [_record_to_ranked_chunk(hit) for hit in reranked[:top_k]]
