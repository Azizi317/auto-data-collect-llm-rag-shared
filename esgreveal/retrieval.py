from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable
import math
import os
import re

from .chunking import DocumentChunk
from .metadata import KPIItem


WORD_RE = re.compile(r"[A-Za-z0-9]+|[가-힣]+")


@dataclass(frozen=True)
class RankedChunk:
    chunk: DocumentChunk
    score: float
    matched_terms: tuple[str, ...] = ()


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in WORD_RE.findall(normalize_for_match(text)):
        if raw.isdigit():
            tokens.append(raw)
            continue
        if re.fullmatch(r"[a-z0-9]+", raw):
            tokens.append(raw)
            continue
        tokens.append(raw)
        compact = raw.replace(" ", "")
        if len(compact) >= 2:
            tokens.extend(compact[index : index + 2] for index in range(len(compact) - 1))
        if len(compact) >= 4:
            tokens.extend(compact[index : index + 3] for index in range(len(compact) - 2))
    return [token for token in tokens if token]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = re.sub(r"\s+", "", normalize_for_match(text))
    if len(compact) <= n:
        return [compact] if compact else []
    return [compact[index : index + n] for index in range(len(compact) - n + 1)]


def _counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left).intersection(right)
    if not common:
        return 0.0
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


@lru_cache(maxsize=1)
def _sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None

    model_name = os.environ.get(
        "ESGREVEAL_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _semantic_similarities(query_text: str, chunk_texts: list[str]) -> list[float]:
    if not chunk_texts:
        return []
    model = _sentence_transformer_model()
    if model is not None:
        try:
            query_embedding = model.encode([query_text], normalize_embeddings=True)
            chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
            scores = np.asarray(chunk_embeddings) @ np.asarray(query_embedding)[0]
            return [float(max(min(score, 1.0), 0.0)) for score in scores.tolist()]
        except Exception:
            pass

    query_counter = Counter(char_ngrams(query_text, n=3))
    return [_counter_cosine(query_counter, Counter(char_ngrams(chunk_text, n=3))) for chunk_text in chunk_texts]


def _semantic_similarity(query_text: str, chunk_text: str) -> float:
    return _semantic_similarities(query_text, [chunk_text])[0]


def query_terms(kpi: KPIItem) -> list[str]:
    return tokenize(" ".join(kpi.all_terms()))


def query_phrases(kpi: KPIItem) -> list[str]:
    phrases = list(kpi.all_terms())
    return [phrase.lower().strip() for phrase in phrases if len(phrase.strip()) >= 2]


def score_chunk(
    chunk: DocumentChunk,
    terms: Iterable[str],
    phrases: Iterable[str],
    kpi: KPIItem,
    semantic: float | None = None,
) -> tuple[float, tuple[str, ...]]:
    tokens = tokenize(chunk.text)
    if not tokens:
        return 0.0, ()

    counts = Counter(tokens)
    text_low = normalize_for_match(chunk.text)
    section_low = normalize_for_match(chunk.section or "")
    outline_low = normalize_for_match(" ".join(chunk.outline_path))
    score = 0.0
    matched_terms: list[str] = []

    for term in terms:
        freq = counts.get(term, 0)
        if freq:
            score += 0.75 + 0.25 * math.log1p(freq)
            matched_terms.append(term)

    for phrase in phrases:
        if not phrase:
            continue
        normalized_phrase = normalize_for_match(phrase)
        if normalized_phrase and normalized_phrase in text_low:
            score += 2.2 + 0.15 * len(normalized_phrase.split())
            matched_terms.append(phrase)

    if kpi.disclosure_type == "numeric" and chunk.kind == "table":
        score *= 1.18
    elif kpi.disclosure_type == "text" and chunk.kind in {"paragraph", "mixed", "ocr"}:
        score *= 1.08

    if section_low:
        section_tokens = tokenize(section_low)
        overlap = len(set(section_tokens).intersection(set(terms)))
        if overlap:
            score += 1.1 * overlap
        for phrase in phrases:
            normalized_phrase = normalize_for_match(phrase)
            if normalized_phrase and normalized_phrase in section_low:
                score += 1.2
                matched_terms.append(f"section:{phrase}")

    if outline_low:
        outline_tokens = tokenize(outline_low)
        outline_overlap = len(set(outline_tokens).intersection(set(terms)))
        if outline_overlap:
            score += 0.85 * outline_overlap
        for phrase in phrases:
            normalized_phrase = normalize_for_match(phrase)
            if normalized_phrase and normalized_phrase in outline_low:
                score += 0.9
                matched_terms.append(f"outline:{phrase}")

    semantic_score = semantic if semantic is not None else _semantic_similarity(" ".join(kpi.all_terms()), chunk.text)
    score += semantic_score * 3.6

    if chunk.kind == "table":
        score += 0.4
    if chunk.kind == "heading":
        score *= 0.92

    score *= 1.0 + min(0.25, len(counts) / max(len(tokens), 1))
    return score, tuple(dict.fromkeys(matched_terms))


def rank_chunks(chunks: list[DocumentChunk], kpi: KPIItem, top_k: int = 3) -> list[RankedChunk]:
    terms = query_terms(kpi)
    phrases = query_phrases(kpi)
    query_text = " ".join(kpi.all_terms())
    semantic_scores = _semantic_similarities(query_text, [chunk.text for chunk in chunks])
    ranked = [
        RankedChunk(chunk=chunk, score=score, matched_terms=matched_terms)
        for chunk, semantic in zip(chunks, semantic_scores, strict=False)
        for score, matched_terms in [score_chunk(chunk, terms, phrases, kpi, semantic)]
    ]
    ranked = [item for item in ranked if item.score > 0]
    ranked.sort(
        key=lambda item: (
            item.score,
            item.chunk.kind == "table",
            item.chunk.page_end,
            len(item.chunk.text),
        ),
        reverse=True,
    )
    return ranked[:top_k]
