from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import os
import time

from .agent_retrieval import rank_knowledge_base
from .chunking import build_chunks
from .document import extract_document_blocks, summarize_blocks
from .extraction import extraction_to_output, heuristic_extract
from .llm import LLMCallError, LLMConfig, call_llm, load_llm_config_from_env
from .metadata import KPIItem, default_metadata, load_metadata
from .prompts import build_prompt
from .retrieval import RankedChunk, rank_chunks


@dataclass(frozen=True)
class KPIResult:
    metadata: dict[str, Any]
    prompt: str
    retrieved_context: list[dict[str, Any]]
    retrieval_backend: str
    output: Any


@dataclass(frozen=True)
class PipelineRun:
    source_path: str
    document_summary: dict[str, Any]
    results: list[KPIResult]


class PipelineLLMError(RuntimeError):
    def __init__(self, *, kpi: KPIItem, source_path: str, backend: str, model: str, message: str) -> None:
        self.kpi = kpi
        self.source_path = source_path
        self.backend = backend
        self.model = model
        self.message = message
        super().__init__(message)


def load_kpis(metadata_path: str | Path | None = None) -> list[KPIItem]:
    return default_metadata() if metadata_path is None else load_metadata(metadata_path)


def _kb_is_ready(kb_dir: str | Path | None) -> bool:
    if kb_dir is None:
        return False
    root = Path(kb_dir)
    return (root / "records.jsonl").exists() and (root / "text_vectors.npy").exists()


def _serialize_retrieved(chunks: Iterable[RankedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "page_start": item.chunk.page_start,
            "page_end": item.chunk.page_end,
            "section": item.chunk.section,
            "section_level": item.chunk.section_level,
            "outline_path": list(item.chunk.outline_path),
            "kind": item.chunk.kind,
            "score": item.score,
            "matched_terms": list(item.matched_terms),
            "text": item.chunk.text,
        }
        for item in chunks
    ]


def _normalize_output(kpi: KPIItem, output: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(output)
    normalized.setdefault("disclosure_kpi", kpi.kpi_ko)
    normalized.setdefault("topic", kpi.topic_ko)
    normalized.setdefault("value", None)
    normalized.setdefault("unit", None)
    normalized.setdefault("target", None)
    normalized.setdefault("action", None)
    normalized.setdefault("evidence", None)
    normalized.setdefault("confidence", None)
    normalized.setdefault("source_page", None)
    normalized.setdefault("matched_terms", [])
    normalized.setdefault("warnings", [])
    normalized.setdefault("llm_backend_used", None)
    normalized.setdefault("llm_model_used", None)
    return normalized


def _make_output(config: LLMConfig | None, prompt: str, kpi: KPIItem, ranked: list[RankedChunk]) -> dict[str, Any]:
    if config is None:
        output = extraction_to_output(heuristic_extract(kpi, ranked))
        output["note"] = "LLM 설정이 없어 규칙 기반 추출 결과를 반환했습니다."
        output["warnings"] = ["LLM disabled; heuristic fallback used."]
        return _normalize_output(kpi, output)

    output = call_llm(config, prompt)
    warnings = list(output.pop("_llm_warnings", []) or [])
    backend_used = output.pop("_llm_backend_used", None)
    model_used = output.pop("_llm_model_used", None)
    output["warnings"] = warnings
    output["llm_backend_used"] = backend_used or config.backend
    output["llm_model_used"] = model_used or config.model
    return _normalize_output(kpi, output)


def _retrieve_for_kpi(
    kpi: KPIItem,
    source_path: str | Path,
    chunks: list[RankedChunk] | list[Any],
    kb_dir: str | Path | None,
    top_k: int,
    retrieval_initial_k: int,
) -> tuple[list[RankedChunk], str]:
    if _kb_is_ready(kb_dir):
        try:
            ranked = rank_knowledge_base(
                kb_dir=Path(kb_dir),
                kpi=kpi,
                top_k=top_k,
                initial_k=retrieval_initial_k,
                source_path=source_path,
            )
            if ranked:
                return ranked, "knowledge_vector_base"
        except Exception as exc:
            return rank_chunks(list(chunks), kpi, top_k=top_k), f"chunk_fallback_warning:{exc}"

    ranked = rank_chunks(list(chunks), kpi, top_k=top_k)
    return ranked, "chunk_fallback"


def run_pipeline(
    pdf_path: str | Path,
    metadata_path: str | Path | None = None,
    top_k: int = 3,
    llm_config: LLMConfig | None = None,
    kb_dir: str | Path | None = None,
    retrieval_initial_k: int = 20,
    worker: Any = None,
    _progress_offset: int = 0,
    _progress_total: int = 100,
) -> PipelineRun:
    def _report(done: int, total: int, message: str) -> None:
        if worker is None:
            return
        value = _progress_offset + int(done * _progress_total / total) if total else _progress_offset
        worker.progress(value, 100, message)

    source_path = str(Path(pdf_path))
    pdf_name = Path(pdf_path).name
    kpis = load_kpis(metadata_path)
    n_kpis = max(len(kpis), 1)

    _report(0, n_kpis, f"[{pdf_name}] 문서 파싱 중...")
    blocks = extract_document_blocks(pdf_path)
    chunks = build_chunks(blocks)
    summary = asdict(summarize_blocks(blocks, page_count=max((block.page_num for block in blocks), default=0)))
    summary["retrieval_backend"] = "knowledge_vector_base" if _kb_is_ready(kb_dir) else "chunk_fallback"
    summary["knowledge_base_dir"] = str(kb_dir) if _kb_is_ready(kb_dir) else None
    summary["warnings"] = []
    config = llm_config or load_llm_config_from_env()

    llm_workers = int(os.environ.get("ESGREVEAL_LLM_WORKERS", "4"))
    results: list[KPIResult | None] = [None] * n_kpis
    warnings_list: list[tuple[str, str]] = []
    started = time.monotonic()
    completed_count = 0

    def _llm_task(index: int, kpi: KPIItem, prompt: str, ranked: list[RankedChunk]) -> tuple[int, dict[str, Any]]:
        return index, _make_output(config, prompt, kpi, ranked)

    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        pending: dict[Future[tuple[int, dict[str, Any]]], tuple[int, KPIItem, str, list[RankedChunk], str]] = {}

        for i, kpi in enumerate(kpis):
            elapsed = time.monotonic() - started
            eta_text = ""
            if i > 0:
                eta = elapsed / i * (n_kpis - i)
                eta_text = f"  남은 시간 약 {int(eta // 60)}분 {int(eta % 60)}초"
            _report(i, n_kpis, f"[{pdf_name}] KPI {i + 1}/{n_kpis}: {kpi.kpi_ko}{eta_text}")

            ranked, backend = _retrieve_for_kpi(
                kpi=kpi,
                source_path=source_path,
                chunks=chunks,
                kb_dir=kb_dir,
                top_k=top_k,
                retrieval_initial_k=retrieval_initial_k,
            )
            prompt = build_prompt(kpi, ranked)
            future = executor.submit(_llm_task, i, kpi, prompt, ranked)
            pending[future] = (i, kpi, prompt, ranked, backend)

        for future in as_completed(pending):
            i, kpi, prompt, ranked, backend = pending[future]
            try:
                index, output = future.result()
            except LLMCallError as exc:
                raise PipelineLLMError(
                    kpi=kpi,
                    source_path=source_path,
                    backend=exc.backend,
                    model=exc.model,
                    message=str(exc),
                ) from exc
            if backend.startswith("chunk_fallback_warning"):
                warnings_list.append((kpi.id, backend))
            for warning in output.get("warnings", []) or []:
                warnings_list.append((kpi.id, warning))
            results[index] = KPIResult(
                metadata=asdict(kpi),
                prompt=prompt,
                retrieved_context=_serialize_retrieved(ranked),
                retrieval_backend=backend,
                output=output,
            )
            completed_count += 1
            _report(completed_count, n_kpis, f"[{pdf_name}] LLM 완료 {completed_count}/{n_kpis}")

    for kpi_id, warning in warnings_list:
        summary["warnings"].append(f"{kpi_id}: {warning}")

    _report(n_kpis, n_kpis, f"[{pdf_name}] 완료")
    return PipelineRun(source_path=source_path, document_summary=summary, results=[r for r in results if r is not None])


def run_batch(
    input_path: str | Path,
    metadata_path: str | Path | None = None,
    top_k: int = 3,
    recursive: bool = False,
    llm_config: LLMConfig | None = None,
    kb_dir: str | Path | None = None,
    retrieval_initial_k: int = 20,
    worker: Any = None,
) -> list[PipelineRun]:
    source = Path(input_path)
    if source.is_file():
        return [
            run_pipeline(
                source,
                metadata_path=metadata_path,
                top_k=top_k,
                llm_config=llm_config,
                kb_dir=kb_dir,
                retrieval_initial_k=retrieval_initial_k,
                worker=worker,
                _progress_offset=0,
                _progress_total=100,
            )
        ]

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(source.glob(pattern))
    n_docs = max(len(pdf_paths), 1)
    runs: list[PipelineRun] = []
    for i, path in enumerate(pdf_paths):
        offset = int(i * 100 / n_docs)
        span = int(100 / n_docs)
        runs.append(
            run_pipeline(
                path,
                metadata_path=metadata_path,
                top_k=top_k,
                llm_config=llm_config,
                kb_dir=kb_dir,
                retrieval_initial_k=retrieval_initial_k,
                worker=worker,
                _progress_offset=offset,
                _progress_total=span,
            )
        )
    return runs


def _result_to_dict(result: KPIResult) -> dict[str, Any]:
    return {
        "metadata": result.metadata,
        "prompt": result.prompt,
        "retrieved_context": result.retrieved_context,
        "retrieval_backend": result.retrieval_backend,
        "output": result.output,
    }


def results_to_json(run: PipelineRun) -> str:
    return json.dumps(
        {
            "source_path": run.source_path,
            "document_summary": run.document_summary,
            "results": [_result_to_dict(result) for result in run.results],
        },
        ensure_ascii=False,
        indent=2,
    )


def batch_results_to_json(runs: list[PipelineRun]) -> str:
    return json.dumps(
        [
            {
                "source_path": run.source_path,
                "document_summary": run.document_summary,
                "results": [_result_to_dict(result) for result in run.results],
            }
            for run in runs
        ],
        ensure_ascii=False,
        indent=2,
    )


def results_to_markdown(run: PipelineRun) -> str:
    def yn(value: Any) -> str:
        if value is True:
            return "예"
        if value is False:
            return "아니오"
        return "미확인"

    def fmt(value: Any) -> str:
        if value is None or value == "":
            return "미확인"
        return str(value)

    summary = run.document_summary
    document_kind = summary.get("document_kind")
    document_kind_label = {
        "academic_paper": "논문/연구자료",
        "esg_report": "ESG 보고서",
    }.get(str(document_kind), "미확인")

    lines = ["# ESGReveal 추출 결과", ""]
    lines.extend(
        [
            "## 문서 개요",
            f"- 원본 파일: {run.source_path}",
            f"- 문서 유형: {document_kind_label}",
            f"- 추출 모드: {fmt(summary.get('extraction_mode'))}",
            f"- Retrieval Backend: {fmt(summary.get('retrieval_backend'))}",
            f"- Knowledge Base Dir: {fmt(summary.get('knowledge_base_dir'))}",
            f"- Warnings: {fmt('; '.join(summary.get('warnings', [])) if summary.get('warnings') else None)}",
            f"- OCR 페이지 수: {fmt(summary.get('ocr_page_count'))}",
            f"- 페이지 수: {fmt(summary.get('page_count'))}",
            f"- 블록 수: {fmt(summary.get('block_count'))}",
            f"- 제목 수: {fmt(summary.get('heading_count'))}",
            f"- 표 블록 수: {fmt(summary.get('table_count'))}",
            f"- 섹션 수: {fmt(summary.get('section_count'))}",
            "",
        ]
    )

    for index, result in enumerate(run.results, start=1):
        meta = result.metadata
        output = result.output if isinstance(result.output, dict) else {}
        k_esg_refs = ", ".join(
            f"{code} {label}".strip()
            for code, label in zip(meta.get("k_esg_codes", []), meta.get("k_esg_labels", []), strict=False)
        ) or "직접 대응 없음"
        expression = meta.get("expression_template", {})
        lines.extend(
            [
                f"## {index}. {fmt(meta.get('kpi_ko'))}",
                f"- Retrieval: {result.retrieval_backend}",
                f"- Category: {fmt(meta.get('category'))}",
                f"- Aspect: {fmt(meta.get('aspect_ko'))} / {fmt(meta.get('aspect_en'))}",
                f"- KPI: {fmt(meta.get('kpi_ko'))} / {fmt(meta.get('kpi_en'))}",
                f"- Topic: {fmt(meta.get('topic_ko'))} / {fmt(meta.get('topic_en'))}",
                f"- Quantity: {fmt(meta.get('quantity_ko'))} / {fmt(meta.get('quantity_en'))}",
                f"- Unit: {fmt(meta.get('unit'))}",
                f"- Standard Family: {fmt(meta.get('standard_family'))}",
                f"- Standard Refs: {', '.join(meta.get('standard_refs', [])) if meta.get('standard_refs') else '미확인'}",
                f"- K-ESG Crosscheck: {k_esg_refs}",
                f"- ExpressionTemplate: {fmt(expression.get('name'))}",
                f"- Disclosure Type: {fmt(meta.get('disclosure_type'))}",
                f"- Obligation: {fmt(meta.get('obligation'))}",
                f"- 공시 여부: {yn(output.get('disclosed'))}",
                f"- Disclosure KPI: {fmt(output.get('disclosure_kpi'))}",
                f"- Topic Output: {fmt(output.get('topic'))}",
                f"- 값: {fmt(output.get('value'))}",
                f"- 단위: {fmt(output.get('unit'))}",
                f"- 목표: {fmt(output.get('target'))}",
                f"- 조치: {fmt(output.get('action'))}",
                f"- 신뢰도: {fmt(output.get('confidence'))}",
                f"- 출처 페이지: {fmt(output.get('source_page'))}",
                f"- 근거: {fmt(str(output.get('evidence', ''))[:260])}",
                f"- Warning: {fmt('; '.join(output.get('warnings', [])) if output.get('warnings') else None)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
