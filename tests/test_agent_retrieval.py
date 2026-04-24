from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from esgreveal.agent_retrieval import (
    KnowledgeVectorBase,
    _should_use_modelscope_reranker,
    build_query_text,
    rank_knowledge_base,
)
from esgreveal.chunking import DocumentChunk
from esgreveal.document import DocumentBlock, DocumentSummary
from esgreveal.knowledge_base import KnowledgeBaseRecord
from esgreveal.metadata import default_metadata
from esgreveal.pipeline import results_to_json, run_pipeline


class _StubEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = np.asarray([vector], dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vector


class AgentRetrievalTests(unittest.TestCase):
    def test_query_text_contains_metadata_terms(self) -> None:
        kpi = default_metadata()[0]
        text = build_query_text(kpi)
        self.assertIn("scope", text)
        self.assertIn(normalized := "tco2e", text)
        self.assertTrue(normalized)

    def test_korean_reranker_uses_cross_encoder_path(self) -> None:
        self.assertFalse(_should_use_modelscope_reranker("dragonkue/bge-reranker-v2-m3-ko"))
        self.assertTrue(_should_use_modelscope_reranker("iic/nlp_corom_passage-ranking_chinese-base-ecom"))

    def test_rank_knowledge_base_prefers_matching_source_and_table_keyword(self) -> None:
        kpi = default_metadata()[0]
        records = (
            KnowledgeBaseRecord(
                record_id="r1",
                component="table_contents",
                source_path=str(Path("demo.pdf")),
                page_start=7,
                page_end=7,
                section="환경",
                section_level=1,
                outline_path=("환경", "배출"),
                text="Scope 1 배출량 | tCO2e | 366",
                vector_text="Scope 1 배출량 | 환경 | 환경 > 배출",
                metadata={"mapping_type": "table_keyword", "keyword": "Scope 1 배출량", "ontology_kpi_ids": [kpi.id]},
            ),
            KnowledgeBaseRecord(
                record_id="r2",
                component="textual_contents",
                source_path=str(Path("other.pdf")),
                page_start=9,
                page_end=9,
                section="환경",
                section_level=1,
                outline_path=("환경",),
                text="The company disclosed energy plans.",
                vector_text="energy plans",
                metadata={},
            ),
        )
        vectors = np.asarray([[1.0, 0.0], [0.1, 0.9]], dtype=np.float32)
        bundle = KnowledgeVectorBase(root_dir="kb", records=records, vectors=vectors)

        with (
            patch("esgreveal.agent_retrieval.load_knowledge_vector_base", return_value=bundle),
            patch("esgreveal.agent_retrieval._query_embedder", return_value=_StubEmbedder([1.0, 0.0])),
            patch("esgreveal.agent_retrieval._corom_model", return_value=None),
        ):
            ranked = rank_knowledge_base("kb", kpi, top_k=1, initial_k=2, source_path="demo.pdf")

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].chunk.page_start, 7)
        self.assertEqual(ranked[0].chunk.kind, "table")

    def test_pipeline_uses_knowledge_base_when_available(self) -> None:
        kpi = default_metadata()[0]
        with TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir)
            (kb_dir / "records.jsonl").write_text("", encoding="utf-8")
            np.save(kb_dir / "text_vectors.npy", np.zeros((0, 0), dtype=np.float32))

            blocks = [
                DocumentBlock(page_num=1, order=0, kind="heading", text="환경", section="환경"),
                DocumentBlock(page_num=1, order=1, kind="paragraph", text="Scope 1 배출량은 366 tCO2e 입니다.", section="환경"),
            ]
            chunks = [
                DocumentChunk(
                    page_start=1,
                    page_end=1,
                    section="환경",
                    section_level=1,
                    outline_path=("환경",),
                    kind="paragraph",
                    text="Scope 1 배출량은 366 tCO2e 입니다.",
                    source_blocks=1,
                )
            ]
            kb_ranked = [
                type(
                    "RankedLike",
                    (),
                    {
                        "chunk": chunks[0],
                        "score": 9.5,
                        "matched_terms": ("scope 1", "배출량"),
                    },
                )()
            ]

            with (
                patch("esgreveal.pipeline.load_kpis", return_value=[kpi]),
                patch("esgreveal.pipeline.extract_document_blocks", return_value=blocks),
                patch("esgreveal.pipeline.build_chunks", return_value=chunks),
                patch(
                    "esgreveal.pipeline.summarize_blocks",
                    return_value=DocumentSummary(
                        page_count=1,
                        block_count=2,
                        heading_count=1,
                        table_count=0,
                        section_count=1,
                        sections=("환경",),
                        outline_count=1,
                        document_kind="esg_report",
                        ocr_page_count=0,
                        extraction_mode="text",
                    ),
                ),
                patch("esgreveal.pipeline.rank_knowledge_base", return_value=kb_ranked),
                patch("esgreveal.pipeline.load_llm_config_from_env", return_value=None),
            ):
                run = run_pipeline("demo.pdf", kb_dir=kb_dir, top_k=1)

        self.assertEqual(run.document_summary["retrieval_backend"], "knowledge_vector_base")
        self.assertEqual(run.results[0].retrieval_backend, "knowledge_vector_base")
        payload = results_to_json(run)
        self.assertIn('"retrieval_backend": "knowledge_vector_base"', payload)


if __name__ == "__main__":
    unittest.main()
