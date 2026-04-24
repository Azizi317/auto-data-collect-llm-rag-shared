from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

from esgreveal.llm import LLMCallError, LLMConfig, normalize_llm_output
from esgreveal.metadata import default_metadata
from esgreveal.pipeline import KPIResult, PipelineRun, _make_output, results_to_json, results_to_markdown, run_pipeline
from esgreveal.retrieval import RankedChunk
from esgreveal.chunking import DocumentChunk


class OutputTests(unittest.TestCase):
    def test_json_and_markdown_are_utf8_friendly(self) -> None:
        item = default_metadata()[0]
        run = PipelineRun(
            source_path="demo.pdf",
            document_summary={
                "page_count": 1,
                "block_count": 1,
                "heading_count": 1,
                "table_count": 0,
                "section_count": 1,
                "sections": ["환경"],
                "document_kind": "esg_report",
                "ocr_page_count": 0,
                "extraction_mode": "text",
            },
            results=[
                KPIResult(
                    metadata=asdict(item),
                    prompt="질문",
                    retrieved_context=[],
                    retrieval_backend="knowledge_vector_base",
                    output={
                        "disclosed": True,
                        "value": "366",
                        "unit": "tCO2e",
                        "target": None,
                        "action": None,
                        "evidence": "Scope 1 배출량은 366 tCO2e입니다.",
                        "confidence": 0.93,
                        "source_page": 72,
                        "matched_terms": ["scope 1", "배출량"],
                    },
                )
            ],
        )
        payload = results_to_json(run)
        markdown = results_to_markdown(run)
        self.assertIn("환경", payload)
        self.assertNotIn("\\u", payload)
        self.assertIn("ESGReveal", markdown)
        self.assertIn("Quantity", markdown)
        self.assertIn("ExpressionTemplate", markdown)

    def test_table4_style_response_is_normalized(self) -> None:
        payload = {
            "disclosure": 1,
            "data": {
                "KPI": "에너지 사용량",
                "Topic": "직접 및 간접 에너지 소비",
                "Value": "101,588",
                "Unit": "MWh",
                "Target": None,
                "Action": None,
                "Evidence": "Energy consumption | 101,588 MWh",
                "SourcePage": 12,
                "MatchedTerms": ["에너지 사용량"],
            },
        }
        normalized = normalize_llm_output(payload)
        self.assertIs(normalized["disclosed"], True)
        self.assertEqual(normalized["disclosure_kpi"], "에너지 사용량")
        self.assertEqual(normalized["topic"], "직접 및 간접 에너지 소비")
        self.assertEqual(normalized["value"], "101,588")
        self.assertEqual(normalized["unit"], "MWh")

    def test_make_output_does_not_silently_fallback_when_llm_fails(self) -> None:
        item = default_metadata()[0]
        ranked = [
            RankedChunk(
                chunk=DocumentChunk(page_start=1, page_end=1, section="환경", kind="paragraph", text="테스트"),
                score=1.0,
                matched_terms=("scope 1",),
            )
        ]
        config = LLMConfig(backend="gemini_cli", model="gemini-3-pro-preview")
        with patch("esgreveal.pipeline.call_llm", side_effect=LLMCallError("gemini_cli", "gemini-3-pro-preview", "quota")):
            with self.assertRaises(LLMCallError):
                _make_output(config, "prompt", item, ranked)


class SmokeTests(unittest.TestCase):
    def test_real_korean_esg_reports_smoke(self) -> None:
        reports_dir = Path("reports/2025_kr")
        pdfs = sorted(reports_dir.glob("*.pdf"))
        self.assertGreaterEqual(len(pdfs), 5, "reports/2025_kr 폴더에 PDF가 충분하지 않습니다.")

        for pdf_path in pdfs[:5]:
            with self.subTest(pdf=pdf_path.name):
                run = run_pipeline(pdf_path, top_k=1)
                self.assertTrue(run.results)
                self.assertIn(run.document_summary["document_kind"], {"esg_report", "academic_paper"})
                first = run.results[0].output
                self.assertIsInstance(first, dict)
                self.assertIn("source_page", first)


if __name__ == "__main__":
    unittest.main()
