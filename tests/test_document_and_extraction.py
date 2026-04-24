from __future__ import annotations

import unittest

from esgreveal.chunking import build_chunks
from esgreveal.chunking import DocumentChunk
from esgreveal.document import DocumentBlock, summarize_blocks
from esgreveal.extraction import heuristic_extract
from esgreveal.metadata import KPIItem
from esgreveal.retrieval import RankedChunk


class DocumentAndExtractionTests(unittest.TestCase):
    def test_academic_paper_classification(self) -> None:
        blocks = [
            DocumentBlock(page_num=1, order=0, kind="heading", text="Abstract", source="pypdf"),
            DocumentBlock(page_num=1, order=1, kind="paragraph", text="We propose a method.", section="Abstract", source="pypdf"),
            DocumentBlock(page_num=2, order=0, kind="heading", text="Introduction", source="pypdf"),
            DocumentBlock(page_num=2, order=1, kind="paragraph", text="This paper studies ESGReveal.", section="Introduction", source="pypdf"),
            DocumentBlock(page_num=3, order=0, kind="heading", text="Methods", source="pypdf"),
            DocumentBlock(page_num=3, order=1, kind="paragraph", text="We evaluate the pipeline.", section="Methods", source="pypdf"),
        ]
        summary = summarize_blocks(blocks, page_count=3)
        self.assertEqual(summary.document_kind, "academic_paper")

    def test_esg_report_classification(self) -> None:
        blocks = [
            DocumentBlock(page_num=1, order=0, kind="heading", text="Sustainability Report", source="pymupdf"),
            DocumentBlock(page_num=1, order=1, kind="paragraph", text="Environment, social and governance performance overview.", section="Sustainability Report", source="pymupdf"),
            DocumentBlock(page_num=2, order=0, kind="heading", text="Environment", source="pymupdf"),
            DocumentBlock(page_num=2, order=1, kind="paragraph", text="Scope 1 emissions were 366 tCO2e.", section="Environment", source="pymupdf"),
        ]
        summary = summarize_blocks(blocks, page_count=2)
        self.assertEqual(summary.document_kind, "esg_report")

    def test_esg_report_with_academic_style_section_titles_stays_esg_report(self) -> None:
        blocks = [
            DocumentBlock(page_num=1, order=0, kind="heading", text="Sustainability Report 2025", source="pymupdf"),
            DocumentBlock(
                page_num=1,
                order=1,
                kind="paragraph",
                text="This sustainability report summarizes Scope 1 and Scope 2 emissions, energy, and waste performance.",
                section="Sustainability Report 2025",
                source="pymupdf",
            ),
            DocumentBlock(page_num=2, order=0, kind="heading", text="Introduction", source="pymupdf"),
            DocumentBlock(
                page_num=2,
                order=1,
                kind="paragraph",
                text="The report covers greenhouse gas emissions and energy management for 2024.",
                section="Introduction",
                source="pymupdf",
            ),
            DocumentBlock(page_num=3, order=0, kind="heading", text="Results", source="pymupdf"),
            DocumentBlock(
                page_num=3,
                order=1,
                kind="paragraph",
                text="Scope 1 emissions decreased and the GRI index is included in the appendix.",
                section="Results",
                source="pymupdf",
            ),
            DocumentBlock(page_num=4, order=0, kind="heading", text="Conclusion", source="pymupdf"),
            DocumentBlock(
                page_num=4,
                order=1,
                kind="paragraph",
                text="Environment and governance disclosures are summarized for the reporting period.",
                section="Conclusion",
                source="pymupdf",
            ),
        ]
        summary = summarize_blocks(blocks, page_count=4)
        self.assertEqual(summary.document_kind, "esg_report")

    def test_heuristic_ignores_year_like_numbers(self) -> None:
        item = KPIItem(
            id="TEST",
            category="E",
            standard_family="ISSB S2",
            standard_refs=("ISSB S2",),
            aspect_ko="환경",
            aspect_en="Environment",
            kpi_ko="Scope 1 배출량",
            kpi_en="Scope 1 greenhouse gas emissions",
            topic_ko="직접 배출",
            topic_en="Direct emissions",
            quantity_ko="절대값",
            quantity_en="Absolute value",
            unit="tCO2e",
            search_terms=("Scope 1", "배출", "온실가스"),
            knowledge=("연도보다 실제 수치를 우선한다.",),
        )
        chunk = RankedChunk(
            chunk=DocumentChunk(
                page_start=12,
                page_end=12,
                section="Environment",
                kind="table",
                text="Year | Item | Value | Unit\n2025 | Scope 1 emissions | 366 | tCO2e",
            ),
            score=1.0,
            matched_terms=("scope 1", "emissions"),
        )
        output = heuristic_extract(item, [chunk])
        self.assertEqual(output.value, "366")
        self.assertEqual(output.unit, "tCO2e")
        self.assertEqual(output.source_page, 12)

    def test_build_chunks_prefers_sentence_boundary_when_splitting(self) -> None:
        blocks = [
            DocumentBlock(
                page_num=1,
                order=0,
                kind="paragraph",
                text="First sentence ends here. Second sentence is also complete. Third sentence keeps going for a while.",
                section="Environment",
                section_level=1,
                outline_path=("Environment",),
            )
        ]
        chunks = build_chunks(blocks, max_chars=55, min_chars=10)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].text.endswith("."))
        self.assertTrue(chunks[1].text.startswith("Second") or chunks[1].text.startswith("Third"))


if __name__ == "__main__":
    unittest.main()
