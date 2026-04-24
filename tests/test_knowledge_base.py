from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np

from esgreveal.chunking import DocumentChunk
from esgreveal.document import DocumentBlock
from esgreveal.knowledge_base import _build_kpi_term_index, _make_table_records, build_knowledge_base
from esgreveal.metadata import default_metadata


class KnowledgeBaseTests(unittest.TestCase):
    def test_build_knowledge_base_writes_component_files(self) -> None:
        blocks = [
            DocumentBlock(page_num=1, order=0, kind="heading", text="1. Environment", section="1. Environment"),
            DocumentBlock(
                page_num=1,
                order=1,
                kind="paragraph",
                text="Total energy consumption was 101,588 MWh in 2025.",
                section="1. Environment",
                section_level=1,
                outline_path=("1. Environment",),
            ),
            DocumentBlock(
                page_num=2,
                order=0,
                kind="table",
                text="Metric | Value\nEnergy consumption | 101,588 MWh",
                section="1. Environment",
                section_level=1,
                outline_path=("1. Environment",),
                table_index=1,
            ),
        ]
        chunks = [
            DocumentChunk(
                page_start=1,
                page_end=1,
                section="1. Environment",
                section_level=1,
                outline_path=("1. Environment",),
                kind="paragraph",
                text="Total energy consumption was 101,588 MWh in 2025.",
                source_blocks=1,
            ),
            DocumentChunk(
                page_start=2,
                page_end=2,
                section="1. Environment",
                section_level=1,
                outline_path=("1. Environment",),
                kind="table",
                text="Metric | Value\nEnergy consumption | 101,588 MWh",
                source_blocks=1,
            ),
        ]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "dummy.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%test\n")
            output_dir = root / "kb"

            with (
                patch("esgreveal.knowledge_base.extract_document_blocks", return_value=blocks),
                patch("esgreveal.knowledge_base.build_chunks", return_value=chunks),
                patch("esgreveal.knowledge_base.T5Summarizer.summarize_many", return_value=["Energy use summary"]),
                patch(
                    "esgreveal.knowledge_base.M3EEmbedder.encode",
                    return_value=np.asarray(
                        [
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [0.5, 0.5],
                            [0.2, 0.8],
                        ],
                        dtype=np.float32,
                    ),
                ),
            ):
                build = build_knowledge_base(pdf_path, output_dir, use_faiss=False, summary_mode="extractive")

            self.assertEqual(build.record_count, 4)
            self.assertEqual(build.component_counts["textual_contents"], 1)
            self.assertEqual(build.component_counts["document_outlines"], 1)
            self.assertEqual(build.component_counts["table_contents"], 2)
            self.assertTrue((output_dir / "records.jsonl").exists())
            self.assertTrue((output_dir / "textual_contents.jsonl").exists())
            self.assertTrue((output_dir / "document_outlines.jsonl").exists())
            self.assertTrue((output_dir / "table_contents.jsonl").exists())
            self.assertTrue((output_dir / "text_vectors.npy").exists())
            self.assertIsNone(build.faiss_index_path)
            self.assertEqual(build.summary_mode, "extractive")

            outline_lines = (output_dir / "document_outlines.jsonl").read_text(encoding="utf-8").splitlines()
            outline_record = json.loads(outline_lines[0])
            self.assertEqual(outline_record["metadata"]["linked_count"], 3)

            table_lines = (output_dir / "table_contents.jsonl").read_text(encoding="utf-8").splitlines()
            table_records = [json.loads(line) for line in table_lines]
            keyword_records = [record for record in table_records if record["metadata"].get("mapping_type") == "table_keyword"]
            self.assertEqual(len(keyword_records), 1)
            self.assertEqual(keyword_records[0]["metadata"]["keyword"], "Energy consumption")

            all_records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            textual_record = next(record for record in all_records if record["component"] == "textual_contents")
            self.assertEqual(textual_record["summary"], "Energy use summary")
            self.assertIn("101,588 MWh", textual_record["vector_text"])

    def test_build_knowledge_base_resumes_from_pdf_checkpoints(self) -> None:
        blocks = [
            DocumentBlock(page_num=1, order=0, kind="heading", text="1. Environment", section="1. Environment"),
            DocumentBlock(
                page_num=1,
                order=1,
                kind="paragraph",
                text="Total energy consumption was 101,588 MWh in 2025.",
                section="1. Environment",
                section_level=1,
                outline_path=("1. Environment",),
            ),
        ]
        chunks = [
            DocumentChunk(
                page_start=1,
                page_end=1,
                section="1. Environment",
                section_level=1,
                outline_path=("1. Environment",),
                kind="paragraph",
                text="Total energy consumption was 101,588 MWh in 2025.",
                source_blocks=1,
            )
        ]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "dummy.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%test\n")
            output_dir = root / "kb"

            with (
                patch("esgreveal.knowledge_base.extract_document_blocks", return_value=blocks) as extract_mock,
                patch("esgreveal.knowledge_base.build_chunks", return_value=chunks),
                patch("esgreveal.knowledge_base.T5Summarizer.summarize_many", return_value=["Energy use summary"]),
                patch(
                    "esgreveal.knowledge_base.M3EEmbedder.encode",
                    return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ),
            ):
                first_build = build_knowledge_base(pdf_path, output_dir, use_faiss=False)

            checkpoint_files = list((output_dir / "_checkpoints").glob("*.json"))
            self.assertTrue(checkpoint_files)
            self.assertEqual(extract_mock.call_count, 1)
            self.assertEqual(first_build.record_count, 2)

            with (
                patch(
                    "esgreveal.knowledge_base.extract_document_blocks",
                    side_effect=AssertionError("checkpoint resume should skip PDF parsing"),
                ),
                patch("esgreveal.knowledge_base.build_chunks", return_value=chunks),
                patch("esgreveal.knowledge_base.T5Summarizer.summarize_many", return_value=["Energy use summary"]),
                patch(
                    "esgreveal.knowledge_base.M3EEmbedder.encode",
                    return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ),
            ):
                second_build = build_knowledge_base(pdf_path, output_dir, use_faiss=False)

            self.assertEqual(second_build.record_count, 2)
            self.assertEqual(second_build.resumed_source_count, 1)

    def test_strategy_like_table_rows_are_pruned(self) -> None:
        blocks = [
            DocumentBlock(
                page_num=12,
                order=0,
                kind="table",
                text="구분 | 단위 | 2025\n에너지 사용량 | MWh | 101,588\nRE100 목표 | % | 100\nScope 1 배출량 | tCO2e | 366",
                section="환경",
                section_level=1,
                outline_path=("환경", "에너지 및 배출"),
                table_index=1,
            )
        ]

        records = _make_table_records(Path("dryrun.pdf"), blocks, _build_kpi_term_index(default_metadata()))
        keyword_records = [record for record in records if record.metadata.get("mapping_type") == "table_keyword"]
        keywords = [record.metadata["keyword"] for record in keyword_records]

        self.assertIn("에너지 사용량", keywords)
        self.assertIn("Scope 1 배출량", keywords)
        self.assertNotIn("RE100 목표", keywords)


if __name__ == "__main__":
    unittest.main()
