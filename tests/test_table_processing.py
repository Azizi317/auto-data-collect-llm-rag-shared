from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import fitz

from esgreveal.chunking import build_chunks
from esgreveal.document import extract_document_blocks


class TableProcessingTests(unittest.TestCase):
    def _make_table_pdf(self) -> Path:
        tmp_dir = Path(tempfile.gettempdir())
        pdf_path = tmp_dir / "esgreveal_table_test.pdf"

        doc = fitz.open()
        page = doc.new_page(width=420, height=560)
        page.insert_text((50, 30), "Energy consumption")

        tables = [
            (60, "Scope 1", "366 tCO2e"),
            (260, "Scope 2", "101,588 tCO2e"),
        ]
        for top, label, value in tables:
            for x in (50, 180, 360):
                page.draw_line((x, top), (x, top + 80), color=(0, 0, 0), width=1)
            for y in (top, top + 40, top + 80):
                page.draw_line((50, y), (360, y), color=(0, 0, 0), width=1)
            page.insert_text((60, top + 22), label)
            page.insert_text((190, top + 22), value)

        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_table_blocks_are_extracted(self) -> None:
        pdf_path = self._make_table_pdf()
        blocks = extract_document_blocks(pdf_path)
        table_blocks = [block for block in blocks if block.kind == "table" and block.source == "table"]
        self.assertGreaterEqual(len(table_blocks), 2)
        self.assertTrue(any("366" in block.text for block in table_blocks))
        self.assertTrue(any("101,588" in block.text for block in table_blocks))

    def test_table_blocks_remain_separate_chunks(self) -> None:
        pdf_path = self._make_table_pdf()
        blocks = extract_document_blocks(pdf_path)
        chunks = build_chunks(blocks)
        table_chunks = [chunk for chunk in chunks if chunk.kind == "table"]
        self.assertGreaterEqual(len(table_chunks), 2)
        self.assertTrue(any("Scope 1" in chunk.text for chunk in table_chunks))
        self.assertTrue(any("Scope 2" in chunk.text for chunk in table_chunks))


if __name__ == "__main__":
    unittest.main()
