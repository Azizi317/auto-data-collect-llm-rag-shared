from __future__ import annotations

import unittest

from esgreveal.chunking import DocumentChunk
from esgreveal.extraction import heuristic_extract
from esgreveal.metadata import default_metadata
from esgreveal.retrieval import RankedChunk


class TableSemanticExtractionTests(unittest.TestCase):
    def _kpi(self, kpi_id: str):
        return next(item for item in default_metadata() if item.id == kpi_id)

    def test_energy_kpi_prefers_actual_quantity_over_re100_target(self) -> None:
        energy_kpi = self._kpi("K-ENV-N04")
        ranked = [
            RankedChunk(
                chunk=DocumentChunk(
                    page_start=3,
                    page_end=3,
                    section="Environment",
                    kind="table",
                    text=(
                        "Item | 2024 | Unit\n"
                        "Energy consumption | 101,588 | MWh\n"
                        "Renewable energy ratio | 100% | RE100 target"
                    ),
                ),
                score=8.5,
                matched_terms=("Energy consumption", "MWh"),
            )
        ]

        result = heuristic_extract(energy_kpi, ranked)
        self.assertTrue(result.disclosed)
        self.assertEqual(result.value, "101,588")
        self.assertEqual(result.unit, "MWh")
        self.assertEqual(result.source_page, 3)
        self.assertIn("Energy consumption", result.evidence or "")

    def test_table_expression_prefers_latest_year(self) -> None:
        scope2_kpi = self._kpi("K-ENV-N02")
        ranked = [
            RankedChunk(
                chunk=DocumentChunk(
                    page_start=8,
                    page_end=8,
                    section="Environment",
                    kind="table",
                    text=(
                        "Category | 2022 | 2023 | 2024 | Unit\n"
                        "Scope 2 emissions | 98,120 | 99,001 | 101,588 | tCO2e"
                    ),
                ),
                score=9.1,
                matched_terms=("Scope 2", "tCO2e"),
            )
        ]

        result = heuristic_extract(scope2_kpi, ranked)
        self.assertEqual(result.value, "101,588")
        self.assertEqual(result.unit, "tCO2e")
        self.assertIn("2024", result.evidence or "")

    def test_scope1_rejects_scope12_total(self) -> None:
        scope1_kpi = self._kpi("K-ENV-N01")
        ranked = [
            RankedChunk(
                chunk=DocumentChunk(
                    page_start=11,
                    page_end=11,
                    section="Environment",
                    kind="table",
                    text=(
                        "Category | 2024 | Unit\n"
                        "Scope 1+2 total emissions | 184,807 | tCO2e\n"
                        "Scope 1 emissions | 42,315 | tCO2e"
                    ),
                ),
                score=9.0,
                matched_terms=("Scope 1", "tCO2e"),
            )
        ]

        result = heuristic_extract(scope1_kpi, ranked)
        self.assertEqual(result.value, "42,315")
        self.assertIn("Scope 1 emissions", result.evidence or "")

    def test_table_expression_handles_multiline_scope_header(self) -> None:
        scope2_kpi = self._kpi("K-ENV-N02")
        ranked = [
            RankedChunk(
                chunk=DocumentChunk(
                    page_start=10,
                    page_end=10,
                    section="Environment",
                    kind="table",
                    text=(
                        "Category | 2023 | 2024\n"
                        " | Scope 1 | Scope 2\n"
                        "Domestic emissions | 11,111 | 22,222\n"
                        "Unit | tCO2e | tCO2e"
                    ),
                ),
                score=8.8,
                matched_terms=("Scope 2", "tCO2e"),
            )
        ]

        result = heuristic_extract(scope2_kpi, ranked)
        self.assertEqual(result.value, "22,222")
        self.assertEqual(result.unit, "tCO2e")


if __name__ == "__main__":
    unittest.main()
