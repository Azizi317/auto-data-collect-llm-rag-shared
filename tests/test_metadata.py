from __future__ import annotations

from collections import Counter
import unittest

from esgreveal.metadata import default_metadata


class MetadataTests(unittest.TestCase):
    def test_metadata_uses_paper_terms(self) -> None:
        items = default_metadata()
        self.assertEqual(len(items), 70)

        first = items[0]
        self.assertTrue(first.aspect_ko)
        self.assertTrue(first.aspect_en)
        self.assertTrue(first.kpi_ko)
        self.assertTrue(first.kpi_en)
        self.assertTrue(first.topic_ko)
        self.assertTrue(first.topic_en)
        self.assertTrue(first.quantity_ko)
        self.assertTrue(first.quantity_en)
        self.assertTrue(first.expression_template.name)

    def test_all_items_have_extensions_and_expression_templates(self) -> None:
        for item in default_metadata():
            with self.subTest(item=item.id):
                self.assertTrue(item.standard_family)
                self.assertGreaterEqual(len(item.standard_refs), 1)
                self.assertGreaterEqual(len(item.search_terms), 1)
                self.assertGreaterEqual(len(item.knowledge), 1)
                self.assertGreaterEqual(len(item.output_fields), 1)
                self.assertTrue(item.k_esg_guideline)
                self.assertIsNotNone(item.k_esg_alignment)
                self.assertTrue(item.expression_template.value_mode)
                self.assertGreaterEqual(len(item.expression_template.output_fields), 1)

    def test_indicator_distribution_matches_paper_summary(self) -> None:
        items = default_metadata()
        numeric = [item for item in items if item.disclosure_type == "numeric"]
        textual = [item for item in items if item.disclosure_type == "text"]
        self.assertEqual(len(numeric), 34)
        self.assertEqual(len(textual), 36)

        by_category_numeric = Counter(item.category for item in numeric)
        by_category_textual = Counter(item.category for item in textual)
        self.assertEqual(by_category_numeric, Counter({"E": 12, "S": 18, "G": 4}))
        self.assertEqual(by_category_textual, Counter({"E": 14, "S": 15, "G": 7}))


if __name__ == "__main__":
    unittest.main()
