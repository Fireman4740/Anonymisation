from __future__ import annotations

import unittest
from pathlib import Path

from atlas_anno.dashboard.analysis import load_dashboard_data, summarize_dashboard_data


class DashboardAnalysisTest(unittest.TestCase):
    def test_loads_current_pilot_artifacts(self) -> None:
        data = load_dashboard_data(Path("data"), batch="pilot_100", strategy="masking")

        self.assertEqual(len(data.raw_docs), 100)
        self.assertEqual(len(data.annotations), 100)
        self.assertEqual(len(data.anonymized), 100)
        self.assertEqual(len(data.attack_pairs), 300)
        self.assertEqual(len(data.attacks_structured), 300)
        self.assertEqual(len(data.attacks_llm), 300)
        self.assertFalse(data.missing_files)

    def test_summarizes_quality_and_report_metrics(self) -> None:
        data = load_dashboard_data(Path("data"), batch="pilot_100", strategy="masking")
        summary = summarize_dashboard_data(data)

        self.assertEqual(summary.coverage["raw_docs"], 100)
        self.assertEqual(summary.coverage["attack_pairs"], 300)
        self.assertEqual(summary.duplicates["attack_pairs"], 0)
        self.assertEqual(summary.duplicates["attacks_structured_pair"], 0)
        self.assertEqual(summary.duplicates["attacks_llm_pair"], 0)
        self.assertEqual(summary.report_metrics["privacy_score"], 0.856)
        self.assertEqual(summary.report_metrics["reid_top1"], 0.4233)
        self.assertEqual(summary.report_metrics["span_f1"], 0.0)
        self.assertEqual(summary.report_metrics["self_bleu"], 0.9229)
        self.assertEqual(summary.report_metrics["distinct_2"], 0.1419)

    def test_summarizes_linguistic_diversity_factors(self) -> None:
        data = load_dashboard_data(Path("data"), batch="pilot_100", strategy="masking")
        summary = summarize_dashboard_data(data)

        self.assertEqual(summary.factor_counts["register"]["courant"], 80)
        self.assertEqual(summary.factor_counts["register"]["familier"], 15)
        self.assertEqual(summary.factor_counts["register"]["soutenu"], 5)
        self.assertEqual(summary.factor_counts["address_form"]["vous"], 82)
        self.assertEqual(summary.factor_counts["address_form"]["tu"], 18)
        self.assertEqual(summary.linguistic_flags["self_bleu_collapse"], True)
        self.assertEqual(summary.linguistic_flags["distinct_2_low"], True)
        self.assertEqual(summary.linguistic_flags["cell_coverage_low"], True)

    def test_document_rows_include_drilldown_fields(self) -> None:
        data = load_dashboard_data(Path("data"), batch="pilot_100", strategy="masking")
        summary = summarize_dashboard_data(data)
        first = summary.document_rows[0]

        self.assertIn("doc_id", first)
        self.assertIn("text", first)
        self.assertIn("anonymized_text", first)
        self.assertIn("span_count", first)
        self.assertIn("attack_top1_success", first)
        self.assertIn("register", first)
        self.assertIn("address_form", first)


if __name__ == "__main__":
    unittest.main()
