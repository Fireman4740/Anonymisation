from __future__ import annotations

import unittest

from atlas_anno.reporting.builder import build_consolidated_report, build_html, build_markdown
from atlas_anno.storage import save_report


class ReportingTest(unittest.TestCase):
    def test_report_rendering(self) -> None:
        save_report("masking", "spans", {"summary": {"f1": 1.0}})
        save_report("masking", "privacy", {"summary": {"privacy_score": 0.8}})
        save_report("masking", "reid", {"summary": {"top1": 0.2}})
        save_report("masking", "utility", {"summary": {"utility_score": 0.7}})
        report = build_consolidated_report("masking")
        markdown = build_markdown("masking", report)
        html = build_html("masking", report)
        self.assertIn("privacy_score", markdown)
        self.assertIn("<html>", html)

