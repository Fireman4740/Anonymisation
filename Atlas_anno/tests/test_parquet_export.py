from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from atlas_anno.annotation.preannotator import run_preannotation_command
from atlas_anno.anonymization.baselines import run_anonymization_command
from atlas_anno.export.parquet_export import export_parquet_batch
from atlas_anno.generation.pipeline import run_generate_dataset_command


class ParquetExportTest(unittest.TestCase):
    def test_export_parquet_uses_writer_backend(self) -> None:
        run_generate_dataset_command(100, "disabled")
        run_preannotation_command("hybrid-llm")
        run_anonymization_command("masking")

        def fake_writer(rows, path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"rows={len(rows)}", encoding="utf-8")
            return path

        with patch("atlas_anno.export.parquet_export._pyarrow_writer", side_effect=fake_writer):
            exported = export_parquet_batch("pilot_100")
        self.assertIn("worlds", exported)
        self.assertTrue(Path(exported["worlds"]).exists())

