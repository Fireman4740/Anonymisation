from __future__ import annotations

import json
import tempfile
import unittest

from atlas_anno.generation.pipeline import run_generate_dataset_command
from atlas_anno.annotation.preannotator import run_preannotation_command
from atlas_anno.review.label_studio import export_label_studio_review_pack, import_label_studio_review_pack
from atlas_anno.storage import label_config_path, label_studio_tasks_path, load_batch_manifest, load_reviewed_documents, reviewed_annotations_path
from atlas_anno.io import read_json


class ReviewExportTest(unittest.TestCase):
    def test_label_studio_export_contains_predictions(self) -> None:
        run_generate_dataset_command(100, "disabled")
        run_preannotation_command("disabled", reasoning_workers=1, resume_enabled=False, cache_enabled=False)
        export_label_studio_review_pack("pilot_100")
        tasks = read_json(label_studio_tasks_path("pilot_100"))
        self.assertEqual(len(tasks), 100)
        self.assertIn("predictions", tasks[0])
        self.assertTrue(label_config_path("pilot_100").exists())

    def test_label_studio_import_creates_reviewed_annotations(self) -> None:
        run_generate_dataset_command(100, "disabled", resume_enabled=False, cache_enabled=False)
        run_preannotation_command("disabled", reasoning_workers=1, resume_enabled=False, cache_enabled=False)
        export_label_studio_review_pack("pilot_100")
        tasks = read_json(label_studio_tasks_path("pilot_100"))
        exported = []
        for task in tasks:
            exported.append(
                {
                    "id": task["id"],
                    "data": task["data"],
                    "annotations": [{"result": task["predictions"][0]["result"]}],
                }
            )

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            handle.write(json.dumps(exported, ensure_ascii=False, indent=2))
            export_path = handle.name

        import_label_studio_review_pack("pilot_100", export_path)
        reviewed = load_reviewed_documents()
        manifest = load_batch_manifest("pilot_100")
        self.assertEqual(len(reviewed), 100)
        self.assertTrue(reviewed_annotations_path().exists())
        self.assertEqual(reviewed[0].metadata.get("review_status"), "reviewed")
        self.assertIn("review_roundtrip", manifest)
