from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from atlas_anno.annotation.preannotator import run_preannotation_command
from atlas_anno.generation.pipeline import run_generate_dataset_command
from atlas_anno.review.label_studio import export_label_studio_review_pack
from atlas_anno.review.label_studio_api import (
    LabelStudioAPI,
    LabelStudioSettings,
    create_project_from_batch,
    export_batch_annotations,
    import_batch_tasks,
)


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class LabelStudioAPITest(unittest.TestCase):
    def setUp(self) -> None:
        run_generate_dataset_command(100, "disabled", resume_enabled=False, cache_enabled=False)
        run_preannotation_command("disabled", reasoning_workers=1, resume_enabled=False, cache_enabled=False)
        export_label_studio_review_pack("pilot_100")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".env", delete=False) as handle:
            handle.write("LABEL_STUDIO_URL=http://127.0.0.1:8080\n")
            handle.write("LABEL_STUDIO_API_TOKEN=test-token\n")
            handle.write("LABEL_STUDIO_TOKEN=\n")
            handle.write("LABEL_STUDIO_PROJECT_ID=\n")
            self.env_path = handle.name
        self.addCleanup(lambda: os.path.exists(self.env_path) and os.unlink(self.env_path))

    def test_create_project_uses_label_config(self) -> None:
        with patch("atlas_anno.review.label_studio_api.urllib.request.urlopen", return_value=_FakeResponse({"id": 42, "title": "Atlas"})):
            payload = create_project_from_batch("pilot_100", "Atlas pilot_100 review", env_path=self.env_path)
        self.assertEqual(payload["id"], 42)

    def test_import_tasks_posts_payload(self) -> None:
        with patch("atlas_anno.review.label_studio_api.urllib.request.urlopen", return_value=_FakeResponse({"task_count": 100})):
            payload = import_batch_tasks("pilot_100", "99", env_path=self.env_path)
        self.assertEqual(payload["task_count"], 100)

    def test_export_annotations_writes_output(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            output_path = handle.name
        with patch("atlas_anno.review.label_studio_api.urllib.request.urlopen", return_value=_FakeResponse([{"id": "doc_000001"}])):
            exported_path = export_batch_annotations("pilot_100", "99", output_path, env_path=self.env_path)
        with open(exported_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload[0]["id"], "doc_000001")


class LabelStudioAuthTest(unittest.TestCase):
    def test_personal_access_token_refreshes_before_api_call(self) -> None:
        requests = []

        def fake_urlopen(request, timeout=30):
            requests.append(request)
            if request.full_url.endswith("/api/token/refresh/"):
                return _FakeResponse({"access": "access-123"})
            return _FakeResponse({"id": 42})

        client = LabelStudioAPI(
            LabelStudioSettings(
                url="http://127.0.0.1:8080",
                api_token="",
                personal_access_token="refresh-token-123",
                project_id="",
            )
        )

        with patch("atlas_anno.review.label_studio_api.urllib.request.urlopen", side_effect=fake_urlopen):
            payload = client.create_or_update_project(title="Atlas", label_config="<View />")

        self.assertEqual(payload["id"], 42)
        self.assertEqual(requests[0].full_url, "http://127.0.0.1:8080/api/token/refresh/")
        self.assertEqual(requests[1].get_header("Authorization"), "Bearer access-123")
