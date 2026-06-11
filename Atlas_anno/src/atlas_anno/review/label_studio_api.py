from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from atlas_anno.paths import project_root
from atlas_anno.review.label_studio import export_label_studio_review_pack, import_label_studio_review_pack
from atlas_anno.settings import parse_dotenv
from atlas_anno.storage import label_config_path, label_studio_export_path, label_studio_project_id_path, label_studio_tasks_path


@dataclass(frozen=True)
class LabelStudioSettings:
    url: str
    api_token: str
    personal_access_token: str
    project_id: str


def load_label_studio_settings(
    env_path: str | None = None,
    *,
    token_override: str | None = None,
    url_override: str | None = None,
) -> LabelStudioSettings:
    """Load Label Studio settings: .env file values, then os.environ overrides, then explicit overrides.

    Priority (highest to lowest): token_override/url_override > os.environ > .env file.
    """
    env_file = Path(env_path) if env_path else project_root() / ".env"
    env_values = parse_dotenv(env_file)

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name, env_values.get(name, default))

    return LabelStudioSettings(
        url=(url_override or get("LABEL_STUDIO_URL", "http://127.0.0.1:8080")).rstrip("/"),
        api_token=get("LABEL_STUDIO_API_TOKEN"),
        personal_access_token=token_override or get("LABEL_STUDIO_TOKEN"),
        project_id=get("LABEL_STUDIO_PROJECT_ID"),
    )


class LabelStudioAPI:
    def __init__(self, settings: LabelStudioSettings) -> None:
        self.settings = settings
        self._cached_access_token: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        headers: Dict[str, str] | None = None,
        retry_auth: bool = True,
    ) -> Any:
        url = f"{self.settings.url}{path}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request_headers = self._headers() if headers is None else headers
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and retry_auth and headers is None and self.settings.personal_access_token:
                self._cached_access_token = None
                return self._request(method, path, payload=payload, headers=headers, retry_auth=False)
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Label Studio API error {exc.code} on {path}: {message}") from exc
        return json.loads(raw) if raw else {}

    def _get_access_token(self) -> str:
        if self._cached_access_token:
            return self._cached_access_token
        if not self.settings.personal_access_token:
            return ""
        payload = self._request(
            "POST",
            "/api/token/refresh/",
            {"refresh": self.settings.personal_access_token},
            headers={"Content-Type": "application/json"},
        )
        access = str(payload.get("access") or "")
        if not access:
            raise RuntimeError("Label Studio token refresh succeeded without returning an access token")
        self._cached_access_token = access
        return access

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.personal_access_token:
            headers["Authorization"] = f"Bearer {self._get_access_token()}"
        elif self.settings.api_token:
            headers["Authorization"] = f"Token {self.settings.api_token}"
        return headers

    def create_or_update_project(self, *, title: str, label_config: str, project_id: str | None = None, force_create: bool = False) -> Dict[str, Any]:
        """Create or update a project.
        
        Args:
            title: Project title
            label_config: Label configuration XML
            project_id: Project ID to update (None or empty uses settings.project_id, if available)
            force_create: If True, always create new project (POST) instead of updating
        """
        target_project = project_id or self.settings.project_id if not force_create else None
        payload = {"title": title, "label_config": label_config}
        if target_project and not force_create:
            return self._request("PATCH", f"/api/projects/{target_project}", payload)
        return self._request("POST", "/api/projects", payload)

    def list_projects(self) -> list[Dict[str, Any]]:
        result = self._request("GET", "/api/projects")
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        if isinstance(result, list):
            return result
        return []

    def import_tasks(self, *, project_id: str, tasks: list[dict[str, Any]]) -> Any:
        return self._request("POST", f"/api/projects/{project_id}/import", tasks)

    def export_annotations(self, *, project_id: str, export_type: str = "JSON") -> Any:
        query = urllib.parse.urlencode({"exportType": export_type})
        return self._request("GET", f"/api/projects/{project_id}/export?{query}")


def _resolve_or_create_project_id(batch: str, title: str, api: LabelStudioAPI) -> str:
    """Return the Label Studio project ID for *batch*, creating it if needed.

    Resolution order:
    1. Cache file ``data/review/{batch}/.label_studio_project_id``
    2. API search — first project whose title contains *batch*
    3. Auto-create using the batch label config and *title*
    """
    cache = label_studio_project_id_path(batch)
    if cache.exists():
        cached = cache.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    for project in api.list_projects():
        if batch in str(project.get("title", "")):
            pid = str(project["id"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(pid, encoding="utf-8")
            return pid

    label_config = label_config_path(batch).read_text(encoding="utf-8")
    result = api.create_or_update_project(title=title, label_config=label_config)
    pid = str(result["id"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(pid, encoding="utf-8")
    return pid


def create_project_from_batch(
    batch: str,
    title: str,
    env_path: str | None = None,
    project_id: str | None = None,
    *,
    token: str | None = None,
    url: str | None = None,
    force_create: bool = False,
) -> Dict[str, Any]:
    settings = load_label_studio_settings(env_path, token_override=token, url_override=url)
    label_config = label_config_path(batch).read_text(encoding="utf-8")
    api = LabelStudioAPI(settings)
    result = api.create_or_update_project(title=title, label_config=label_config, project_id=project_id, force_create=force_create)
    pid = str(result["id"])
    cache = label_studio_project_id_path(batch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(pid, encoding="utf-8")
    return result


def import_batch_tasks(
    batch: str,
    project_id: str | None = None,
    env_path: str | None = None,
    *,
    token: str | None = None,
    url: str | None = None,
) -> Any:
    settings = load_label_studio_settings(env_path, token_override=token, url_override=url)
    api = LabelStudioAPI(settings)
    if not project_id:
        project_id = _resolve_or_create_project_id(batch, f"Atlas {batch}", api)
    tasks = json.loads(label_studio_tasks_path(batch).read_text(encoding="utf-8"))
    return api.import_tasks(project_id=project_id, tasks=tasks)


def export_batch_annotations(
    batch: str,
    project_id: str | None = None,
    output_path: str | None = None,
    env_path: str | None = None,
    *,
    token: str | None = None,
    url: str | None = None,
) -> str:
    settings = load_label_studio_settings(env_path, token_override=token, url_override=url)
    api = LabelStudioAPI(settings)
    if not project_id:
        project_id = _resolve_or_create_project_id(batch, f"Atlas {batch}", api)
    payload = api.export_annotations(project_id=project_id)
    out = Path(output_path) if output_path else label_studio_export_path(batch)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def sync_batch_reviews(batch: str, input_path: str) -> Dict[str, object]:
    return import_label_studio_review_pack(batch, input_path)


def prepare_batch_for_label_studio(batch: str, selection: str = "all") -> Dict[str, str]:
    return export_label_studio_review_pack(batch, selection=selection)
