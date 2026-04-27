from __future__ import annotations

import hashlib
from pathlib import Path

from atlas_anno.paths import project_root
from atlas_anno.schemas import PromptSpec


def prompt_path(prompt_name: str) -> Path:
    return project_root() / "prompts" / prompt_name / "system.txt"


def load_prompt_spec(prompt_name: str) -> PromptSpec:
    path = prompt_path(prompt_name)
    system_prompt = path.read_text(encoding="utf-8").strip()
    version = hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:12]
    return PromptSpec(
        prompt_name=prompt_name,
        system_prompt=system_prompt,
        version=version,
        path=str(path),
    )
