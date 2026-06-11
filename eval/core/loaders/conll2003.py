from __future__ import annotations

import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Span = Tuple[int, int, str]
DatasetDoc = Tuple[str, str, List[Span]]

_ORIGINAL_CONLL_URL = "https://data.deepai.org/conll2003.zip"
_CLEANCONLL_RAW_BASE = "https://raw.githubusercontent.com/flairNLP/CleanCoNLL/main/data"

_ORIGINAL_SPLIT_FILES = {
    "train": "train.txt",
    "dev": "valid.txt",
    "test": "test.txt",
}

_CLEANCONLL_ANNOTATION_FILES = {
    "train": "cleanconll_annotations.train",
    "dev": "cleanconll_annotations.dev",
    "test": "cleanconll_annotations.test",
}

_CLEANCONLL_PATCH_FILES = {
    "train": "train_tokens.patch",
    "dev": "dev_tokens.patch",
    "test": "test_tokens.patch",
}

_PATCH_HEADER_RE = re.compile(
    r"^(?P<old_start>\d+)(?:,(?P<old_end>\d+))?"
    r"(?P<op>[acd])"
    r"(?P<new_start>\d+)(?:,(?P<new_end>\d+))?$"
)


@dataclass(frozen=True)
class _PatchCommand:
    old_start: int
    old_end: int
    op: str
    new_lines: List[str]
    old_lines: List[str]


def get_conll2003_variant(default: str = "clean") -> str:
    raw = os.getenv("CONLL2003_VARIANT", default).strip().lower()
    if raw in {"clean", "cleanconll"}:
        return "clean"
    if raw in {"original", "orig", "legacy"}:
        return "original"
    raise ValueError(
        f"Unsupported CONLL2003_VARIANT={raw!r}. Expected 'clean' or 'original'."
    )


def get_conll2003_dataset_name(split: str = "test") -> str:
    norm_split = _normalize_split(split)
    prefix = "cleanconll" if get_conll2003_variant() == "clean" else "conll2003"
    return f"{prefix}/{norm_split}"


def build_docs_from_conll2003(
    limit: Optional[int] = None,
    split: str = "test",
    *,
    variant: Optional[str] = None,
) -> List[DatasetDoc]:
    norm_split = _normalize_split(split)
    effective_variant = (variant or get_conll2003_variant()).strip().lower()

    base_cache = Path(__file__).resolve().parent / "datasets"
    if effective_variant == "clean":
        file_path = _ensure_cleanconll_split(base_cache / "cleanconll_cache", norm_split)
        ner_col = 4
    elif effective_variant == "original":
        original_files = _ensure_original_conll(base_cache / "conll2003_cache")
        file_path = original_files[norm_split]
        ner_col = 3
    else:
        raise ValueError(
            f"Unsupported variant={effective_variant!r}. Expected 'clean' or 'original'."
        )

    return _load_docs_from_conll_file(file_path, ner_col=ner_col, limit=limit)


def _normalize_split(split: str) -> str:
    s = split.strip().lower()
    if s in {"valid", "val", "dev"}:
        return "dev"
    if s in {"train", "test"}:
        return s
    raise ValueError(f"Unsupported split={split!r}. Expected train/dev/test.")


def _download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)


def _ensure_original_conll(cache_dir: Path) -> Dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "conll2003.zip"

    if not zip_path.exists():
        print(f"Downloading original CoNLL-2003 from {_ORIGINAL_CONLL_URL}...")
        _download_file(_ORIGINAL_CONLL_URL, zip_path)

    expected_files = {split: cache_dir / filename for split, filename in _ORIGINAL_SPLIT_FILES.items()}
    if not all(path.exists() for path in expected_files.values()):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)

    return expected_files


def _ensure_cleanconll_split(cache_dir: Path, split: str) -> Path:
    output_dir = cache_dir / "cleanconll"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cleanconll.{split}"

    if output_path.exists():
        return output_path

    original_files = _ensure_original_conll(cache_dir / "conll03")
    assets = _ensure_cleanconll_assets(cache_dir)

    tokens_path = cache_dir / "tokens_updated" / f"{split}_tokens_updated.txt"
    tokens_path.parent.mkdir(parents=True, exist_ok=True)

    original_token_lines = _extract_first_column_lines(original_files[split])
    patch_commands = _parse_normal_diff_patch(assets["patches"][split].read_text(encoding="utf-8").splitlines())
    updated_tokens = _apply_normal_diff_patch(original_token_lines, patch_commands)
    tokens_path.write_text("\n".join(updated_tokens) + "\n", encoding="utf-8")

    annotation_lines = assets["annotations"][split].read_text(encoding="utf-8").splitlines()
    merged_lines = _merge_cleanconll_tokens_and_annotations(updated_tokens, annotation_lines)
    output_path.write_text("\n".join(merged_lines) + "\n", encoding="utf-8")
    return output_path


def _ensure_cleanconll_assets(cache_dir: Path) -> Dict[str, Dict[str, Path]]:
    annotations_dir = cache_dir / "cleanconll_annotations"
    patches_dir = cache_dir / "patch_files"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    annotation_paths: Dict[str, Path] = {}
    patch_paths: Dict[str, Path] = {}

    for split, filename in _CLEANCONLL_ANNOTATION_FILES.items():
        dest = annotations_dir / filename
        if not dest.exists():
            url = f"{_CLEANCONLL_RAW_BASE}/cleanconll_annotations/{filename}"
            print(f"Downloading CleanCoNLL annotations from {url}...")
            _download_file(url, dest)
        annotation_paths[split] = dest

    for split, filename in _CLEANCONLL_PATCH_FILES.items():
        dest = patches_dir / filename
        if not dest.exists():
            url = f"{_CLEANCONLL_RAW_BASE}/patch_files/{filename}"
            print(f"Downloading CleanCoNLL patch file from {url}...")
            _download_file(url, dest)
        patch_paths[split] = dest

    return {"annotations": annotation_paths, "patches": patch_paths}


def _extract_first_column_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                lines.append("")
                continue
            lines.append(line.split()[0])
    return lines


def _parse_normal_diff_patch(lines: Iterable[str]) -> List[_PatchCommand]:
    materialized = list(lines)
    commands: List[_PatchCommand] = []
    idx = 0

    while idx < len(materialized):
        header = materialized[idx].strip()
        idx += 1
        if not header:
            continue

        match = _PATCH_HEADER_RE.match(header)
        if not match:
            raise ValueError(f"Unsupported patch header: {header!r}")

        old_start = int(match.group("old_start"))
        old_end = int(match.group("old_end") or match.group("old_start"))
        op = match.group("op")

        old_lines: List[str] = []
        new_lines: List[str] = []

        if op in {"c", "d"}:
            while idx < len(materialized) and materialized[idx].startswith("<"):
                old_lines.append(_strip_patch_prefix(materialized[idx]))
                idx += 1

        if op == "c":
            if idx >= len(materialized) or materialized[idx].strip() != "---":
                raise ValueError(f"Missing '---' separator after change header: {header!r}")
            idx += 1

        if op in {"c", "a"}:
            while idx < len(materialized) and materialized[idx].startswith(">"):
                new_lines.append(_strip_patch_prefix(materialized[idx]))
                idx += 1

        commands.append(
            _PatchCommand(
                old_start=old_start,
                old_end=old_end,
                op=op,
                new_lines=new_lines,
                old_lines=old_lines,
            )
        )

    return commands


def _strip_patch_prefix(line: str) -> str:
    if len(line) >= 2 and line[1] == " ":
        return line[2:]
    return line[1:] if line else ""


def _apply_normal_diff_patch(original_lines: List[str], commands: List[_PatchCommand]) -> List[str]:
    result: List[str] = []
    cursor = 1

    for command in commands:
        if command.op == "a":
            result.extend(original_lines[max(cursor - 1, 0) : command.old_start])
            result.extend(command.new_lines)
            cursor = command.old_start + 1
            continue

        if command.old_start > 0:
            original_slice = original_lines[command.old_start - 1 : command.old_end]
            if command.old_lines and original_slice != command.old_lines:
                raise ValueError(
                    "Patch precondition failed: source lines do not match expected diff payload."
                )

        result.extend(original_lines[max(cursor - 1, 0) : max(command.old_start - 1, 0)])

        if command.op == "c":
            result.extend(command.new_lines)
        elif command.op != "d":
            raise ValueError(f"Unsupported patch operation: {command.op!r}")

        cursor = command.old_end + 1

    result.extend(original_lines[max(cursor - 1, 0) :])
    return result


def _merge_cleanconll_tokens_and_annotations(
    token_lines: List[str],
    annotation_lines: List[str],
) -> List[str]:
    if len(token_lines) != len(annotation_lines):
        raise ValueError(
            f"Token/annotation line count mismatch: {len(token_lines)} != {len(annotation_lines)}"
        )

    merged: List[str] = []
    for token, annotation_line in zip(token_lines, annotation_lines):
        if not token and not annotation_line:
            merged.append("")
            continue

        if not annotation_line:
            merged.append(token)
            continue

        parts = annotation_line.split("\t")
        if len(parts) < 5:
            raise ValueError(f"Unexpected CleanCoNLL annotation line: {annotation_line!r}")

        merged.append("\t".join([token, *parts[1:]]))

    return merged


def _load_docs_from_conll_file(file_path: Path, *, ner_col: int, limit: Optional[int]) -> List[DatasetDoc]:
    docs: List[DatasetDoc] = []
    tokens: List[str] = []
    ner_tags: List[str] = []
    guid = 0

    with open(file_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                if tokens:
                    docs.append(_convert_to_doc(str(guid), tokens, ner_tags))
                    guid += 1
                    if limit is not None and guid >= limit:
                        return docs
                    tokens = []
                    ner_tags = []
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            token = parts[0]

            if token == "-DOCSTART-":
                if tokens:
                    docs.append(_convert_to_doc(str(guid), tokens, ner_tags))
                    guid += 1
                    if limit is not None and guid >= limit:
                        return docs
                    tokens = []
                    ner_tags = []
                continue

            if len(parts) <= ner_col:
                raise ValueError(
                    f"Expected at least {ner_col + 1} columns in {file_path}, got: {line!r}"
                )

            tokens.append(token)
            ner_tags.append(parts[ner_col].strip())

    if tokens and (limit is None or guid < limit):
        docs.append(_convert_to_doc(str(guid), tokens, ner_tags))

    return docs


def _convert_to_doc(doc_id: str, tokens: List[str], ner_tags: List[str]) -> DatasetDoc:
    text = ""
    spans: List[Span] = []

    current_idx = 0
    current_entity: Optional[str] = None
    entity_start = 0

    for token, ner_tag in zip(tokens, ner_tags):
        token_len = len(token)

        if ner_tag != "O":
            tag_type = ner_tag[2:]
            prefix = ner_tag[:2]

            if prefix == "B-":
                if current_entity is not None:
                    spans.append((entity_start, current_idx - 1, current_entity))
                current_entity = tag_type
                entity_start = current_idx
            elif prefix == "I-":
                if current_entity != tag_type:
                    if current_entity is not None:
                        spans.append((entity_start, current_idx - 1, current_entity))
                    current_entity = tag_type
                    entity_start = current_idx
        else:
            if current_entity is not None:
                spans.append((entity_start, current_idx - 1, current_entity))
                current_entity = None

        text += token + " "
        current_idx += token_len + 1

    if current_entity is not None:
        spans.append((entity_start, current_idx - 1, current_entity))

    return doc_id, text[:-1], spans
