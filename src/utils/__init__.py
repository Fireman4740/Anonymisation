"""Lazy exports for utility helpers to avoid circular imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "is_valid_email",
    "is_valid_phone",
    "is_valid_iban",
    "is_valid_nir",
    "find_all_regex",
    "deduplicate_spans",
    "merge_overlapping_spans",
    "PseudoMapper",
    "WHITELIST_WORDS",
]

_PERSONAL_INFO_ATTRS = {
    "is_valid_email",
    "is_valid_phone",
    "is_valid_iban",
    "is_valid_nir",
}

_TEXT_SANITIZER_ATTRS = {
    "find_all_regex",
    "deduplicate_spans",
    "merge_overlapping_spans",
}


def __getattr__(name: str) -> Any:
    if name in _PERSONAL_INFO_ATTRS:
        module = import_module(".personal_info", __name__)
        return getattr(module, name)
    if name in _TEXT_SANITIZER_ATTRS:
        module = import_module(".text_sanitizer", __name__)
        return getattr(module, name)
    if name == "PseudoMapper":
        module = import_module(".utils_pseudo", __name__)
        return getattr(module, name)
    if name == "WHITELIST_WORDS":
        module = import_module(".whitelist_words", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
