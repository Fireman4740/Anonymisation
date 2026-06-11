from __future__ import annotations

from eval.core.loaders.conll2003 import (
    _apply_normal_diff_patch,
    _merge_cleanconll_tokens_and_annotations,
    _parse_normal_diff_patch,
    get_conll2003_dataset_name,
    get_conll2003_variant,
)


def test_get_conll2003_variant_defaults_to_clean(monkeypatch):
    monkeypatch.delenv("CONLL2003_VARIANT", raising=False)
    assert get_conll2003_variant() == "clean"
    assert get_conll2003_dataset_name("valid") == "cleanconll/dev"


def test_get_conll2003_variant_accepts_original(monkeypatch):
    monkeypatch.setenv("CONLL2003_VARIANT", "original")
    assert get_conll2003_variant() == "original"
    assert get_conll2003_dataset_name("test") == "conll2003/test"


def test_apply_normal_diff_patch_handles_change_delete_and_add():
    original = ["SOCCER-WORLD", "", "TEAM", "FINAL"]
    patch_lines = [
        "1c1,3",
        "< SOCCER-WORLD",
        "---",
        "> SOCCER",
        "> -",
        "> WORLD",
        "2d3",
        "< ",
        "4a6",
        "> !",
    ]

    commands = _parse_normal_diff_patch(patch_lines)
    patched = _apply_normal_diff_patch(original, commands)

    assert patched == ["SOCCER", "-", "WORLD", "TEAM", "FINAL", "!"]


def test_merge_cleanconll_tokens_and_annotations_restores_token_column():
    tokens = ["SOCCER", "-", "WORLD", ""]
    annotations = [
        "[TOKEN]\tNN\tO\tO\tO",
        "[TOKEN]\t:\tO\tO\tO",
        "[TOKEN]\tNN\tO\tO\tO",
        "",
    ]

    merged = _merge_cleanconll_tokens_and_annotations(tokens, annotations)

    assert merged == [
        "SOCCER\tNN\tO\tO\tO",
        "-\t:\tO\tO\tO",
        "WORLD\tNN\tO\tO\tO",
        "",
    ]
