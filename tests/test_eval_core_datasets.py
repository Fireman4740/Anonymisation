from __future__ import annotations

from eval.core.datasets import get_allowed_labels, load_local_dataset_docs, uses_news_ner_profile


def test_cleanconll2003_local_loader_uses_clean_variant(monkeypatch):
    calls = {}

    def fake_build_docs_from_conll2003(*, limit, split, variant=None):
        calls["limit"] = limit
        calls["split"] = split
        calls["variant"] = variant
        return [("doc-1", "John Doe works at Acme", [(0, 8, "PER")])]

    monkeypatch.setattr("eval.core.datasets.build_docs_from_conll2003", fake_build_docs_from_conll2003)

    docs = load_local_dataset_docs(
        dataset_kind="cleanconll2003",
        dataset_path="/tmp/cleanconll.test",
        split="test",
        limit=5,
    )

    assert len(docs) == 1
    assert calls == {"limit": 5, "split": "test", "variant": "clean"}


def test_cleanconll2003_uses_news_ner_scope():
    assert uses_news_ner_profile("cleanconll2003") is True
    assert get_allowed_labels("cleanconll2003") == frozenset({"PER", "ORG", "LOC", "MISC"})
