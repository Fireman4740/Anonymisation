import pytest

from src.utils.validation import count_placeholder_types, validate_anonymization


def test_count_placeholder_types():
    text = "Contact [MAIL_1] ou [MAIL_2] et [PER_A]."
    counts = count_placeholder_types(text)
    assert counts["MAIL"] == 2
    assert counts["PER"] == 1


def test_validate_anonymization_detects_issues():
    anonymized = "Parlez à [MAIL_1]"
    issues = validate_anonymization(
        original="",
        anonymized=anonymized,
        expected_counts={"[MAIL_": 2},
        forbidden_patterns=["@example.com"],
    )
    assert any("⚠️" in issue for issue in issues)
    assert all("example" not in issue for issue in issues)


def test_validate_anonymization_passes_when_ok():
    anonymized = "Contact [MAIL_1] et [MAIL_2]"
    issues = validate_anonymization(
        original="",
        anonymized=anonymized,
        expected_counts={"MAIL": 2},
        forbidden_patterns=["secret"],
    )
    assert issues == []
