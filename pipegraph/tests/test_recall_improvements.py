# -*- coding: utf-8 -*-

import os
from src.nodes.detection.deterministic.validators import Validators
from src.nodes.detection.detection_node import _passes_ai_length_filter
from src.utils.entity_utils import normalize_entity_type
from src.utils.span_utils import SOURCE_PRIORITY

def test_french_nir_accepts_13_digit_form_without_key():
    assert Validators.french_ssn("1 84 12 76 451 089") is True


def test_us_ssn_validator_accepts_and_rejects_expected_formats():
    assert Validators.us_ssn("123-45-6789") is True
    assert Validators.us_ssn("000-45-6789") is False


def test_validator_routing_separates_nir_and_us_ssn():
    assert Validators.get_validator("nir") is Validators.french_ssn
    assert Validators.get_validator("ssn") is Validators.us_ssn


def test_llm_detection_prompt_covers_missing_recall_labels():
    prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../prompts/llm_review_system.txt"))
    with open(prompt_path, "r", encoding="utf-8") as f:
        _SYSTEM_PROMPT = f.read()

    for label in [
        "PROJECT",
        "ERROR_CODE",
        "STYLE",
        "NIR",
        "SSN",
        "NATIONALITY",
        "MARITAL STATUS",
        "EMPLOYMENT STATUS",
        "EDUCATIONAL BACKGROUND",
        "CITIZENSHIP STATUS",
        "MEDICAL",
        "RACE",
    ]:
        assert label in _SYSTEM_PROMPT


def test_news_ner_profile_projects_generic_labels_to_conll_space():
    assert normalize_entity_type("Organization", profile="news_ner") == "ORG"
    assert normalize_entity_type("GPE", profile="news_ner") == "LOC"
    assert normalize_entity_type("Event", profile="news_ner") == "MISC"
    assert normalize_entity_type("Nationality", profile="conll2003") == "MISC"


def test_short_uppercase_acronyms_bypass_default_min_len_filter():
    assert _passes_ai_length_filter({"start": 0, "end": 2, "value": "EU"}, 3) is True
    assert _passes_ai_length_filter({"start": 0, "end": 4, "value": "U.N."}, 3) is True
    assert _passes_ai_length_filter({"start": 0, "end": 2, "value": "Jo"}, 3) is False


def test_source_priority_includes_llm_and_heuristic_recall_sources():
    assert SOURCE_PRIORITY["heuristic"] > SOURCE_PRIORITY["gliner"]
    assert SOURCE_PRIORITY["llm_review"] >= SOURCE_PRIORITY["llm"]
    assert SOURCE_PRIORITY["llm_verified"] > SOURCE_PRIORITY["gliner"]
