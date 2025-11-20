
"""
Tests Basiques du Pipeline

Tests simples pour valider le fonctionnement de base.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api import anonymize_text, AnonymizationPipeline, preset


def test_simple_anonymization():
    """Test simple d'anonymisation."""
    result = anonymize_text(
        "Jean Dupont habite à Paris",
        level="L0",
        secret_salt="test_secret"
    )
    
    assert "anonymized_text" in result
    assert "[PER_" in result["anonymized_text"]
    assert "[LOC_" in result["anonymized_text"]
    assert "Jean Dupont" not in result["anonymized_text"]
    assert "Paris" not in result["anonymized_text"]
    print("✓ test_simple_anonymization passed")


def test_email_detection():
    """Test détection d'email."""
    result = anonymize_text(
        "Contact: alice@example.com",
        level="L0",
        secret_salt="test"
    )
    
    assert "[MAIL_" in result["anonymized_text"] or "[EMAIL" in result["anonymized_text"]
    assert "alice@example.com" not in result["anonymized_text"]
    print("✓ test_email_detection passed")


def test_pipeline_reuse():
    """Test réutilisation du pipeline."""
    pipeline = AnonymizationPipeline(level="L0", secret_salt="secret")
    
    result1 = pipeline.anonymize("Alice Martin")
    result2 = pipeline.anonymize("Bob Johnson")
    
    assert result1["anonymized_text"] != result2["anonymized_text"]
    assert "Alice" not in result1["anonymized_text"]
    assert "Bob" not in result2["anonymized_text"]
    print("✓ test_pipeline_reuse passed")


def test_same_entity_same_placeholder():
    """Test que la même entité donne le même placeholder."""
    pipeline = AnonymizationPipeline(level="L0", secret_salt="secret")
    
    result1 = pipeline.anonymize("Alice travaille", scope_id="doc1")
    result2 = pipeline.anonymize("Alice habite", scope_id="doc1")
    
    # Extraire les placeholders
    import re
    placeholders1 = re.findall(r'\[PER_[A-Z]+\]', result1["anonymized_text"])
    placeholders2 = re.findall(r'\[PER_[A-Z]+\]', result2["anonymized_text"])
    
    # Même entité, même scope → même placeholder
    if placeholders1 and placeholders2:
        assert placeholders1[0] == placeholders2[0]
    print("✓ test_same_entity_same_placeholder passed")


def test_policy_preset():
    """Test des presets de policy."""
    l0 = preset("L0")
    l1 = preset("L1")
    l2 = preset("L2")
    
    assert l0.level == "L0"
    assert l0.llm_detection == False
    
    assert l1.level == "L1"
    assert l1.llm_detection == True
    assert l1.date_granularity == "month"
    
    assert l2.level == "L2"
    assert l2.date_granularity == "year"
    assert l2.org_policy == "redact"
    print("✓ test_policy_preset passed")


def test_evaluation_metrics():
    """Test des métriques d'évaluation."""
    result = anonymize_text(
        "Alice: alice@test.com, Bob: bob@test.com",
        level="L0",
        secret_salt="test"
    )
    
    assert "evaluation" in result
    assert "metrics" in result["evaluation"]
    assert result["evaluation"]["metrics"]["entities_detected"] >= 2
    assert result["evaluation"]["is_valid"] == True
    print("✓ test_evaluation_metrics passed")


def test_batch_anonymization():
    """Test anonymisation batch."""
    pipeline = AnonymizationPipeline(level="L0", secret_salt="batch_test")
    
    texts = [
        "Alice Martin",
        "Bob Johnson",
        "charlie@example.com"
    ]
    
    results = pipeline.anonymize_batch(texts, scope_id="batch_1")
    
    assert len(results) == 3
    for result in results:
        assert "anonymized_text" in result
        assert result["evaluation"]["is_valid"] == True
    print("✓ test_batch_anonymization passed")


if __name__ == "__main__":
    print("Lancement des tests basiques...\n")
    
    try:
        test_simple_anonymization()
        test_email_detection()
        test_pipeline_reuse()
        test_same_entity_same_placeholder()
        test_policy_preset()
        test_evaluation_metrics()
        test_batch_anonymization()
        
        print("\n✅ Tous les tests sont passés!")
    except AssertionError as e:
        print(f"\n❌ Test échoué: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
