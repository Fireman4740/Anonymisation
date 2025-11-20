# 🚀 Guide de Démarrage Rapide

Ce guide vous permet de démarrer rapidement avec le pipeline d'anonymisation.

## Installation

```bash
# Cloner le projet
cd /home/ubuntu/anonymization_pipeline_refactored

# Installer les dépendances (à adapter selon votre environnement)
pip install torch  # Pour NER GLiNER
pip install gliner  # Modèles NER
pip install schwifty  # Pour IBAN/BIC (optionnel)
pip install geonamescache  # Pour villes (optionnel)
```

## Premier Test (30 secondes)

```python
from api import anonymize_text

# Test simple
result = anonymize_text(
    "Jean Dupont habite à Paris, email: jean@example.com",
    level="L0",
    secret_salt="test_secret"
)

print(result["anonymized_text"])
# Sortie: [PER_ABC] habite à [LOC_XYZ], email: [MAIL_DEF]
```

## Exemples par Niveau

### Niveau L0 - Basic (Regex + NER)

**Utilisation** : Anonymisation rapide et déterministe sans LLM.

```python
from api import AnonymizationPipeline

# Créer le pipeline
pipeline = AnonymizationPipeline(
    level="L0",
    secret_salt="my_production_secret"
)

# Texte à anonymiser
text = """
Alice Martin travaille chez Google France.
Email: alice.martin@google.com
Téléphone: +33 6 12 34 56 78
Adresse IP: 192.168.1.100
"""

# Anonymiser
result = pipeline.anonymize(text, scope_id="doc_001")

print("=== Texte Anonymisé ===")
print(result["anonymized_text"])

print("\n=== Entités Détectées ===")
for entity in result["audit"]["entities"]:
    print(f"- {entity['etype']}: {entity['surface']} → détecté par {entity['source']}")

print("\n=== Métriques ===")
print(f"Entités détectées: {result['evaluation']['metrics']['entities_detected']}")
print(f"Entités remplacées: {result['evaluation']['metrics']['entities_replaced']}")
print(f"Ratio de longueur: {result['evaluation']['metrics']['length_ratio']:.2f}")
```

**Sortie attendue**:
```
[PER_ABC] travaille chez [ORG_XYZ] France.
Email: [MAIL_DEF]
Téléphone: [TELEPHONE_GHI]
Adresse IP: [IP_JKL]
```

### Niveau L1 - Advanced (L0 + LLM + RUPTA)

**Utilisation** : Anonymisation sophistiquée avec paraphrase et optimisation privacy-utility.

```python
pipeline = AnonymizationPipeline(
    level="L1",
    secret_salt="production_secret",
    # Personnalisation
    date_granularity="month",
    org_policy="generalize",
    paraphrase_intensity=2
)

text = """
Marie Curie, née le 7 novembre 1867 à Varsovie, 
était une physicienne et chimiste polonaise et naturalisée française.
Elle a reçu le Prix Nobel de Physique en 1903.
"""

result = pipeline.anonymize(text, scope_id="biography_001")

print("=== Texte Original ===")
print(text)

print("\n=== Texte Anonymisé et Paraphrasé ===")
print(result["anonymized_text"])

if result["audit"]["paraphrase_applied"]:
    print("\n✓ Paraphrase appliquée")

if result["audit"]["rupta_applied"]:
    print("✓ RUPTA appliqué")
    rupta = result["audit"]["rupta"]
    print(f"  - Privacy score: {rupta['privacy']}")
    print(f"  - Utility score: {rupta['utility']}")
    print(f"  - Itérations: {rupta['iterations']}")
```

### Niveau L2 - Maximum (Généralisation Agressive)

**Utilisation** : Protection maximale avec redaction complète.

```python
pipeline = AnonymizationPipeline(
    level="L2",
    secret_salt="secret"
)

text = """
Bob Johnson a envoyé 1,500,000 EUR à Alice le 15/06/2024.
Transaction ID: TXN-2024-001234
IP Source: 203.0.113.42
"""

result = pipeline.anonymize(text)

print(result["anonymized_text"])
# Sortie avec généralisation maximale:
# [REDACTED] a envoyé [REDACTED] à [REDACTED] le [DATE_2024].
# Transaction ID: [REDACTED]
# IP Source: [REDACTED]
```

## Cas d'Usage Pratiques

### Cas 1 : Documents Médicaux

```python
from api import AnonymizationPipeline

pipeline = AnonymizationPipeline(
    level="L1",
    secret_salt="medical_secret_2024",
    date_granularity="quarter",  # Trimestre pour dates médicales
    org_policy="generalize",
    mapping_retention="discard"  # Ne pas conserver les mappings
)

medical_record = """
Patient: Dr. Jean Dupont
Date de naissance: 12 mars 1975
Numéro de sécurité sociale: 1 75 03 75 056 123 45
Diagnostic: Hypertension artérielle
Médecin traitant: Dr. Marie Lambert (marie.lambert@hopital-paris.fr)
Hôpital: Centre Hospitalier de Paris
Date de consultation: 15 juin 2024
"""

result = pipeline.anonymize(medical_record, scope_id="patient_record_12345")

# Sauvegarder le résultat anonymisé
with open("anonymized_record.txt", "w") as f:
    f.write(result["anonymized_text"])

print("Document médical anonymisé sauvegardé.")
```

### Cas 2 : Logs Applicatifs

```python
pipeline = AnonymizationPipeline(
    level="L0",
    secret_salt="logs_secret",
    # Personnaliser pour logs
    date_granularity="none",  # Garder timestamps précis
    skip_regex_tags={"DATE"}  # Ne pas anonymiser les dates
)

log_entries = [
    "[2024-11-03 10:23:45] User alice@company.com logged in from 192.168.1.50",
    "[2024-11-03 10:24:12] API call to https://api.service.com/users/123",
    "[2024-11-03 10:25:03] Error: Failed to send email to bob@customer.com",
]

# Anonymiser tous les logs
anonymized_logs = pipeline.anonymize_batch(
    log_entries,
    scope_id="logs_2024_11_03"
)

for original, result in zip(log_entries, anonymized_logs):
    print(f"Original: {original}")
    print(f"Anonymisé: {result['anonymized_text']}")
    print()
```

**Sortie**:
```
Original: [2024-11-03 10:23:45] User alice@company.com logged in from 192.168.1.50
Anonymisé: [2024-11-03 10:23:45] User [MAIL_ABC] logged in from [IP_XYZ]

Original: [2024-11-03 10:24:12] API call to https://api.service.com/users/123
Anonymisé: [2024-11-03 10:24:12] API call to [URL_DEF]

Original: [2024-11-03 10:25:03] Error: Failed to send email to bob@customer.com
Anonymisé: [2024-11-03 10:25:03] Error: Failed to send email to [MAIL_GHI]
```

### Cas 3 : Commentaires Clients

```python
pipeline = AnonymizationPipeline(
    level="L1",
    secret_salt="feedback_secret",
    paraphrase_intensity=2
)

feedbacks = [
    "Alice Martin: Le service est excellent, merci à Bob de l'équipe support!",
    "Jean Dupont: Problème résolu rapidement par Marie (marie@support.fr)",
    "Client #12345: RAS, tout fonctionne parfaitement."
]

for feedback in feedbacks:
    result = pipeline.anonymize(feedback)
    print(f"Anonymisé: {result['anonymized_text']}\n")
```

### Cas 4 : RGPD - Export de Données

```python
from api import anonymize_text

def export_user_data_anonymized(user_data: dict) -> dict:
    """Exporte les données utilisateur en les anonymisant pour RGPD."""
    
    anonymized_data = {}
    
    for key, value in user_data.items():
        if isinstance(value, str):
            result = anonymize_text(
                value,
                level="L1",
                secret_salt="rgpd_export_secret",
                scope_id=f"user_{user_data.get('id', 'unknown')}"
            )
            anonymized_data[key] = result["anonymized_text"]
        else:
            anonymized_data[key] = value
    
    return anonymized_data

# Exemple
user = {
    "id": 12345,
    "name": "Alice Martin",
    "email": "alice.martin@example.com",
    "phone": "+33 6 12 34 56 78",
    "address": "15 rue de la Paix, 75001 Paris",
    "notes": "Cliente VIP depuis 2020"
}

anonymized_user = export_user_data_anonymized(user)

import json
print(json.dumps(anonymized_user, indent=2, ensure_ascii=False))
```

## Configuration GPU

Pour accélérer la détection NER :

```bash
# Forcer CUDA
export NER_FORCE_DEVICE=cuda

# Activer FP16 (précision mixte)
export NER_HALF_PRECISION=1

# Preset GLiNER
export GLINER_PRESET=best  # Ou "fast", "balanced", "accuracy", "pii"
```

Test GPU :
```python
from layer1_detection.ner.gliner_ensemble import warm_up_models

# Préchauffer les modèles
warm_up_models(gliner_preset="balanced")

print("Modèles préchauffés et prêts !")
```

## Intégration dans Votre Application

### FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api import AnonymizationPipeline

app = FastAPI()
pipeline = AnonymizationPipeline(level="L1", secret_salt="api_secret")

class AnonymizeRequest(BaseModel):
    text: str
    scope_id: str = "default"

class AnonymizeResponse(BaseModel):
    anonymized_text: str
    is_valid: bool

@app.post("/anonymize", response_model=AnonymizeResponse)
def anonymize_endpoint(request: AnonymizeRequest):
    try:
        result = pipeline.anonymize(request.text, scope_id=request.scope_id)
        return AnonymizeResponse(
            anonymized_text=result["anonymized_text"],
            is_valid=result["evaluation"]["is_valid"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lancer : uvicorn main:app --reload
```

### Flask

```python
from flask import Flask, request, jsonify
from api import AnonymizationPipeline

app = Flask(__name__)
pipeline = AnonymizationPipeline(level="L0", secret_salt="flask_secret")

@app.route("/anonymize", methods=["POST"])
def anonymize():
    data = request.json
    text = data.get("text")
    
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    result = pipeline.anonymize(text, scope_id=data.get("scope_id", "default"))
    
    return jsonify({
        "anonymized_text": result["anonymized_text"],
        "metrics": result["evaluation"]["metrics"]
    })

# Lancer : flask run
```

### CLI Tool

```python
# anonymize_cli.py
import sys
from api import anonymize_text

def main():
    if len(sys.argv) < 2:
        print("Usage: python anonymize_cli.py <text> [level]")
        sys.exit(1)
    
    text = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else "L0"
    
    result = anonymize_text(text, level=level, secret_salt="cli_secret")
    
    print(result["anonymized_text"])

if __name__ == "__main__":
    main()

# Utilisation :
# python anonymize_cli.py "Alice Martin: alice@example.com" L1
```

## Debugging

### Mode Verbose

```bash
# Activer logs NER
export NER_DEBUG=1
```

```python
result = anonymize_text("Mon texte", level="L0")

# Voir détails
print("=== Audit Complet ===")
import json
print(json.dumps(result["audit"], indent=2, ensure_ascii=False))
```

### Vérifier la Validation

```python
result = pipeline.anonymize("Mon texte")

if not result["evaluation"]["is_valid"]:
    print("⚠️ ERREURS DE VALIDATION:")
    for error in result["evaluation"]["validation_errors"]:
        print(f"  - {error}")

if result["evaluation"]["warnings"]:
    print("⚠️ AVERTISSEMENTS:")
    for warning in result["evaluation"]["warnings"]:
        print(f"  - {warning}")
```

## Prochaines Étapes

1. **Lire** : [ARCHITECTURE.md](ARCHITECTURE.md) pour comprendre le fonctionnement interne
2. **Explorer** : [API_REFERENCE.md](API_REFERENCE.md) pour la documentation complète
3. **Tester** : Lancer les tests avec `pytest tests/`
4. **Contribuer** : Voir `CONTRIBUTING.md` (si disponible)

## Ressources

- GitHub : [Votre repo]
- Documentation : `docs/`
- Issues : [Votre issue tracker]
- Support : [Votre email/forum]

---

**Bon anonymisation ! 🔒**
