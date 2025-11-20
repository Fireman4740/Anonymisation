
# 🔒 Pipeline d'Anonymisation - Architecture Refactorisée

**Version 2.0** - Architecture en couches claire et maintenable

## 📋 Vue d'Ensemble

Ce pipeline d'anonymisation offre une solution complète pour protéger les données sensibles dans vos textes avec une architecture moderne en 3 couches :

- **Couche 1 - Détection** : Regex, NER (GLiNER), GPU optimization, LLM avancé
- **Couche 2 - Transformation** : Pseudonymisation, généralisation, paraphrase LLM, RUPTA
- **Couche 3 - Évaluation** : Métriques, validation, qualité

## ✨ Caractéristiques Principales

### 🎯 Architecture Claire
- **3 couches bien séparées** : Détection → Transformation → Évaluation
- **Orchestrateur léger** (~300 lignes) : Coordination sans logique métier
- **Injection de dépendances** : Testable et composable
- **Pas de code dupliqué** : RUPTA centralisé, une seule implémentation NER

### 🚀 Performance
- **Support GPU** : Accélération automatique si disponible
- **NER GLiNER** : Détection haute qualité sans HuggingFace legacy
- **Cache intelligent** : Modèles chargés une seule fois

### 🔧 Flexibilité
- **3 niveaux prédéfinis** : L0 (basic), L1 (advanced + LLM), L2 (maximum)
- **Policy customisable** : Contrôle fin sur chaque aspect
- **API simple** : Fonctionnelle ou orientée objet

## 🏗️ Architecture

```
anonymization_pipeline_refactored/
├── layer1_detection/           # Couche 1 : Détection
│   ├── regex/                  # Patterns PII
│   ├── ner/                    # GLiNER + GPU
│   └── detection_service.py    # Service unifié
├── layer2_transformation/      # Couche 2 : Transformation
│   ├── replacement/            # Placeholders
│   ├── generalization/         # Policy-driven
│   ├── paraphrase/             # LLM stylométrique
│   ├── rupta/                  # Privacy-Utility
│   └── transformation_service.py
├── layer3_evaluation/          # Couche 3 : Évaluation
│   └── evaluator.py            # Métriques & validation
├── orchestrator/               # Orchestration
│   └── orchestrator.py         # Coordination (~300 lignes)
├── api/                        # API publique
│   ├── policy.py               # Configuration
│   └── pipeline.py             # Interface utilisateur
├── utils/                      # Utilitaires
│   ├── pseudo_mapper.py        # HMAC pseudonymisation
│   ├── whitelist.py            # Mots tolérés
│   └── config.py               # Config loader
└── docs/                       # Documentation
```

## 🚀 Démarrage Rapide

### Installation

```bash
# Cloner et installer
cd /home/ubuntu/anonymization_pipeline_refactored
pip install -r requirements.txt  # À créer avec vos dépendances
```

### Utilisation Basique (Niveau L0)

```python
from api import anonymize_text

# Anonymisation simple avec regex + NER
result = anonymize_text(
    "Jean Dupont habite à Paris, email: jean@example.com",
    level="L0",
    secret_salt="my_secret_key"
)

print(result["anonymized_text"])
# [PER_ABC] habite à [LOC_XYZ], email: [MAIL_DEF]
```

### Utilisation Avancée (Niveau L1)

```python
from api import AnonymizationPipeline

# Pipeline avec LLM + RUPTA
pipeline = AnonymizationPipeline(
    level="L1",
    secret_salt="my_secret_key",
    # Overrides optionnels
    date_granularity="month",
    paraphrase_intensity=2
)

result = pipeline.anonymize(
    "Marie Curie, née le 7 novembre 1867, était une physicienne.",
    scope_id="document_123"
)

print(result["anonymized_text"])
# Texte paraphrasé avec placeholders et dates généralisées
```

### Mode Batch

```python
texts = [
    "Alice travaille chez Google",
    "Bob habite à Lyon",
    "Charlie a envoyé un email à alice@gmail.com"
]

results = pipeline.anonymize_batch(texts, scope_id="batch_001")

for i, result in enumerate(results):
    print(f"Texte {i+1}: {result['anonymized_text']}")
```

## 📊 Niveaux d'Anonymisation

### L0 - Basic (Regex + NER)
- Détection regex (emails, téléphones, IPs, etc.)
- Détection NER GLiNER (personnes, lieux, organisations)
- Remplacement par placeholders typés
- **Pas de LLM** : Rapide et déterministe

### L1 - Advanced (L0 + LLM + RUPTA)
- Tout L0 +
- Détection LLM avancée (clustering, co-référence)
- Généralisation dates (mois), orgs, IPs (CIDR)
- Paraphrase stylométrique
- Audit de risque + hardening automatique
- Optimisation RUPTA (privacy-utility trade-off)

### L2 - Maximum (L1 + Généralisation Agressive)
- Tout L1 +
- Généralisation dates (année uniquement)
- Redaction complète des orgs
- Paraphrase intensive
- Seuils de risque très stricts

## 🔧 Configuration Avancée

### Policy Personnalisée

```python
from api import AnonymizationPolicy

policy = AnonymizationPolicy(
    level="L1",
    placeholder_style="generic",  # ou "typed"
    date_granularity="quarter",   # "none", "week", "month", "quarter", "year"
    org_policy="generalize",      # "keep", "generalize", "redact"
    llm_detection=True,
    llm_paraphrase=True,
    paraphrase_intensity=2,       # 0-3
    risk_threshold=50,            # 0-100
    rupta_enabled=True,
    rupta_max_iterations=5,
)

result = anonymize_text(
    "Mon texte sensible",
    level="L1",
    secret_salt="secret",
    # Passer la policy en override
    **policy.to_dict()
)
```

### GPU Acceleration

Le pipeline détecte automatiquement le GPU disponible (CUDA, MPS, CPU). Pour forcer un device :

```bash
export NER_FORCE_DEVICE=cuda  # ou "mps" ou "cpu"
export NER_HALF_PRECISION=1   # FP16 sur CUDA
```

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Architecture détaillée des 3 couches
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Documentation complète de l'API
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Exemples d'utilisation pas à pas

## 🧪 Tests

```bash
# Tests unitaires
python -m pytest tests/unit/

# Tests d'intégration
python -m pytest tests/integration/

# Test rapide
python examples/simple_example.py
```

## 🔍 Métriques et Évaluation

Chaque anonymisation retourne des métriques détaillées :

```python
result = pipeline.anonymize("Mon texte")

# Métriques
print(result["evaluation"]["metrics"])
# {
#   "entities_detected": 5,
#   "entities_replaced": 5,
#   "replacement_rate": 1.0,
#   "placeholder_count": 5,
#   "length_ratio": 0.95
# }

# Validation
print(result["evaluation"]["is_valid"])  # True/False
print(result["evaluation"]["warnings"])   # Avertissements éventuels
```

## ⚙️ Différences avec l'Ancienne Version

### ✅ Améliorations

| Aspect | Avant | Après |
|--------|-------|-------|
| **Architecture** | Monolithique (678 lignes orchestrator) | 3 couches séparées (~300 lignes orchestrator) |
| **Code dupliqué** | RUPTA implémenté 2 fois | Une seule implémentation centralisée |
| **NER** | GLiNER + HuggingFace legacy | GLiNER uniquement (simplifié) |
| **Testabilité** | Couplage fort | Injection de dépendances |
| **Maintenabilité** | Difficile | Responsabilités claires |
| **Code total** | ~5000 lignes | ~4200 lignes (-16%) |

### 🔄 Migration

Pour migrer de l'ancienne version :

1. **Imports** : `from api import anonymize_text` (au lieu de `from src.orchestrator import ...`)
2. **API** : Compatible à 99% (même signature de base)
3. **Breaking change** : HF NER retiré (utilisez GLiNER à la place)

## 🤝 Contribution

Contributions bienvenues ! Veuillez :
1. Créer une issue pour discuter des changements majeurs
2. Suivre le style de code existant
3. Ajouter des tests pour les nouvelles fonctionnalités
4. Mettre à jour la documentation

## 📝 Licence

[À définir selon votre projet]

## 🙏 Remerciements

- **GLiNER** pour la détection NER haute qualité
- **OpenRouter** pour l'accès LLM unifié
- **RUPTA** pour l'optimisation privacy-utility

## 📞 Support

- Documentation : `docs/`
- Issues : [Votre repo GitHub]
- Email : [Votre email]

---

**Version 2.0** - Architecture refactorisée pour la clarté et la maintenabilité  
Novembre 2025
