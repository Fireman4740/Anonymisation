# 🔒 Pipeline d'Anonymisation Avancé

> Système d'anonymisation de texte hybride combinant regex, NER, LLM et optimisation RUPTA pour une protection maximale de la vie privée tout en préservant l'utilité des données.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Vue d'Ensemble

Ce projet fournit un pipeline d'anonymisation de texte de niveau production capable de détecter et anonymiser automatiquement les informations personnellement identifiables (PII) et les données sensibles dans du texte en français et autres langues.

### Points Forts

- **🎚️ 3 Niveaux d'Anonymisation** : L0 (Basic), L1 (Advanced), L2 (Maximum)
- **🔍 Détection Hybride** : Regex + NER (GLiNER) + LLM avec déduplication intelligente
- **🛡️ Protection Renforcée** : Patterns avancés (IBAN, BIC, secrets, API keys, IPv6)
- **🤖 LLM Optionnel** : Paraphrase stylométrique et optimisation RUPTA
- **⚡ GPU Ready** : Support automatique GPU/CPU avec optimisations
- **📊 Évaluation Intégrée** : Métriques, validation et rapports détaillés
- **🔧 Hautement Configurable** : Policies personnalisables et overrides fins

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│               COUCHE 3 : ÉVALUATION                     │
│            Métriques, Validation, Warnings              │
├─────────────────────────────────────────────────────────┤
│               COUCHE 2 : TRANSFORMATION                 │
│     Pseudonymisation, Généralisation, LLM, RUPTA        │
├─────────────────────────────────────────────────────────┤
│               COUCHE 1 : DÉTECTION                      │
│           Regex, NER (GLiNER), Advanced Patterns        │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Démarrage Rapide

### Installation

```bash
# Cloner le projet
git clone <votre-repo>
cd Anonymisation

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # ou `venv\Scripts\activate` sur Windows

# Installer les dépendances
pip install -r pipeline/requirements.txt

# Installation optionnelle pour fonctionnalités avancées
pip install schwifty phonenumbers  # Pour IBAN/BIC et téléphones internationaux
```


cd /mnt/f/IA/Anonymisation/eval/streamlit_app && conda run -n ano --no-capture-output streamlit run app.py --server.headless true --server.port 8501

### Premier Test (30 secondes)

```python
import sys
sys.path.append('pipeline')

from src.core.orchestrator import anonymize_text

# Test simple niveau L0 (regex + NER)
result = anonymize_text(
    "Jean Dupont habite à Paris, email: jean@example.com, tél: +33 6 12 34 56 78",
    level="L0",
    secret_salt="test_secret"
)

print(result["anonymized_text"])
# Sortie: [PER_ABC] habite à [LOC_XYZ], email: [MAIL_DEF], tél: [TELEPHONE_GHI]
```

### Lancer l'API

```bash
# Démarrer le serveur FastAPI
cd pipeline
uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000

# Tester l'API
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice Martin: alice@example.com"}'
```

## 📚 Documentation

La documentation complète est disponible dans le dossier [`pipeline/docs/`](pipeline/docs/) :

- **[QUICKSTART.md](pipeline/QUICKSTART.md)** - Guide de démarrage avec exemples pratiques
- **[ARCHITECTURE.md](pipeline/docs/ARCHITECTURE.md)** - Architecture détaillée du système
- **[API_REFERENCE.md](pipeline/docs/API_REFERENCE.md)** - Référence complète de l'API
- **[ROADMAP.md](pipeline/docs/ROADMAP.md)** - Feuille de route et évolutions futures

### Exemples par Niveau

#### Niveau L0 - Basic (Regex + NER)
Anonymisation rapide et déterministe sans LLM.

```python
from src.core.orchestrator import AnonymizationPipeline

pipeline = AnonymizationPipeline(level="L0", secret_salt="my_secret")
result = pipeline.anonymize("Marie Curie (marie@curie.fr) a découvert le radium.")

print(result["anonymized_text"])
# [PER_ABC] ([MAIL_DEF]) a découvert le radium.
```

#### Niveau L1 - Advanced (L0 + LLM + RUPTA)
Anonymisation sophistiquée avec paraphrase et optimisation privacy-utility.

```python
pipeline = AnonymizationPipeline(
    level="L1",
    secret_salt="prod_secret",
    paraphrase_intensity=2,
    date_granularity="month"
)

result = pipeline.anonymize(
    "Jean Dupont, né le 15/03/1985, travaille chez Google France.",
    scope_id="bio_001"
)

print(result["anonymized_text"])
# La paraphrase et RUPTA sont appliqués automatiquement
```

#### Niveau L2 - Maximum (Généralisation Agressive)
Protection maximale avec redaction complète.

```python
pipeline = AnonymizationPipeline(level="L2", secret_salt="secret")
result = pipeline.anonymize("Montant: 150,000 EUR transféré le 15/06/2024")

print(result["anonymized_text"])
# Montant: [REDACTED] transféré le [DATE_2024]
```

## 🎯 Cas d'Usage

### Documents Médicaux
```python
pipeline = AnonymizationPipeline(
    level="L1",
    secret_salt="medical_2024",
    date_granularity="quarter",
    mapping_retention="discard"
)
```

### Logs Applicatifs
```python
pipeline = AnonymizationPipeline(
    level="L0",
    date_granularity="none",
    skip_regex_tags={"DATE"}  # Conserver les timestamps
)
```

### Conformité RGPD
```python
pipeline = AnonymizationPipeline(
    level="L1",
    org_policy="generalize",
    paraphrase_intensity=2
)
```

## 🧪 Évaluation et Tests

### Exécuter les Tests

```bash
cd pipeline

# Tests unitaires
pytest tests/

# Tests d'évaluation exhaustifs
python evaluation/test_exhaustif.py \
    --dataset evaluation/datasets/default_cases.json \
    --output evaluation/reports/run.json
```

### Datasets d'Évaluation

Le projet inclut plusieurs datasets de test :

- **`default_cases.json`** - 12 cas de test standard (regex, NER, LLM)
- **`hard_realistic_cases.json`** - Cas difficiles et réalistes
- **DB-bio** - Dataset biomédical
- **PersonalReddit** - Posts Reddit synthétiques
- **TAB** - Text Anonymization Benchmark (ECHR)

### Métriques

Les rapports d'évaluation incluent :
- Nombre d'entités détectées/remplacées
- Taux de validation (forbidden patterns, compteurs d'entités)
- Privacy score et Utility score (RUPTA)
- Temps d'exécution par couche

## 🛠️ Configuration

### Variables d'Environnement

```bash
# API Server
export PIPELINE_SECRET_SALT="production_secret"
export PIPELINE_DEFAULT_LEVEL="L1"
export PIPELINE_SCOPE_PREFIX="customer"

# GPU/Performance
export NER_FORCE_DEVICE="cuda"
export NER_HALF_PRECISION="1"
export GLINER_PRESET="balanced"  # fast, balanced, best, accuracy, pii

# Debug
export NER_DEBUG="1"
```

### Policy Configuration

```python
from src.core.policy import AnonymizationPolicy

# Créer une policy personnalisée
policy = AnonymizationPolicy(
    level="L1",
    date_granularity="month",
    org_policy="generalize",
    paraphrase_intensity=2,
    rupta_target_privacy=0.9,
    rupta_target_utility=0.7
)

pipeline = AnonymizationPipeline(policy=policy)
```

### Patterns Personnalisés

Les patterns regex sont configurables via `pipeline/patterns_config.yaml` :

```yaml
patterns:
  API_KEY:
    regex: 'sk-[a-zA-Z0-9]{32,64}'
    priority: 100
    validator: null
  
  CUSTOM_ID:
    regex: 'CID-\d{6}'
    priority: 50
    validator: null
```

## 📊 Structure du Projet

```
Anonymisation/
├── README.md                          # Ce fichier
├── pyproject.toml                     # Configuration du projet Python
├── requirements.txt                   # Dépendances principales
├── Dataset/                           # Datasets d'évaluation
│   ├── evaluation/                    # Benchmarks (DB-bio, Reddit, TAB)
│   └── génération_data/              # Scripts de génération
├── eval/                              # Scripts d'évaluation
│   ├── benchmark_pipeline.py
│   ├── evaluate_pipeline.py
│   └── metrics.py
├── pipeline/                          # Pipeline principal
│   ├── QUICKSTART.md                 # Guide de démarrage rapide
│   ├── config.json                   # Configuration globale
│   ├── patterns_config.yaml          # Patterns regex personnalisés
│   ├── docs/                         # Documentation détaillée
│   │   ├── ARCHITECTURE.md
│   │   ├── API_REFERENCE.md
│   │   └── ROADMAP.md
│   ├── src/                          # Code source
│   │   ├── core/                     # Orchestrateur et policies
│   │   ├── services/                 # Services (detection, transformation, etc.)
│   │   ├── llm/                      # Clients LLM et reasoner
│   │   ├── rupta/                    # Optimisation RUPTA
│   │   └── utils/                    # Utilitaires
│   ├── scripts/
│   │   └── api_server.py            # Serveur FastAPI
│   ├── tests/                        # Tests unitaires
│   └── evaluation/                   # Tests d'évaluation
└── results/                          # Résultats d'évaluation
```

## 🔧 Développement

### Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

### Tests en Développement

```bash
# Tests avec coverage
pytest --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_pipeline.py -v

# Tests d'évaluation avec fail-fast
python evaluation/test_exhaustif.py --fail-fast
```

## 📈 Performance

### Benchmarks (GPU Tesla T4)

| Niveau | Texte (500 mots) | Temps moyen | Entités détectées |
|--------|------------------|-------------|-------------------|
| L0     | Article blog     | 0.8s        | 15-20            |
| L1     | Document médical | 3.2s        | 25-30            |
| L2     | Logs techniques  | 0.5s        | 10-15            |

### Optimisations

- **GPU automatique** : Détection GPU/CPU avec fallback
- **Batch processing** : Traitement par lot pour volumes importants
- **Caching NER** : Modèles chargés une seule fois
- **FP16** : Précision mixte pour accélération GPU

## 🔐 Sécurité

### Bonnes Pratiques

- ✅ Utiliser un `secret_salt` fort et unique par environnement
- ✅ Définir `mapping_retention="discard"` pour données sensibles
- ✅ Activer validation des forbidden patterns
- ✅ Utiliser L1/L2 pour documents médicaux/financiers
- ✅ Tester sur datasets représentatifs avant production

### Limitations

- La paraphrase LLM peut parfois modifier le sens
- NER peut manquer des entités ambiguës ou contextuelles
- RUPTA nécessite plusieurs itérations (impact performance)

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🤝 Support

- **Issues** : [GitHub Issues](votre-repo/issues)
- **Documentation** : [`pipeline/docs/`](pipeline/docs/)
- **Email** : your.email@example.com

## 🙏 Remerciements

- [GLiNER](https://github.com/urchade/GLiNER) pour le NER zero-shot
- [OpenRouter](https://openrouter.ai/) pour l'accès LLM
- Communauté NLP française

---

**Développé avec ❤️ pour la protection de la vie privée**
