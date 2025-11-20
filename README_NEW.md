# 🔒 Anonymisation - Advanced Text Anonymization System

**Version 2.0** - Architecture Modulaire avec optimisation Privacy-Utility (RUPTA)

---

## 🎯 Aperçu

Système d'anonymisation de texte avancé combinant :
- **Détection multi-couches** : Regex, NER (GLiNER), LLM
- **Optimisation GPU** : Accélération 6-10x pour NER
- **RUPTA** : Optimisation automatique privacy-utility avec LLM
- **Architecture modulaire** : Services composables avec injection de dépendances

## ✨ Fonctionnalités

### 🔍 Détection Robuste
- **Regex validés** : Email, téléphone, NIR, IBAN, IP, URLs
- **NER multilingue** : GLiNER avec ensemble de modèles
- **Support GPU** : Optimisation automatique pour GPU 24GB+
- **LLM detection** : Détection contextuelle avancée (niveau L1)

### 🛡️ Privacy-Utility Optimization (RUPTA)
- **Optimisation itérative** : Equilibre automatique privacy vs utility
- **Métriques combinées** : Re-identification risk + utility preservation
- **Multilingue** : Support FR, EN, DE, ES, IT, PT, NL, etc.
- **Configurable** : Seuils personnalisables

### ⚡ Performance
- **Mode Standard** : ~8-10s pour 50 phrases
- **Mode GPU** : ~1-1.5s pour 50 phrases (**6-10x plus rapide**)
- **Batching intelligent** : Auto-tuning selon VRAM disponible
- **Multi-threading** : Parallélisation des modèles NER

## 📦 Installation

### Prérequis
- Python 3.11+
- GPU NVIDIA avec CUDA (optionnel, pour accélération)
- 24GB+ VRAM (recommandé pour mode GPU)

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/yourusername/anonymisation.git
cd anonymisation

# Installer les dépendances
pip install -e .

# Ou avec requirements.txt
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et ajouter votre OPENROUTER_API_KEY
```

## 🚀 Démarrage Rapide

### API Flask

```bash
# Lancer l'API
python -m api.app

# L'API est disponible sur http://localhost:8000
```

### Utilisation en ligne de commande

```python
from src.core import anonymize_text

text = "Mon email est jean.dupont@example.com et mon IP est 192.168.1.10"
result = anonymize_text(
    value=text,
    level="L0",  # L0 = sans LLM, L1 = avec LLM
    scope_id="demo-scope",
    secret_salt="demo-secret"
)

print(result["text"])
# "Mon email est [EMAIL_ABC] et mon IP est [IP_XYZ]"
```

### Avec RUPTA (optimisation privacy-utility)

```python
result = anonymize_text(
    value=text,
    level="L1",  # Activer LLM
    rupta_enabled=True,
    rupta_max_iterations=3,
    rupta_utility_threshold=80
)
```

## 📁 Architecture

```
Anonymisation/
├── 📁 config/              # Configuration
│   └── default.json
├── 📁 src/                 # Code source
│   ├── core/              # Modules core (orchestrator, policy)
│   ├── services/          # Services (detectors, generalizers, llm)
│   ├── llm/               # Clients LLM
│   ├── rupta/             # Module RUPTA
│   └── utils/             # Utilitaires
├── 📁 api/                # API Flask
├── 📁 tests/              # Tests
├── 📁 scripts/            # Scripts d'évaluation
├── 📁 docs/               # Documentation
└── 📁 datasets/           # Datasets d'évaluation
```

## 🔧 Configuration

### Configuration par défaut

```json
{
  "llm": {
    "provider": "lmstudio",
    "base_url": "http://localhost:1234/v1",
    "models": {
      "detect": "openai/gpt-oss-20b",
      "paraphrase": "openai/gpt-oss-20b",
      "audit": "openai/gpt-oss-20b"
    }
  },
  "ner_gpu": {
    "enabled": true,
    "vram_gb": 24,
    "batch_size": 64,
    "gliner_preset": "best"
  },
  "rupta": {
    "enabled": false,
    "p_threshold": 10,
    "max_iterations": 3,
    "utility_threshold": 80
  }
}
```

### Providers LLM supportés

- **LM Studio** (local) : `"provider": "lmstudio"`
- **OpenRouter** (cloud) : `"provider": "openrouter"`
- **Ollama** (local) : `"provider": "ollama"`

## 📊 Niveaux d'Anonymisation

| Niveau | Description | LLM | NER | RUPTA |
|--------|-------------|-----|-----|-------|
| **L0** | Regex + NER uniquement | ❌ | ✅ | ❌ |
| **L1** | L0 + LLM detection/paraphrase/audit | ✅ | ✅ | Optionnel |

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/

# Tests d'intégration
pytest tests/integration/

# Tests avec couverture
pytest --cov=src tests/

# Test de compatibilité Python 3.11
python tests/test_python311_compat.py
```

## 📖 Documentation

- **[Guides](docs/guides/)** : Démarrage rapide, Python 3.11, API
- **[Architecture](docs/architecture/)** : Refactoring, diagrammes, migration
- **[Features](docs/features/)** : RUPTA, NER GPU, Auto-fallback

## 🎯 Exemples

### Évaluation RUPTA sur DB-Bio

```bash
# Baseline
python scripts/eval/eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline

# RUPTA
python scripts/eval/eval_rupta_dbbio.py --split test --n_samples 10

# Comparaison
python scripts/eval/compare_baseline_rupta.py
```

### Benchmark NER GPU vs Standard

```bash
python scripts/benchmarks/benchmark_ner_gpu.py
```

## 🔬 Métriques

**Privacy** :
- Re-identification Rank : Position de la vraie personne dans les candidats
- Not Identified Rate : % de textes non ré-identifiables

**Utility** :
- Classification Accuracy : Préservation de l'occupation/label
- Utility Score : Score de cohérence sémantique

## 🤝 Contribution

Les contributions sont bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

## 📝 Changelog

Voir [CHANGELOG.md](CHANGELOG.md) pour l'historique des modifications.

## 📄 License

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

## 🆘 Support

- **Documentation** : [docs/](docs/)
- **Issues** : [GitHub Issues](https://github.com/yourusername/anonymisation/issues)
- **Discussions** : [GitHub Discussions](https://github.com/yourusername/anonymisation/discussions)

---

**Développé avec ❤️ pour la privacy et l'utility**
