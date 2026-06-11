# 🔒 Pipeline d'Anonymisation Avancé

> Système d'anonymisation de texte hybride (regex + NER + LLM), structuré autour d'un pipeline LangGraph (PipeGraph) avec outillage d'évaluation et génération de données.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Vue d'ensemble

Ce dépôt regroupe trois volets complémentaires :

- **PipeGraph** (dossier `pipegraph/`) : pipeline d'anonymisation LangGraph.
- **Évaluation** (dossier `eval/`) : runner officiel, benchmarks, rapports et application Streamlit.
- **Atlas_anno** (dossier `Atlas_anno/`) : génération et préannotation de datasets.

## 🧱 Architecture (PipeGraph)

- Détection : regex + NER (GLiNER / spaCy / Flair)
- Transformation : anonymisation, pseudonymisation, généralisation
- LLM optionnels : détection, audit (RUPTA), paraphrase

La configuration PipeGraph se fait via `pipegraph/config.json`.

## 🚀 Démarrage rapide

### 1) Installer l'environnement

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2) Installer les dépendances PipeGraph

```bash
pip install -r pipegraph/requirements.txt
```

### 3) Lancer une exécution de démonstration

```bash
python pipegraph/main.py
```

## 🐍 API Python

```python
from pipegraph import anonymize, anonymize_file

# Mode minimal sans LLM (regex + NER uniquement, 100% local)
result = anonymize("Je m'appelle Jean Dupont, jean.dupont@example.com", no_llm=True)
print(result.anonymized_text)
print(result.entities)        # offsets sur original_text, provenance incluse
print(result.privacy_score)   # score d'audit LLM (None si no_llm)

# Mode complet avec LLM (OPENROUTER_API_KEY requis dans .env)
result = anonymize("...", {"anon_strategy": "mask", "scope_id": "doc-42"})

# Fichier
anonymize_file("data/input.txt", "outputs/anonymized.txt", no_llm=True)
```

## 💻 CLI

```bash
# Anonymiser un texte (sans LLM)
python -m pipegraph anonymize --text "Jean Dupont habite à Paris" --no-llm

# Anonymiser un fichier
python -m pipegraph anonymize-file --input data/input.txt --output outputs/anonymized.txt

# Sortie JSON complète (texte original masqué par défaut, --show-original pour l'inclure)
python -m pipegraph anonymize --text "..." --json

# Overrides de config runtime
python -m pipegraph anonymize --text "..." --config-overrides '{"anon_strategy": "mask"}'

# Baselines et ablations = fichiers de config, pas des forks de code
python -m pipegraph anonymize --text "..." --config configs/baselines/regex_only.json
python -m pipegraph anonymize --text "..." --config configs/ablations/no_paraphrase.json
```

Configs disponibles : voir [configs/README.md](configs/README.md)
(`regex_only`, `ner_only`, `no_llm`, `full_llm` + 4 ablations).

## 🧪 Mode mock LLM (tests offline)

Tous les nœuds LLM passent par `src/nodes/llm/provider.py` (`get_llm_client`).
Pour exécuter le pipeline complet sans aucun appel réseau :

```bash
PIPEGRAPH_LLM_MOCK=1 python -m pipegraph anonymize --text "..."
# ou par config runtime : {"llm_mock": true}
```

Le mock renvoie des réponses no-op sûres par rôle (détection vide, audit
score 0, vérification keep-all) et enregistre les prompts (`mock.calls`).

## 📊 Évaluation

Installer les dépendances communes (si besoin) :

```bash
pip install -r requirements.txt
```

Point d'entrée recommandé : **CLI unifiée**

```bash
python -m eval list-datasets
python -m eval run --dataset tab --config configs/evaluation/no_llm.json
python -m eval run --dataset all --config configs/evaluation/full_llm.json
python -m eval ablation --dataset tab --ablation-config configs/evaluation/ablations/default.json
python -m eval compare --runs runs/evaluation/A runs/evaluation/B --output runs/comparison/
python -m eval report --run runs/evaluation/A
```

Autres points d'entrée :

| Script | Usage |
| --- | --- |
| `python scripts/evaluate.py` | Wrapper leger vers la CLI unifiee |
| `python -m eval.run_pipeline_evaluation` | Moteur officiel dataset-aware (appele par la CLI unifiee) |

Les datasets sont declares dans `eval.registry`; leur logique specifique est
rangee par dataset dans `eval/core/dataset_adapters/`.

### Runner officiel

```bash
# Évaluation officielle rapide sans risque LLM
python -m eval.run_pipeline_evaluation \
  --datasets tab dbbio anonymization conll2003 \
  --skip-risk --limit 50

# Avec RAT-Bench niveau 1
python -m eval.run_pipeline_evaluation \
  --datasets tab dbbio ratbench conll2003 anonymization \
  --ratbench-levels 1 --ratbench-languages english --limit 50
```

### Streamlit

```bash
streamlit run eval/streamlit_app/app.py
```

Les sorties officielles sont écrites dans `artifacts/eval-runs/<run-id>/`. Le script `evaluate.py` écrit dans `eval/evaluation/reports/` et `eval/evaluation/runs/`.

## 🧪 Tests

```bash
pytest pipegraph/tests/
pytest tests/
pytest Atlas_anno/tests/
```

## 🧭 Documentation par module

- `docs/README.md` — documentation détaillée du dépôt
- `pipegraph/README.md` — pipeline LangGraph
- `eval/README.md` — runner officiel, scripts d'évaluation et formats de sortie
- `docs/evaluation/README.md` — détails datasets, métriques et rapports
- `Atlas_anno/README.md` — génération et préannotation de données

## 📁 Structure du repo

```
Anonymisation/
├── pipegraph/           # Pipeline LangGraph
├── eval/                # Benchmarks + Streamlit
├── Atlas_anno/          # Génération / préannotation
├── scripts/             # Wrappers CLI et scripts opérationnels
├── tests/               # Tests liés aux datasets / éval
├── requirements.txt     # Dépendances communes
└── README.md
```

## 🤖 LLM local (Ollama)

> **Architecture :** Ollama tourne sur Windows, l'évaluation Streamlit tourne dans WSL. L'IP du host Windows est détectée automatiquement via la passerelle WSL2 (`/proc/net/route`).

### 1) Démarrer Ollama sur Windows

```powershell
# Session courante — écoute sur toutes les interfaces (requis pour WSL)
$env:OLLAMA_HOST = "http://0.0.0.0:11434"
ollama serve
```

Pour rendre le changement permanent (nouvelles sessions PowerShell) :

```powershell
setx OLLAMA_HOST "http://0.0.0.0:11434"
# Relancer PowerShell, puis :
ollama serve
```

> Par défaut Ollama écoute uniquement sur `127.0.0.1`, qui est inaccessible depuis WSL.
> `0.0.0.0` expose le serveur sur toutes les interfaces, y compris la passerelle WSL2.

### 2) Configurer `pipegraph/config.json`

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama3.2:latest",
    "base_url": "http://localhost:11434/v1"
  }
}
```

> L'URL `localhost` est automatiquement convertie vers l'IP du host Windows en WSL2. Aucune modification manuelle n'est nécessaire.

### 3) Dans l'interface Streamlit

Sélectionner le provider **ollama** dans la sidebar. La liste de modèles disponibles se charge automatiquement depuis l'API Ollama.

---

## 🔐 Variables d'environnement

- `OPENROUTER_API_KEY` (si provider OpenRouter activé)
- `.env` doit rester limité aux clés API; la configuration PipeGraph vit dans `pipegraph/config.json`.

## 📝 Licence

MIT, voir [LICENSE](LICENSE).
