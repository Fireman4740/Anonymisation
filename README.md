# 🔒 Pipeline d'Anonymisation Avancé

> Système d'anonymisation de texte hybride (regex + NER + LLM), structuré autour d'un pipeline LangGraph (PipeGraph) avec outillage d'évaluation et génération de données.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Vue d'ensemble

Ce dépôt regroupe trois volets complémentaires :

- **PipeGraph** (dossier `pipegraph/`) : pipeline d'anonymisation LangGraph.
- **Évaluation** (dossier `eval/`) : benchmarks, rapports et application Streamlit.
- **Atlas_anno** (dossier `Atlas_anno/`) : génération et préannotation de datasets.

## 🧱 Architecture (PipeGraph)

- Détection : regex + NER (GLiNER / spaCy / Flair)
- Transformation : anonymisation, pseudonymisation, généralisation
- LLM optionnels : détection, audit (RUPTA), paraphrase

La configuration se fait via `pipegraph/config.json` et `pipegraph/config/pipeline_config.yaml`.

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

## 📊 Évaluation

Installer les dépendances communes (si besoin) :

```bash
pip install -r requirements.txt
```

Exemples :

```bash
# Benchmark complet (sans risque LLM)
python eval/run_full_benchmark.py --skip-risk --limit 50

# TAB
python eval/cli/tab.py --split test --limit 50 --save-run

# RAT-Benchm
python eval/cli/ratbench.py --level 1 --limit 50 --save-run
```

### Streamlit

```bash
streamlit run eval/streamlit_app/app.py
```

Les rapports sont écrits dans `eval/evaluation/reports/` et `eval/evaluation/runs/`.

## 🧪 Tests

```bash
pytest pipegraph/tests/
pytest tests/
pytest Atlas_anno/tests/
```

## 🧭 Documentation par module

- `pipegraph/README.md` — pipeline LangGraph
- `eval/README.md` — scripts d'évaluation et formats de sortie
- `Atlas_anno/README.md` — génération et préannotation de données

## 📁 Structure du repo

```
Anonymisation/
├── pipegraph/           # Pipeline LangGraph
├── eval/                # Benchmarks + Streamlit
├── Atlas_anno/          # Génération / préannotation
├── tests/               # Tests liés aux datasets / éval
├── run_ablation_full.sh # Ablations PipeGraph
├── requirements.txt     # Dépendances communes
└── README.md
```

## 🔐 Variables d'environnement

- `OPENROUTER_API_KEY` (si LLM activé)
- `.env` (optionnel, lu par certains scripts)

## 📝 Licence

MIT, voir [LICENSE](LICENSE).
