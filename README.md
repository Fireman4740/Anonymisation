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

## 📊 Évaluation

Installer les dépendances communes (si besoin) :

```bash
pip install -r requirements.txt
```

Deux points d'entrée :

| Script | Usage |
| --- | --- |
| `python eval/evaluate.py` | Script unifié — benchmark, ablation, dataset standalone |
| `python -m eval.run_pipeline_evaluation` | Runner officiel ARC/ResearchClaw |

### `eval/evaluate.py` — commandes rapides

```bash
# Benchmark multi-dataset sans LLM
python eval/evaluate.py benchmark \
  --datasets tab dbbio anonymization \
  --limit 50 --no-llm --skip-risk

# Benchmark avec métriques strictes (standard CoNLL/PII-Bench)
python eval/evaluate.py benchmark \
  --datasets tab ratbench --limit 50 --strict

# RAT-Bench niveau 1 uniquement
python eval/evaluate.py benchmark \
  --datasets ratbench --levels 1 --limit 50 --skip-risk

# Ablation des nœuds sur TAB
python eval/evaluate.py ablation \
  --dataset tab --suite nodes --limit 20

# Évaluation standalone d'un dataset
python eval/evaluate.py dataset \
  --dataset personalreddit --limit 100 --save-run
```

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
├── tests/               # Tests liés aux datasets / éval
├── run_ablation_full.sh # Ablations PipeGraph
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
