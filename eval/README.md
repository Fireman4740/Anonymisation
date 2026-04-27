# Évaluation du pipeline d'anonymisation

Ce dossier regroupe l'évaluation **PipeGraph locale** (benchmarks, rapports et application Streamlit).

## Organisation actuelle

- `eval/core/`
  - helpers partagés pour le chargement PipeGraph, la normalisation de config et les agrégations.
- `eval/cli/`
  - points d'entrée CLI par dataset (`tab`, `ratbench`, `evaluate_ratbench_risk`).
- `eval/pipegraph_eval_local.py`
  - noyau d'exécution document par document et métriques de spans.
- `eval/run_full_benchmark.py`
  - évaluation complète unifiée (détection + leaks + risque).
- `eval/run_ablation.py`
  - exécution des suites d'ablation.
- `eval/streamlit_app/`
  - interface d'analyse et de comparaison des rapports.

## Commandes rapides

```bash
# Benchmark complet (sans risque LLM)
python eval/run_full_benchmark.py --skip-risk --limit 50

# TAB
python eval/cli/tab.py --split test --limit 50 --save-run

# RAT-Bench
python eval/cli/ratbench.py --level 1 --limit 50 --save-run

# Streamlit
streamlit run eval/streamlit_app/app.py
```

## Format de sortie

Les rapports détaillés sont sauvegardés au format canonique `meta + data` :

- `meta` : contexte d'exécution, dataset, config, agrégats
- `data` : liste détaillée des documents évalués

Les outputs sont écrits dans :

- `eval/evaluation/reports/`
- `eval/evaluation/runs/`

Le chargeur Streamlit reste compatible avec les anciens formats historiques (`details`, ou liste brute de documents).

---

## Historique (legacy API)

Les scripts ci-dessous ciblent l'ancien pipeline FastAPI. Ils sont conservés comme référence, mais ne sont pas utilisés dans le flux PipeGraph local :

- `eval/benchmark_pipeline.py`
- `eval/eval_rupta_dbbio.py`

Ces scripts supposent un endpoint `/anonymize` exposé par un serveur API externe.
