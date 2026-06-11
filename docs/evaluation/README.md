# Evaluation

Le point d'entree officiel est `python -m eval`. La CLI passe par
`eval.api`, puis par le runner dataset-aware `eval.run_pipeline_evaluation`.

## Points D'Entree

| Commande | Role |
| --- | --- |
| `python -m eval` | CLI unifiee: run, ablation, compare, report, list-datasets |
| `python scripts/evaluate.py` | Wrapper leger vers `python -m eval` |
| `python -m eval.run_pipeline_evaluation` | Runner officiel, utile pour ARC/ResearchClaw |
| `streamlit run eval/streamlit_app/app.py` | Interface d'analyse |

## Commandes

```bash
python -m eval list-datasets
python -m eval run --dataset tab --config configs/evaluation/no_llm.json
python -m eval run --dataset all --config configs/evaluation/full_llm.json
python -m eval ablation --dataset tab \
  --config configs/evaluation/no_llm.json \
  --ablation-config configs/evaluation/ablations/default.json
python -m eval compare --runs runs/evaluation/A runs/evaluation/B --output runs/comparison/
python -m eval report --run runs/evaluation/A --format markdown
```

## Architecture

- `eval/registry.py`: source de verite pour noms, alias et capabilities.
- `eval/core/dataset_adapters/`: logique specifique par dataset.
- `eval/run_pipeline_evaluation.py`: cycle commun d'evaluation, scoring et artefacts.
- `eval/api.py`: API Python pour CLI, Streamlit et notebooks.

## Datasets

Datasets enregistres:

- `tab`
- `dbbio`
- `anonymization`
- `ratbench`
- `conll2003`
- `personalreddit`

RAT-Bench expand automatiquement les combinaisons langue/niveau via
`--ratbench-languages` et `--ratbench-levels`.

## Sorties

Un run officiel contient:

```text
runs/evaluation/<run-id>/
  run_config.json
  candidate_effective_config.json
  summary.json
  summary.md
  manifest.json
  datasets/<key>/documents.jsonl
  datasets/<key>/metrics.json
```

## Documentation Liee

- [Datasets et sources](datasets.md)
- [Metriques et rapports](metrics-and-reports.md)
- [Ablations](ablations.md)
