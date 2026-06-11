# Evaluation PipeGraph

Le package `eval/` contient l'evaluation locale de PipeGraph: registry de
datasets, adapters, runner officiel, API Python, CLI et Streamlit.

## Point D'Entree

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

`scripts/evaluate.py` est le seul wrapper script conserve. Les anciens scripts
legacy redondants ont ete retires.

## API Python

```python
from eval.api import EvaluationRunner, compare_runs, load_predictions

runner = EvaluationRunner.from_config("configs/evaluation/no_llm.json")
payload = runner.run(dataset="tab")
summary = runner.run_ablation(
    dataset="tab",
    ablation_config="configs/evaluation/ablations/minimal.json",
)
```

## Architecture

| Chemin | Role |
| --- | --- |
| `eval/api.py` | Facade Python: run, ablation, compare, acces aux artefacts |
| `eval/cli/main.py` | CLI unifiee exposee par `python -m eval` |
| `eval/registry.py` | Source de verite des datasets |
| `eval/core/dataset_adapters/` | Logique specifique par dataset |
| `eval/run_pipeline_evaluation.py` | Runner officiel dataset-aware |
| `eval/core/` | Config, metrics, reporting, I/O, profils |
| `eval/core/pipeline.py` | Execution document par document |
| `eval/core/loaders/` | Loaders de datasets specialises |
| `eval/streamlit_app/` | Interface d'analyse |

## Datasets

Le registry expose `tab`, `dbbio`, `anonymization`, `ratbench`, `conll2003` et
`personalreddit`. Ajouter un dataset se fait via un nouvel adapter et son
enregistrement dans `eval/registry.py`.

## Artefacts

Chaque run ecrit:

```text
run_config.json
candidate_effective_config.json
summary.json
summary.md
manifest.json
datasets/<key>/documents.jsonl
datasets/<key>/metrics.json
```
