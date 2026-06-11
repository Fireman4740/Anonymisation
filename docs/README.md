# Documentation du dépôt Anonymisation

Cette documentation décrit le dépôt sous l'angle opérationnel : comment le pipeline PipeGraph anonymise les textes, comment l'évaluation est structurée, et où trouver les sources des datasets utilisés.

## Vue d'ensemble

Le dépôt contient trois volets complémentaires :

- `pipegraph/` : pipeline d'anonymisation LangGraph, avec détection hybride, anonymisation, modules LLM et boucle RUPTA.
- `eval/` : runner officiel d'évaluation, loaders de datasets, rapports JSON/Markdown et application Streamlit d'analyse.
- `Atlas_anno/` : génération, préannotation et revue de datasets synthétiques. Ce volet garde sa documentation propre dans [`Atlas_anno/README.md`](../Atlas_anno/README.md) et [`Atlas_anno/docs/`](../Atlas_anno/docs/).

## Documentation principale

- [Pipeline PipeGraph](pipeline/README.md)
- [Architecture du pipeline](pipeline/architecture.md)
- [Configuration du pipeline](pipeline/configuration.md)
- [Composants du pipeline](pipeline/components.md)
- [Évaluation](evaluation/README.md)
- [Datasets et sources](evaluation/datasets.md)
- [Métriques et rapports](evaluation/metrics-and-reports.md)
- [Ablations](evaluation/ablations.md)

## Chemins importants

| Chemin | Rôle |
| --- | --- |
| [`pipegraph/main.py`](../pipegraph/main.py) | Démonstration locale du pipeline. |
| [`pipegraph/src/graph.py`](../pipegraph/src/graph.py) | Construction du graphe LangGraph. |
| [`pipegraph/config.json`](../pipegraph/config.json) | Configuration non secrète PipeGraph. |
| [`eval/cli/main.py`](../eval/cli/main.py) | CLI unifiée d'évaluation (`python -m eval`). |
| [`eval/api.py`](../eval/api.py) | API Python d'évaluation (runner, ablations, compare). |
| [`eval/run_pipeline_evaluation.py`](../eval/run_pipeline_evaluation.py) | Moteur officiel multi-datasets et multi-axes. |
| [`eval/streamlit_app/app.py`](../eval/streamlit_app/app.py) | Interface Streamlit de visualisation. |
| [`eval/datasets/`](../eval/datasets/) | Datasets locaux et caches de téléchargement. |

## Démarrage rapide

Installer les dépendances PipeGraph :

```bash
pip install -r pipegraph/requirements.txt
```

Lancer une exécution de démonstration :

```bash
python pipegraph/main.py
```

Lancer une évaluation rapide sans risque LLM :

```bash
python -m eval.run_pipeline_evaluation \
  --datasets tab dbbio anonymization conll2003 \
  --skip-risk \
  --limit 50
```

Lancer l'interface d'analyse :

```bash
streamlit run eval/streamlit_app/app.py
```

## Notes de périmètre

Cette documentation ne remplace pas les README historiques des sous-modules. Elle sert d'index détaillé et cohérent pour naviguer dans le dépôt. Les éléments Atlas sont volontairement résumés ici afin d'éviter une duplication avec la documentation existante d'`Atlas_anno`.
