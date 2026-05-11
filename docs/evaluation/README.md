# Évaluation

Le dossier `eval/` contient l'outillage d'évaluation local de PipeGraph : loaders de datasets, calculs de métriques, scripts d'évaluation, rapports JSON/Markdown et interface Streamlit.

## Points d'entrée

| Script | Usage |
| --- | --- |
| [`eval/evaluate.py`](../../eval/evaluate.py) | **Point d'entrée unifié** — benchmark, ablation, dataset standalone |
| `python -m eval.run_pipeline_evaluation` | Runner officiel ARC/ResearchClaw avec protocole complet |
| `streamlit run eval/streamlit_app/app.py` | Interface graphique d'analyse |

---

## `eval/evaluate.py` — script unifié

### Sous-commande `benchmark`

Évaluation multi-dataset avec métriques complètes (partial + strict, leaks, BLEU, R_succ).

```bash
python eval/evaluate.py benchmark [options]
```

| Argument | Valeurs | Défaut | Description |
| --- | --- | --- | --- |
| `--datasets` | `tab` `dbbio` `anonymization` `ratbench` `conll2003` `personalreddit` | `tab dbbio anonymization` | Datasets (un ou plusieurs) |
| `--limit` | entier | aucun | Max documents par dataset |
| `--split` | `test` `dev` `train` | `test` | Split |
| `--levels` | `1` `2` `3` | `1 2 3` | Niveaux RAT-Bench |
| `--language` | `english` `mandarin` `spanish` | `english` | Langue RAT-Bench |
| `--skip-risk` | flag | — | Ignorer l'axe risque RAT-Bench |
| `--strict` | flag | — | Afficher les métriques strict-match (offsets + label exacts) en priorité |
| `--save-runs` | flag | — | Sauvegarder les runs document-level |
| `--out` | chemin | `eval/evaluation/reports/` | Répertoire de sortie |
| `--no-llm` | flag | — | Désactiver tous les modules LLM |
| `--with-llm` | flag | — | Forcer LLM activé |
| `--llm-provider` | chaîne | config | Provider LLM |
| `--llm-model` | chaîne | config | Modèle LLM |
| `--profile` | `auto` + profils disponibles | `auto` | Profil d'évaluation |
| `--eval-mode` | `canonical` `benchmark` `both` | `both` | Mode de projection des labels |
| `--masking-mode` | `production` `benchmark` | `benchmark` | Politique de masquage |

Exemples :

```bash
# Rapide sans LLM
python eval/evaluate.py benchmark --datasets tab dbbio --limit 50 --no-llm --skip-risk

# Avec métriques strictes (standard CoNLL / PII-Bench)
python eval/evaluate.py benchmark --datasets tab ratbench --limit 50 --strict

# RAT-Bench niveau 1 anglais uniquement
python eval/evaluate.py benchmark --datasets ratbench --levels 1 --language english --limit 50 --skip-risk

# Tous les datasets avec sauvegarde des runs
python eval/evaluate.py benchmark \
  --datasets tab dbbio anonymization ratbench conll2003 personalreddit \
  --limit 100 --strict --skip-risk --save-runs
```

### Sous-commande `ablation`

Grille d'ablation sur un seul dataset.

```bash
python eval/evaluate.py ablation [options]
```

| Argument | Valeurs | Défaut | Description |
| --- | --- | --- | --- |
| `--dataset` | voir datasets | `tab` | Dataset cible |
| `--suite` | `nodes` `ner_presets` `ner_ensemble` `ner_threshold` `ner_vote` `anon_strategy` `detection_mode` `full` `custom` `list` | `nodes` | Suite d'ablation |
| `--custom-config` | chemin JSON | — | Config pour `--suite custom` |
| `--limit` | entier | `50` | Documents par config |
| `--split` | `test` `dev` `train` | `test` | Split |
| `--language` | `english` `mandarin` `spanish` | `english` | Langue (RAT-Bench) |
| `--level` | `1` `2` `3` | aucun | Niveau RAT-Bench |
| `--parallel-configs` | entier | `1` | Configs évaluées en parallèle |
| `--save-runs` | flag | — | Sauvegarder les runs |
| `--out` | chemin | — | Répertoire de sortie |
| LLM + profil | — | — | Identiques à `benchmark` |

Exemples :

```bash
# Lister les suites disponibles
python eval/evaluate.py ablation --suite list

# Ablation des nœuds sur TAB
python eval/evaluate.py ablation --dataset tab --suite nodes --limit 20

# Ablation NER sur RAT-Bench L1
python eval/evaluate.py ablation --dataset ratbench --suite ner_presets --level 1 --limit 30

# Suite complète en parallèle
python eval/evaluate.py ablation --dataset tab --suite full --limit 50 --parallel-configs 4
```

### Sous-commande `dataset`

Évaluation standalone d'un seul dataset, sortie par document.

```bash
python eval/evaluate.py dataset --dataset <nom> [options]
```

| Argument | Valeurs | Défaut | Description |
| --- | --- | --- | --- |
| `--dataset` | voir datasets | **requis** | Dataset |
| `--split` | `test` `dev` `train` | `test` | Split |
| `--language` | `english` `mandarin` `spanish` | `english` | Langue (RAT-Bench) |
| `--level` | `1` `2` `3` | aucun | Niveau RAT-Bench |
| `--limit` | entier | aucun | Max documents |
| `--save-run` | flag | — | Sauvegarder le run |
| `--strict` | flag | — | Métriques strict-match |
| `--out` | chemin | — | Répertoire de sortie |
| LLM + profil | — | — | Identiques à `benchmark` |

Exemples :

```bash
python eval/evaluate.py dataset --dataset tab --split test --limit 50
python eval/evaluate.py dataset --dataset ratbench --level 1 --limit 30 --strict
python eval/evaluate.py dataset --dataset personalreddit --limit 100 --save-run
python eval/evaluate.py dataset --dataset conll2003 --split dev --limit 200 --no-llm
```

---

## Runner officiel `run_pipeline_evaluation`

Utilisé pour le protocole ARC/ResearchClaw. Produit un dossier complet sous `artifacts/eval-runs/<timestamp>/`.

```bash
# Rapide sans risque LLM
python -m eval.run_pipeline_evaluation \
  --datasets tab dbbio anonymization conll2003 --skip-risk --limit 50

# Avec RAT-Bench niveau 1
python -m eval.run_pipeline_evaluation \
  --datasets tab dbbio ratbench conll2003 anonymization \
  --ratbench-levels 1 --ratbench-languages english --limit 50

# Candidat ARC/ResearchClaw
python -m eval.run_pipeline_evaluation \
  --candidate artifacts/improvement_tests/candidates/mon_candidat.json \
  --datasets tab dbbio ratbench conll2003 anonymization \
  --ratbench-levels 1 2 3 --ratbench-languages english \
  --llm-provider openrouter --output artifacts/eval-runs/manual-run
```

---

## Organisation

| Chemin | Rôle |
| --- | --- |
| [`eval/evaluate.py`](../../eval/evaluate.py) | Script unifié (benchmark / ablation / dataset) |
| [`eval/core/`](../../eval/core/) | Fonctions partagées : bootstrap, config, datasets, métriques, reporting |
| [`eval/pipegraph_eval_local.py`](../../eval/pipegraph_eval_local.py) | Exécution document par document, métriques de spans (strict + partial + BLEU) |
| [`eval/run_pipeline_evaluation.py`](../../eval/run_pipeline_evaluation.py) | Runner officiel ARC/ResearchClaw |
| [`eval/run_full_benchmark.py`](../../eval/run_full_benchmark.py) | Wrapper de compatibilité → runner officiel |
| [`eval/run_ablation.py`](../../eval/run_ablation.py) | Suites d'ablation (importées par `evaluate.py`) |
| [`eval/run_store.py`](../../eval/run_store.py) | Sauvegarde et lecture des runs |
| [`eval/ratbench_loader.py`](../../eval/ratbench_loader.py) | Loader et métriques RAT-Bench |
| [`eval/conll2003_loader.py`](../../eval/conll2003_loader.py) | Loader CoNLL-2003 / CleanCoNLL |
| [`eval/streamlit_app/`](../../eval/streamlit_app/) | Interface Streamlit d'analyse et comparaison |
| [`eval/datasets/`](../../eval/datasets/) | Datasets locaux, scripts de chargement et caches |

## Sorties

| Dossier | Contenu |
| --- | --- |
| `artifacts/eval-runs/<run-id>/` | Runs officiels ARC/ResearchClaw |
| `eval/evaluation/reports/` | Rapports JSON/Markdown (`evaluate.py benchmark`, historique) |
| `eval/evaluation/runs/` | Runs document-level (`evaluate.py --save-runs`, Streamlit) |
| `eval/datasets/*/cache/` | Datasets téléchargés et caches |

## Axes d'évaluation

| Axe | Description |
| --- | --- |
| `span_detection` | Strict exact (offsets+label), relaxed overlap, F2, métriques par label |
| `anonymization_leakage` | Fuite des valeurs gold dans le texte anonymisé |
| `ratbench_reid_risk` | R_succ (θ=0.2), avg_risk, attaque LLM OpenRouter |
| `utility_preservation` | BLEU (original vs. anonymisé), proxy DB-bio |
| `runtime` | Durée, secondes/document, erreurs et timeouts |

## RAT-Bench et OpenRouter

L'axe `ratbench_reid_risk` nécessite `OPENROUTER_API_KEY`. Sans clé :

- le run continue avec `risk_degraded` par défaut ;
- `--require-risk` fait échouer le dataset RAT-Bench (runner officiel) ;
- `--skip-risk` désactive explicitement cet axe.

## Documentation liée

- [Datasets et sources](datasets.md)
- [Métriques et rapports](metrics-and-reports.md)
- [Ablations](ablations.md)
