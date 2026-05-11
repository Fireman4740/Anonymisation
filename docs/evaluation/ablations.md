# Ablations

Les ablations sont exécutées par [`eval/run_ablation.py`](../../eval/run_ablation.py). Elles comparent plusieurs configurations PipeGraph sur un même dataset afin d'isoler l'impact des composants, des modèles ou des paramètres.

## Commandes utiles

Lister les suites :

```bash
python eval/run_ablation.py --list-suites
```

Afficher une grille sans l'exécuter :

```bash
python eval/run_ablation.py --suite nodes --configs-only
```

Exécuter une ablation rapide :

```bash
python eval/run_ablation.py --dataset tab --limit 10 --suite nodes
```

Exécuter une ablation RAT-Bench :

```bash
python eval/run_ablation.py --dataset ratbench --level 1 --limit 50 --suite ner_ensemble
```

Sauvegarder les runs pour Streamlit :

```bash
python eval/run_ablation.py --dataset tab --limit 50 --suite full --with-llm --save-runs
```

## Options principales

| Option | Description |
| --- | --- |
| `--dataset` | `tab`, `dbbio`, `anonymization`, `ratbench`, `conll2003`. |
| `--split` | Split `test`, `dev` ou `train` quand applicable. |
| `--language` | Langue RAT-Bench : `english`, `mandarin`, `spanish`. |
| `--level` | Niveau RAT-Bench : `1`, `2`, `3`. |
| `--limit` | Nombre maximum de documents par configuration. |
| `--suite` | Suite d'ablation. |
| `--custom-config` | Grille JSON personnalisée. |
| `--save-runs` | Sauvegarde chaque run détaillé. |
| `--out` | Chemin du résumé JSON. |
| `--with-llm` | Force les modules LLM actifs dans la grille. |
| `--parallel-configs` | Nombre de configurations évaluées en parallèle, ou `auto`. |

## Suites disponibles

### `nodes`

Compare les grands blocs du pipeline :

- regex uniquement ;
- NER uniquement ;
- regex + NER ;
- regex + NER + LLM detection ;
- regex + NER + LLM detection + verification ;
- regex + NER + LLM verification uniquement ;
- pipeline complet avec RUPTA.

Objectif : mesurer la contribution de chaque famille de noeuds.

### `ner_presets`

Compare les presets GLiNER :

- `fast`
- `balanced`
- `pii`
- `multitask`
- `accuracy`
- `best`
- `full`

Objectif : mesurer le compromis vitesse, couverture et précision.

### `ner_ensemble`

Compare des modèles GLiNER individuels et des combinaisons :

- modèles seuls ;
- paires ;
- triplets ;
- ensemble complet.

Objectif : identifier l'apport du multi-modèle et du consensus.

### `ner_threshold`

Fait varier `gliner_threshold`, par exemple de `0.10` à `0.60`.

Objectif : mesurer l'effet du seuil de confiance sur les faux positifs et les fuites.

### `ner_vote`

Fait varier `ner_min_vote`.

Objectif : contrôler le niveau de consensus requis entre modèles NER. Une valeur basse favorise le recall ; une valeur haute favorise la précision.

### `anon_strategy`

Compare les stratégies d'anonymisation :

- `pseudo`
- `generalize`
- `mask`
- `redact`
- politique mixte par type

Objectif : mesurer l'effet de la transformation sur les leaks et les métriques downstream.

### `detection_mode`

Compare :

- `serial`
- `parallel`

Objectif : mesurer l'impact du mode d'exécution sur le temps et vérifier que les résultats restent cohérents.

### `full`

Concatène toutes les suites prédéfinies. C'est une suite longue.

### `custom`

Charge une grille depuis `--custom-config`.

Format attendu :

```json
[
  {
    "name": "custom_run",
    "description": "Description courte",
    "config": {
      "enable_detection": true,
      "enable_deterministic": true,
      "enable_ai": true,
      "enable_anonymization": true,
      "disable_llm": true
    }
  }
]
```

## Sorties

Le script affiche un tableau comparatif avec :

- precision ;
- recall ;
- F2 ;
- leaks ;
- durée ;
- description.

Le résumé JSON est écrit par défaut dans :

```text
eval/evaluation/reports/ablation_<suite>_<dataset>.json
```

Avec `--save-runs`, chaque configuration écrit aussi un run détaillé dans :

```text
eval/evaluation/runs/
```

## Parallélisme

`--parallel-configs auto` choisit automatiquement :

- davantage de parallélisme quand aucun LLM n'est actif ;
- un parallélisme plus modéré quand des appels LLM sont présents.

Pour déboguer, utiliser :

```bash
python eval/run_ablation.py --dataset tab --suite nodes --parallel-configs 1
```

## Bonnes pratiques

- Commencer avec `--limit 10` ou `--limit 50`.
- Utiliser `--configs-only` avant une grille longue.
- Utiliser `--save-runs` seulement quand une inspection Streamlit est nécessaire.
- Éviter `--with-llm` sans clé API ou provider local fonctionnel.
- Comparer prioritairement le recall et F2 pour l'anonymisation, puis examiner les faux positifs et leaks document par document.
