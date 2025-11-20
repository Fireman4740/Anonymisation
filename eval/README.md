# Benchmark du pipeline d'anonymisation

Ce dossier contient les ressources nécessaires pour évaluer automatiquement les performances du pipeline via l'API FastAPI exposée par `pipeline/scripts/api_server.py`.

## `benchmark_pipeline.py`

Script CLI qui:

- charge un ou plusieurs jeux de données (JSON ou JSONL, dossier ou fichier) contenant des exemples de texte;
- envoie chaque exemple à l'endpoint `/anonymize` de l'API FastAPI;
- mesure la latence, le débit et le taux de succès par dataset;
- peut exécuter une phase de chauffe et générer un rapport JSON détaillé.

### Installation des dépendances

Le script utilise les dépendances déjà déclarées à la racine du projet (`requests`, etc.). Assurez-vous d'installer `requirements.txt` à la racine ou d'activer l'environnement Conda/Wheel approprié (`conda activate ano`).

### Exemples d'utilisation

1. **Benchmark rapide sur le dataset par défaut**

   ```bash
   python benchmark_pipeline.py \
       --dataset datasets/data/anonymization_dataset.json \
       --limit 50 \
       --warmup 5 \
       --api-url http://localhost:8000 \
       --concurrency 4 \
       --output results/bench_default.json
   ```

2. **Évaluer un dossier complet de JSONL**

   ```bash
   python benchmark_pipeline.py \
       --dataset datasets/DB-bio \
       --per-dataset-limit 100 \
       --shuffle --seed 42 \
       --api-url http://0.0.0.0:8000 \
       --global-overrides '{"llm_paraphrase": false}'
   ```

### Paramètres importants

| Option | Description |
| --- | --- |
| `--dataset` | Fichiers ou dossiers JSON/JSONL à ingérer. Plusieurs chemins possibles. |
| `--limit` | Nombre d'exemples mesurés après warmup (global). |
| `--warmup` | Requêtes ignorées dans les statistiques pour laisser l'API chauffer. |
| `--concurrency` | Nombre de threads simultanés. |
| `--global-overrides` | JSON d'overrides appliqué à chaque requête. |
| `--output` | Sauvegarde un rapport détaillé (stats + résultats unitaires). |

### Prérequis

- L'API FastAPI doit être démarrée (`uvicorn pipeline.scripts.api_server:app --host 0.0.0.0 --port 8000`).
- Les jeux de données doivent contenir un champ texte (`text`, `original_text`, `prompt`, etc.).
- Pour les datasets volumineux, ajuster `--timeout` et `--concurrency` selon les ressources disponibles.

### Résultats

Le script affiche un tableau récapitulatif par dataset (taux de succès, latences moyennes/p95/pmax) et une ligne globale avec le débit moyen. Les rapports JSON contiennent chaque mesure détaillée, ce qui permet une analyse ultérieure (tableur, Pandas, etc.).

## `eval_rupta_dbbio.py`

Évalue la performance privacy/utility sur le dataset **DB-Bio** en utilisant l'API `/anonymize`.

```bash
python eval_rupta_dbbio.py \
        --dataset-dir eval/datasets/DB-bio \
        --split test \
        --n-samples 50 \
        --api-url http://localhost:8000 \
        --output results/dbbio_run.json
```

- `--use-baseline` désactive RUPTA côté pipeline pour comparer l'état actuel.
- `--p-threshold` et `--utility-threshold` ajustent les seuils de succès privacy/utility.
- `--global-overrides '{"llm_paraphrase": false}'` permet de transmettre des overrides API.
- Le script instancie `OpenRouterClient` : configurez `config.json` ou `OPENROUTER_API_KEY` avant exécution.

Chaque run produit
    - les réponses API (succès/erreurs, latence)
    - une évaluation privacy (rang de ré-identification, entités sensibles)
    - une évaluation utility (confiance de classification, entités confuses le cas échéant)
    - un résumé agrégé + rapport JSON optionnel.
