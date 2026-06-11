# Atlas_anno

`Atlas_anno` est un projet Python autonome pour :

- générer un lot synthétique d'anonymisation
- préannoter les documents
- évaluer la qualité de l'anonymisation
- exporter un pack de revue pour Label Studio
- réimporter les annotations humaines dans un format Atlas canonique

Le projet n'importe aucun code runtime depuis le reste du dépôt. Les secrets et endpoints sont lus depuis `.env`.

## Modèles utilisés

Configurables via `.env` :

- `ATLAS_MODEL_REASONING` — modèle pour les tâches structurées (ex. `tencent/hy3-preview`)
- `ATLAS_MODEL_CREATIVE` — modèle pour la génération de textes (ex. `tencent/hy3-preview`)

## Environnements recommandés

Utiliser deux environnements conda séparés :

- `ano` pour `Atlas_anno`
- `label-studio` pour le serveur Label Studio local

Cela évite les conflits `numpy/scipy` entre Atlas et Label Studio.

## Variables d'environnement

Copier `.env.example` vers `.env` :

```bash
cp /mnt/f/IA/Anonymisation/Atlas_anno/.env.example /mnt/f/IA/Anonymisation/Atlas_anno/.env
```

Variables utiles :

- `OPENROUTER_API_KEY` — clé API OpenRouter
- `OPENROUTER_BASE_URL` — endpoint (défaut : `https://openrouter.ai/api/v1/chat/completions`)
- `ATLAS_MODEL_REASONING` — modèle structuré (ex. `tencent/hy3-preview`)
- `ATLAS_MODEL_CREATIVE` — modèle créatif (ex. `tencent/hy3-preview`)
- `ATLAS_HTTP_TIMEOUT_SECONDS` — timeout HTTP (défaut : 60)
- `LABEL_STUDIO_URL` — URL du serveur (défaut : `http://127.0.0.1:8080`)
- `LABEL_STUDIO_TOKEN` — JWT refresh token Label Studio
- `LABEL_STUDIO_PROJECT_ID` — ID du projet (résolu automatiquement si absent)

## Préparation des environnements

### 1. Environnement Atlas

Activation :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano
```

### 2. Environnement Label Studio local

Création et installation du serveur :

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n label-studio python=3.11 -y
conda activate label-studio
python -m pip install label-studio
```

---

## Pipeline complète de création du dataset

Toutes les commandes s'exécutent dans le répertoire `Atlas_anno` avec l'environnement `ano`.

```bash
# Activation (à faire une seule fois par session)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano
cd /mnt/f/IA/Anonymisation/Atlas_anno
```

### Étape 1 — Générer le dataset

Génère mondes, personnages, scénarios, textes et paires d'attaque en une seule commande.

```bash
atlas generate-dataset \
  --documents 100 \
  --llm-mode primary-fallback \
  --reasoning-workers 6 \
  --creative-workers 4 \
  --no-resume \
  --no-cache
```

| Option | Valeurs | Description |
|--------|---------|-------------|
| `--llm-mode` | `primary-fallback` | LLM avec fallback déterministe si JSON invalide |
| `--llm-mode` | `disabled` | Génération 100 % déterministe (sans appel API) |
| `--resume` / `--no-resume` | — | Reprendre ou repartir de zéro |
| `--cache` / `--no-cache` | — | Réutiliser ou ignorer les réponses LLM en cache |

### Étape 2 — Valider le dataset

```bash
atlas validate-dataset
```

### Étape 3 — Construire les paires d'attaque

```bash
atlas build-attack-pairs
```

### Étape 4 — Calibrer la difficulté

```bash
atlas calibrate-difficulty
```

### Étape 5 — Préannoter les documents

```bash
atlas preannotate \
  --mode hybrid-llm \
  --batch pilot_100 \
  --reasoning-workers 6 \
  --no-resume \
  --no-cache
```

| Option | Valeurs | Description |
|--------|---------|-------------|
| `--mode` | `hybrid-llm` | LLM + règles symboliques |
| `--mode` | `disabled` | Règles symboliques uniquement (sans appel LLM) |
| `--resume` | — | Reprendre après une coupure réseau |

### Étape 6 — Anonymiser les documents

```bash
atlas run-anonymizer --strategy masking --mode auto
```

Stratégies disponibles : `masking`, `generalization`, `rewrite_balanced`.

### Étape 7 — Attaques adversariales

```bash
# Attaque symbolique (règles)
atlas attack-structured --strategy masking

# Attaque LLM (adversaire)
atlas attack-llm --strategy masking
```

### Étape 8 — Évaluation

```bash
atlas eval-spans   --strategy masking   # Précision / Rappel des spans
atlas eval-privacy --strategy masking   # Score de protection vie privée
atlas eval-reid    --strategy masking   # Risque de ré-identification
atlas eval-utility --strategy masking   # Impact sur l'utilité du texte
```

### Étape 9 — Construire le rapport

```bash
atlas build-report --strategy masking
```

### Étape 10 — Exporter

```bash
# Export Parquet (pour analyse)
atlas export-parquet --batch pilot_100

# Export pack Label Studio (pour revue humaine)
atlas export-review-pack --target label-studio --batch pilot_100 --selection all
```

---

### Pipeline complète en une seule commande

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && \
cd /mnt/f/IA/Anonymisation/Atlas_anno && \
atlas generate-dataset --documents 100 --llm-mode primary-fallback --reasoning-workers 6 --creative-workers 4 --no-resume --no-cache && \
atlas validate-dataset && \
atlas build-attack-pairs && \
atlas calibrate-difficulty && \
atlas preannotate --mode hybrid-llm --batch pilot_100 --reasoning-workers 6 --no-resume --no-cache && \
atlas run-anonymizer --strategy masking --mode auto && \
atlas attack-structured --strategy masking && \
atlas attack-llm --strategy masking && \
atlas eval-spans --strategy masking && \
atlas eval-privacy --strategy masking && \
atlas eval-reid --strategy masking && \
atlas eval-utility --strategy masking && \
atlas build-report --strategy masking && \
atlas export-parquet --batch pilot_100 && \
atlas export-review-pack --target label-studio --batch pilot_100 --selection all && \
echo "=== Pipeline terminée ==="
```

> **En cas de coupure réseau** pendant `preannotate` ou `attack-llm`, relancer uniquement l'étape concernée avec `--resume`.

---

## Workflow Label Studio

### 1. Démarrer le serveur Label Studio

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate label-studio
label-studio start
```

Interface accessible sur `http://127.0.0.1:8080`.

### 2. Créer le projet Label Studio

Le project ID est résolu automatiquement et mis en cache dans `data/review/{batch}/.label_studio_project_id`.

```bash
python scripts/label_studio/create_project.py --batch pilot_100 --title "Atlas pilot_100 review"
```

Options avancées :

```bash
# Surcharger le token ou l'URL
python scripts/label_studio/create_project.py --batch pilot_100 --token "eyJ..." --url "http://localhost:8080"
```

### 3. Importer les tâches

Le project ID est résolu automatiquement depuis le cache :

```bash
python scripts/label_studio/import_tasks.py --batch pilot_100
```

### 4. Exporter les annotations après revue

```bash
python scripts/label_studio/export_annotations.py --batch pilot_100
```

### 5. Réimporter les annotations dans Atlas

```bash
atlas import-review-pack --target label-studio --batch pilot_100 \
  --input data/review/pilot_100/label_studio_export.json
```

---

## Artefacts produits

| Fichier | Description |
|---------|-------------|
| `data/raw_docs/raw_docs.jsonl` | Documents texte générés |
| `data/annotations/preannotations.jsonl` | Pré-annotations LLM/symboliques |
| `data/annotations/reviewed_annotations.jsonl` | Annotations après revue humaine |
| `data/anonymized/masking.jsonl` | Documents anonymisés |
| `data/attacks/pairs.jsonl` | Paires d'attaque |
| `data/reports/masking_spans.json` | Éval Précision/Rappel |
| `data/reports/masking_privacy.json` | Éval protection vie privée |
| `data/reports/masking_reid.json` | Éval ré-identification |
| `data/reports/masking_utility.json` | Éval utilité |
| `data/reports/masking_report.json` | Rapport global |
| `data/parquet/pilot_100/*.parquet` | Exports Parquet |
| `data/review/pilot_100/label_studio_tasks.json` | Tâches Label Studio |
| `data/review/pilot_100/label_config.xml` | Configuration interface Label Studio |
| `data/logs/llm_runs.jsonl` | Journal des appels LLM |

---

## Diagnostic

Afficher les derniers runs LLM :

```bash
atlas inspect-llm-runs --limit 20
```

## Dépannage

- Si OpenRouter est lent ou coupe, ajouter `--resume` et relancer l'étape concernée.
- Si tu ne veux aucun appel LLM, utiliser `--llm-mode disabled` (generate-dataset) ou `--mode disabled` (preannotate).
- Si `label-studio command not found`, activer l'environnement `label-studio`.
- Le project ID Label Studio est sauvegardé automatiquement dans `data/review/{batch}/.label_studio_project_id` à la création.
- Les warnings `SECRET_KEY` au premier démarrage de Label Studio ne sont pas bloquants.
