# Atlas_anno

`Atlas_anno` est un projet Python autonome pour :

- générer un lot synthétique d’anonymisation
- préannoter les documents
- exporter un pack de revue pour Label Studio
- réimporter les annotations humaines dans un format Atlas canonique

Le projet n’importe aucun code runtime depuis le reste du dépôt. Les secrets et endpoints sont lus depuis `.env`.

## Modèles utilisés

- `aion-labs/aion-2.0` pour les tâches structurées et complexes
- `mistralai/mistral-small-creative` pour les tâches créatives

## Environnements recommandés

Utiliser deux environnements conda séparés :

- `ano` pour `Atlas_anno`
- `label-studio` pour le serveur Label Studio local

Cela évite les conflits `numpy/scipy` entre Atlas et Label Studio.

## Variables d’environnement

Copier `.env.example` vers `.env` :

```bash
cp /mnt/f/IA/Anonymisation/Atlas_anno/.env.example /mnt/f/IA/Anonymisation/Atlas_anno/.env
```

Variables utiles :

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL`
- `ATLAS_HTTP_TIMEOUT_SECONDS`
- `LABEL_STUDIO_URL`
- `LABEL_STUDIO_TOKEN`
- `LABEL_STUDIO_PROJECT_ID`

Valeurs par défaut attendues pour Label Studio local :

```dotenv
LABEL_STUDIO_URL=http://127.0.0.1:8080
LABEL_STUDIO_TOKEN=
LABEL_STUDIO_PROJECT_ID=
```

`LABEL_STUDIO_TOKEN` correspond au personal access token / refresh token. Le client garde aussi la compatibilité avec l’ancien `LABEL_STUDIO_API_TOKEN` si nécessaire.

## Préparation des environnements

### 1. Environnement Atlas

Si `ano` existe déjà et que le projet tourne, tu peux garder ton environnement actuel.

Activation :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano
```

### 2. Environnement Label Studio local

Création et installation du serveur :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda create -n label-studio python=3.11 -y && conda activate label-studio && python -m pip install --upgrade pip && python -m pip install label-studio
```

## Commandes Atlas en WSL

Toutes les commandes Atlas ci-dessous s’exécutent dans `ano`.

### Générer le lot pilote

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && PYTHONPATH=src python -m atlas_anno.cli generate-dataset --documents 100 --llm-mode primary-fallback --reasoning-workers 12 --creative-workers 8 --resume --cache
```

### Préannotation rapide

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && PYTHONPATH=src python -m atlas_anno.cli preannotate --mode hybrid-llm --batch pilot_100 --reasoning-workers 12 --resume --cache
```

### Préannotation sans LLM

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && PYTHONPATH=src python -m atlas_anno.cli preannotate --mode disabled --batch pilot_100 --reasoning-workers 1 --no-resume --no-cache
```

### Export du pack de revue Label Studio

Tous les documents :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && PYTHONPATH=src python -m atlas_anno.cli export-review-pack --target label-studio --batch pilot_100 --selection all
```

Seulement les documents prioritaires :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && PYTHONPATH=src python -m atlas_anno.cli export-review-pack --target label-studio --batch pilot_100 --selection review-required
```

## Workflow Label Studio local

Le flux recommandé est :

1. générer et préannoter dans `ano`
2. lancer le serveur dans `label-studio`
3. créer le projet Label Studio
4. importer les tâches
5. annoter dans l’interface web
6. exporter les annotations
7. réimporter les annotations dans Atlas

### 1. Démarrer Label Studio local

Cette commande doit être lancée dans l’environnement `label-studio` :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate label-studio && cd /mnt/f/IA/Anonymisation/Atlas_anno && bash scripts/label_studio/start_local.sh
```

Ou directement :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate label-studio && label-studio start
```

L’interface sera en général sur :

```text
http://127.0.0.1:8080
```

### 2. Créer ou mettre à jour le projet Label Studio

Cette commande peut être lancée depuis `ano` :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && python scripts/label_studio/create_project.py --batch pilot_100 --title "Atlas pilot_100 review"
```

Si la réponse contient un identifiant de projet, mets-le dans `.env` comme `LABEL_STUDIO_PROJECT_ID`.

### 3. Importer les tâches dans le projet

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && python scripts/label_studio/import_tasks.py --batch pilot_100 --project-id "$LABEL_STUDIO_PROJECT_ID"
```

### 4. Exporter les annotations revues depuis Label Studio

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && python scripts/label_studio/export_annotations.py --batch pilot_100 --project-id "$LABEL_STUDIO_PROJECT_ID" --output data/review/pilot_100/label_studio_export.json
```

### 5. Réimporter les annotations revues dans Atlas

Commande CLI Atlas :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && PYTHONPATH=src python -m atlas_anno.cli import-review-pack --target label-studio --batch pilot_100 --input data/review/pilot_100/label_studio_export.json
```

Commande helper équivalente :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && python scripts/label_studio/sync_reviewed.py --batch pilot_100 --input data/review/pilot_100/label_studio_export.json
```

## Artefacts produits

- `data/worlds/worlds.jsonl`
- `data/characters/characters.jsonl`
- `data/scenarios/scenarios.jsonl`
- `data/raw_docs/raw_docs.jsonl`
- `data/annotations/preannotations.jsonl`
- `data/annotations/reviewed_annotations.jsonl`
- `data/review/pilot_100/label_studio_tasks.json`
- `data/review/pilot_100/label_config.xml`
- `data/review/pilot_100/label_studio_export.json`
- `data/batches/pilot_100/manifest.json`
- `data/logs/llm_runs.jsonl`

## Diagnostic

Afficher les derniers runs LLM :

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ano && cd /mnt/f/IA/Anonymisation/Atlas_anno && PYTHONPATH=src python -m atlas_anno.cli inspect-llm-runs --limit 20
```

## Dépannage

- Si OpenRouter est lent, garde `--resume --cache` et relance la même commande.
- Si tu ne veux aucun appel LLM pendant la préannotation, utilise `--mode disabled`.
- Si `label-studio command not found in PATH`, tu n’es pas dans l’environnement `label-studio` ou le package n’est pas installé.
- Si `LABEL_STUDIO_PROJECT_ID` est vide, passe `--project-id` explicitement aux scripts.
- Les warnings sur `SECRET_KEY` au premier démarrage de Label Studio local ne sont pas bloquants.
