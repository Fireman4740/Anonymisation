# 🔧 Scripts Utilitaires

Ce dossier contient les scripts d'évaluation, de test et d'automatisation.

## 📜 Scripts Disponibles

### Suite de benchmarks

**`run_benchmarks.py`**

- Point d'entrée unique (sous-commandes `ner`, `pipeline`, `compare`, `quick`)
- Support des jeux DB-Bio, PersonalReddit, TAB
- Génération optionnelle de rapports JSON/Markdown

```bash
# Comparer NER standard vs GPU
python scripts/run_benchmarks.py ner --mode both --text-size mediumr

# Évaluer une politique complète
python scripts/run_benchmarks.py pipeline --dataset all --samples 10 --policy L1 --output results/eval_all.json

# Générer un rapport comparatif
python scripts/run_benchmarks.py compare --baseline results/baseline.json --rupta results/rupta.json --output results/report.md

# Workflow rapide (NER + DB-Bio)
python scripts/run_benchmarks.py quick --samples 3 --policy L1
```

### Téléchargement de Datasets

**`download_datasets.py`**

- Télécharge DB-Bio et PersonalReddit depuis Google Drive
- Mode interactif avec menu de sélection
- Extraction automatique

```bash
python scripts/download_datasets.py
```

### Exemples RUPTA

**`examples_rupta.py`**

- Démonstrations d'utilisation RUPTA
- Tests de privacy/utility evaluation
- Exemple d'optimisation complète

```bash
python scripts/examples_rupta.py
```

### Test Python 3.11

**`test_python311_compat.py`**

- Vérifie la compatibilité Python 3.11
- Teste les imports RUPTA
- Valide le baseline sans NER

```bash
python scripts/test_python311_compat.py
```

### Script Bash

**`quickstart.sh`** - Démarrage interactif

```bash
./scripts/quickstart.sh
```

- Vérifie l'environnement
- Télécharge les datasets si nécessaire
- Lance un test rapide

## ⚙️ Configuration Requise

### Variables d'environnement

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Dépendances

```bash
pip install -r requirements.txt
```

## 📊 Ordre d'Utilisation Recommandé

1. **Test compatibilité**

   ```bash
   python scripts/test_python311_compat.py
   ```

2. **Télécharger datasets**

   ```bash
   python scripts/download_datasets.py
   ```

3. **Tester avec exemples**

   ```bash
   python scripts/examples_rupta.py
   ```

4. **Évaluation complète**

   ```bash
   python scripts/run_benchmarks.py pipeline --dataset all --samples 10 --policy L1 --output results/eval_all.json
   ```

## 🔗 Scripts Principaux (Racine)

Certains scripts restent à la racine car utilisés fréquemment :

- **../eval_rupta_dbbio.py** - Évaluation RUPTA principale
- **../eval_tab.py** - Évaluation TAB benchmark
- **../main_eval.py** - API Flask
