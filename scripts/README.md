# 🔧 Scripts Utilitaires

Ce dossier contient les scripts d'évaluation, de test et d'automatisation.

## 📜 Scripts Disponibles

### Téléchargement de Datasets

**`download_datasets.py`**

- Télécharge DB-Bio et PersonalReddit depuis Google Drive
- Mode interactif avec menu de sélection
- Extraction automatique

```bash
python scripts/download_datasets.py
```

### Comparaison Baseline vs RUPTA

**`compare_baseline_rupta.py`**

- Compare les résultats baseline et RUPTA
- Génère un rapport markdown détaillé
- Calcule les améliorations privacy/utility

```bash
python scripts/compare_baseline_rupta.py \
  --baseline results/old/baseline.json \
  --rupta results/old/rupta.json \
  --output comparison_report.md
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

### Scripts Bash

**`quickstart.sh`** - Démarrage interactif

```bash
./scripts/quickstart.sh
```

- Vérifie l'environnement
- Télécharge les datasets si nécessaire
- Lance un test rapide

**`run_eval.sh`** - Évaluation complète

```bash
./scripts/run_eval.sh
```

- Évaluation baseline (10 exemples)
- Évaluation RUPTA (10 exemples)
- Génération du rapport comparatif

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
   ./scripts/run_eval.sh
   ```

## 🔗 Scripts Principaux (Racine)

Certains scripts restent à la racine car utilisés fréquemment :

- **../eval_rupta_dbbio.py** - Évaluation RUPTA principale
- **../eval_tab.py** - Évaluation TAB benchmark
- **../main_eval.py** - API Flask
