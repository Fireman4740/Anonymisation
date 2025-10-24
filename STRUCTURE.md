# 📁 Structure du Projet - Organisation Finale

## 🎯 Vue d'ensemble

Le projet a été nettoyé et organisé pour ne garder que **l'essentiel** :

- Code source fonctionnel
- Documentation principale
- Scripts d'évaluation
- Configuration

## 📂 Arborescence Complète

```
Anonymisation/
│
├── 📄 README.md                    # Documentation générale du système
├── 📄 README_RUPTA.md              # Documentation RUPTA complète
├── 📄 START_HERE.md                # Guide de démarrage rapide
├── 📄 QUICKSTART_RUPTA.md          # Guide RUPTA rapide
├── 📄 TODO_RUPTA.md                # Liste des tâches
├── 📄 BASELINE_FIX.md              # Documentation du fix baseline (IMPORTANT)
│
├── ⚙️  config.json                  # Configuration du système
├── 📦 requirements.txt             # Dépendances Python
├── 🔧 .gitignore                   # Fichiers à ignorer
├── 🔒 .env                         # Variables d'environnement (OPENROUTER_API_KEY)
│
├── 🐍 eval_rupta_dbbio.py          # Script principal d'évaluation RUPTA
├── 🐍 eval_tab.py                  # Évaluation TAB
├── 🐍 main_eval.py                 # API Flask
├── 📓 notebook_eval.ipynb          # Notebook de démonstration
├── 📊 comparison_report.md         # Dernier rapport de comparaison
│
├── 📁 src/                         # Code source principal
│   ├── __init__.py
│   ├── config_loader.py            # Chargement config
│   ├── llm_reasoner_openrouter.py  # Raisonnement LLM
│   ├── ner_ensemble.py             # NER ensemble
│   ├── openrouter_client.py        # Client API OpenRouter
│   ├── orchestrator.py             # Orchestrateur principal
│   ├── personal_info.py            # Détection infos personnelles
│   ├── policy.py                   # Politiques d'anonymisation
│   ├── text_sanitizer.py           # Nettoyage de texte
│   ├── utils_pseudo.py             # Utilitaires pseudonymisation
│   ├── whitelist_words.py          # Mots en liste blanche
│   │
│   └── 📁 rupta/                   # Module RUPTA
│       ├── __init__.py
│       ├── optimizer.py            # Optimisation itérative
│       ├── privacy_evaluator.py    # Évaluation privacy
│       ├── prompts_fr.py           # Prompts en français
│       └── utility_evaluator.py    # Évaluation utility
│
├── 📁 scripts/                     # Scripts utilitaires
│   ├── download_datasets.py        # Téléchargement datasets
│   ├── compare_baseline_rupta.py   # Comparaison résultats
│   ├── examples_rupta.py           # Exemples d'utilisation
│   ├── test_python311_compat.py    # Test compatibilité Python 3.11
│   ├── quickstart.sh               # Script démarrage interactif
│   └── run_eval.sh                 # Script d'évaluation complète
│
├── 📁 docs/                        # Documentation technique
│   ├── integration/                # Documentation intégration RUPTA
│   │   ├── PLAN_INTEGRATION_RUPTA.md
│   │   └── RECAP_INTEGRATION_RUPTA.md
│   │
│   └── python311/                  # Documentation Python 3.11
│       ├── PYTHON_311_COMPAT.md
│       ├── PYTHON_311_SOLUTION.md
│       ├── RECAP_FINAL_PYTHON311.md
│       ├── CHECKLIST_PYTHON311.md
│       └── MISSION_ACCOMPLIE.txt
│
├── 📁 results/                     # Résultats d'évaluation
│   └── old/                        # Anciens résultats archivés
│       ├── baseline.json
│       ├── rupta.json
│       ├── results_dbbio.json
│       ├── test_new_baseline.json
│       └── preds_tab_L0_test.json
│
└── 📁 Dataset/                     # Datasets d'évaluation
    ├── data/                       # Données d'origine
    │   ├── anonymization_dataset.json
    │   └── max_anonymization_dataset.json
    │
    └── evaluation/                 # Datasets RUPTA
        ├── DB-Bio/                 # Biographies célébrités
        │   ├── README.md
        │   ├── train.jsonl
        │   ├── dev.jsonl
        │   └── test.jsonl (239 exemples)
        │
        └── PersonalReddit/         # Commentaires Reddit
            ├── README.md
            ├── train.jsonl
            └── test.jsonl
```

## 📋 Fichiers Essentiels à la Racine

### Documentation (6 fichiers)

1. **README.md** - Vue générale du système d'anonymisation
2. **README_RUPTA.md** - Documentation complète RUPTA
3. **START_HERE.md** - Guide ultra-rapide (3 commandes)
4. **QUICKSTART_RUPTA.md** - Guide détaillé RUPTA
5. **TODO_RUPTA.md** - Tâches et roadmap
6. **BASELINE_FIX.md** - Fix critique du baseline (privacy_rank 1.0→10.0)

### Configuration (2 fichiers)

7. **config.json** - Configuration système et RUPTA
8. **requirements.txt** - Dépendances Python

### Scripts Principaux (4 fichiers)

9. **eval_rupta_dbbio.py** - Évaluation RUPTA sur DB-Bio
10. **eval_tab.py** - Évaluation TAB benchmark
11. **main_eval.py** - API Flask d'anonymisation
12. **notebook_eval.ipynb** - Notebook de démonstration

### Rapports (1 fichier)

13. **comparison_report.md** - Dernier rapport de comparaison

## 🗑️ Fichiers Supprimés

- ✅ `__pycache__/` - Cache Python (régénérable)
- ✅ `debug_dump/` - Données de debug volumineuses
- ✅ `.DS_Store` - Fichier système macOS
- ✅ `src/__pycache__/` - Cache modules

## 📦 Organisation Améliorée

### Avant (désorganisé)

```
Racine avec 25+ fichiers mélangés
Documentation, scripts, résultats tout ensemble
Fichiers cache et debug visibles
```

### Après (organisé)

```
✅ Racine : 13 fichiers essentiels
✅ docs/ : Documentation technique archivée
✅ scripts/ : Tous les scripts d'évaluation
✅ results/ : Résultats archivés
✅ Plus de cache visible
```

## 🚀 Commandes Rapides

### Démarrage

```bash
# Guide ultra-rapide
cat START_HERE.md

# Script interactif
./scripts/quickstart.sh

# Évaluation complète
./scripts/run_eval.sh
```

### Développement

```bash
# Tests
python scripts/test_python311_compat.py

# Exemples
python scripts/examples_rupta.py

# Évaluation
python eval_rupta_dbbio.py --split test --n_samples 10
```

### Documentation

```bash
# Général
cat README.md

# RUPTA
cat README_RUPTA.md

# Python 3.11
cat docs/python311/PYTHON_311_SOLUTION.md

# Fix baseline
cat BASELINE_FIX.md
```

## 📊 Statistiques

- **Fichiers racine** : 25 → 13 (réduction 48%)
- **Documentation** : Organisée en 2 dossiers thématiques
- **Scripts** : 6 scripts centralisés dans `scripts/`
- **Résultats** : Archivés dans `results/old/`
- **Cache** : Supprimé (régénérable)

## ✅ Avantages

1. **Clarté** - Structure logique et intuitive
2. **Maintenance** - Fichiers essentiels faciles à trouver
3. **Git** - `.gitignore` amélioré, moins de fichiers trackés
4. **Navigation** - Documentation et scripts séparés
5. **Performance** - Cache supprimé, dossier plus léger

## 🎯 Points d'Entrée

### Pour commencer

👉 `START_HERE.md` - 3 commandes pour démarrer

### Pour comprendre

👉 `README.md` - Architecture générale
👉 `README_RUPTA.md` - Méthodologie RUPTA

### Pour évaluer

👉 `eval_rupta_dbbio.py` - Script principal
👉 `scripts/run_eval.sh` - Évaluation automatique

### Pour développer

👉 `src/` - Code source
👉 `config.json` - Configuration
👉 `TODO_RUPTA.md` - Roadmap

## 📝 Notes

- **BASELINE_FIX.md** est crucial - garde la trace du fix privacy_rank 1.0→10.0
- **results/old/** conserve les anciens résultats pour comparaison
- **docs/** archive la documentation technique détaillée
- **.env** contient `OPENROUTER_API_KEY` (ne pas commiter)
