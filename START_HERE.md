# 🚀 Démarrage Ultra-Rapide - RUPTA Python 3.11

> 💡 **Nouveau projet ?** Consultez d'abord [`STRUCTURE.md`](STRUCTURE.md) pour comprendre l'organisation.

## ⚡ En 3 commandes

```bash
# 1. Installer gdown
pip install gdown

# 2. Définir clé API
export OPENROUTER_API_KEY=sk-or-v1-...

# 3. Lancer le script interactif
./scripts/quickstart.sh
```

## 🎯 Ou manuellement

### Test simple (30 secondes)

```bash
python eval_rupta_dbbio.py --split test --n_samples 1 --use_baseline
```

### Évaluation complète (5 minutes)

```bash
# Baseline
python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline --output baseline.json

# RUPTA
python eval_rupta_dbbio.py --split test --n_samples 10 --output rupta.json

# Comparaison
python compare_baseline_rupta.py --baseline baseline.json --rupta rupta.json
```

## ✅ Validation Python 3.11

```bash
python test_python311_compat.py
```

Doit afficher :

```
✅ Python 3.11: Supporté
✅ Imports RUPTA: OK
✅ Baseline (regex): Fonctionne
```

## 📚 Documentation

- **Vue d'ensemble** : [STRUCTURE.md](STRUCTURE.md) - Architecture complète 🆕
- **Guide complet** : [README_RUPTA.md](README_RUPTA.md)
- **Quick start** : [QUICKSTART_RUPTA.md](QUICKSTART_RUPTA.md)
- **Solution Python 3.11** : [PYTHON_311_SOLUTION.md](docs/python311/PYTHON_311_SOLUTION.md)
- **Fix baseline** : [BASELINE_FIX.md](BASELINE_FIX.md) ⚠️ Important
- **TODO** : [TODO_RUPTA.md](TODO_RUPTA.md)

## ⚠️ Important

### Python 3.11

- ✅ Regex anonymization
- ✅ RUPTA (privacy + utility)
- ❌ NER complexe (DeepPavlov/GLiNER)

### Clé API requise

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

Ou créez `.env` :

```bash
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env
```

## 💰 Coûts estimés

| Évaluation  | Exemples | Coût (gpt-4o-mini) |
| ----------- | -------- | ------------------ |
| Test rapide | 1        | ~$0.01             |
| Moyenne     | 10       | ~$0.10             |
| Complète    | 100      | ~$1.00             |

## 🐛 Problèmes ?

### Dataset manquant

```bash
python scripts/download_datasets.py
# Option 1 : DB-Bio
```

### Import errors

```bash
pip install geonamescache schwifty tqdm
```

### Clé API invalide

```bash
echo $OPENROUTER_API_KEY  # Vérifier
```

## 🎉 C'est tout !

Le système est maintenant **100% compatible Python 3.11** et prêt à utiliser.
