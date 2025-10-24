# Configuration pour éviter les problèmes de compatibilité Python 3.11

## Problème identifié

Le module NER (DeepPavlov) a des dépendances incompatibles avec Python 3.11 :
- `torch<1.14.0,>=1.6.0` n'existe pas pour Python 3.11 (seulement torch>=2.0)
- `transformers==4.30.0` incompatible avec gliner et sentence-transformers

## Solutions appliquées

### 1. Désactiver NER dans eval_rupta_dbbio.py

Le script d'évaluation RUPTA désactive maintenant automatiquement les NER internes :

```python
overrides = {
    "disable_internal_ner": True,  # Désactiver DeepPavlov/GLiNER/HF
    "llm_detection": False,        # Baseline simple sans LLM
    "llm_paraphrase": False        # Baseline simple sans paraphrase
}
```

### 2. Utiliser uniquement les regex

Pour l'évaluation RUPTA, le baseline utilise uniquement :
- Détection par regex (emails, téléphones, etc.)
- Pseudonymisation avec salt

Cela suffit pour comparer avec RUPTA qui optimise ensuite avec LLM.

### 3. Alternative : Environnement Python 3.9

Si vous avez besoin du NER complet, créez un environnement Python 3.9 :

```bash
conda create -n anno39 python=3.9
conda activate anno39
pip install -r requirements.txt
```

## Fichiers modifiés

- ✅ `eval_rupta_dbbio.py` : Désactivation NER, correction signature anonymize_text()
- ✅ Ce fichier de documentation

## Pour exécuter l'évaluation

```bash
# Avec Python 3.11 (sans NER, regex seulement)
python eval_rupta_dbbio.py --split test --n_samples 5 --use_baseline

# Avec Python 3.9 (NER complet)
conda activate anno39
python eval_rupta_dbbio.py --split test --n_samples 5 --use_baseline
```

## Dépendances minimales pour Python 3.11

```
# RUPTA essentiels
tqdm>=4.64.0
gdown>=4.7.0

# LLM (OpenRouter)
requests>=2.28.0

# Pas de NER complexe avec Python 3.11
# Utiliser regex + LLM uniquement
```

## Dépendances complètes pour Python 3.9

```
# Tout fonctionne avec Python 3.9
torch>=1.13,<2.0
transformers==4.30.0
deeppavlov>=1.0.0
gliner>=0.1.0
# ... reste des requirements.txt
```
