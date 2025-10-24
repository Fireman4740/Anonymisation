# Suppression de DeepPavlov - Changelog

**Date**: 7 Octobre 2025

## 🎯 Objectif

Supprimer la dépendance à DeepPavlov qui nécessitait Python 3.9 et causait des problèmes de compatibilité avec Python 3.11. Le système utilise maintenant **uniquement GLiNER** pour le NER (Named Entity Recognition).

## ✅ Modifications Effectuées

### 1. **src/ner_ensemble.py**

#### Suppressions :
- ❌ Import DeepPavlov (`from deeppavlov import ...`)
- ❌ Variable `_DP_AVAILABLE`
- ❌ Variable `_DP_MODELS`
- ❌ Constante `DEEPPAVLOV_ENTITY_TAGS`
- ❌ Fonction `_maybe_move_deeppavlov_model()`
- ❌ Fonction `_load_dp_models()`
- ❌ Fonction `_normalize_dp_label()`
- ❌ Fonction `_decode_bio_to_spans()`
- ❌ Fonction `_dp_predict()`
- ❌ Fonction `run_deeppavlov_ner_ensemble()` (version standard)
- ❌ Fonction `run_deeppavlov_ner_ensemble()` (version fast mode)
- ❌ Section entière "DeepPavlov" (~300 lignes)

#### Modifications :
- ✅ Docstring mise à jour (suppression mention DeepPavlov)
- ✅ `__all__` mis à jour (suppression de `run_deeppavlov_ner_ensemble` et `DEEPPAVLOV_ENTITY_TAGS`)
- ✅ Fonction `warm_up_models()` : suppression du paramètre `dp_configs`

### 2. **src/orchestrator.py**

#### Suppressions :
- ❌ Import `run_deeppavlov_ner_ensemble` (2 endroits : import principal + fallback)
- ❌ Variables `use_dp` et `dp_cfgs`
- ❌ Overrides `ner_use_deeppavlov` et `ner_dp_configs`
- ❌ Appel `run_deeppavlov_ner_ensemble()`

#### Modifications :
- ✅ Commentaire Step 1bis : "NER — fusion externe + interne (GLiNER + HF)" au lieu de "DP + GLiNER + HF"
- ✅ Logique NER simplifiée : GLiNER en priorité, HF en fallback si pas de résultats GLiNER
- ✅ `merge_ner_lists()` : uniquement `gl_ents` et `hf_ents` au lieu de `dp_ents, gl_ents, hf_ents`

### 3. **scripts/eval_rupta_pipeline.py** (Nouveau - créé précédemment)

- ✅ Aucune dépendance DeepPavlov
- ✅ Nouveau paramètre `--policy` (L0 ou L1) pour tester différentes stratégies
- ✅ Support des 3 datasets : dbbio, reddit, tab

### 4. **scripts/run_rupta_eval.sh** (Mis à jour)

- ✅ Support du paramètre `policy` en argument (L0 ou L1)
- ✅ Noms de fichiers de sortie incluant le policy level : `eval_*_${POLICY}.json`

## 📊 Impact

### Avantages ✅
1. **Compatibilité Python 3.11** : Plus de problèmes avec torch<1.14.0
2. **Simplification** : Moins de dépendances, code plus maintenable
3. **Performance** : GLiNER est plus rapide et précis que DeepPavlov
4. **GPU Support** : Meilleure utilisation du GPU avec GLiNER

### Tests ✅
```bash
# Test avec 2 échantillons TAB
python scripts/eval_rupta_pipeline.py --dataset tab --split test --n_samples 2 --use_rupta --policy L1
```

Résultat : ✅ **Fonctionnel** - Les modèles GLiNER se chargent correctement et l'évaluation RUPTA fonctionne.

## 🔧 Configuration NER Actuelle

### GLiNER (Principal)
- **Modèles par défaut** : 
  - `urchade/gliner_large-v2.1`
  - `urchade/gliner_multi-v2.1`
- **Labels** : `GLINER_ALL_LABELS` (60+ entités incluant PII)
- **Threshold** : 0.35
- **GPU** : Auto-détection CUDA/MPS

### HuggingFace NER (Fallback)
- **Modèle** : `Davlan/bert-base-multilingual-cased-ner-hrl`
- **Usage** : Uniquement si GLiNER ne trouve rien
- **Mode** : Chunked avec stride pour textes longs

## 📝 Commandes Utiles

```bash
# Évaluation avec policy L1 (avec LLM + RUPTA)
./scripts/run_rupta_eval.sh dbbio L1

# Évaluation avec policy L0 (baseline sans LLM)
./scripts/run_rupta_eval.sh dbbio L0

# Test rapide TAB
python scripts/eval_rupta_pipeline.py --dataset tab --n_samples 10 --policy L1

# Évaluation complète 3 datasets
./scripts/run_rupta_eval.sh all L1
```

## 🎯 Prochaines Étapes

1. ✅ Tester l'évaluation complète sur DB-Bio
2. ✅ Tester l'évaluation complète sur PersonalReddit  
3. ✅ Tester l'évaluation complète sur TAB
4. ✅ Comparer les métriques L0 vs L1
5. ✅ Mettre à jour la documentation principale (README.md)

## 📚 Fichiers Modifiés

- `src/ner_ensemble.py` (~300 lignes supprimées)
- `src/orchestrator.py` (~15 lignes modifiées)
- `scripts/eval_rupta_pipeline.py` (nouveau fichier)
- `scripts/run_rupta_eval.sh` (mis à jour)
- `CHANGELOG_DEEPPAVLOV_REMOVAL.md` (ce fichier)

---

**✅ Migration terminée avec succès !**
