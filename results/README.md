# 📊 Résultats d'Évaluation

Ce dossier contient les résultats des évaluations RUPTA.

## 📁 Structure

```
results/
├── README.md           # Ce fichier
└── old/                # Anciens résultats archivés
    ├── baseline.json
    ├── rupta.json
    ├── results_dbbio.json
    ├── test_new_baseline.json
    └── preds_tab_L0_test.json
```

## 📝 Format des Fichiers

### baseline.json / rupta.json
Résultats d'évaluation RUPTA au format :
```json
{
  "config": {
    "split": "test",
    "n_samples": 10,
    "method": "Baseline" ou "RUPTA",
    "p_threshold": 10
  },
  "metrics": {
    "avg_privacy_rank": 10.0,
    "privacy_not_identified_rate": 0.0,
    "avg_utility_confidence": 100.0,
    "utility_preserved_rate": 1.0,
    "n_samples": 10
  },
  "results": [...]
}
```

### Métriques Importantes

**Privacy Metrics**
- `avg_privacy_rank` : Position moyenne de la vraie personne (>10 = non identifié)
- `privacy_not_identified_rate` : % de cas où rank > p_threshold

**Utility Metrics**
- `avg_utility_confidence` : Confiance moyenne dans la prédiction (%)
- `utility_preserved_rate` : % de cas où l'utilité est préservée

## 🎯 Objectifs RUPTA

Un bon compromis privacy-utility montre :
- **Privacy** : avg_privacy_rank > 10 ou non-identified > 60%
- **Utility** : confidence > 80%, preserved > 75%

## 📂 Anciens Résultats (old/)

Les anciens résultats sont archivés pour :
- Comparaison avec nouvelles versions
- Analyse de l'évolution
- Validation des améliorations

### Résultats Archivés

1. **baseline.json** - Évaluation baseline initiale (invalide - voir BASELINE_FIX.md)
2. **rupta.json** - Évaluation RUPTA initiale (invalide)
3. **test_new_baseline.json** - Test après fix baseline (privacy_rank: 10.0 ✅)
4. **results_dbbio.json** - Premier test DB-Bio

⚠️ **Note** : Les résultats baseline.json et rupta.json initiaux sont invalides car le baseline ne masquait pas les noms. Voir `../BASELINE_FIX.md` pour détails.

## 🔄 Générer de Nouveaux Résultats

### Baseline
```bash
python eval_rupta_dbbio.py \
  --split test \
  --n_samples 10 \
  --use_baseline \
  --output results/baseline_new.json
```

### RUPTA
```bash
python eval_rupta_dbbio.py \
  --split test \
  --n_samples 10 \
  --output results/rupta_new.json
```

### Comparaison
```bash
python scripts/compare_baseline_rupta.py \
  --baseline results/baseline_new.json \
  --rupta results/rupta_new.json \
  --output comparison_report.md
```

## 📈 Analyse

Les résultats permettent de :
1. Mesurer l'amélioration RUPTA vs Baseline
2. Valider le compromis privacy-utility
3. Identifier les cas problématiques
4. Optimiser les hyperparamètres (p_threshold, max_iterations)

## 🔒 Confidentialité

Les résultats contiennent :
- Textes anonymisés (pas d'infos sensibles)
- Métriques agrégées
- Pas de données personnelles brutes

Safe to commit dans git.
