# Changelog

Toutes les modifications notables de ce projet sont documentées dans ce fichier.

## [1.1.0] - 2024-01-XX - RUPTA Integration

### ✨ Nouveautés Majeures

#### 🌍 Support Multilingual
- **Prompts universels** : Tous les prompts RUPTA convertis de français vers anglais universel
- **Langues supportées** : Toutes les langues européennes (FR, EN, DE, ES, IT, PT, NL, etc.)
- **Fichier renommé** : `prompts_fr.py` → `prompts_multilang.py`
- **Paramètre langue** : `language="auto"` par défaut (détection automatique)

#### 🔐 RUPTA Privacy-Utility Optimization
- **Intégration complète** au niveau L1 (avec LLM)
- **Optimisation itérative** du tradeoff privacy-utility
- **Métriques combinées** : 50% privacy + 50% utility
- **Configuration flexible** via `AnonymizationPolicy`

### 📊 Nouveaux Paramètres Policy

Ajout de 5 paramètres RUPTA dans `AnonymizationPolicy` :
```python
rupta_enabled: bool = False              # Active RUPTA
rupta_p_threshold: int = 10              # Nombre de candidats pour privacy
rupta_max_iterations: int = 3            # Itérations d'optimisation max
rupta_privacy_threshold: int | None = None  # Seuil privacy (None = cherche non-id)
rupta_utility_threshold: int = 80        # Seuil minimum d'utilité (%)
```

### 🔧 Modifications de Code

#### `src/policy.py`
- Ajout des 5 paramètres RUPTA à `AnonymizationPolicy`
- Preset L1 activé avec RUPTA par défaut :
  - `rupta_enabled=True`
  - Configuration optimale : p_threshold=10, max_iterations=3, utility≥80%

#### `src/orchestrator.py`
- **Nouvelle Step 7** : RUPTA Privacy-Utility Optimization
- Imports ajoutés :
  - `from .rupta.privacy_evaluator import evaluate_reidentification_risk`
  - `from .rupta.utility_evaluator import evaluate_utility_preservation`
- Logique d'optimisation itérative :
  1. Évaluation privacy (re-identification risk)
  2. Évaluation utility (classification confidence)
  3. Calcul récompense combinée
  4. Optimisation (masquage + paraphrase)
  5. Convergence vers meilleur tradeoff
- Retourne `rupta_metrics` dans le résultat

#### `src/rupta/prompts_multilang.py` (renommé)
- 9 prompts convertis de français vers anglais :
  - `PRIVACY_REFLECTION_FR_1` : Génération candidats
  - `PRIVACY_REFLECTION_FR_2` : Identification entités sensibles
  - `PRIVACY_CONFIDENCE_FR` : Score de confiance
  - `UTILITY_REFLECTION_FR_1` : Classification d'occupation
  - `UTILITY_CONFUSED_ENTITIES_FR` : Détection entités confuses
  - `REINFORCEMENT_FR` : Optimisation par récompense
  - `SIMPLE_REWRITING_FR` : Anonymisation
  - `DETECTION_FR` : Détection PII
  - `GENERAL_SYSTEM_FR` : Prompt système

#### `src/rupta/privacy_evaluator.py`
- Import mis à jour : `prompts_fr` → `prompts_multilang`
- Paramètre `language` : `"fr"/"en"` → `"auto"`
- Documentation multilingue complète

#### `src/rupta/utility_evaluator.py`
- Import mis à jour : `prompts_fr` → `prompts_multilang`
- Ajout d'alias `evaluate_utility_preservation()` pour compatibilité

### 📜 Nouveaux Scripts

#### `scripts/eval_rupta_pipeline.py`
Évaluation complète du pipeline sur datasets :
- Support DB-Bio et PersonalReddit
- Mode baseline (L0) et RUPTA (L1)
- Métriques détaillées : privacy rank, utility score, iterations, reward
- Export JSON avec résumé statistique

**Usage** :
```bash
python scripts/eval_rupta_pipeline.py \
    --dataset dbbio \
    --n_samples 50 \
    --use_rupta \
    --output results/eval_dbbio.json
```

#### `scripts/compare_baseline_rupta.py`
Génération de rapport comparatif Markdown :
- Comparaison Baseline vs RUPTA
- Analyse du tradeoff privacy-utility
- Recommandations d'optimisation
- Tableau de métriques

**Usage** :
```bash
python scripts/compare_baseline_rupta.py \
    --results results/eval_dbbio.json \
    --output results/comparison_report.md
```

#### `scripts/test_rupta_integration.py`
Test rapide d'intégration :
- 3 cas multilingues (FR/EN/ES)
- Validation baseline vs RUPTA
- Vérification amélioration privacy/utility

**Usage** :
```bash
python scripts/test_rupta_integration.py
```

#### `scripts/run_rupta_eval.sh`
Launcher bash pour évaluation rapide :
- Commandes : `test`, `pilot`, `dbbio`, `reddit`, `all`, `compare`, `quick`, `full`
- Workflow automatisé
- Scripts colorés et interactifs

**Usage** :
```bash
./scripts/run_rupta_eval.sh quick  # Test + Pilot + Compare
./scripts/run_rupta_eval.sh all    # Évaluation complète
```

### 📚 Nouvelle Documentation

#### `docs/RUPTA_INTEGRATION.md`
Guide complet d'utilisation RUPTA :
- Vue d'ensemble et fonctionnalités
- Quick start et configuration
- Support multilingual
- Évaluation et métriques
- Configuration avancée
- Troubleshooting
- Exemples pratiques

#### `scripts/README.md`
Documentation des scripts d'évaluation :
- Description de chaque script
- Options et arguments
- Workflow recommandé
- Exemples d'usage
- Métriques expliquées

#### `RUPTA_RECAP.md`
Récapitulatif de l'intégration :
- Travaux complétés (checklist)
- Architecture RUPTA
- Configuration par défaut
- Métriques attendues
- Next steps
- Commandes utiles

#### `README.md` (mis à jour)
Ajout section 9.1 RUPTA :
- Fonctionnement
- Configuration
- Utilisation
- Langues supportées
- Commandes d'évaluation

### 🔄 Changements de Compatibilité

#### Imports
**Avant** :
```python
from src.rupta.prompts_fr import PRIVACY_REFLECTION_FR_1
```

**Après** :
```python
from src.rupta.prompts_multilang import PRIVACY_REFLECTION_FR_1
```

#### Paramètre langue
**Avant** :
```python
evaluate_reidentification_risk(..., language="fr")
```

**Après** :
```python
evaluate_reidentification_risk(..., language="auto")  # Détection auto
```

### 📈 Métriques de Performance

#### Attendues sur DB-Bio (1000 samples test)
**Baseline (L0)** :
- Privacy : ~30-40% non-identifiés
- Utility : ~70-75% score moyen
- Temps : ~10 min

**RUPTA (L1)** :
- Privacy : ~70-85% non-identifiés (+40-50%)
- Utility : ~80-85% score moyen (+5-10%)
- Temps : ~30-45 min (3x plus lent)

### 🐛 Corrections

- Import circulaire résolu dans `utility_evaluator.py`
- Alias `evaluate_utility_preservation()` ajouté pour compatibilité scripts
- Typage amélioré (warnings Pylance réduits)

### ⚠️ Breaking Changes

Aucun breaking change majeur. L'ancienne API reste compatible :
- L0 continue de fonctionner sans RUPTA
- L1 active RUPTA par défaut (peut être désactivé via `overrides`)
- Les anciens scripts restent fonctionnels

### 🚀 Migration

Pour activer RUPTA sur du code existant :

```python
# Avant (L0 uniquement)
result = anonymize_text(value=text, level="L0", ...)

# Après (L1 avec RUPTA)
result = anonymize_text(
    value=text,
    level="L1",
    overrides={
        "rupta_ground_truth_people": "Marie Curie",
        "rupta_ground_truth_label": "physicist"
    }
)

# Accès aux métriques RUPTA
rupta_metrics = result.get("rupta_metrics", {})
privacy_rank = rupta_metrics.get("privacy", {}).get("rank")
utility_score = rupta_metrics.get("utility", {}).get("confidence_score")
```

### 🔍 Tests

#### Tests Manuels
- [x] Test d'intégration multilingue (FR/EN/ES)
- [ ] Évaluation pilote DB-Bio (50 samples)
- [ ] Évaluation pilote PersonalReddit (50 samples)
- [ ] Évaluation complète DB-Bio (1000 samples)
- [ ] Évaluation complète PersonalReddit (~1000 samples)

#### Tests Automatisés
- [ ] Tests unitaires RUPTA
- [ ] Tests d'intégration orchestrator
- [ ] Tests de régression

### 📦 Dépendances

Aucune nouvelle dépendance ajoutée. Utilise les packages existants :
- `openrouter` (API LLM)
- Modèles NER existants
- Bibliothèques standard Python

### 🙏 Crédits

- **RUPTA Framework** : [github.com/RUPTA-anonymization/RUPTA](https://github.com/RUPTA-anonymization/RUPTA)
- **Datasets** : DB-Bio (10k biographies), PersonalReddit (8k commentaires)
- **Prompts** : Adaptés et multilingues depuis RUPTA original

---

## [1.0.0] - 2024-01-XX - Initial Release

### Fonctionnalités Initiales

- Détection regex/validateurs (emails, téléphones, NIR, IBAN, etc.)
- NER multilingue (DeepPavlov, GLiNER, HuggingFace)
- Pseudonymisation HMAC stable (`PseudoMapper`)
- Client OpenRouter + LLM Reasoner
- Orchestrateur multi-niveau (L0/L1)
- Paraphrase contrôlée
- Audit LLM + durcissement adaptatif
- API Flask (`main_eval.py`)

---

**Format** : [Version] - Date - Titre
- ✨ Nouveautés
- 🔧 Modifications
- 🐛 Corrections
- ⚠️ Breaking Changes
- 🚀 Migration

[Unreleased]: Changements non encore publiés
