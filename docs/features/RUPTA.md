# 🔐 RUPTA Integration Guide

## Overview

RUPTA (Risk-Utility Privacy Tradeoff Analysis) est maintenant intégré au pipeline d'anonymisation au niveau **L1 (avec LLM)**.

### Fonctionnalités principales

- ✅ **Multilingual**: Supporte toutes les langues européennes (FR, EN, DE, ES, IT, etc.)
- ✅ **Privacy-Utility Optimization**: Équilibre automatique entre protection et utilité
- ✅ **Iterative Refinement**: Optimisation par étapes avec feedback LLM
- ✅ **Evaluation Framework**: Scripts d'évaluation sur DB-Bio et PersonalReddit

---

## 🚀 Quick Start

### 1. Configuration

Le système fonctionne avec deux niveaux de policy :

**L0 (Baseline)**: Sans LLM, uniquement NER/regex

```python
from src.policy import AnonymizationPolicy

policy_l0 = AnonymizationPolicy.preset("L0")
```

**L1 (RUPTA Enabled)**: Avec optimisation privacy-utility

```python
policy_l1 = AnonymizationPolicy.preset("L1")
# Configuration RUPTA par défaut :
# - rupta_enabled=True
# - rupta_p_threshold=10 (10 candidats pour évaluation privacy)
# - rupta_max_iterations=3 (optimisation rapide)
# - rupta_privacy_threshold=None (cherche non-identification)
# - rupta_utility_threshold=80 (maintenir ≥80% d'utilité)
```

### 2. Utilisation de base

```python
from src.orchestrator import anonymize_text

# Anonymisation avec RUPTA (L1)
result = anonymize_text(
    value="Marie Curie, physicienne française, a reçu deux prix Nobel.",
    scope_id="example_001",
    secret_salt="my_secret",
    level="L1",  # Active RUPTA
    overrides={
        "rupta_ground_truth_people": "Marie Curie",
        "rupta_ground_truth_label": "physicist"
    }
)

print(result["anonymized_text"])
print(result["rupta_metrics"])
# {
#   "privacy": {"rank": 999, "non_identified": True},
#   "utility": {"confidence_score": 85, "correct_prediction": True},
#   "iterations": 2,
#   "final_reward": 0.92
# }
```

### 3. Langues supportées

RUPTA fonctionne avec **toutes les langues européennes** :

```python
# Français
anonymize_text("Marie Curie était une physicienne française.", ...)

# Anglais
anonymize_text("Marie Curie was a French physicist.", ...)

# Allemand
anonymize_text("Marie Curie war eine französische Physikerin.", ...)

# Espagnol
anonymize_text("Marie Curie fue una física francesa.", ...)

# Italien
anonymize_text("Marie Curie era una fisica francese.", ...)
```

Le système détecte automatiquement la langue et utilise des prompts multilingues optimisés.

---

## 📊 Evaluation

### Évaluer sur DB-Bio

```bash
# Test avec 50 échantillons (rapide)
python scripts/eval_rupta_pipeline.py \
    --dataset dbbio \
    --split test \
    --n_samples 50 \
    --use_rupta \
    --output results/eval_dbbio_test.json

# Test complet (1000 échantillons)
python scripts/eval_rupta_pipeline.py \
    --dataset dbbio \
    --split test \
    --n_samples 0 \
    --use_baseline \
    --use_rupta \
    --output results/eval_dbbio_full.json
```

### Évaluer sur PersonalReddit

```bash
python scripts/eval_rupta_pipeline.py \
    --dataset reddit \
    --split test \
    --n_samples 50 \
    --use_rupta \
    --output results/eval_reddit_test.json
```

### Comparer Baseline vs RUPTA

```bash
python scripts/compare_baseline_rupta.py \
    --results results/eval_dbbio_test.json \
    --output results/comparison_report.md
```

Génère un rapport Markdown avec :

- Tableau comparatif des métriques
- Analyse du tradeoff privacy-utility
- Recommandations d'optimisation

---

## ⚙️ Advanced Configuration

### Personnaliser les paramètres RUPTA

```python
from src.policy import AnonymizationPolicy

# Configuration agressive (max privacy)
policy_max_privacy = AnonymizationPolicy(
    level_llm=True,
    rupta_enabled=True,
    rupta_p_threshold=20,  # Plus de candidats = meilleure privacy
    rupta_max_iterations=5,  # Plus d'itérations = meilleure optimisation
    rupta_privacy_threshold=None,  # Pas de seuil strict
    rupta_utility_threshold=70  # Tolérance pour perte d'utilité
)

# Configuration équilibrée (défaut)
policy_balanced = AnonymizationPolicy.preset("L1")

# Configuration rapide (moins d'itérations)
policy_fast = AnonymizationPolicy(
    level_llm=True,
    rupta_enabled=True,
    rupta_p_threshold=5,
    rupta_max_iterations=1,
    rupta_utility_threshold=90  # Priorité à l'utilité
)
```

### Paramètres RUPTA

| Paramètre                 | Type      | Défaut                | Description                                               |
| ------------------------- | --------- | --------------------- | --------------------------------------------------------- |
| `rupta_enabled`           | bool      | False (L0), True (L1) | Active RUPTA                                              |
| `rupta_p_threshold`       | int       | 10                    | Nombre de candidats pour évaluation privacy               |
| `rupta_max_iterations`    | int       | 3                     | Nombre max d'itérations d'optimisation                    |
| `rupta_privacy_threshold` | int\|None | None                  | Seuil de rang privacy (None = cherche non-identification) |
| `rupta_utility_threshold` | int       | 80                    | Seuil minimum de confiance d'utilité (%)                  |

---

## 📈 Métriques

### Privacy Metrics

- **Rank**: Position de la vraie identité dans les candidats (1-10, ou 999 si non identifié)
- **Non-identified Rate**: % d'exemples où rank = 999
- **Average Rank**: Rang moyen pour les exemples identifiés (plus élevé = meilleur)

### Utility Metrics

- **Confidence Score**: Confiance de classification (0-100%)
- **Correct Prediction**: Booléen indiquant si la classification est correcte
- **Utility Preserved Rate**: % d'exemples avec confiance ≥ threshold

### Combined Reward

RUPTA optimise une récompense combinée :

```
reward = privacy_reward * 0.5 + utility_reward * 0.5
```

Où :

- `privacy_reward = 1.0` si non-identifié, sinon `rank / p_threshold`
- `utility_reward = confidence_score / 100` si prédiction correcte, sinon `0.0`

---

## 🔧 Troubleshooting

### Erreur: "Module 'prompts_fr' not found"

✅ **Solution**: Les prompts ont été renommés en `prompts_multilang.py`

```bash
# Vérifier que le fichier existe
ls src/rupta/prompts_multilang.py
```

### Privacy trop faible (< 50% non-identifiés)

✅ **Solutions**:

1. Augmenter `rupta_max_iterations` (3 → 5)
2. Augmenter `rupta_p_threshold` (10 → 20)
3. Réduire `rupta_utility_threshold` (80 → 70)

### Utilité trop faible (< 70% préservée)

✅ **Solutions**:

1. Réduire `rupta_max_iterations` (3 → 1)
2. Augmenter `rupta_utility_threshold` (80 → 90)
3. Utiliser un modèle plus puissant (`gpt-4o` au lieu de `gpt-4o-mini`)

### Évaluation trop lente

✅ **Solutions**:

1. Réduire `n_samples` (tester sur 10-50 échantillons)
2. Utiliser `gpt-4o-mini` au lieu de `gpt-4o` (3x plus rapide)
3. Réduire `rupta_max_iterations` (3 → 1)

---

## 📚 Examples

### Exemple complet avec évaluation

```python
from src.orchestrator import anonymize_text
from src.openrouter_client import OpenRouterClient
from src.rupta.privacy_evaluator import evaluate_reidentification_risk
from src.rupta.utility_evaluator import evaluate_utility_preservation

# Texte original
text = "Albert Einstein, physicien théoricien allemand, a développé la théorie de la relativité."
people = "Albert Einstein"
label = "physicist"

# 1. Anonymisation avec RUPTA
result = anonymize_text(
    value=text,
    scope_id="example_einstein",
    secret_salt="secret",
    level="L1",
    overrides={
        "rupta_ground_truth_people": people,
        "rupta_ground_truth_label": label
    }
)

anonymized = result["anonymized_text"]
print(f"Anonymisé: {anonymized}")

# 2. Évaluation manuelle (optionnel)
client = OpenRouterClient()

privacy = evaluate_reidentification_risk(
    client=client,
    anonymized_text=anonymized,
    ground_truth_people=people,
    p_threshold=10,
    model="qwen/qwen3-30b-a3b-instruct-2507"
)

utility = evaluate_utility_preservation(
    client=client,
    anonymized_text=anonymized,
    ground_truth_label=label,
    model="qwen/qwen3-30b-a3b-instruct-2507"
)

print(f"Privacy Rank: {privacy['rank']}")
print(f"Utility Score: {utility['confidence_score']}")
print(f"RUPTA Iterations: {result['rupta_metrics']['iterations']}")
print(f"Final Reward: {result['rupta_metrics']['final_reward']:.2f}")
```

---

## 🗂️ Project Structure

```
src/rupta/
├── prompts_multilang.py         # Prompts multilingues (EN/FR/DE/ES/IT/etc.)
├── privacy_evaluator.py         # Évaluation du risque de ré-identification
├── utility_evaluator.py         # Évaluation de la préservation d'utilité
└── rupta_optimizer.py           # Optimiseur privacy-utility (TODO)

scripts/
├── eval_rupta_pipeline.py       # Script d'évaluation principal
├── compare_baseline_rupta.py    # Comparaison Baseline vs RUPTA
├── examples_rupta.py            # Exemples d'utilisation
└── download_datasets.py         # Téléchargement des datasets

Dataset/evaluation/
├── DB-bio/                      # Biographies de célébrités
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── PersonalReddit/              # Commentaires Reddit synthétiques
    └── Reddit_synthetic/
        ├── train.jsonl
        └── test.jsonl
```

---

## 📝 Next Steps

1. ✅ Multilingual support (DONE)
2. ✅ Policy integration (DONE)
3. ⏳ Orchestrator integration (IN PROGRESS)
4. ⏳ Full evaluation scripts (IN PROGRESS)
5. ⏳ RUPTA optimizer module (TODO)
6. ⏳ Documentation complète (IN PROGRESS)

---

## 📖 References

- **RUPTA Paper**: [Privacy-Utility Tradeoff for Text Anonymization](https://github.com/RUPTA-anonymization/RUPTA)
- **DB-Bio Dataset**: 10k celebrity biographies with 24 occupations
- **PersonalReddit**: 8k synthetic Reddit comments with 35 occupations and 7 personal attributes

---

## 🤝 Contributing

Pour contribuer à l'amélioration de RUPTA :

1. Tester sur de nouveaux datasets
2. Optimiser les prompts pour de meilleures performances
3. Ajuster les paramètres par défaut selon les résultats d'évaluation
4. Ajouter de nouvelles langues si nécessaire

---

_Dernière mise à jour: 2024 - RUPTA Integration v1.0_
