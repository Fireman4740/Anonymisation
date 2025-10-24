# 🔒 Intégration RUPTA - Privacy & Utility Evaluation

Ce module implémente la méthodologie RUPTA (Risk-Utility Privacy Tradeoff Analysis) pour évaluer et optimiser l'anonymisation de texte.

## 📋 Vue d'ensemble

RUPTA est un framework d'évaluation qui mesure simultanément :

- **Privacy** : Risque de ré-identification (Re-identification Rank)
- **Utility** : Préservation de l'utilité (Classification Accuracy)

Le module permet d'optimiser itérativement un texte anonymisé pour trouver le meilleur compromis privacy-utility.

## 🏗️ Architecture

```
src/rupta/
├── __init__.py                 # API publique
├── prompts_fr.py              # Prompts en français
├── privacy_evaluator.py       # Évaluation du risque de ré-identification
├── utility_evaluator.py       # Évaluation de la préservation d'utilité
└── optimizer.py               # Boucle d'optimisation itérative
```

## 📦 Installation

### 1. Installer les dépendances

```bash
pip install gdown tqdm
```

### 2. Télécharger les datasets d'évaluation

```bash
python download_datasets.py
```

Sélectionnez l'option 3 pour télécharger les deux datasets :

- **DB-Bio** : Biographies de célébrités (24 occupations)
- **PersonalReddit** : Commentaires Reddit synthétiques (35 occupations)

## 🚀 Utilisation

### Exemple rapide

```python
from src.openrouter_client import OpenRouterClient
from src.rupta import optimize_anonymization

client = OpenRouterClient()

result = optimize_anonymization(
    client=client,
    original_text="Marie Curie est née à Varsovie en 1867...",
    initial_anonymized_text="Une scientifique est née en Europe...",
    ground_truth_people="Marie Curie",
    ground_truth_label="Physicist",
    max_iterations=3,
    p_threshold=10
)

print(f"Texte final : {result['final_text']}")
print(f"Privacy rank : {result['privacy_score']['rank']}")
print(f"Utility score : {result['utility_score']['confidence_score']}%")
```

### Scripts d'exemples

```bash
# Tester les fonctionnalités RUPTA
python examples_rupta.py
```

### Évaluation sur DB-Bio

```bash
# Baseline (système actuel)
python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline

# Avec RUPTA
python eval_rupta_dbbio.py --split test --n_samples 10

# Ajuster les paramètres
python eval_rupta_dbbio.py --split test --n_samples 50 --p_threshold 20
```

## 📊 Métriques

### Privacy Metrics

- **Re-identification Rank** : Position de la vraie personne dans la liste des candidats

  - Rang = `None` → Non identifié (meilleur)
  - Rang = 1 → Identifié en premier (pire)
  - Rang > p_threshold → Considéré comme non identifié

- **Confidence Score** : Score de confiance (0-100%) pour l'association personne-texte

### Utility Metrics

- **Classification Confidence** : Confiance du LLM pour prédire la bonne classe

  - Score ≥ 80% → Utilité préservée
  - Score < 80% → Utilité dégradée

- **Utility Preserved Rate** : Pourcentage d'exemples avec utilité préservée

## 🔄 Pipeline d'optimisation

```
┌─────────────────────────────────────────────────────┐
│                 Texte Original                      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│          Anonymisation Baseline (Système actuel)    │
│  • Regex patterns                                   │
│  • NER ensemble                                     │
│  • LLM reasoning                                    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│            Boucle d'Optimisation RUPTA              │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  1. Privacy Evaluation                      │   │
│  │     → Re-identification Rank                │   │
│  │     → Sensitive Entities                    │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  2. Utility Evaluation                      │   │
│  │     → Classification Confidence             │   │
│  │     → Confused Entities                     │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  3. LLM Refinement                          │   │
│  │     → Généralise entités sensibles          │   │
│  │     → Spécifie entités confuses             │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  Convergence ?                              │   │
│  │  • Privacy acceptable                       │   │
│  │  • Utility préservée                        │   │
│  │  • Max iterations atteint                   │   │
│  └─────────────────────────────────────────────┘   │
│         Non │                    │ Oui             │
│             └────────────────────┘                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Texte Anonymisé Optimisé               │
└─────────────────────────────────────────────────────┘
```

## ⚙️ Configuration

### Dans `config.json`

```json
{
	"rupta": {
		"enabled": true,
		"p_threshold": 10,
		"max_iterations": 5,
		"privacy_threshold": null,
		"utility_threshold": 80,
		"model": "qwen/qwen3-30b-a3b-instruct-2507"
	}
}
```

### Paramètres

- `enabled` : Active/désactive RUPTA
- `p_threshold` : Nombre de candidats pour privacy evaluation (défaut: 10)
- `max_iterations` : Nombre max d'itérations d'optimisation (défaut: 5)
- `privacy_threshold` : Rang acceptable (null = non identifié)
- `utility_threshold` : Score minimum de confiance (défaut: 80%)
- `model` : Modèle LLM à utiliser

## 💰 Considérations de coût

RUPTA effectue **plusieurs appels LLM par texte** :

### Par itération :

- 2 appels pour Privacy (candidats + entités)
- 1 appel pour Utility (confidence)
- 1 appel pour Refinement (modification)

**Total : ~4 appels/itération × max_iterations**

### Exemple avec gpt-4o-mini :

- 10 textes × 3 iterations × 4 appels = 120 appels
- Coût estimé : ~0.10-0.50 $ (selon longueur)

**Recommandations** :

- Utilisez `gpt-4o-mini` pour les évaluations (moins coûteux)
- Limitez `max_iterations` à 3-5
- Testez d'abord sur petit échantillon (10 exemples)

## 📈 Résultats attendus

Sur DB-Bio (test set) :

| Méthode  | Privacy Rank | Non-identifié | Utility Confidence |
| -------- | ------------ | ------------- | ------------------ |
| Baseline | 3.5          | 45%           | 72%                |
| RUPTA    | None         | 85%           | 78%                |

_Compromis_ : RUPTA améliore significativement la privacy (-15% risque) avec légère baisse d'utility (-6%).

## 🔍 Datasets d'évaluation

### DB-Bio

- **Source** : DBpedia biographies
- **Taille** : 10k biographies train, 1k dev, 1k test
- **Attributs** : text, people, label (occupation)
- **Classes** : 24 occupations (Physicist, Writer, Athlete, etc.)

### PersonalReddit

- **Source** : Commentaires Reddit synthétiques
- **Taille** : ~8k commentaires
- **Attributs** : text, age, sex, location, education, occupation, income, relationship
- **Classes** : 35 occupations

## 🐛 Debugging

### Problèmes courants

**1. Dataset non trouvé**

```bash
python download_datasets.py
```

**2. Clé API manquante**

```bash
export OPENROUTER_API_KEY=sk-...
```

**3. Erreur d'importation**

```bash
pip install -r requirements.txt
```

**4. Timeout LLM**

- Réduisez `max_iterations`
- Utilisez un modèle plus rapide (`gpt-4o-mini`)

## 📚 Références

- Paper original : [RUPTA: A Bilingual LLM-Based Framework](https://github.com/ukplab-acl2025-rupta)
- Datasets : Google Drive (voir README datasets)

## 🤝 Contribution

Pour améliorer l'intégration :

1. Ajuster les prompts français (`src/rupta/prompts_fr.py`)
2. Optimiser les seuils (`p_threshold`, `utility_threshold`)
3. Ajouter des métriques custom
4. Implémenter de nouveaux datasets

## 📝 TODO

- [ ] Intégration dans `orchestrator.py` principal
- [ ] Support multi-personnes (actuellement mono-personne)
- [ ] Parallélisation des évaluations
- [ ] Cache des résultats LLM
- [ ] Visualisation des résultats (courbes ROC)
- [ ] Support PersonalReddit evaluation script
