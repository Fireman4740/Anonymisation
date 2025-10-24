# Plan d'Intégration RUPTA dans le Système d'Anonymisation

## Vue d'ensemble

Le repository **ukplab-acl2025-rupta** propose une méthode d'anonymisation basée sur des LLMs avec trois composants clés :
1. **Privacy Evaluator** - Évalue le risque de ré-identification
2. **Utility Evaluator** - Mesure la préservation de l'utilité pour les tâches en aval
3. **Optimization Component** - Coordonne les modifications itératives

## 1. Datasets d'Évaluation à Intégrer

### 1.1 DB-Bio Dataset
- **Source** : Biographies de célébrités de DBpedia
- **Format** : JSONL avec champs `text`, `people`, `label` (occupation)
- **Localisation prévue** : `./Dataset/evaluation/DB-Bio/`
- **Fichiers** :
  - `train.jsonl`
  - `test.jsonl` 
  - `validation.jsonl`

### 1.2 PersonalReddit Dataset
- **Source** : Commentaires Reddit synthétiques
- **Format** : JSONL avec attributs personnels (age, sex, location, education, etc.)
- **Localisation prévue** : `./Dataset/evaluation/PersonalReddit/`
- **Fichiers** :
  - `train.jsonl`
  - `test.jsonl`

**Action** : Télécharger ces datasets depuis les liens Google Drive fournis dans le README du repo.

## 2. Composants RUPTA à Adapter

### 2.1 Privacy Evaluator (generators/generator_utils.py)

**Fonctions clés à intégrer** :

```python
# À adapter de generic_privacy_reflection()
def evaluate_reidentification_risk(
    model: ModelBase,
    anonymized_text: str,
    ground_truth_people: str,
    p_threshold: int = 10
) -> dict:
    """
    Génère une liste de candidats et vérifie si la vraie personne 
    est identifiable.
    
    Returns:
        {
            'rank': int,  # Position de la vraie personne
            'confidence': str,  # Yes/No
            'entities': list  # Entités sensibles identifiées
        }
    """
```

**Intégration dans votre système** :
- Créer un nouveau module `src/rupta_evaluators.py`
- Adapter les prompts de `people_prompt.py` pour le français
- Utiliser votre `OpenRouterClient` existant

### 2.2 Utility Evaluator

**Fonctions clés** :

```python
# À adapter de generic_utility_reflection()
def evaluate_classification_utility(
    model: ModelBase,
    anonymized_text: str,
    ground_truth_label: str,
    original_text: str
) -> dict:
    """
    Évalue si le texte anonymisé préserve l'information 
    pour la classification.
    
    Returns:
        {
            'confidence_score': int,  # 0-100
            'confirmation': str,  # Yes/No
            'confused_entities': list
        }
    """
```

### 2.3 Optimization Component (reflexion.py)

**Boucle d'optimisation à adapter** :

```python
def optimize_anonymization(
    text: str,
    max_iters: int = 5,
    p_threshold: int = 10,
    privacy_target: float = 0.9,
    utility_target: float = 0.8
) -> dict:
    """
    Boucle itérative :
    1. Anonymiser
    2. Évaluer privacy/utility
    3. Ajuster si nécessaire
    4. Recommencer jusqu'à convergence ou max_iters
    """
```

## 3. Architecture d'Intégration Proposée

### 3.1 Nouvelle Structure de Fichiers

```
src/
├── rupta/
│   ├── __init__.py
│   ├── privacy_evaluator.py      # Adaptation de generic_privacy_reflection
│   ├── utility_evaluator.py      # Adaptation de generic_utility_reflection
│   ├── optimizer.py               # Boucle reflexion adaptée
│   ├── prompts_fr.py              # Prompts traduits en français
│   └── prompts_en.py              # Prompts originaux anglais
├── orchestrator.py                # Mise à jour pour intégrer RUPTA
└── config_loader.py               # Ajout config RUPTA

Dataset/
├── evaluation/
│   ├── DB-Bio/
│   │   ├── train.jsonl
│   │   ├── test.jsonl
│   │   └── validation.jsonl
│   └── PersonalReddit/
│       ├── train.jsonl
│       └── test.jsonl
```

### 3.2 Modification de l'Orchestrator

**Ajout d'un mode "RUPTA" dans l'orchestrator** :

```python
def anonymize_text(
    value: str,
    policy: AnonymizationPolicy,
    # ... paramètres existants ...
    use_rupta: bool = False,
    rupta_config: Optional[dict] = None
) -> dict:
    """
    Si use_rupta=True :
    1. Anonymisation initiale (pipeline actuel)
    2. Évaluation privacy avec Privacy Evaluator
    3. Si risque trop élevé : itérations d'amélioration
    4. Évaluation utility finale
    5. Retour résultat + métriques
    """
```

## 4. Prompts RUPTA à Traduire/Adapter

### 4.1 Privacy Evaluation (people_prompt.py)

**Prompts clés** :
- `PRIVACY_REFLECTION_INSTRUCTION_1` : Identification de candidats
- `PRIVACY_REFLECTION_INSTRUCTION_2` : Vérification et extraction d'entités sensibles
- `PRIVACY_EVALUATION_CONFIDENCE_INSTRUCTION` : Score de confiance

**Adaptation française** :
```python
PRIVACY_REFLECTION_FR_1 = """
Vous êtes expert en identification de personnes à partir de biographies anonymisées.
Générez une liste de {p_threshold} candidats les plus probables décrits par le texte suivant.
Classez-les du plus probable au moins probable.

Texte anonymisé :
{anonymized_text}

Répondez au format JSON :
{format_instructions}
"""
```

### 4.2 Utility Evaluation

**Prompts clés** :
- `UTILITY_REFLECTION_INSTRUCTION_1` : Classification + score de confiance

**Adaptation** :
```python
UTILITY_REFLECTION_FR_1 = """
Vous êtes expert en classification d'occupation à partir de biographies.
Évaluez votre confiance (0-100) pour classifier ce texte anonymisé dans la catégorie : {label}

Texte anonymisé :
{anonymized_text}

Échelle :
- 0 : Aucune confiance
- 1-50 : Quelques détails correspondent
- 51-99 : La plupart des détails correspondent  
- 100 : Correspondance parfaite
"""
```

### 4.3 Optimization/Reinforcement

**Prompts clés** :
- `REINFORCEMENT_INSTRUCTION` : Instructions de modification itérative

## 5. Pipeline d'Intégration Étape par Étape

### Phase 1 : Préparation (Semaine 1)
1. ✅ Télécharger les datasets DB-Bio et PersonalReddit
2. ✅ Créer la structure `src/rupta/`
3. ✅ Traduire les prompts essentiels en français

### Phase 2 : Implémentation Core (Semaine 2-3)
1. Implémenter `privacy_evaluator.py`
   - Fonction d'évaluation de ré-identification
   - Utiliser votre `OpenRouterClient`
   - Parser les réponses JSON (réutiliser logique de `llm_reasoner_openrouter.py`)

2. Implémenter `utility_evaluator.py`
   - Score de confiance pour classification
   - Détection d'entités confuses

3. Implémenter `optimizer.py`
   - Boucle de raffinement itératif
   - Critères d'arrêt (convergence ou max_iters)

### Phase 3 : Intégration Orchestrator (Semaine 4)
1. Ajouter paramètre `use_rupta` dans `anonymize_text()`
2. Créer fonction wrapper `anonymize_with_rupta_eval()`
3. Gérer les métriques de sortie (privacy_score, utility_score, iterations)

### Phase 4 : Évaluation (Semaine 5)
1. Script d'évaluation sur DB-Bio
2. Script d'évaluation sur PersonalReddit
3. Comparaison avec baseline (votre système actuel)
4. Métriques :
   - Privacy : Taux de ré-identification (rank@10)
   - Utility : Accuracy de classification
   - Trade-off : Courbes privacy/utility

## 6. Fichiers de Configuration

### config.json (ajout section RUPTA)

```json
{
  "rupta": {
    "enabled": false,
    "max_iterations": 5,
    "p_threshold": 10,
    "privacy_target_rank": 11,
    "utility_min_confidence": 80,
    "models": {
      "privacy_evaluator": "openai/gpt-4-turbo",
      "utility_evaluator": "openai/gpt-4-turbo",
      "optimizer": "openai/gpt-4-turbo"
    },
    "language": "fr"
  }
}
```

## 7. Scripts d'Évaluation

### eval_rupta_dbbio.py

```python
"""
Évaluation du système avec méthode RUPTA sur DB-Bio dataset
"""
import json
from src.rupta.optimizer import optimize_anonymization
from src.rupta.privacy_evaluator import evaluate_reidentification_risk

def run_evaluation():
    # Charger test set
    with open('Dataset/evaluation/DB-Bio/test.jsonl') as f:
        test_data = [json.loads(line) for line in f]
    
    results = []
    for item in test_data:
        # Anonymiser avec RUPTA
        result = optimize_anonymization(
            text=item['text'],
            ground_truth_people=item['people'],
            ground_truth_label=item['label']
        )
        results.append(result)
    
    # Calculer métriques
    calculate_metrics(results)
```

## 8. Métriques d'Évaluation à Implémenter

### Privacy Metrics
```python
def calculate_privacy_metrics(results):
    """
    - Rank de la vraie personne dans les candidats
    - Taux où rank > p_threshold (succès privacy)
    - Score de confiance moyen
    """
```

### Utility Metrics
```python
def calculate_utility_metrics(results):
    """
    - Accuracy de classification sur texte anonymisé
    - Perte d'accuracy vs texte original
    - Score de confiance moyen
    """
```

### Trade-off Metrics
```python
def calculate_tradeoff(privacy_scores, utility_scores):
    """
    - Courbe Pareto privacy/utility
    - Points optimaux selon seuils
    """
```

## 9. Différences Clés RUPTA vs Votre Système Actuel

| Aspect | Votre Système | RUPTA |
|--------|---------------|-------|
| Détection | Regex + NER + LLM | Principalement LLM guidé |
| Anonymisation | Placeholders typés | Généralisation textuelle |
| Privacy | Audit post-hoc | Évaluation itérative |
| Utility | Pas d'évaluation formelle | Score de classification |
| Optimisation | Politique fixe | Boucle de raffinement |

## 10. Points d'Attention

### 10.1 Coûts API
- RUPTA fait plusieurs appels LLM par texte (détection + privacy eval + utility eval + iterations)
- Prévoir budget conséquent pour évaluation complète
- Possibilité de cacher certains résultats

### 10.2 Langue
- Prompts originaux en anglais
- Datasets en anglais
- Adaptation française nécessaire avec validation

### 10.3 Complémentarité
- RUPTA et votre système ne sont pas mutuellement exclusifs
- Possibilité d'utiliser RUPTA comme **couche d'audit** sur votre pipeline
- Votre système : détection précise (regex+NER) 
- RUPTA : évaluation et raffinement

## 11. Proposition d'Architecture Hybride

```
Texte Original
      ↓
[Votre Pipeline Actuel]
  - Regex + NER
  - Placeholders typés
  - Généralisation basique
      ↓
Texte Anonymisé V1
      ↓
[RUPTA Layer - Optionnel]
  - Privacy Evaluation
  - Si risque > seuil → Raffinement
  - Utility Evaluation
      ↓
Texte Anonymisé Final + Métriques
```

## 12. Prochaines Actions Concrètes

### Immédiat
1. [ ] Télécharger DB-Bio dataset
2. [ ] Télécharger PersonalReddit dataset
3. [ ] Créer dossiers `Dataset/evaluation/`
4. [ ] Créer module `src/rupta/__init__.py`

### Court terme (1-2 semaines)
1. [ ] Traduire prompts essentiels (privacy eval)
2. [ ] Implémenter `privacy_evaluator.py` basique
3. [ ] Tester sur 10 exemples DB-Bio

### Moyen terme (3-4 semaines)
1. [ ] Implémenter `utility_evaluator.py`
2. [ ] Implémenter boucle `optimizer.py`
3. [ ] Intégrer dans orchestrator

### Long terme (1-2 mois)
1. [ ] Évaluation complète sur datasets
2. [ ] Comparaison avec baseline
3. [ ] Publication des résultats

## 13. Ressources Nécessaires

### Datasets
- DB-Bio : [Google Drive Link](https://drive.google.com/file/d/1oXWI2mh_mkrs2bZs4riGgbYbQoA9RNzD/view?usp=sharing)
- PersonalReddit : [Google Drive Link](https://drive.google.com/file/d/1Z6Xs6zgsn7tkdcW5SElRzbSqUhZFLjwX/view?usp=sharing)

### Modèles LLM (via OpenRouter)
- GPT-4 Turbo (recommandé pour évaluation)
- Possibilité d'utiliser modèles moins coûteux pour tests

### Computational
- Les LLM sont appelés via API (pas de GPU local nécessaire)
- RAM suffisante pour charger datasets (~500MB)

---

**Note** : Ce plan est conçu pour une intégration progressive. Vous pouvez commencer par la Phase 1 et itérer selon les résultats.
