# 🚀 Guide de Démarrage Rapide - RUPTA

Guide pas-à-pas pour utiliser l'évaluation et l'optimisation RUPTA dans votre système d'anonymisation.

## ⏱️ Quick Start (5 minutes)

### 1. Installer les dépendances

```bash
pip install gdown tqdm
```

### 2. Télécharger les datasets

```bash
python download_datasets.py
# Sélectionner option 3 (télécharger les deux)
```

### 3. Tester avec les exemples

```bash
# Définir votre clé API
export OPENROUTER_API_KEY=sk-...

# Lancer les exemples
python examples_rupta.py
```

## 📊 Évaluation Complète (30 minutes)

### Étape 1 : Évaluation Baseline

```bash
python eval_rupta_dbbio.py \
  --split test \
  --n_samples 10 \
  --use_baseline \
  --output results_baseline.json
```

**Attendu** :

- ~10 minutes d'exécution
- Évaluation de 10 textes
- Résultats dans `results_baseline.json`

### Étape 2 : Évaluation RUPTA

```bash
python eval_rupta_dbbio.py \
  --split test \
  --n_samples 10 \
  --output results_rupta.json
```

**Attendu** :

- ~20-30 minutes d'exécution (optimisation itérative)
- Résultats dans `results_rupta.json`

### Étape 3 : Comparaison

```bash
python compare_baseline_rupta.py \
  --baseline results_baseline.json \
  --rupta results_rupta.json \
  --output rapport_comparison.md \
  --detailed
```

**Attendu** :

- Rapport markdown avec métriques comparatives
- Analyse du compromis privacy-utility
- Recommandations d'optimisation

## 🔧 Intégration dans votre Pipeline

### Option A : Utilisation standalone

```python
from src.openrouter_client import OpenRouterClient
from src.rupta import optimize_anonymization

client = OpenRouterClient()

# Votre texte original
text = "Marie Curie est née à Varsovie..."

# Anonymisation initiale (votre système actuel)
anonymized = your_anonymization_function(text)

# Optimisation RUPTA
result = optimize_anonymization(
    client=client,
    original_text=text,
    initial_anonymized_text=anonymized,
    ground_truth_people="Marie Curie",
    ground_truth_label="Physicist",
    max_iterations=3
)

final_text = result['final_text']
privacy_score = result['privacy_score']
utility_score = result['utility_score']
```

### Option B : Intégration dans orchestrator (TODO)

```python
# Dans src/orchestrator.py
def anonymize_text_with_rupta(
    text: str,
    config_path: str,
    ground_truth_people: str = None,
    ground_truth_label: str = None
):
    # 1. Anonymisation baseline
    result = anonymize_text(text, config_path)

    # 2. Si RUPTA activé
    config = load_config(config_path)
    if config.get('rupta', {}).get('enabled', False):
        client = OpenRouterClient()
        rupta_result = optimize_anonymization(
            client=client,
            original_text=text,
            initial_anonymized_text=result['anonymized_text'],
            ground_truth_people=ground_truth_people,
            ground_truth_label=ground_truth_label,
            **config['rupta']
        )
        result['anonymized_text'] = rupta_result['final_text']
        result['rupta_metrics'] = {
            'privacy': rupta_result['privacy_score'],
            'utility': rupta_result['utility_score'],
            'iterations': rupta_result['iterations']
        }

    return result
```

## ⚙️ Configuration Optimale

### Pour production (privacy prioritaire)

```json
{
	"rupta": {
		"enabled": true,
		"p_threshold": 20,
		"max_iterations": 5,
		"privacy_threshold": null,
		"utility_threshold": 70,
		"model": "openai/gpt-4o"
	}
}
```

### Pour développement (rapide)

```json
{
	"rupta": {
		"enabled": true,
		"p_threshold": 10,
		"max_iterations": 3,
		"privacy_threshold": null,
		"utility_threshold": 80,
		"model": "qwen/qwen3-30b-a3b-instruct-2507"
	}
}
```

### Pour équilibre privacy-utility

```json
{
	"rupta": {
		"enabled": true,
		"p_threshold": 15,
		"max_iterations": 4,
		"privacy_threshold": null,
		"utility_threshold": 75,
		"model": "qwen/qwen3-30b-a3b-instruct-2507"
	}
}
```

## 📈 Métriques et Interprétation

### Privacy (Re-identification Rank)

| Rang | Interprétation             | Action                    |
| ---- | -------------------------- | ------------------------- |
| None | ✅ Non identifié           | Optimal                   |
| 1-3  | ❌ Facilement identifiable | Augmenter généralisation  |
| 4-10 | ⚠️ Risque modéré           | Acceptable selon contexte |
| >10  | ✅ Difficile à identifier  | Bon                       |

### Utility (Classification Confidence)

| Score  | Interprétation             | Action                 |
| ------ | -------------------------- | ---------------------- |
| ≥90%   | ✅ Excellente préservation | Optimal                |
| 80-89% | ✅ Bonne préservation      | Acceptable             |
| 70-79% | ⚠️ Préservation modérée    | Ajuster seuils         |
| <70%   | ❌ Utilité dégradée        | Réduire généralisation |

## 💰 Estimation des Coûts

### Modèle gpt-4o-mini (recommandé)

| Tâche         | Textes | Iterations | Appels LLM | Coût estimé |
| ------------- | ------ | ---------- | ---------- | ----------- |
| Test rapide   | 10     | 3          | ~120       | $0.10       |
| Eval moyenne  | 50     | 3          | ~600       | $0.50       |
| Eval complète | 100    | 5          | ~2000      | $2.00       |

### Modèle gpt-4o

| Tâche         | Textes | Iterations | Appels LLM | Coût estimé |
| ------------- | ------ | ---------- | ---------- | ----------- |
| Test rapide   | 10     | 3          | ~120       | $0.50       |
| Eval moyenne  | 50     | 3          | ~600       | $2.50       |
| Eval complète | 100    | 5          | ~2000      | $10.00      |

**Formule** : `Coût = n_textes × max_iterations × 4 appels × prix_par_appel`

## 🐛 Troubleshooting

### Problème : Dataset non trouvé

```bash
# Vérifier les datasets
python download_datasets.py
# Option 4 : Vérifier

# Si manquant, télécharger
# Option 3 : Télécharger les deux
```

### Problème : Clé API invalide

```bash
# Vérifier la clé
echo $OPENROUTER_API_KEY

# Redéfinir si nécessaire
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Problème : Timeout LLM

```python
# Réduire le nombre d'itérations
python eval_rupta_dbbio.py --n_samples 5
```

### Problème : Métriques peu concluantes

**Privacy trop faible** :

- Augmenter `p_threshold` (20-30)
- Utiliser plus d'itérations (5-7)

**Utility trop basse** :

- Réduire `max_iterations` (2-3)
- Augmenter `utility_threshold` (85-90)

## 📚 Ressources

- **Documentation complète** : [README_RUPTA.md](README_RUPTA.md)
- **Plan d'intégration** : [PLAN_INTEGRATION_RUPTA.md](PLAN_INTEGRATION_RUPTA.md)
- **Exemples de code** : [examples_rupta.py](examples_rupta.py)
- **Repository RUPTA** : https://github.com/ukplab-acl2025-rupta

## ✅ Checklist de Déploiement

- [ ] Dépendances installées (`gdown`, `tqdm`)
- [ ] Datasets téléchargés (DB-Bio, PersonalReddit)
- [ ] Clé API OpenRouter configurée
- [ ] Tests exemples réussis (`examples_rupta.py`)
- [ ] Évaluation baseline effectuée (10+ échantillons)
- [ ] Évaluation RUPTA effectuée (10+ échantillons)
- [ ] Rapport de comparaison généré
- [ ] Configuration RUPTA optimisée dans `config.json`
- [ ] Intégration dans orchestrator (optionnel)
- [ ] Tests de régression passés

## 🎯 Prochaines Étapes

1. **Évaluation à grande échelle** (100+ exemples)
2. **Optimisation des hyperparamètres** (grid search)
3. **Support multi-personnes** (plusieurs entités à protéger)
4. **Visualisation des résultats** (courbes, graphiques)
5. **Cache LLM** (réduire les coûts)
6. **Parallélisation** (accélérer l'évaluation)

## 🤝 Support

Pour toute question ou problème :

1. Vérifier [README_RUPTA.md](README_RUPTA.md)
2. Consulter le [PLAN_INTEGRATION_RUPTA.md](PLAN_INTEGRATION_RUPTA.md)
3. Tester avec [examples_rupta.py](examples_rupta.py)
