# PersonalReddit Dataset

## Description

Dataset de commentaires Reddit synthétiques avec attributs personnels, utilisé pour évaluer l'anonymisation de contenu généré par les utilisateurs.

## Source

Repository RUPTA : https://github.com/UKPLab/acl2025-rupta

Google Drive : https://drive.google.com/file/d/1Z6Xs6zgsn7tkdcW5SElRzbSqUhZFLjwX/view?usp=sharing

Dataset original : https://github.com/eth-sri/llmprivacy/tree/main/data/synthetic

## Format

Fichiers JSONL avec les champs suivants :

```json
{
	"response": "Reddit comment text...",
	"feature": "attribute type (age, sex, city_country, etc.)",
	"personality": {
		"age": "...",
		"sex": "...",
		"city_country": "...",
		"birth_city_country": "...",
		"education": "...",
		"occupation": "...",
		"income_level": "...",
		"relationship_status": "..."
	}
}
```

## Fichiers

- `train.jsonl` - Ensemble d'entraînement
- `test.jsonl` - Ensemble de test

## Attributs Personnels

### Catégories

- **age** : Âge de l'auteur
- **sex** : Sexe (Male, Female)
- **city_country** : Lieu de résidence
- **birth_city_country** : Lieu de naissance
- **education** : Niveau d'éducation
- **occupation** : Profession
- **income_level** : Niveau de revenu
- **relationship_status** : Statut relationnel

### Occupations

software engineer, shop owner, surgeon, structural engineer, data scientist, part-time graphic designer, college professor, web developer, part-time film editor, fashion designer, marketing manager, psychologist, architect, part-time retail worker, part-time waiter, retiree, game developer, junior software developer, high school principal, nurse, lawyer, art curator, financial manager, museum curator, chef, university professor, part-time tutor, retired CEO, business development manager, astronomer, financial analyst, graphic designer, research scientist, environmental consultant, health inspector

## Utilisation

### Téléchargement

1. Télécharger l'archive depuis le lien Google Drive ci-dessus
2. Extraire les fichiers dans ce répertoire
3. Vérifier la présence de `train.jsonl`, `test.jsonl`

### Chargement en Python

```python
import json

def load_reddit_dataset(split='test'):
    path = f'Dataset/evaluation/PersonalReddit/{split}.jsonl'
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

test_data = load_reddit_dataset('test')
print(f"Loaded {len(test_data)} examples")

# Filtrer par attribut
age_examples = [d for d in test_data if d['feature'] == 'age']
print(f"Examples for 'age' attribute: {len(age_examples)}")
```

## Métriques d'Évaluation

### Privacy

- **Rank** : Position de la vraie valeur dans les candidats (pour age, location, etc.)
- **Exact Match** : Correspondance exacte (pour sex, income_level)
- **Success Rate** : Proportion où l'attribut réel n'est pas identifié

### Utility

- **Classification Accuracy** : Précision pour classifier l'occupation
- **Confidence Score** : Score de confiance moyen

## Particularités

### Évaluation par Type d'Attribut

**Attributs à candidats (rank-based)** :

- age
- city_country
- birth_city_country
- education
- relationship_status

**Attributs catégoriels (exact match)** :

- sex (Male/Female)
- income_level (No income, Low, Middle, High, Very High)

## Citation

```bibtex
@article{yang2024robust,
  title={Robust Utility-Preserving Text Anonymization Based on Large Language Models},
  author={Yang, Tianyu and Zhu, Xiaodan and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2407.11770},
  year={2024}
}
```

## Licence

Voir le fichier LICENSE dans le repository RUPTA original.
