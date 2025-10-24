# DB-Bio Dataset

## Description

Dataset de biographies de célébrités extraites de DBpedia, utilisé pour évaluer l'anonymisation.

## Source

Repository RUPTA : https://github.com/UKPLab/acl2025-rupta

Google Drive : https://drive.google.com/file/d/1oXWI2mh_mkrs2bZs4riGgbYbQoA9RNzD/view?usp=sharing

## Format

Fichiers JSONL avec les champs suivants :

```json
{
    "text": "Biography text of the celebrity...",
    "people": "Celebrity Name",
    "label": "Occupation category (e.g., Tennis Player, Chef, etc.)"
}
```

## Fichiers

- `train.jsonl` - Ensemble d'entraînement
- `test.jsonl` - Ensemble de test  
- `validation.jsonl` - Ensemble de validation

## Catégories d'Occupation

Chef, Classical Music Artist, Table Tennis Player, Entomologist, Lacrosse Player, Astronaut, Medician, Fashion Designer, Horse Trainer, Ambassador, Photographer, Engineer, Formula One Racer, Comedian, Martial Artist, Chess Player, Painter, Soccer Player, Tennis Player, Architect, Cyclist, Basketball Player, Congressman, Baseball Player

## Utilisation

### Téléchargement

1. Télécharger l'archive depuis le lien Google Drive ci-dessus
2. Extraire les fichiers dans ce répertoire
3. Vérifier la présence de `train.jsonl`, `test.jsonl`, `validation.jsonl`

### Chargement en Python

```python
import json

def load_dbbio_dataset(split='test'):
    path = f'Dataset/evaluation/DB-Bio/{split}.jsonl'
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

test_data = load_dbbio_dataset('test')
print(f"Loaded {len(test_data)} examples")
```

## Métriques d'Évaluation

### Privacy
- **Rank** : Position de la vraie personne dans les candidats générés
- **Success Rate** : Proportion de cas où rank > p_threshold (défaut: 10)

### Utility
- **Classification Accuracy** : Précision de classification d'occupation
- **Confidence Score** : Score de confiance moyen (0-100)

## Citation

```bibtex
@article{yang2024robust,
  title={Robust Utility-Preserving Text Anonymization Based on Large Language Models},
  author={Yang, Tianyu and Zhu, Xiaodan and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2407.11770},
  year={2024}
}
```
