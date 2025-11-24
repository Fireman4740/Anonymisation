# Interface d'Analyse Streamlit

Cette interface Streamlit permet d'analyser visuellement les résultats de l'évaluation du pipeline d'anonymisation.

## Fonctionnalités

- 📊 **Visualisation des métriques globales** : Précision, Rappel, F2-Score, nombre de fuites
- 🔍 **Mise en évidence des erreurs dans le texte** :
  - ✓ **True Positives** (vert) : Entités correctement détectées
  - ✗ **False Negatives** (rouge) : Entités manquées
  - ⚠ **False Positives** (orange) : Sur-détections
  - 🚨 **Fuites** (rose) : Informations sensibles non masquées
- 🔎 **Filtres avancés** :
  - Afficher uniquement les documents avec erreurs
  - Afficher uniquement les documents avec fuites
  - Filtrer par rappel/précision minimum
- 📑 **Navigation facile** :
  - Sélection de différents rapports
  - Tri par différents critères
  - Pagination

## Installation

1. Installer les dépendances supplémentaires :
```bash
pip install streamlit
```

Ou ajouter `streamlit>=1.28.0` à votre `requirements.txt`.

## Utilisation

1. **Générer les rapports d'évaluation** (si ce n'est pas déjà fait) :
```bash
cd /chemin/vers/Anonymisation/eval
python evaluate_pipeline.py
```

Cela créera des fichiers `*_details.json` dans `eval/evaluation/reports/`.

2. **Lancer l'interface Streamlit** :
```bash
cd /chemin/vers/Anonymisation/eval/streamlit_app
streamlit run app.py
```

3. **Ouvrir dans votre navigateur** :
L'application s'ouvrira automatiquement à `http://localhost:8501`

## Structure de l'Interface

### Sidebar
- **Configuration** : Sélection du rapport à analyser
- **Filtres** : Options pour filtrer les documents affichés
- **Légende** : Explication des couleurs utilisées

### Vue Principale
- **Métriques globales** : Vue d'ensemble des performances
- **Liste des documents** : Documents filtrés avec leurs métriques
- **Détails par document** :
  - Texte annoté avec mise en évidence des erreurs
  - Liste détaillée des False Negatives, False Positives et Fuites

## Exemples de Cas d'Usage

### Identifier les documents avec le plus de fuites
1. Cocher "Afficher uniquement les documents avec fuites"
2. Trier par "leaks_count" en ordre décroissant

### Analyser les faux négatifs
1. Ajuster "Rappel minimum" à 0.0
2. Trier par "recall" en ordre croissant
3. Observer les entités manquées en rouge dans le texte

### Vérifier les sur-détections
1. Ajuster "Précision minimum" à 0.0
2. Trier par "precision" en ordre croissant
3. Observer les sur-détections en orange dans le texte

## Notes Techniques

- L'application charge les données depuis `eval/evaluation/reports/*_details.json`
- Les rapports doivent être générés avec la version la plus récente de `evaluate_pipeline.py`
- La mise en évidence fusionne intelligemment les annotations qui se chevauchent
- Les fuites sont détectées en cherchant les entités sensibles dans le texte anonymisé

## Dépannage

### "Aucun rapport trouvé"
Vérifiez que vous avez exécuté `evaluate_pipeline.py` et que les fichiers JSON sont présents dans `eval/evaluation/reports/`.

### Les textes sont tronqués
Les rapports ne contiennent que les 200 premiers caractères de chaque document (champ `text_snippet`). Pour voir le texte complet, modifiez `evaluate_pipeline.py` pour stocker le texte entier.

### Erreurs de chevauchement
Si plusieurs annotations se chevauchent, l'application priorise dans cet ordre : LEAK > FN > TP > FP.
