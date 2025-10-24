# 📋 TODO - Intégration RUPTA

Liste des tâches restantes pour finaliser l'intégration RUPTA.

## ✅ Terminé (Phase 1 + Adaptation Python 3.11)

### Implémentation Core

- [x] Analyse du repository RUPTA original
- [x] Création du plan d'intégration détaillé
- [x] Création de l'architecture des modules (`src/rupta/`)
- [x] Traduction des prompts en français
- [x] Implémentation `privacy_evaluator.py`
- [x] Implémentation `utility_evaluator.py`
- [x] Implémentation `optimizer.py`
- [x] Adaptation à l'API OpenRouterClient existante
- [x] Création structure datasets (`Dataset/evaluation/`)
- [x] Documentation README datasets (DB-Bio, PersonalReddit)
- [x] Script de téléchargement datasets
- [x] Script d'exemples (`examples_rupta.py`)
- [x] Script d'évaluation DB-Bio
- [x] Script de comparaison Baseline vs RUPTA
- [x] Configuration RUPTA dans `config.json`
- [x] Documentation complète (README_RUPTA.md)
- [x] Guide de démarrage rapide (QUICKSTART_RUPTA.md)

### Adaptation Python 3.11 ✨ NOUVEAU

- [x] Identification problème dépendances NER
- [x] Correction signature `anonymize_text()` dans `eval_rupta_dbbio.py`
- [x] Désactivation NER incompatibles (DeepPavlov/GLiNER)
- [x] Script de validation (`test_python311_compat.py`)
- [x] Script interactif de démarrage (`quickstart.sh`)
- [x] Documentation solution (`PYTHON_311_SOLUTION.md`)
- [x] Guide ultra-rapide (`START_HERE.md`)
- [x] Checklist validation (`CHECKLIST_PYTHON311.md`)
- [x] Tests passés avec succès ✅

## 🔄 En Cours (Phase 2)

### Datasets

- [ ] **Télécharger DB-Bio** depuis Google Drive

  - Fichier : https://drive.google.com/file/d/1oXWI2mh_mkrs2bZs4riGgbYbQoA9RNzD/view
  - Destination : `Dataset/evaluation/DB-Bio/`
  - Commande : `python download_datasets.py` → Option 1

- [ ] **Télécharger PersonalReddit** depuis Google Drive

  - Fichier : https://drive.google.com/file/d/1Z6Xs6zgsn7tkdcW5SElRzbSqUhZFLjwX/view
  - Destination : `Dataset/evaluation/PersonalReddit/`
  - Commande : `python download_datasets.py` → Option 2

- [ ] **Vérifier les datasets**
  - Commande : `python download_datasets.py` → Option 4
  - Tester chargement : Option 5

### Évaluation

- [ ] **Test initial sur 5 exemples DB-Bio**

  ```bash
  python eval_rupta_dbbio.py --split test --n_samples 5 --use_baseline
  python eval_rupta_dbbio.py --split test --n_samples 5
  ```

- [ ] **Évaluation Baseline (10 exemples)**

  ```bash
  python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline --output results_baseline_10.json
  ```

- [ ] **Évaluation RUPTA (10 exemples)**

  ```bash
  python eval_rupta_dbbio.py --split test --n_samples 10 --output results_rupta_10.json
  ```

- [ ] **Comparaison initiale**
  ```bash
  python compare_baseline_rupta.py --baseline results_baseline_10.json --rupta results_rupta_10.json --detailed
  ```

## 🚀 À Faire (Phase 3)

### Intégration dans le Pipeline Principal

- [ ] **Modifier `src/orchestrator.py`**

  - Ajouter fonction `anonymize_with_rupta()`
  - Intégrer appel conditionnel basé sur `config.json`
  - Gérer les paramètres ground_truth (optionnels)
  - Retourner métriques RUPTA dans le résultat

- [ ] **Créer wrapper haut niveau**

  ```python
  # Dans src/orchestrator.py
  def anonymize_text_with_evaluation(
      text: str,
      config_path: str,
      ground_truth_people: Optional[str] = None,
      ground_truth_label: Optional[str] = None,
      use_rupta: bool = None
  ) -> Dict[str, Any]:
      """Anonymise avec évaluation optionnelle RUPTA"""
      pass
  ```

- [ ] **Tests d'intégration**
  - Créer `tests/test_rupta_integration.py`
  - Tester avec/sans RUPTA activé
  - Vérifier backward compatibility
  - Tester gestion des erreurs LLM

### Script PersonalReddit

- [ ] **Créer `eval_rupta_reddit.py`**

  - Adapter à la structure PersonalReddit
  - Gérer les 7 attributs (age, sex, location, etc.)
  - Évaluation multi-attributs
  - Métriques agrégées par attribut

- [ ] **Évaluation sur PersonalReddit**
  ```bash
  python eval_rupta_reddit.py --split test --n_samples 10 --use_baseline
  python eval_rupta_reddit.py --split test --n_samples 10
  ```

### Optimisation des Prompts

- [ ] **Affiner les prompts français**

  - Tester variantes de `PRIVACY_REFLECTION_FR_1`
  - Optimiser `UTILITY_REFLECTION_FR_1` pour mieux préserver contexte
  - Améliorer `REINFORCEMENT_FR` pour guidage plus précis

- [ ] **A/B Testing des prompts**
  - Créer `src/rupta/prompts_fr_variants.py`
  - Comparer performances sur 50 exemples
  - Sélectionner meilleurs prompts

### Optimisation des Hyperparamètres

- [ ] **Grid Search**

  - Tester `p_threshold`: [5, 10, 15, 20, 25]
  - Tester `max_iterations`: [2, 3, 4, 5, 7]
  - Tester `utility_threshold`: [70, 75, 80, 85, 90]
  - Créer script `optimize_hyperparams.py`

- [ ] **Trouver configuration optimale**
  - Balance privacy-utility
  - Minimiser coût LLM
  - Documenter recommandations

## 🔧 Améliorations (Phase 4)

### Performance

- [ ] **Cache LLM**

  - Implémenter cache des réponses LLM identiques
  - Réduire coûts pour évaluations répétées
  - Stocker dans SQLite ou Redis

- [ ] **Parallélisation**

  - Évaluer plusieurs textes en parallèle
  - Utiliser `asyncio` pour appels API concurrents
  - Limiter rate limiting OpenRouter

- [ ] **Early stopping**
  - Arrêter itérations si convergence détectée tôt
  - Sauvegarder tokens LLM

### Fonctionnalités

- [ ] **Support multi-personnes**

  - Gérer listes de personnes (pas seulement une)
  - Évaluation privacy pour chaque personne
  - Agrégation des scores

- [ ] **Multi-label classification**

  - Supporter plusieurs labels par texte
  - Utility evaluation pour chaque label
  - Métriques micro/macro

- [ ] **Modes d'évaluation**
  - Mode "privacy-first" : maximiser privacy
  - Mode "utility-first" : minimiser perte utility
  - Mode "balanced" : compromis équilibré

### Visualisation

- [ ] **Dashboard de résultats**

  - Graphiques privacy vs utility
  - Courbes ROC
  - Distribution des rangs
  - Interface Streamlit ou Gradio

- [ ] **Export des résultats**
  - CSV pour analyse Excel
  - LaTeX pour publications
  - HTML avec graphiques interactifs

### Tests

- [ ] **Tests unitaires**

  - `tests/test_privacy_evaluator.py`
  - `tests/test_utility_evaluator.py`
  - `tests/test_optimizer.py`
  - Coverage > 80%

- [ ] **Tests d'intégration**

  - `tests/test_rupta_pipeline.py`
  - Tester scenarios réels
  - Mock des appels LLM pour CI/CD

- [ ] **Tests de régression**
  - Benchmarks sur DB-Bio
  - Assurer pas de dégradation
  - Automated testing dans CI

## 📊 Évaluation à Grande Échelle (Phase 5)

### Benchmarks Complets

- [ ] **DB-Bio full evaluation**

  - Test set complet (~1000 exemples)
  - Baseline vs RUPTA
  - Rapport détaillé

- [ ] **PersonalReddit full evaluation**
  - Test set complet
  - Évaluation par attribut
  - Comparaison avec paper RUPTA original

### Comparaison avec State-of-the-Art

- [ ] **Implémenter baselines**

  - Presidio (Microsoft)
  - NER simple (spaCy)
  - Regex-based
  - Random masking

- [ ] **Benchmarking complet**
  - Tableau comparatif
  - Analyse statistique
  - Publication résultats

## 📝 Documentation

- [ ] **Tutoriel vidéo**

  - Démo de l'installation
  - Exemple d'utilisation
  - Interprétation des résultats

- [ ] **Paper technique**

  - Méthodologie d'intégration
  - Résultats sur datasets français
  - Adaptation cross-linguale

- [ ] **API Documentation**
  - Docstrings complètes
  - Sphinx documentation
  - Hébergement Read the Docs

## 🐛 Bugs Connus

Aucun pour l'instant. À documenter lors des tests.

## 💡 Idées Futures

- [ ] **Fine-tuning des prompts avec GPT-4**

  - Génération automatique de variantes
  - Sélection par performance

- [ ] **Active Learning**

  - Sélectionner exemples les plus informatifs
  - Réduire nombre d'évaluations nécessaires

- [ ] **Transfer Learning**

  - Adapter modèle privacy/utility sur données françaises
  - Améliorer performance cross-linguale

- [ ] **Reinforcement Learning**
  - Apprendre politique d'optimisation
  - Minimiser nombre d'itérations

## 📅 Timeline Recommandée

### Semaine 1

- ✅ Implémentation des modules core (FAIT)
- ✅ Documentation (FAIT)
- 🔄 Téléchargement datasets
- 🔄 Tests initiaux (5-10 exemples)

### Semaine 2

- Évaluation moyenne échelle (50-100 exemples)
- Optimisation hyperparamètres
- Intégration dans orchestrator
- Tests de régression

### Semaine 3

- PersonalReddit evaluation
- Optimisation prompts français
- Implémentation cache LLM
- Visualisations

### Semaine 4

- Évaluation grande échelle (full datasets)
- Benchmarking vs baselines
- Documentation finale
- Publication résultats

## 🎯 Critères de Succès

### Phase 2 (MVP)

- [x] Code fonctionnel et testé
- [ ] Datasets téléchargés et validés
- [ ] Évaluation sur ≥10 exemples
- [ ] Comparaison baseline vs RUPTA documentée

### Phase 3 (Production)

- [ ] Intégration dans pipeline principal
- [ ] Tests automatisés (coverage >80%)
- [ ] Documentation complète
- [ ] Performance acceptable (<30s par texte)

### Phase 4 (Optimisé)

- [ ] Hyperparamètres optimisés
- [ ] Cache LLM implémenté
- [ ] Coût réduit de 50%
- [ ] Visualisations disponibles

### Phase 5 (Research)

- [ ] Évaluation full datasets
- [ ] Comparaison state-of-the-art
- [ ] Publication des résultats
- [ ] Contribution open-source

---

**Dernière mise à jour** : Phase 1 complétée
**Prochaine tâche prioritaire** : Télécharger datasets et lancer premiers tests
