# 📚 Documentation du Pipeline d'Anonymisation

Bienvenue dans la documentation complète du pipeline d'anonymisation. Cette section contient tous les guides, références et analyses techniques du projet.

## 📖 Guides Principaux

### 🚀 [QUICKSTART.md](../QUICKSTART.md)
**Guide de démarrage rapide** - Commencez ici !
- Installation en 5 minutes
- Premiers tests et exemples
- Cas d'usage pratiques (médical, logs, RGPD)
- Configuration GPU
- API clé en main

### 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md)
**Architecture du système** - Comprenez le fonctionnement interne
- Vue d'ensemble des 3 couches (Détection, Transformation, Évaluation)
- Flux de données et orchestration
- Services et composants
- Diagrammes détaillés
- Principes de conception

### 📚 [API_REFERENCE.md](API_REFERENCE.md)
**Référence API complète** - Toutes les fonctions et classes
- API publique (`anonymize_text`, `AnonymizationPipeline`)
- Policy et configuration
- Services avancés
- Paramètres et overrides
- Exemples de code

### 🗺️ [ROADMAP.md](ROADMAP.md)
**Feuille de route** - Évolutions futures
- Objectifs clés
- Jalons par lot de travail
- Plan temporel
- Améliorations prévues (patterns déterministes, AdvancedAnonymizer, validation QA)

## 📊 Analyses et Rapports

### [analysis_report.md](analysis_report.md)
**Rapport d'analyse détaillé** - Analyse complète du système (1882 lignes)
- Contexte et problèmes identifiés
- Recommandations d'architecture
- Analyse approfondie de chaque composant
- Stratégies de refactoring

### [anonymisation_solutions.md](anonymisation_solutions.md)
**Solutions techniques** - Résolution des problèmes spécifiques (597 lignes)
- Problèmes critiques du pipeline hybride
- Solutions basées sur l'état de l'art
- Librairies recommandées (schwifty, phonenumbers, Flair)
- Implémentation de patterns avancés

### [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
**Résumé d'implémentation** - Suivi des tâches accomplies
- ✅ Architecture en 3 couches
- ✅ Orchestrateur refactorisé
- ✅ API publique
- ✅ Documentation complète
- ✅ Tests et exemples

## 🎯 Navigation Recommandée

### Pour Débutants
1. 📖 Lire le [README principal](../../README.md)
2. 🚀 Suivre le [QUICKSTART.md](../QUICKSTART.md)
3. 🧪 Tester avec les exemples

### Pour Développeurs
1. 🏗️ Comprendre l'[ARCHITECTURE.md](ARCHITECTURE.md)
2. 📚 Explorer l'[API_REFERENCE.md](API_REFERENCE.md)
3. 📊 Lire [analysis_report.md](analysis_report.md)
4. 🗺️ Consulter la [ROADMAP.md](ROADMAP.md)

### Pour Intégrateurs
1. 🚀 [QUICKSTART.md](../QUICKSTART.md) - Section "Intégration dans Votre Application"
2. 📚 [API_REFERENCE.md](API_REFERENCE.md) - Exemples FastAPI/Flask
3. ⚙️ Configuration et variables d'environnement

### Pour Contributeurs
1. 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture complète
2. 📊 [analysis_report.md](analysis_report.md) - Analyse technique
3. 🗺️ [ROADMAP.md](ROADMAP.md) - Prochaines étapes
4. 🔧 [anonymisation_solutions.md](anonymisation_solutions.md) - Solutions techniques

## 📁 Structure de la Documentation

```
docs/
├── README.md                      # Ce fichier (index)
├── ARCHITECTURE.md                # Architecture du système
├── API_REFERENCE.md               # Référence API complète
├── ROADMAP.md                     # Feuille de route
├── analysis_report.md             # Rapport d'analyse détaillé
├── anonymisation_solutions.md     # Solutions techniques
└── IMPLEMENTATION_SUMMARY.md      # Résumé d'implémentation
```

## 🔗 Liens Utiles

- [README Principal](../../README.md)
- [QUICKSTART](../QUICKSTART.md)
- [Code Source](../src/)
- [Tests](../tests/)
- [Évaluation](../evaluation/)
- [Scripts](../scripts/)

## 📝 Notes de Version

### Version 2.0.0 (Actuelle)
- ✅ Architecture refactorisée en 3 couches
- ✅ Support RUPTA et optimisation privacy-utility
- ✅ API FastAPI intégrée
- ✅ Patterns avancés (IBAN, BIC, secrets, IPv6)
- ✅ Support GPU automatique
- ✅ Documentation complète

---

**💡 Conseil** : Si vous cherchez quelque chose de spécifique, utilisez la recherche de votre éditeur (Ctrl+F / Cmd+F) dans les documents Markdown ou consultez l'[API_REFERENCE.md](API_REFERENCE.md) pour les fonctions.

**🆘 Besoin d'aide ?** Commencez par le [QUICKSTART.md](../QUICKSTART.md) ou consultez les exemples dans le code.
