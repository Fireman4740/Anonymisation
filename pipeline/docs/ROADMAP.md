# Roadmap d'exécution – Pipeline d'anonymisation avancé

Cette feuille de route synthétise les solutions identifiées par l'agent de recherche et les traduit en étapes concrètes pour renforcer le pipeline hybride (regex + NER + LLM + RUPTA).

## Objectifs clés

1. **Détection déterministe renforcée** : couvrir secrets, IBAN/BIC, cartes, téléphones multi-pays, IPv6 et schémas d'URL non-HTTP via un fichier de patterns versionné.
2. **Pipeline avancé** : introduire `AdvancedAnonymizer` (Phase 1 regex + validations, Phase 2 NER Flair/spaCy) et l'intégrer à la détection existante.
3. **Sécurité LLM/RUPTA** : réduire l'intensité de paraphrase, préserver la multiplicité des entités et s'assurer que RUPTA respecte les occurrences et patterns interdits.
4. **Validation & QA** : appliquer des vérifications post-traitement (forbidden patterns, compteurs d'entités) et automatiser l'évaluation via `test_exhaustif.py` sur `hard_realistic_cases.json`.

## Jalons par lot de travail

| Lot                              | Actions                                                                   | Résultats attendus |
| -------------------------------- | ------------------------------------------------------------------------- | ------------------ |
| **LT1 – Patterns déterministes** | • Créer `patterns_config.yaml` (priorités, validations Luhn & Schwifty) \ |

• Ajouter chargeur YAML et watchers \
• Étendre les regex aux schémas secrets/API/IPv6 | Couverture complète des cas "secrets/API/financiers" avant toute détection NER. |
| **LT2 – AdvancedAnonymizer** | • Implémenter Phase 1 (regex, Schwifty, phonenumbers) \
• Implémenter Phase 2 (Flair `flair/ner-french`, spaCy pour future fusion) \
• Gestion des overlaps + mapping d'entités \
• Intégrer au `DetectionService` via un flag `use_advanced_anonymizer` | Résultats NER/regex uniformisés, prêtes à être consommés par le mapper de placeholders. |
| **LT3 – LLM & RUPTA** | • Réduire `paraphrase_intensity` à 1 (LOW) \
• Ajouter `paraphrase_preserve_multiplicity` + prompt explicite \
• Enrichir overrides RUPTA (`rupta_preserve_entity_counts`) \
• Surface warnings lorsque les validations échouent et repli sur le texte anonymisé pré-LLM | Paraphrases conservant les occurrences, RUPTA aligné avec les contraintes d'entités. |
| **LT4 – Validation & QA** | • Ajouter `validate_anonymization()` (forbidden patterns + compteurs) \
• Injecter `expected_placeholder_counts` depuis les datasets/tests \
• Mettre à jour `hard_realistic_cases.json` (overrides globales + nouveaux attributs) \
• Exécuter `test_exhaustif.py` sur dataset dur + rapporter résultats | Boucle QA automatisée et rapports exploitables. |

## Plan temporel (3 jours ouvrés)

| Jour                                | Actions détaillées                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| **Jour 1 – Patterns & dépendances** | • Installer libs locales (`schwifty`, `phonenumbers`, `flair`, `spacy`, `pyyaml`) \ |

• Créer `patterns_config.yaml` et loaders \
• Étendre `requirements.txt` + doc d'installation \
• Vérifier détection sur `case_cloud_logs_indirect` (secrets/forbidden) |
| **Jour 2 – AdvancedAnonymizer** | • Implémenter la classe + tests unitaires ciblés \
• Intégrer au `DetectionService` et exposer flag dans overrides \
• Couvrir IBAN/BIC, téléphones internationaux, IPv6/URL spéciaux \
• Valider sur `case_long_b2b_investigation` & `case_contract_multilingual` |
| **Jour 3 – LLM/RUPTA & QA** | • Ajuster prompts/paraphrase, ajouter validations post-LLM \
• Appliquer overrides globaux (`paraphrase_preserve_multiplicity`, `rupta_preserve_entity_counts`) \
• Ajouter `validate_anonymization` + warning/reporting \
• Lancer `test_exhaustif.py --dataset evaluation/datasets/hard_realistic_cases.json` et consigner le rapport |

## Deliverables

- `docs/ROADMAP.md` (ce document) + mise à jour de `docs/anonymisation_solutions.md`
- `patterns_config.yaml` versionné + chargeur Python
- Nouveau module `AdvancedAnonymizer` avec tests unitaires de régression
- Mise à jour du pipeline LLM/RUPTA + validations
- Rapport d'évaluation (`evaluation/reports/test_results.json`) documentant le run final

## Risques & mitigations

- **Poids des modèles Flair/spaCy** : prévoir fallback silencieux (désactivation automatique si non installés) pour ne pas bloquer la CI.
- **Temps d'exécution** : limiter l'usage LLM aux cas `L1`, prévoir un mode `--skip-llm` pour QA rapide.
- **Couverture dataset** : surveiller `expected_pattern_counts` et `forbidden_patterns` lors des régressions; les exposer comme overrides pour que le pipeline puisse réagir en temps réel.

---

Cette roadmap sert de référence pour coordonner développement, validation et documentation jusqu'à l'obtention d'un taux de réussite de 100% sur le dataset "hard".
