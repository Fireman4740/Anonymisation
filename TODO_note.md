# Journal d'avancement 25-11

J'ai travaillé sur les points suivants aujourd'hui :

- création d'un nouveau pippeline d'annonymistion en utilisant langgraph. Je me suis basé sur le pipeline existant dans src/pipeline/ mais en utilisant langgraph pour la gestion des étapes.
- Je suis reparti a la base du code pour mieux comprendre comment intégrer langgraph.
- J'ai commencé a implémenter la détéction des PII en utilisant langgraph.
- J'ai fait un interface de visualistation des erreurs dans streamlit pour mieux comprendre les erreurs du pipeline.

Next steps:
Je veux continuer l'application streamlit de visualisation en intégra directement les scripts d'évaluation.
Il faut que je formalise un script d'évaluation pour chaque dataset car ils n'ont la meme utiliser et métriques.
Je veux racorder le pipeline graphe au streamlit pour évaluer au fur et a mesure des modifications du pipeline.

Pour le pipeline je souhaite me consentrer pour le moment sur la détection la plus fiable possible des PII et du masquage.
Les prochaines étapes sont d'optimiser le code qui utilse les ner et scpacy. Il faut refaire le code de merge en regex et Ner.
J'aimerai tester un genre de concil avec les ner et un llm ultra rapide pour augmenter les performaces.

## TO DO list

- [ ] **Streamlit** : Intégrer directement les scripts d'évaluation dans l'application de visualisation.
- [ ] **Évaluation** : Formaliser un script d'évaluation spécifique pour chaque dataset (métriques et usages différents).
  - [ ] **TAB (Text Anonymization Benchmark)** : Dataset de référence juridique (CEDH).
  - [ ] **Gretel AI Synthetic PII Finance** : Données synthétiques multilingues (dont FR).
  - [ ] **AI4Privacy PII Masking** : Grand volume pour validation NER.
  - [ ] **DB-bio** : Données biomédicales/textes cliniques pour transfert anonymisation.
  - [ ] **PersonalReddit** : Posts personnels (style narratif) pour stress-test généralisation PII.
- [ ] **Intégration** : Raccorder le pipeline LangGraph au Streamlit pour une évaluation en temps réel des modifications.
- [ ] **Pipeline (Détection)** : Optimiser le code utilisant les NER et spaCy.
- [ ] **Pipeline (Merge)** : Refaire le code de fusion entre les résultats Regex et NER.
- [ ] **Expérimentation** : Tester une approche "conseil" combinant NER et un LLM rapide pour améliorer les performances de détection.
