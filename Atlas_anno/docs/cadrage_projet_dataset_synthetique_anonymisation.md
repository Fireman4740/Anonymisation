# Projet Atlas — Dataset synthétique d’anonymisation textuelle et banc d’évaluation par réidentification

## 1. Résumé exécutif

Ce projet vise à construire un système complet pour générer, annoter, anonymiser et évaluer un dataset synthétique de textes réalistes (support client, emails, texte libre multi-domaines) afin de mesurer la qualité de pipelines d’anonymisation textuelle.

L’objectif n’est pas seulement de masquer les identifiants directs, mais aussi d’évaluer si un attaquant peut encore réidentifier une personne fictive à partir d’indices indirects, de quasi-identifiants, de combinaisons rares d’attributs ou de signaux de style.

Le projet repose sur quatre briques :

1. **Un générateur de population fictive** qui crée des personnes cohérentes avec des attributs, biographies, rôles, historiques, signatures d’écriture et univers organisationnels.
2. **Un générateur de textes** qui produit des tickets, emails, conversations et documents courts/moyens à partir de ces personnages.
3. **Un moteur de pré-annotation et d’anonymisation** piloté par LLM.
4. **Un framework d’évaluation** qui mesure à la fois la qualité de l’anonymisation, l’utilité résiduelle et le risque de réidentification par un modèle attaquant.

Le projet est volontairement **synthétique**, afin d’éviter les risques juridiques et éthiques liés à la redistribution de données réelles, tout en restant **pertinent** grâce à des personnages riches, des contextes réalistes et des scénarios d’attaque structurés.

---

## 2. Objectifs du projet

### 2.1 Objectif principal

Créer un benchmark reproductible permettant de comparer plusieurs pipelines d’anonymisation textuelle sur trois axes :

- **protection de la vie privée**,
- **résistance à la réidentification**,
- **préservation de l’utilité métier**.

### 2.2 Sous-objectifs

- Générer un corpus synthétique réaliste de support client, emails et texte libre.
- Représenter des identifiants directs et indirects.
- Simuler plusieurs niveaux d’attaquants.
- Évaluer si des LLM ou des attaquants structurés peuvent retrouver le bon individu fictif.
- Produire un benchmark simple à étendre.
- Rendre le pipeline industrialisable.

### 2.3 Ce que le projet ne cherche pas à faire

- Publier un benchmark fondé sur des données réelles identifiantes.
- Garantir une anonymisation “absolue”.
- Remplacer l’évaluation humaine et juridique.

---

## 3. Hypothèse centrale

Un texte n’est pas suffisamment anonymisé si :

- les identifiants directs ont disparu,
- mais qu’un attaquant peut encore retrouver la personne grâce à des indices indirects comme l’âge, le rôle, l’équipe, l’ancienneté, un événement rare, une expertise unique ou une combinaison atypique.

Le bon signal d’évaluation n’est donc pas seulement :

- “est-ce que les noms ont été masqués ?”

mais surtout :

- “est-ce qu’un attaquant peut encore retrouver la bonne personne ou réduire l’espace candidat à un nombre très faible ?”

---

## 4. Périmètre fonctionnel

### 4.1 Types de textes couverts

Le benchmark couvre trois familles de textes :

1. **Support client**
   - tickets courts
   - tickets longs
   - descriptions de bugs
   - échanges support / client
   - messages d’escalade

2. **Emails**
   - email unique
   - email avec signature
   - thread court
   - relance interne
   - coordination de projet

3. **Texte libre multi-domaines**
   - demande RH
   - message IT interne
   - demande administrative
   - retour utilisateur
   - note contextuelle

### 4.2 Niveaux d’évaluation

Le projet évalue :

- détection des spans sensibles,
- qualité d’anonymisation,
- risque résiduel,
- réidentification ciblée,
- linkability entre documents,
- utilité sur tâches aval.

---

## 5. Choix techniques imposés

### 5.1 Orchestrateur LLM

Le système utilisera l’API OpenRouter comme couche d’accès unique aux modèles.

### 5.2 Politique de routage modèle

- **Modèle principal pour tâches complexes, analytiques, structurées, multi-étapes et raisonnement** : `aion-labs/aion-2.0`
- **Modèle principal pour tâches créatives, génération stylistique, variation de ton, enrichissement textuel et reformulation** : `mistralai/mistral-small-creative`

### 5.3 Rôle recommandé des modèles

#### aion-labs/aion-2.0

À utiliser pour :

- génération de population cohérente avec contraintes,
- contrôle de cohérence globale d’un personnage,
- extraction structurée,
- annotation complexe,
- détection de combinaisons à risque,
- scoring explicable,
- génération de mondes adversariaux,
- planification d’anonymisation,
- agent attaquant structuré,
- audit final.

#### mistralai/mistral-small-creative

À utiliser pour :

- génération de variantes stylistiques,
- génération de tickets réalistes,
- génération d’emails naturels,
- enrichissement de formulations,
- réécriture anonymisée fluide,
- génération de paraphrases,
- augmentation de diversité linguistique.

### 5.4 Principe d’architecture

Le système doit séparer :

- la **vérité structurée** (personnages, attributs, causalité),
- la **surface textuelle** (texte généré),
- la **vérité d’annotation**,
- la **sortie anonymisée**,
- la **sortie attaquant**.

---

## 6. Cas d’usage cibles

### Cas 1 — Benchmark de pipeline d’anonymisation

Entrée : texte brut.

Sortie : texte anonymisé.

Évaluation : le pipeline doit protéger l’identité tout en conservant l’utilité.

### Cas 2 — Benchmark d’attaquant

Entrée : texte anonymisé + liste de candidats.

Sortie : ranking de personnes plausibles.

Évaluation : plus l’attaquant réussit, plus le risque résiduel est fort.

### Cas 3 — Benchmark de linkability

Entrée : deux textes anonymisés.

Sortie : probabilité qu’ils viennent de la même personne.

### Cas 4 — Benchmark privacy / utility

Comparer plusieurs stratégies :

- masking,
- généralisation,
- suppression,
- reformulation,
- réécriture privacy-first,
- réécriture utility-first.

---

## 7. Principes de design du dataset

### 7.1 Pourquoi un dataset synthétique

Le choix synthétique permet :

- contrôle total de la vérité terrain,
- génération de cas rares,
- maîtrise des quasi-identifiants,
- absence de données personnelles réelles,
- création de scénarios de difficulté ciblés,
- meilleure redistribution du benchmark.

### 7.2 Risque à éviter

Un dataset synthétique pauvre serait peu utile si :

- les textes sont trop artificiels,
- les quasi-identifiants sont trop évidents,
- les personnages sont trop simples,
- les attaquants sont trop faibles,
- les signaux de style sont absents,
- les univers organisationnels sont trop petits ou incohérents.

### 7.3 Principe directeur

Le dataset doit être **simple à construire**, mais **riche dans les interactions entre attributs**.

---

## 8. Vue d’ensemble de l’architecture

```text
Population Generator
    -> Character Profiles
    -> Organization Worlds
    -> Event Worlds

Scenario Generator
    -> Support / Email / Free-text briefs

Text Generator
    -> Raw texts
    -> Context / thread metadata

Annotation Agent
    -> Direct identifiers
    -> Quasi-identifiers
    -> Sensitive attributes
    -> Compositional links

Anonymization Baselines
    -> Masking
    -> Generalization
    -> Rewriting

Attacker Suite
    -> Structured attacker
    -> LLM attacker
    -> Linkability attacker

Evaluation Suite
    -> Span metrics
    -> Privacy metrics
    -> Re-id metrics
    -> Utility metrics
    -> Global reports
```

---

## 9. Modèle conceptuel du dataset

Le dataset repose sur cinq couches logiques.

### 9.1 Couche 1 — Population fictive

Chaque personne fictive a :

- un identifiant interne stable,
- un prénom / nom synthétique,
- une langue,
- un ton d’écriture,
- un métier,
- un niveau de séniorité,
- une équipe,
- une organisation,
- une localisation,
- une tranche d’âge,
- une ancienneté,
- des expertises,
- des certifications,
- des événements marquants,
- des attributs potentiellement sensibles,
- des particularités rares,
- des tics de style.

### 9.2 Couche 2 — Monde organisationnel

Un “world” contient :

- entreprises fictives,
- départements,
- équipes,
- organigrammes,
- projets,
- produits,
- incidents,
- calendriers,
- historiques,
- spécialisations rares.

Le world sert à rendre certaines combinaisons identifiantes.

### 9.3 Couche 3 — Scénarios documentaires

Chaque document est généré à partir d’un brief qui précise :

- type de texte,
- auteur,
- destinataire,
- but du message,
- ton,
- contraintes de domaine,
- attributs qui doivent apparaître,
- attributs qui doivent rester implicites,
- bruit textuel,
- niveau de difficulté.

### 9.4 Couche 4 — Annotations gold

Le dataset contient :

- spans identifiants directs,
- spans quasi-identifiants,
- spans sensibles,
- relations compositionnelles,
- labels doc-level,
- monde de candidats pour la réidentification,
- cibles d’anonymisation de référence.

### 9.5 Couche 5 — Sorties d’évaluation

Pour chaque document, on stocke :

- sortie du pipeline anonymiseur,
- score privacy,
- score utilité,
- score réidentification,
- score linkability,
- diagnostics d’échec.

---

## 10. Schéma minimal des personnages fictifs

```json
{
  "person_id": "p_0042",
  "full_name": "Nadia Mercier",
  "age_range": "25-29",
  "gender": "female",
  "language": "fr",
  "country": "France",
  "organization_id": "org_02",
  "department": "AI Solutions",
  "team": "LLM Ops",
  "role": "AI Support Engineer",
  "seniority": "mid",
  "tenure_years": 2,
  "degrees": ["PhD"],
  "skills": ["python", "rag", "prompting", "vector db"],
  "rare_traits": ["only_phd_under_30_in_team"],
  "certifications": ["azure-ai-architect"],
  "events": ["post_merger_hire_2024", "owner_incident_1178"],
  "style_profile": {
    "formality": "medium",
    "signature_pattern": "thanks_name",
    "verbosity": "short",
    "emoji_usage": "none",
    "favorite_connectors": ["donc", "par contre"]
  },
  "sensitive_attributes": []
}
```

---

## 11. Schéma minimal des documents

```json
{
  "doc_id": "doc_000001",
  "domain": "support_ticket",
  "unit_type": "single_message",
  "language": "fr",
  "author_id": "p_0042",
  "target_person_ids": ["p_0042"],
  "world_id": "world_enterprise_02",
  "text": "Bonjour, je suis docteure en IA dans l’équipe LLM Ops et j’ai moins de 30 ans. Je gère seule l’intégration du connecteur X depuis l’incident de mars.",
  "scenario": {
    "difficulty": "hard",
    "required_signals": ["degree", "age_range", "unique_responsibility", "team"],
    "implicit_signals": ["rare_profile"],
    "document_goal": "request_help"
  },
  "annotations": {
    "spans": [],
    "relations": [],
    "doc_labels": {}
  },
  "candidate_worlds": {
    "public": ["p_0007", "p_0042", "p_0054", "p_0098"],
    "org_internal": ["p_0042", "p_0054"],
    "insider": ["p_0042"]
  }
}
```

---

## 12. Ontologie d’annotation

### 12.1 Identifiants directs

- PERSON_NAME
- EMAIL
- PHONE
- USERNAME
- ADDRESS
- URL
- ACCOUNT_ID
- ORDER_ID
- IP_ADDRESS
- ORG_NAME_STRONG
- PROJECT_NAME_STRONG

### 12.2 Quasi-identifiants explicites

- ROLE
- DEGREE
- AGE
- AGE_RANGE
- TENURE
- TEAM
- DEPARTMENT
- LOCATION
- NATIONALITY
- CERTIFICATION
- SKILL_RARE
- PRODUCT_CONTEXT
- EVENT_DATE
- CUSTOMER_SEGMENT

### 12.3 Quasi-identifiants implicites

- UNIQUE_ROLE
- ONLY_PERSON_WITH_X
- SMALL_GROUP_HINT
- POST_EVENT_LINK
- ORG_HISTORY_LINK
- RARE_RESPONSIBILITY
- COMPOSITIONAL_QID

### 12.4 Attributs sensibles

- HEALTH
- ETHNICITY
- RELIGION
- DISABILITY
- FAMILY_STATUS
- SEXUAL_ORIENTATION
- LEGAL
- FINANCIAL

### 12.5 Signaux de liaison / style

- SIGNATURE_PATTERN
- WRITING_TIC
- JARGON_PATTERN
- SAME_AUTHOR_SIGNAL
- THREAD_LINK_HINT

### 12.6 Labels documentaires

- direct_risk_level
- qid_risk_level
- compositional_risk_level
- style_risk_level
- overall_reid_risk
- utility_sensitivity
- human_review_required

---

## 13. Règles de vérité terrain

### 13.1 Règle 1

Chaque document doit avoir un auteur principal connu.

### 13.2 Règle 2

Chaque signal injecté doit être traçable à un attribut du personnage ou du monde.

### 13.3 Règle 3

Chaque document difficile doit contenir au moins une combinaison risquée, pas seulement un identifiant direct.

### 13.4 Règle 4

Les signaux de réidentification doivent varier entre :

- explicites,
- implicites,
- compositionnels,
- stylistiques.

### 13.5 Règle 5

Le dataset doit contenir des cas où le pipeline de masking simple échoue.

---

## 14. Génération des mondes fictifs

### 14.1 Objectif

Construire des univers suffisamment riches pour que certains attributs soient anodins seuls mais identifiants en combinaison.

### 14.2 Structure d’un world

- 1 à 5 entreprises fictives
- 3 à 10 départements
- 5 à 50 personnages
- quelques profils rares
- historique de projets
- incidents marquants
- structure hiérarchique
- technologies maîtrisées
- calendrier d’événements

### 14.3 Difficulté progressive

#### Niveau easy

Les candidats sont nombreux.

#### Niveau medium

Les combinaisons réduisent le pool à quelques candidats.

#### Niveau hard

Une combinaison de 2 à 4 indices suffit presque à identifier.

#### Niveau adversarial

Le texte contient peu de PII directes, mais la combinaison contextuelle ou le style reste suffisant.

---

## 15. Génération des personnages

### 15.1 Distribution

Chaque world doit contenir :

- profils communs,
- profils rares,
- doublons partiels,
- profils confondables,
- profils uniques.

### 15.2 Types de rareté à injecter

- seul docteur d’une équipe,
- seule personne certifiée sur un outil,
- plus jeune / plus ancien d’une équipe,
- seul responsable d’un connecteur,
- profil hybride rare,
- personne associée à un incident unique,
- combinaison âge + rôle + ancienneté atypique.

### 15.3 Contrôle qualité population

Le générateur doit refuser :

- les populations incohérentes,
- les univers trop homogènes,
- les rares profils trop artificiels,
- les collisions impossibles,
- les attributs contradictoires.

---

## 16. Génération des scénarios textuels

### 16.1 Familles de scénarios

#### Support client

- signalement d’incident
- demande de réinitialisation
- intégration cassée
- problème d’accès
- configuration produit
- escalade technique
- retour utilisateur expert

#### Emails

- demande d’aide
- coordination interne
- relance projet
- incident urgent
- partage de contexte
- réponse à support

#### Texte libre

- note explicative
- message d’introduction
- auto-description contextuelle
- commentaire post-incident

### 16.2 Paramètres de scénario

- longueur
- formalisme
- urgence
- présence de signature
- quantité de bruit
- densité de QIDs
- degré d’implicite
- type de destinataire
- présence d’historique de thread

---

## 17. Agent de génération du dataset

Le système de génération sera composé de sous-agents spécialisés.

### 17.1 Agent A — World Builder

**But** : générer un monde organisationnel cohérent.

**Entrée** : paramètres de taille, diversité, niveau de rareté.

**Sortie** : organisations, équipes, événements, relations.

**Modèle recommandé** : `aion-labs/aion-2.0`

### 17.2 Agent B — Character Builder

**But** : générer des personnages cohérents et divers.

**Entrée** : world, distributions cibles.

**Sortie** : profils structurés complets.

**Modèle recommandé** : `aion-labs/aion-2.0`

### 17.3 Agent C — Scenario Planner

**But** : définir le brief documentaire et les signaux à injecter.

**Entrée** : personnage, world, domaine, niveau de difficulté.

**Sortie** : scenario spec.

**Modèle recommandé** : `aion-labs/aion-2.0`

### 17.4 Agent D — Surface Text Generator

**But** : produire le texte naturel.

**Entrée** : scenario spec + style profile.

**Sortie** : texte brut.

**Modèle recommandé** : `mistralai/mistral-small-creative`

### 17.5 Agent E — Annotation Pre-Labeler

**But** : proposer les spans et risques.

**Entrée** : texte + vérité structurée.

**Sortie** : annotations candidates.

**Modèle recommandé** : `aion-labs/aion-2.0`

### 17.6 Agent F — Consistency Auditor

**But** : vérifier que le texte correspond bien à la vérité du personnage.

**Entrée** : texte + profil + scénario.

**Sortie** : validation ou rejet.

**Modèle recommandé** : `aion-labs/aion-2.0`

---

## 18. Agent de pré-annotation

### 18.1 But

Produire une pré-annotation de haute qualité pour réduire le travail humain.

### 18.2 Sorties attendues

- spans détectés,
- labels proposés,
- justification,
- score de confiance,
- signaux compositionnels,
- suggestion d’anonymisation,
- drapeau de revue humaine.

### 18.3 Stratégie recommandée

Pipeline hybride :

1. règles regex,
2. extraction déterministe,
3. LLM span annotation,
4. LLM compositional risk,
5. audit final.

### 18.4 Critères de passage en revue humaine

- contradiction entre règles et LLM,
- confiance faible,
- document hard ou adversarial,
- présence sensible,
- composition risquée non résolue.

---

## 19. Agent anonymiseur

### 19.1 Rôle

Prendre un texte et produire plusieurs variantes anonymisées.

### 19.2 Stratégies à comparer

- **masking** : remplacements par placeholders,
- **generalization** : abstraire les détails,
- **rewrite_privacy_first** : maximiser la protection,
- **rewrite_utility_first** : préserver davantage le sens,
- **balanced** : compromis privacy / utilité.

### 19.3 Sorties minimales

- anonymized_text,
- actions_performed,
- rationale,
- estimated_privacy_gain,
- estimated_utility_loss.

### 19.4 Règle clé

Le pipeline doit pouvoir être évalué sans dépendre de sa justification textuelle.

---

## 20. Agent attaquant

Le benchmark doit intégrer plusieurs attaquants.

### 20.1 Structured Attacker

Utilise les attributs extraits et filtre la liste des candidats.

Sortie :

- ensemble candidat,
- ranking,
- attributs causaux.

### 20.2 LLM Attacker

Reçoit :

- texte anonymisé,
- liste de candidats,
- profils candidats.

Retourne :

- top-k candidats,
- justification,
- niveau de confiance.

### 20.3 Linkability Attacker

Entrée : deux documents anonymisés.

Sortie : score same-author / different-author.

### 20.4 Adversarial Style Attacker

But : exploiter style, signature, tics, structure.

---

## 21. Mondes d’attaquants

Chaque exemple doit être évalué sous plusieurs niveaux de connaissance.

### 21.1 Public

Connaît des informations génériques et publiques.

### 21.2 Organizational

Connaît les équipes, les rôles et quelques événements internes.

### 21.3 Insider

Connaît les incidents, responsabilités, historiques et profils rares.

### 21.4 Pourquoi c’est critique

Un texte peut être anonyme pour le public mais trivialement réidentifiable pour un collègue interne.

---

## 22. Protocoles d’évaluation

### 22.1 Bloc A — Évaluation de spans

Mesures :

- precision,
- recall,
- F1,
- strict match,
- overlap match,
- macro par label,
- micro global.

### 22.2 Bloc B — Évaluation du risque résiduel

Mesures :

- nombre de spans à risque restants,
- densité résiduelle,
- rareté résiduelle,
- compositional risk score,
- style risk score.

### 22.3 Bloc C — Réidentification ciblée

Mesures :

- top-1,
- top-3,
- top-5,
- MRR,
- taille moyenne du pool candidat,
- taux de singleton,
- entropie résiduelle.

### 22.4 Bloc D — Linkability

Mesures :

- ROC-AUC,
- EER,
- pairwise F1,
- cluster purity.

### 22.5 Bloc E — Utilité

Mesures recommandées :

- intention classification,
- routing support,
- topic classification,
- résumé,
- similarité sémantique,
- extraction de problème et solution.

### 22.6 Bloc F — Pareto privacy / utility

Le benchmark doit comparer les pipelines sur une frontière privacy / utilité.

---

## 23. Scores globaux recommandés

### 23.1 Privacy Preservation Score

Score agrégé basé sur :

- suppression des identifiants directs,
- réduction des quasi-identifiants,
- baisse de succès de l’attaquant,
- réduction de la linkability.

### 23.2 Utility Preservation Score

Score agrégé basé sur :

- maintien du sens métier,
- performance sur tâches aval,
- lisibilité,
- cohérence.

### 23.3 Re-identification Risk Score

Score centré sur l’attaquant.

### 23.4 Global Benchmark Score

Combinaison pondérée configurable.

---

## 24. Splits du dataset

### 24.1 train

Documents standards pour développement.

### 24.2 dev

Pour calibration des prompts et hyperparamètres.

### 24.3 test_standard

Mix équilibré.

### 24.4 test_hard

Profils rares et combinaisons dangereuses.

### 24.5 test_linkability

Multi-documents par auteur.

### 24.6 test_open_world

Certains auteurs ne sont pas présents dans la liste des candidats.

### 24.7 test_cross_domain

Transfert entre emails, support et free text.

### 24.8 Règles de split

- split par auteur,
- split par thread,
- éviter les fuites de style,
- contrôler les distributions de rareté.

---

## 25. Spécification des scripts Python

### 25.1 `generate_worlds.py`

Génère les mondes fictifs.

### 25.2 `generate_characters.py`

Génère les profils fictifs.

### 25.3 `generate_scenarios.py`

Crée les briefs documentaires.

### 25.4 `generate_texts.py`

Produit les textes synthétiques.

### 25.5 `preannotate.py`

Produit une pré-annotation.

### 25.6 `validate_dataset.py`

Vérifie cohérence, duplicats, collisions et couverture.

### 25.7 `run_anonymizer.py`

Exécute un pipeline d’anonymisation.

### 25.8 `attack_structured.py`

Attaquant par filtrage et ranking structuré.

### 25.9 `attack_llm.py`

Attaquant LLM.

### 25.10 `attack_linkability.py`

Attaquant pairwise / cluster.

### 25.11 `eval_spans.py`

Scores de détection et transformation.

### 25.12 `eval_privacy.py`

Scores de risque résiduel.

### 25.13 `eval_reid.py`

Scores de réidentification.

### 25.14 `eval_utility.py`

Scores de tâches aval.

### 25.15 `build_report.py`

Rapport final HTML / Markdown / JSON.

---

## 26. Structure de repo recommandée

```text
project/
  configs/
    models.yaml
    generation.yaml
    attacks.yaml
    evaluation.yaml
  data/
    worlds/
    characters/
    scenarios/
    raw_docs/
    annotations/
    anonymized/
    reports/
  docs/
    product_requirements.md
    annotation_guidelines.md
    dataset_card.md
    benchmark_protocol.md
    threat_model.md
  prompts/
    world_builder/
    character_builder/
    scenario_planner/
    text_generator/
    preannotation/
    anonymizer/
    attacker/
  src/
    generation/
    annotation/
    anonymization/
    attacks/
    evaluation/
    utils/
  tests/
  notebooks/
  README.md
```

---

## 27. Contrats d’API internes

### 27.1 OpenRouter adapter

Le code applicatif ne doit jamais dépendre directement d’un modèle particulier.

Créer une interface commune :

- `generate_json()`
- `generate_text()`
- `score_candidates()`
- `rewrite_text()`

### 27.2 Routage recommandé

- JSON strict, extraction, audit, ranking : `aion-labs/aion-2.0`
- génération créative et réécriture naturelle : `mistralai/mistral-small-creative`

### 27.3 Résilience

Prévoir :

- retries,
- timeouts,
- validation JSON,
- fallback de prompt,
- journalisation des coûts.

---

## 28. Spécification des prompts

### 28.1 Prompt Character Builder

Le prompt doit imposer :

- cohérence métier,
- distribution d’attributs,
- rareté contrôlée,
- style stable,
- pas de caricature,
- diversité réaliste.

### 28.2 Prompt Text Generator

Le prompt doit imposer :

- naturel,
- pas de liste mécanique,
- intégration indirecte des indices,
- ton adapté au domaine,
- variation syntaxique,
- contrôle de longueur.

### 28.3 Prompt Pre-Annotation

Le prompt doit retourner :

- spans,
- labels,
- type de risque,
- justification,
- niveau de confiance.

### 28.4 Prompt Attacker

Le prompt doit forcer :

- choix parmi une liste de candidats,
- justification fondée sur indices,
- top-k,
- confiance,
- impossibilité de répondre hors liste si mode closed-world.

---

## 29. Contrôle qualité du dataset

### 29.1 Qualité minimale attendue

- cohérence personnage / texte,
- diversité lexicale,
- densité de signaux contrôlée,
- couverture de tous les labels,
- équilibre des domaines,
- présence de cas faciles, moyens et difficiles.

### 29.2 Reject rules

Rejeter un exemple si :

- attribut impossible,
- texte contradictoire,
- style robotique,
- QID absent dans un exemple hard,
- annotations incohérentes,
- candidat correct absent du candidate pool.

### 29.3 Tableaux de bord internes

- distribution des labels,
- distribution des scores de rareté,
- longueur moyenne,
- diversité par domaine,
- réussite baseline attacker,
- réussite masking baseline.

---

## 30. Jeux de baselines à livrer

### Baseline 1 — Regex + masking

Point de départ simple.

### Baseline 2 — NER + masking

Détection plus robuste mais encore limitée.

### Baseline 3 — LLM generalization

Pipeline LLM qui abstrait les détails.

### Baseline 4 — LLM rewrite balanced

Réécriture fluide orientée compromis.

### Baseline 5 — Adversarial refinement

Boucle anonymiseur -> attaquant -> anonymiseur.

---

## 31. Roadmap de développement

## Phase 0 — Cadrage

### Livrables

- vision produit,
- menace et threat model,
- ontologie v1,
- architecture technique,
- conventions JSON,
- plan d’évaluation.

### Durée estimative

1 semaine.

## Phase 1 — Génération de population

### Livrables

- générateur de worlds,
- générateur de personnages,
- validateurs de cohérence.

### Critères d’acceptation

- mondes cohérents,
- profils variés,
- profils rares contrôlés.

## Phase 2 — Génération documentaire

### Livrables

- générateur de scénarios,
- générateur de textes,
- génération multi-domaines.

### Critères d’acceptation

- textes naturels,
- diversité suffisante,
- couverture de signaux.

## Phase 3 — Pré-annotation

### Livrables

- pipeline hybride règles + LLM,
- schéma d’annotation,
- exports JSONL.

## Phase 4 — Attaquants et évaluation

### Livrables

- structured attacker,
- LLM attacker,
- linkability attacker,
- suite de métriques.

## Phase 5 — Baselines d’anonymisation

### Livrables

- masking baseline,
- generalization baseline,
- rewrite baseline,
- adversarial loop.

## Phase 6 — Rapport et packaging

### Livrables

- dataset card,
- benchmark protocol,
- scripts reproductibles,
- scoreboard.

---

## 32. MVP recommandé

### 32.1 Taille cible MVP

- 3 worlds
- 60 à 120 personnages
- 2 000 à 5 000 documents
- 3 domaines
- 3 niveaux de difficulté

### 32.2 Ce que le MVP doit déjà inclure

- au moins un attaquant structuré,
- un attaquant LLM,
- un anonymiseur baseline,
- des annotations gold synthétiques,
- un rapport privacy / utilité.

### 32.3 Ce que le MVP peut attendre

- annotations humaines massives,
- benchmark public final,
- leaderboard externe,
- multiplicité de langues.

---

## 33. V2 recommandée

- ajout multi-langue,
- ajout de threads plus longs,
- ajout de conversations support multi-tours,
- ajout de documents plus subtils,
- ajout de styles plus riches,
- ajout de feedback humain,
- ajout d’active learning,
- ajout de profils sensibles plus fins.

---

## 34. Critères de succès du projet

Le projet sera considéré réussi si :

1. le dataset est réaliste et cohérent,
2. les baselines simples échouent sur une partie non triviale des exemples,
3. les attaquants réussissent mieux quand l’anonymisation est faible,
4. les scores privacy / utilité permettent de distinguer clairement les pipelines,
5. les résultats sont reproductibles,
6. les diagnostics d’échec sont actionnables.

---

## 35. Risques projet

### 35.1 Risque 1 — Dataset trop artificiel

**Mitigation** : enrichir les styles, faire auditer des échantillons, injecter diversité contrôlée.

### 35.2 Risque 2 — Attaquant irréaliste

**Mitigation** : définir plusieurs niveaux d’attaquant et plusieurs mondes de connaissance.

### 35.3 Risque 3 — Score trop dépendant du prompt

**Mitigation** : prompts versionnés, seeds, réplications, protocole standard.

### 35.4 Risque 4 — Sur-optimisation au benchmark

**Mitigation** : splits durs, open-world, cross-domain.

### 35.5 Risque 5 — Anonymisation destructrice

**Mitigation** : ajouter des tâches aval et un score utilité.

---

## 36. Décisions de gouvernance à figer

- langue MVP : français seul ou bilingue FR/EN,
- closed-world only ou open-world inclus dès le départ,
- granularité message ou thread,
- publication publique ou benchmark interne d’abord,
- poids des scores privacy / utilité,
- présence ou non de signaux sensibles au MVP.

---

## 37. Recommandation de stack technique

- Python 3.11+
- Pydantic pour les schémas
- Typer ou Click pour CLI
- Pandas / Polars pour l’analyse
- Jinja2 pour prompts
- LiteLLM ou client OpenAI-compatible pour OpenRouter
- Pytest pour tests
- MLflow ou Weights & Biases pour traçabilité expérimentale
- DVC optionnel pour versionnement des artefacts

---

## 38. Exigences non fonctionnelles

### Reproductibilité

- seeds versionnées,
- prompts versionnés,
- modèles versionnés,
- configs gelées.

### Observabilité

- logs par étape,
- coût par appel LLM,
- taux d’échec JSON,
- métriques de validation.

### Maintenabilité

- contrats de données stricts,
- séparation claire génération / annotation / évaluation.

### Auditabilité

- chaque document doit pouvoir être retracé jusqu’au world, personnage, scénario et prompt.

---

## 39. Plan de développement de l’agent de dev IA

L’agent de dev IA devra agir comme un assistant d’ingénierie qui aide à construire le projet par étapes.

### Capacités attendues

- générer les schémas JSON,
- proposer des prompts,
- coder les scripts de génération,
- coder les scripts d’évaluation,
- écrire des tests,
- proposer des analyses d’échec,
- générer des rapports de benchmark.

### Boucle de travail recommandée

1. lire la spec,
2. générer ou modifier le module ciblé,
3. écrire les tests,
4. exécuter validation,
5. produire un résumé des écarts.

### Règle clé

L’agent ne doit pas mélanger vérité terrain et texte généré. Il doit préserver la séparation des artefacts.

---

## 40. Plan d’implémentation prioritaire

### Sprint 1

- modèles OpenRouter branchés,
- adaptateur LLM,
- schémas Pydantic,
- world builder minimal,
- character builder minimal.

### Sprint 2

- scenario generator,
- text generator,
- validation de cohérence,
- premières sorties JSONL.

### Sprint 3

- pré-annotation,
- baseline masking,
- eval spans.

### Sprint 4

- structured attacker,
- eval re-id,
- eval privacy.

### Sprint 5

- LLM attacker,
- eval linkability,
- rapport HTML.

### Sprint 6

- rewrite baseline,
- privacy / utility frontier,
- packaging docs.

---

## 41. Livrables finaux attendus

1. Spec produit complète
2. Threat model
3. Ontologie d’annotation
4. Dataset schema
5. Prompt pack
6. Générateurs de données
7. Pré-annotateur
8. Baselines d’anonymisation
9. Attaquants
10. Scripts d’évaluation
11. Rapport consolidé
12. Dataset card
13. Guide de reproduction

---

## 42. Recommandation finale

Commencer simple, mais pas naïf.

Le MVP ne doit pas chercher la taille maximale. Il doit chercher :

- cohérence des mondes,
- richesse des quasi-identifiants,
- existence de cas de réidentification indirecte,
- bon protocole d’attaque,
- bonne séparation privacy / utilité.

La bonne stratégie est :

1. construire un petit benchmark synthétique solide,
2. mesurer quelles anonymisations échouent vraiment,
3. enrichir progressivement la complexité.

---

## 43. Annexe — Paramètres de sortie attendus par domaine

### Support client

- plus direct,
- plus orienté problème,
- souvent plus dense en détails opérationnels,
- fort potentiel de quasi-identifiants métier.

### Emails

- plus de signatures,
- plus de structure sociale,
- plus de style personnel,
- fort potentiel de linkability.

### Texte libre

- plus de diversité,
- plus d’indices implicites,
- utile pour tester la robustesse générale.

---

## 44. Annexe — Politique de décision d’anonymisation

### Masquer

Quand l’élément est un identifiant direct clair.

### Généraliser

Quand l’élément apporte de la valeur métier mais augmente la rareté.

### Supprimer

Quand l’élément n’apporte pas de valeur et augmente le risque.

### Réécrire

Quand le sens métier doit être conservé mais les détails causent la réidentification.

---

## 45. Annexe — Formats de sortie recommandés

- JSONL pour documents et annotations
- Parquet pour analyses tabulaires
- YAML pour config et ontologie
- Markdown pour documentation
- HTML pour reporting benchmark

---

## 46. Décision recommandée pour démarrer

Démarrer avec :

- langue française,
- support client + emails,
- closed-world + insider world,
- 100 personnages,
- 3 000 documents,
- 1 structured attacker,
- 1 LLM attacker,
- 3 baselines anonymisation,
- 1 tâche aval de classification.

C’est le meilleur compromis entre simplicité, pertinence et vitesse d’exécution.

