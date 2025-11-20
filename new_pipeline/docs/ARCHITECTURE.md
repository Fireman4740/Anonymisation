# 🏗️ Architecture du Pipeline d'Anonymisation

## Vue d'Ensemble

Le pipeline est organisé en **3 couches indépendantes** avec un orchestrateur léger pour la coordination.

```
┌───────────────────────────────────────────────────────────┐
│                COUCHE 3 : ÉVALUATION                      │
│                   (Métriques & Validation)                │
├───────────────────────────────────────────────────────────┤
│  • Métriques de base (nb entités, ratios)                 │
│  • Validation placeholders                                │
│  • Warnings et erreurs                                    │
│                                                           │
│  Module: layer3_evaluation/evaluator.py                  │
└───────────────────────────────────────────────────────────┘
         ▲
         │
┌───────────────────────────────────────────────────────────┐
│     COUCHE 2 : TRANSFORMATION                             │
│     (Pseudonymisation & Anonymisation)                    │
├───────────────────────────────────────────────────────────┤
│  2a. Remplacement Direct                                  │
│      • Appliquer détections → placeholders                │
│      • PseudoMapper (HMAC stable)                         │
│                                                           │
│  2b. Généralisation Policy-Driven                         │
│      • Dates → granularité (month/quarter/year)           │
│      • Orgs → généralisation/redact                       │
│                                                           │
│  2c. Transformations LLM (L1+ uniquement)                 │
│      • Paraphrase stylométrique                           │
│      • Audit + Hardening loop                             │
│      • RUPTA optimization (privacy-utility)               │
│                                                           │
│  Module: layer2_transformation/transformation_service.py  │
└───────────────────────────────────────────────────────────┘
         ▲
         │
┌───────────────────────────────────────────────────────────┐
│          COUCHE 1 : DÉTECTION                             │
│          (Identification des Entités)                     │
├───────────────────────────────────────────────────────────┤
│  1a. Détection Regex                                      │
│      • Patterns PII (email, phone, NIR, IBAN)             │
│      • Patterns techniques (IP, URL, UUID)                │
│      • Patterns financiers (montants)                     │
│      • Module: layer1_detection/regex/text_sanitizer.py   │
│                                                           │
│  1b. Détection NER                                        │
│      • GLiNER (multi-modèles, voting)                     │
│      • GPU Pipeline (optionnel, auto-fallback)            │
│      • Modules: layer1_detection/ner/                     │
│                                                           │
│  1c. Détection LLM Avancée (L1+ uniquement)               │
│      • Clustering d'entités                               │
│      • Co-référence resolution                            │
│      • Détection contextuelle                             │
│                                                           │
│  Service Unifié: layer1_detection/detection_service.py    │
│  • Combine regex + NER + LLM                              │
│  • Déduplication intelligente                             │
│  • Priority: regex > ner-gpu > ner                        │
└───────────────────────────────────────────────────────────┘
         ▲
         │
┌───────────────────────────────────────────────────────────┐
│           COUCHE 0 : ORCHESTRATION                        │
│           (Coordination Simple)                           │
├───────────────────────────────────────────────────────────┤
│  • orchestrator/orchestrator.py (~300 lignes)             │
│    - Injection de dépendances (services)                  │
│    - Coordination du flux L0/L1/L2                        │
│    - Assemblage du résultat final                         │
│    - Gestion d'erreur globale                             │
│                                                           │
│  • api/policy.py (configuration)                          │
│    - Presets L0/L1/L2                                     │
│    - Paramètres configurables                             │
│    - Sérialisation/désérialisation                        │
└───────────────────────────────────────────────────────────┘
```

## Flux de Traitement

### Mode L0 (Sans LLM)
```
Input Text
    │
    ▼
┌─────────────────┐
│  COUCHE 1       │
│  Détection      │
│  • Regex        │
│  • NER GLiNER   │
│  • GPU (opt)    │
└─────────────────┘
    │
    ▼ List[DetectedEntity]
┌─────────────────┐
│  COUCHE 2       │
│  Transform      │
│  • Remplacer    │
│  • Généraliser  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  COUCHE 3       │
│  Évaluation     │
│  • Métriques    │
│  • Validation   │
└─────────────────┘
    │
    ▼
Anonymized Text
```

### Mode L1 (Avec LLM + RUPTA)
```
Input Text
    │
    ▼
┌─────────────────┐
│  Regex + NER    │
└─────────────────┘
    │
    ▼ Seeds
┌─────────────────┐
│  LLM Detection  │
│  • Clustering   │
│  • Co-référence │
└─────────────────┘
    │
    ▼ Enhanced Entities
┌─────────────────┐
│  Remplacements  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Généralisation │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Paraphrase LLM │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Audit+Hardening│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  RUPTA Optim    │
│  • Privacy eval │
│  • Utility eval │
│  • Itérative    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Évaluation     │
└─────────────────┘
    │
    ▼
Final Text
```

## Modules Détaillés

### Couche 1 - Détection

#### DetectionService
**Fichier**: `layer1_detection/detection_service.py`

**Responsabilités**:
- Coordonner détection regex et NER
- Gérer le GPU pipeline (auto-fallback)
- Dédupliquer les entités (priorité regex > ner-gpu > ner)
- Retourner des `DetectedEntity` structurées

**API Principale**:
```python
class DetectionService:
    def detect_regex(text, skip_tags) -> List[DetectedEntity]
    def detect_ner(text, external_ner) -> List[DetectedEntity]
    def detect_all(text, skip_regex_tags, external_ner) -> List[DetectedEntity]
```

#### Text Sanitizer (Regex)
**Fichier**: `layer1_detection/regex/text_sanitizer.py`

**Patterns Supportés**:
- **PII**: Email, Phone FR, NIR (SSN), IBAN, BIC
- **Dates**: FR/EN (multiples formats)
- **Technical**: IPv4/6, URL, UUID, Path, MAC
- **Financial**: Montants avec devises
- **Legal**: Articles, lois, codes

**Validation Strategy Pattern**:
- `IbanValidator`: Validation IBAN complète
- `BicValidator`: Validation BIC
- `FrenchSSNValidator`: NIR avec clé Luhn
- `LuhnValidator`: Cartes bancaires

#### NER GLiNER
**Fichier**: `layer1_detection/ner/gliner_ensemble.py`

**Presets**:
- `fast`: gliner_small-v2.1
- `balanced`: gliner_medium-v2.1
- `accuracy`: gliner_large + gliner_multi
- `pii`: gliner_multi_pii-v1
- `best`: 4 modèles avec voting

**Features**:
- Auto-detect device (CUDA/MPS/CPU)
- FP16 sur CUDA (optionnel)
- Sentence splitting intelligent
- Model caching global
- Voting avec pondération

### Couche 2 - Transformation

#### TransformationService
**Fichier**: `layer2_transformation/transformation_service.py`

**Responsabilités**:
- Appliquer les remplacements (entities → placeholders)
- Généralisation policy-driven
- Paraphrase LLM (si L1+)
- Audit + Hardening loop (si L1+)
- RUPTA optimization (si L1+ et activé)

**API Principale**:
```python
class TransformationService:
    def apply_replacements(text, entities) -> (text, replacements)
    def apply_generalization(text) -> (text, generalizations)
    def apply_paraphrase(text, temperature) -> (text, error)
    def apply_audit_and_hardening(text, original) -> (text, audit, rounds)
    def apply_rupta(...) -> (text, metrics)
    def transform(text, entities, ...) -> TransformationResult
```

#### GeneralizationService
**Fichier**: `layer2_transformation/generalization/generalizer.py`

**Généralisations**:
- **Dates**: none/week/month/quarter/year
- **Orgs**: keep/generalize/redact
- **IPs**: keep/cidr24/redact (futur)

**Policy Escalation**:
```python
escalate_policy(policy) -> AnonymizationPolicy  # Immutable
```

#### PseudoMapper
**Fichier**: `utils/pseudo_mapper.py`

**Fonction**: Générer des placeholders stables et déterministes

```python
mapper = PseudoMapper(secret="salt", scope_id="doc123")
placeholder = mapper.placeholder("PER", "Jean Dupont")
# → [PER_ABC]  (toujours le même pour cette personne dans ce scope)
```

### Couche 3 - Évaluation

#### Evaluator
**Fichier**: `layer3_evaluation/evaluator.py`

**Métriques**:
- `entities_detected`: Nombre d'entités détectées
- `entities_replaced`: Nombre d'entités remplacées
- `replacement_rate`: Taux de remplacement
- `placeholder_count`: Nombre de placeholders
- `length_ratio`: Ratio longueur original/anonymisé

**Validation**:
- Placeholders bien formés
- Texte non vide
- Warnings si taux de remplacement > 50%

### Orchestrateur

#### anonymize_text_refactored
**Fichier**: `orchestrator/orchestrator.py`

**Responsabilités (UNIQUEMENT coordination)**:
1. Charger policy et appliquer overrides
2. Créer services (detection, transformation, evaluation)
3. Appeler Couche 1 (détection)
4. Appeler Couche 2 (transformation)
5. Appeler Couche 3 (évaluation)
6. Assembler le résultat
7. Gestion d'erreur globale

**Pas de logique métier** : tout est délégué aux services !

## Injection de Dépendances

L'orchestrateur accepte des services pré-configurés :

```python
# Créer services custom
detection_service = DetectionService(use_gliner=False)  # Regex only
transformation_service = TransformationService(policy, mapper)
evaluator = Evaluator()

# Injecter dans orchestrateur
result = anonymize_text_refactored(
    "Mon texte",
    scope_id="test",
    secret_salt="secret",
    level="L0",
    detection_service=detection_service,
    transformation_service=transformation_service,
    evaluator=evaluator,
)
```

**Avantages**:
- Tests unitaires faciles (mock services)
- Composition flexible
- Pas de couplage fort

## Gestion d'Erreur

### Par Couche
Chaque service retourne des tuples `(result, error)` :

```python
text, error = llm_service.paraphrase(text)
if error:
    # Gérer l'erreur
    pass
```

### Globale
L'orchestrateur catch toutes les exceptions et retourne un résultat valide avec l'erreur :

```python
{
    "anonymized_text": original_text,  # Fallback sur original
    "audit": {
        "error": "Message d'erreur",
        "traceback": "...",
        "entities": []
    },
    "evaluation": {
        "is_valid": False,
        "validation_errors": [...]
    }
}
```

## Performance

### Cache et Lazy Loading
- **Modèles GLiNER** : Chargés une seule fois, cachés globalement
- **GPU Pipeline** : Créé une seule fois, réutilisé

### GPU Optimization
- Auto-detect CUDA/MPS/CPU
- Fallback automatique si OOM
- FP16 sur CUDA (optionnel)

### Parallélisation
- GPU Pipeline utilise ThreadPoolExecutor pour inférence batch
- Sentence splitting pour optimiser la mémoire

## Extensibilité

### Ajouter un Nouveau Détecteur
1. Créer dans `layer1_detection/`
2. Retourner des `List[DetectedEntity]`
3. Intégrer dans `DetectionService.detect_all()`

### Ajouter une Nouvelle Généralisation
1. Créer dans `layer2_transformation/generalization/`
2. Ajouter méthode dans `GeneralizationService`
3. Appeler depuis `apply_all()`

### Ajouter une Nouvelle Métrique
1. Ajouter dans `Evaluator.evaluate()`
2. Retourner dans `metrics` dict

## Principes de Design

### Separation of Concerns
Chaque couche a une responsabilité unique :
- Couche 1 : **Identifier**
- Couche 2 : **Transformer**
- Couche 3 : **Évaluer**

### Single Responsibility Principle
Chaque module/service a un rôle clair et unique.

### Dependency Injection
Les services sont injectables pour faciliter les tests et la composition.

### Immutability (Policy)
Les policies ne mutent pas, `escalate_policy()` retourne une nouvelle instance.

### Fail-Safe
En cas d'erreur, retourner un résultat valide avec l'erreur plutôt que de crasher.

---

**Version 2.0** - Architecture en couches claire et maintenable
