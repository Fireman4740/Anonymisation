# 📚 Référence API Complète

## Table des Matières
- [API Publique](#api-publique)
  - [anonymize_text()](#anonymize_text)
  - [AnonymizationPipeline](#anonymizationpipeline)
- [Policy](#policy)
  - [AnonymizationPolicy](#anonymizationpolicy)
  - [preset()](#preset)
- [Services (Avancé)](#services-avancé)
  - [DetectionService](#detectionservice)
  - [TransformationService](#transformationservice)
  - [Evaluator](#evaluator)

---

## API Publique

### anonymize_text()

Fonction pratique pour anonymiser un texte (API fonctionnelle).

#### Signature
```python
def anonymize_text(
    text: str,
    level: str = "L0",
    scope_id: str = "default",
    secret_salt: str = "default_secret",
    ner_results: Optional[List[dict]] = None,
    **overrides
) -> Dict[str, Any]
```

#### Paramètres
- **text** (str) : Texte à anonymiser
- **level** (str, optionnel) : Niveau d'anonymisation
  - `"L0"` : Basic (regex + NER)
  - `"L1"` : Advanced (+ LLM + RUPTA)
  - `"L2"` : Maximum (généralisation agressive)
  - Default: `"L0"`
- **scope_id** (str, optionnel) : ID de scope pour pseudonymisation stable
  - Default: `"default"`
- **secret_salt** (str, optionnel) : Secret HMAC pour pseudonymisation
  - Default: `"default_secret"`
  - ⚠️ **Important** : Utilisez un secret fort en production
- **ner_results** (List[dict], optionnel) : Résultats NER pré-calculés
  - Format: `[{"start": int, "end": int, "entity_group": str}, ...]`
- ****overrides** : Overrides de policy (voir [AnonymizationPolicy](#anonymizationpolicy))

#### Retour
Dictionnaire avec :
```python
{
    "anonymized_text": str,          # Texte anonymisé final
    "audit": {
        "entities": List[dict],      # Entités détectées
        "replacements": List[dict],  # Remplacements effectués
        "generalizations": List[dict], # Généralisations effectuées
        "paraphrase_applied": bool,  # Paraphrase appliquée
        "rupta_applied": bool,       # RUPTA appliqué
        ...                          # Métadonnées additionnelles
    },
    "evaluation": {
        "is_valid": bool,            # Validation réussie
        "metrics": dict,             # Métriques calculées
        "validation_errors": List[str], # Erreurs de validation
        "warnings": List[str]        # Avertissements
    },
    "policy": dict                   # Policy utilisée
}
```

#### Exemples

**Basique (L0)**:
```python
from api import anonymize_text

result = anonymize_text(
    "Jean Dupont habite à Paris, email: jean@example.com",
    level="L0",
    secret_salt="my_secret"
)
print(result["anonymized_text"])
# [PER_ABC] habite à [LOC_XYZ], email: [MAIL_DEF]
```

**Avec overrides**:
```python
result = anonymize_text(
    "Marie est née le 15 juin 1990",
    level="L1",
    secret_salt="secret",
    date_granularity="month",
    paraphrase_intensity=2
)
```

**Batch processing**:
```python
texts = ["Texte 1", "Texte 2", "Texte 3"]
results = [anonymize_text(t, secret_salt="secret") for t in texts]
```

---

### AnonymizationPipeline

Classe pour utilisation orientée objet avec configuration réutilisable.

#### Constructeur
```python
class AnonymizationPipeline:
    def __init__(
        self,
        level: str = "L0",
        secret_salt: str = "default_secret",
        policy: Optional[AnonymizationPolicy] = None,
        **overrides
    )
```

##### Paramètres
- **level** (str) : Niveau d'anonymisation ("L0", "L1", "L2")
- **secret_salt** (str) : Secret HMAC
- **policy** (AnonymizationPolicy, optionnel) : Policy personnalisée
- ****overrides** : Overrides de policy

#### Méthodes

##### anonymize()
```python
def anonymize(
    self,
    text: str,
    scope_id: str = "default",
    ner_results: Optional[List[dict]] = None,
    **additional_overrides
) -> Dict[str, Any]
```

Anonymise un texte avec la configuration du pipeline.

**Paramètres**:
- **text** (str) : Texte à anonymiser
- **scope_id** (str) : ID de scope
- **ner_results** (List[dict], optionnel) : NER pré-calculés
- ****additional_overrides** : Overrides additionnels pour cet appel

**Retour** : Même format que `anonymize_text()`

##### anonymize_batch()
```python
def anonymize_batch(
    self,
    texts: List[str],
    scope_id: str = "default",
    **overrides
) -> List[Dict[str, Any]]
```

Anonymise un lot de textes.

**Paramètres**:
- **texts** (List[str]) : Liste de textes
- **scope_id** (str) : ID de scope commun
- ****overrides** : Overrides additionnels

**Retour** : Liste de résultats

#### Exemples

**Configuration réutilisable**:
```python
from api import AnonymizationPipeline

# Créer le pipeline une fois
pipeline = AnonymizationPipeline(
    level="L1",
    secret_salt="production_secret",
    date_granularity="quarter",
    org_policy="generalize"
)

# Utiliser plusieurs fois
result1 = pipeline.anonymize("Premier texte", scope_id="doc1")
result2 = pipeline.anonymize("Deuxième texte", scope_id="doc2")
```

**Batch processing**:
```python
pipeline = AnonymizationPipeline(level="L0", secret_salt="secret")

documents = [
    "Alice travaille chez Google",
    "Bob habite à Lyon",
    "Charlie: charlie@gmail.com"
]

results = pipeline.anonymize_batch(documents, scope_id="batch_001")

for i, result in enumerate(results):
    print(f"Doc {i+1}: {result['anonymized_text']}")
```

**Overrides par appel**:
```python
pipeline = AnonymizationPipeline(level="L1")

# Override ponctuel
result = pipeline.anonymize(
    "Texte sensible",
    scope_id="special",
    paraphrase_intensity=3,  # Override pour cet appel
    risk_threshold=30
)
```

---

## Policy

### AnonymizationPolicy

Classe de configuration définissant les stratégies d'anonymisation.

#### Attributs

##### Niveau et Style
- **level** (str) : `"L0"`, `"L1"`, ou `"L2"`
- **placeholder_style** (str) : `"typed"` ou `"generic"`
  - `"typed"` : `[PER_ABC]`, `[ORG_XYZ]`, etc.
  - `"generic"` : `[REDACTED]` pour tout

##### Policies de Généralisation
- **date_granularity** (str) : `"none"`, `"week"`, `"month"`, `"quarter"`, `"year"`
  - `"none"` : Pas de généralisation
  - `"month"` : `[DATE_2024-06]`
  - `"quarter"` : `[DATE_2024-Q2]`
  - `"year"` : `[DATE_2024]`
- **org_policy** (str) : `"keep"`, `"generalize"`, `"redact"`
  - `"keep"` : `[ORG_GOOGLE]`
  - `"generalize"` : `[ORG]`
  - `"redact"` : `[REDACTED]`
- **ip_policy** (str) : `"keep"`, `"cidr24"`, `"redact"`

##### Features LLM (L1+ uniquement)
- **llm_detection** (bool) : Activer détection LLM avancée
- **llm_paraphrase** (bool) : Activer paraphrase stylométrique
- **llm_audit** (bool) : Activer audit de risque
- **paraphrase_intensity** (int) : Intensité 0-3
- **risk_threshold** (int) : Seuil 0-100 pour hardening
- **max_hardening_rounds** (int) : Nombre max de tours de durcissement

##### Features RUPTA (L1+ uniquement)
- **rupta_enabled** (bool) : Activer optimisation RUPTA
- **rupta_max_iterations** (int) : Nombre max d'itérations
- **rupta_p_threshold** (float) : Seuil p-value pour privacy
- **rupta_privacy_threshold** (int, optionnel) : Seuil privacy rank minimum
- **rupta_utility_threshold** (float) : Seuil utility score minimum

##### Autres
- **mapping_retention** (str) : `"keep"` ou `"discard"`
  - `"keep"` : Conserver les mappings dans le résultat
  - `"discard"` : Supprimer pour sécurité

#### Méthodes

##### to_dict()
```python
def to_dict(self) -> Dict[str, Any]
```
Convertit la policy en dictionnaire.

##### from_dict()
```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> AnonymizationPolicy
```
Crée une policy depuis un dictionnaire.

#### Exemple

```python
from api import AnonymizationPolicy

# Policy personnalisée
policy = AnonymizationPolicy(
    level="L1",
    placeholder_style="typed",
    date_granularity="quarter",
    org_policy="generalize",
    ip_policy="cidr24",
    llm_detection=True,
    llm_paraphrase=True,
    paraphrase_intensity=2,
    risk_threshold=60,
    rupta_enabled=True,
    rupta_max_iterations=3,
    mapping_retention="discard"
)

# Utiliser
result = anonymize_text(
    "Mon texte",
    secret_salt="secret",
    **policy.to_dict()
)

# Ou
pipeline = AnonymizationPipeline(policy=policy, secret_salt="secret")
```

---

### preset()

Fonction utilitaire pour obtenir une policy pré-configurée.

#### Signature
```python
def preset(level: str) -> AnonymizationPolicy
```

#### Paramètres
- **level** (str) : `"L0"`, `"L1"`, ou `"L2"`

#### Retour
`AnonymizationPolicy` configurée selon le niveau.

#### Presets

**L0 - Basic**:
```python
AnonymizationPolicy(
    level="L0",
    placeholder_style="typed",
    date_granularity="none",
    org_policy="keep",
    llm_detection=False,
    llm_paraphrase=False,
    rupta_enabled=False
)
```

**L1 - Advanced**:
```python
AnonymizationPolicy(
    level="L1",
    placeholder_style="typed",
    date_granularity="month",
    org_policy="generalize",
    llm_detection=True,
    llm_paraphrase=True,
    llm_audit=True,
    paraphrase_intensity=2,
    rupta_enabled=True
)
```

**L2 - Maximum**:
```python
AnonymizationPolicy(
    level="L2",
    placeholder_style="generic",
    date_granularity="year",
    org_policy="redact",
    llm_detection=True,
    llm_paraphrase=True,
    paraphrase_intensity=3,
    risk_threshold=40,
    rupta_enabled=True
)
```

#### Exemple
```python
from api import preset

policy = preset("L1")
policy.date_granularity = "quarter"  # Personnaliser
```

---

## Services (Avancé)

Pour utilisation avancée avec injection de dépendances.

### DetectionService

Service de détection unifié (regex + NER).

#### Constructeur
```python
from layer1_detection import DetectionService

service = DetectionService(
    gpu_pipeline=None,
    use_gliner=True,
    gliner_models=None,
    gliner_labels=None,
    gliner_threshold=0.35,
    gliner_preset="balanced"
)
```

#### Méthodes

##### detect_regex()
```python
def detect_regex(
    self,
    text: str,
    skip_tags: Optional[Set[str]] = None
) -> List[DetectedEntity]
```

##### detect_ner()
```python
def detect_ner(
    self,
    text: str,
    external_ner: Optional[List[dict]] = None
) -> List[DetectedEntity]
```

##### detect_all()
```python
def detect_all(
    self,
    text: str,
    skip_regex_tags: Optional[Set[str]] = None,
    external_ner: Optional[List[dict]] = None
) -> List[DetectedEntity]
```

---

### TransformationService

Service de transformation unifié.

#### Constructeur
```python
from layer2_transformation import TransformationService

service = TransformationService(
    policy=policy,
    pseudo_mapper=mapper,
    llm_service=None
)
```

#### Méthodes

##### apply_replacements()
```python
def apply_replacements(
    self,
    text: str,
    entities: List[DetectedEntity]
) -> Tuple[str, List[Dict[str, Any]]]
```

##### apply_generalization()
```python
def apply_generalization(
    self,
    text: str
) -> Tuple[str, List[Generalization]]
```

##### transform()
```python
def transform(
    self,
    text: str,
    entities: List[DetectedEntity],
    original_text: Optional[str] = None,
    ground_truth_people: Optional[str] = None,
    ground_truth_label: Optional[str] = None
) -> TransformationResult
```

---

### Evaluator

Service d'évaluation et validation.

#### Constructeur
```python
from layer3_evaluation import Evaluator

evaluator = Evaluator()
```

#### Méthodes

##### evaluate()
```python
def evaluate(
    self,
    original_text: str,
    anonymized_text: str,
    entities_detected: int,
    entities_replaced: int
) -> EvaluationResult
```

##### validate_placeholders()
```python
def validate_placeholders(
    self,
    text: str
) -> Tuple[bool, List[str]]
```

---

**Version 2.0** - Référence API Complète
