# Composants PipeGraph

Cette page décrit les principaux composants du pipeline PipeGraph et leurs responsabilités.

## Détection hybride

Le noeud [`DetectionNode`](../../pipegraph/src/nodes/detection/detection_node.py) orchestre deux familles de détecteurs :

- le détecteur déterministe ;
- le détecteur NER IA.

Il peut les exécuter en série ou en parallèle selon `detection_mode` ou `pipeline.nodes.detection.execution_mode`.

### Détection déterministe

Le détecteur déterministe est basé sur des patterns et validateurs, configurés par [`pipegraph/config/patterns_config.yaml`](../../pipegraph/config/patterns_config.yaml).

Il cible les entités structurées ou fortement régulières, par exemple :

- emails ;
- téléphones ;
- identifiants techniques ;
- IBAN, cartes ou comptes selon les patterns disponibles ;
- IP et autres formats détectables par règles.

Les entités déterministes ont une priorité élevée lors de la fusion, car leurs offsets et types sont souvent plus fiables pour les formats structurés.

### NER IA

Le détecteur IA est implémenté autour de [`AINerDetector`](../../pipegraph/src/nodes/detection/ai_ner/detector.py). Il supporte plusieurs providers :

- GLiNER ;
- Flair ;
- spaCy.

Le mode GLiNER peut fonctionner avec des presets ou une liste explicite de modèles. Les détections sont normalisées, filtrées par seuil, puis agrégées avec les résultats déterministes.

Les paramètres d'ablation les plus importants sont :

- `gliner_preset` ;
- `gliner_models` ;
- `gliner_threshold` ;
- `ner_min_vote` ;
- `ner_min_len` ;
- `entity_profile`.

## Fusion et normalisation des entités

Après détection, les entités sont converties au format canonique, puis fusionnées via les helpers de `src.utils.span_utils`.

Les objectifs sont :

- dédupliquer les entités identiques ;
- résoudre les chevauchements ;
- donner priorité aux sources les plus fiables ;
- normaliser les labels selon le profil cible (`pii`, `news_ner`, `conll2003`, etc.).

## Anonymisation

Le noeud [`AnonymizationNode`](../../pipegraph/src/nodes/anonymisation/anonymization_node.py) applique les remplacements sur le texte.

Contraintes importantes :

- les offsets sont interprétés sur `original_text` ;
- le noeud repart toujours de `original_text` ;
- les entités sont appliquées de la fin vers le début pour ne pas casser les offsets ;
- les pseudonymes peuvent être stabilisés par `scope_id`.

### Stratégies disponibles

| Stratégie | Effet |
| --- | --- |
| `pseudo` | Génère un pseudonyme cohérent via `PseudoMapper`. |
| `mask` | Masque partiellement selon le type. |
| `generalize` | Remplace par `[TYPE]`. |
| `redact` | Remplace par `[TYPE_REDACTED]`. |

La stratégie globale peut être surchargée par une politique par type dans le YAML ou dans `state.config`.

## Noeuds LLM

Les noeuds LLM vivent dans [`pipegraph/src/nodes/llm/`](../../pipegraph/src/nodes/llm/). Ils sont optionnels et pilotés par `config.json` et les flags runtime.

### `llm_review`

Le noeud de revue LLM ajoute une couche complémentaire à la détection classique. Il est utile pour les quasi-identifiants et les formulations moins régulières.

### `llm_verification`

Le noeud de vérification contrôle ou filtre les entités issues des étapes LLM. Il sert à réduire les faux positifs avant la seconde passe d'anonymisation.

### `llm_audit`

Le noeud d'audit évalue le texte anonymisé et produit un `privacy_score` ainsi que du feedback structuré dans `llm_feedback`.

### `llm_paraphrase`

Le noeud de paraphrase réécrit le texte anonymisé lorsque RUPTA considère que le risque reste trop élevé.

## RUPTA

RUPTA est la boucle adversariale :

```text
llm_audit -> llm_paraphrase -> llm_audit
```

Elle cherche à réduire le risque de ré-identification tout en conservant l'utilité du texte. Le routeur arrête la boucle quand le score passe sous le seuil ou quand `max_iterations` est atteint.

## Client LLM

Le client LLM centralise :

- le provider ;
- le modèle ;
- les timeouts ;
- les retries ;
- le modèle de fallback ;
- le support ou non de sorties structurées.

Les providers configurés doivent exposer une API compatible avec les appels attendus par les noeuds LLM.

## Gestion des ressources

`GraphResources` évite de recréer les modèles et le graphe à chaque document. C'est important pour :

- les benchmarks multi-documents ;
- les modèles NER coûteux à charger ;
- la libération des caches GPU ;
- les runs Streamlit et les ablations.
