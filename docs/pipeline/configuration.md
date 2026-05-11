# Configuration PipeGraph

PipeGraph combine une configuration statique dans les fichiers du dossier `pipegraph/` et des overrides runtime passés par les scripts d'évaluation.

## `pipegraph/config.json`

[`pipegraph/config.json`](../../pipegraph/config.json) est la source des paramètres PipeGraph non secrets: LLM, features optionnelles, RUPTA, sécurité de développement, GPU/NER, détection et runtime.

### Bloc `llm`

Le bloc `llm` décrit le provider actif par défaut. Le défaut du projet est `ollama`, attendu en local sur `http://localhost:11434/v1`. OpenRouter reste disponible via `llm_provider=openrouter` dans les overrides runtime ou en modifiant `llm.provider`.

| Clé | Rôle |
| --- | --- |
| `provider` | Provider logique, par exemple `ollama` ou `openrouter`. |
| `model` | Modèle par défaut. |
| `base_url` | Endpoint API compatible OpenAI. |
| `api_key` | Clé locale ou vide selon le provider. |
| `retry_count` | Nombre de tentatives. |
| `fallback_model` | Modèle de secours, si configuré. |
| `supports_response_format` | Indique si le provider supporte le JSON schema/response format. |
| `timeout_seconds` | Timeout des appels LLM. |

### Bloc `openrouter`

Le bloc `openrouter` sert aux runs distants et aux benchmarks avec appels LLM. Quand `llm.provider` vaut `openrouter`, son champ `model` est la source de vérité du modèle par défaut.

- `base_url` ;
- `model` ;
- `fallback_model` ;
- `timeout_seconds` ;
- `retry_count` ;
- `supports_response_format`.

Les appels OpenRouter nécessitent `OPENROUTER_API_KEY`.

### Bloc `features`

Ce bloc active ou désactive les noeuds LLM au niveau global :

| Clé | Effet |
| --- | --- |
| `llm_detection` | Active la revue/détection LLM. |
| `llm_verification` | Active la vérification LLM. |
| `llm_paraphrase` | Active la paraphrase LLM. |
| `llm_audit` | Active l'audit de risque. |

### Bloc `rupta`

RUPTA pilote la boucle adversariale audit/paraphrase :

| Clé | Effet |
| --- | --- |
| `enabled` | Active la boucle RUPTA. |
| `p_threshold` | Seuil de risque au-dessus duquel paraphraser. |
| `max_iterations` | Nombre maximum d'itérations. |
| `privacy_threshold` | Seuil additionnel éventuel de confidentialité. |
| `utility_threshold` | Seuil de préservation d'utilité. |
| `temperature` | Température LLM pour les opérations associées. |

### Bloc `ner_gpu`

Ce bloc décrit les préférences d'exécution GPU pour le NER GLiNER :

- VRAM disponible ;
- batch size ;
- nombre maximum de modèles parallèles ;
- FP16 ;
- `torch.compile` ;
- preset GLiNER ;
- préchargement des modèles.

### Blocs `security`, `detection` et `runtime`

Ces blocs remplacent les anciennes variables non secrètes du `.env` racine :

- `security` contient les valeurs de développement pour la pseudonymisation ;
- `detection` contient le seuil par défaut et le chemin des patterns ;
- `runtime` contient les paramètres globaux comme `debug`, `log_level` et les timeouts.

## Section `pipeline`

La section `pipeline` de [`pipegraph/config.json`](../../pipegraph/config.json) configure les noeuds du pipeline.

### Détection

Le noeud `detection` supporte :

- `enabled` : active le noeud complet ;
- `execution_mode` : `serial` ou `parallel` ;
- `deterministic.enabled` : active regex/algo ;
- `deterministic.patterns_config_path` : chemin des patterns ;
- `ai_ner.enabled` : active le NER IA ;
- `ai_ner.provider` : `gliner`, `flair` ou `spacy` ;
- `ai_ner.threshold` : seuil de confiance ;
- `ai_ner.gliner.preset` : preset GLiNER ;
- `ai_ner.gliner.label_profile` : profil de labels, par exemple `pii`.

Les presets GLiNER documentés dans le YAML incluent `fast`, `balanced`, `accuracy`, `pii`, `multitask`, `best` et `full`.

### Anonymisation

Le noeud `anonymization` définit une stratégie globale et une politique par type :

| Stratégie | Description |
| --- | --- |
| `pseudo` | Remplacement cohérent via pseudonymes. |
| `mask` | Masquage partiel adapté au type. |
| `generalize` | Remplacement par `[TYPE]`. |
| `redact` | Remplacement par `[TYPE_REDACTED]`. |

La politique peut spécialiser les actions par type, par exemple `PERSON -> pseudo`, `LOC -> generalize`, `EMAIL -> mask`, `IBAN -> redact`.

### Noeuds LLM

La section `pipeline.nodes` contient des blocs pour :

- `llm_detection` ;
- `llm_audit` ;
- `llm_paraphrase`.

Leur activation effective dépend à la fois de `config.json` et des overrides runtime.

## Overrides runtime

Les scripts d'évaluation injectent des clés dans `state.config`. Ces valeurs servent aux benchmarks et ablations sans modifier les fichiers statiques.

### Détection

| Clé | Description |
| --- | --- |
| `enable_detection` | Active/désactive le noeud de détection. |
| `enable_deterministic` | Active/désactive le détecteur déterministe. |
| `enable_ai` | Active/désactive le NER IA. |
| `detection_mode` | Force `serial` ou `parallel`. |
| `ner_provider` | Force `gliner`, `flair` ou `spacy`. |
| `gliner_preset` | Force un preset GLiNER. |
| `gliner_models` | Force une liste explicite de modèles GLiNER. |
| `gliner_threshold` | Force le seuil GLiNER. |
| `entity_profile` | Profil de normalisation des entités. |
| `gliner_label_profile` | Alias de profil pour GLiNER. |
| `gliner_labels` | Liste explicite de labels GLiNER. |
| `ner_min_vote` | Vote minimum pour conserver une entité NER. |
| `ner_min_len` | Longueur minimale des entités IA. |

### Anonymisation

| Clé | Description |
| --- | --- |
| `enable_anonymization` | Active/désactive l'anonymisation. |
| `anon_strategy` | Stratégie globale runtime. |
| `anon_policy` | Politique par type, en override de `config.json`. |
| `anon_clear_yaml_policy` | Ignore la politique statique si `true` (nom conservé pour compatibilité runtime). |
| `scope_id` | Scope de cohérence des pseudonymes. |

### LLM et RUPTA

| Clé | Description |
| --- | --- |
| `disable_llm` | Désactive l'ensemble des noeuds LLM côté runtime. |
| `llm_detection` | Active/désactive la détection/revue LLM. |
| `llm_verification` | Active/désactive la vérification LLM. |
| `llm_audit` | Active/désactive l'audit LLM. |
| `llm_paraphrase` | Active/désactive la paraphrase LLM. |
| `rupta_enabled` | Active/désactive la boucle RUPTA. |
| `rupta_max_iterations` | Nombre maximum d'itérations RUPTA. |
| `rupta_p_threshold` | Seuil de `privacy_score`. |

## Variables d'environnement

| Variable | Utilisation |
| --- | --- |
| `OPENROUTER_API_KEY` | Requise uniquement pour les appels OpenRouter et le risque RAT-Bench. |
| `.env` | Réservé aux clés API et secrets d'accès externes. |
| `CONLL2003_VARIANT` | Choix `clean` ou `original` dans le loader CoNLL2003. |

## Recommandations pratiques

- Utiliser `disable_llm=True` ou `--no-llm` pour les benchmarks rapides de détection pure.
- Utiliser `--skip-risk` quand `OPENROUTER_API_KEY` n'est pas disponible.
- Garder les overrides runtime dans les scripts d'évaluation pour les expériences reproductibles, plutôt que modifier les fichiers statiques à chaque run.
