# Architecture PipeGraph

PipeGraph est implémenté comme un `StateGraph` LangGraph. Chaque noeud reçoit un `PipelineState`, produit une mise à jour partielle, puis LangGraph fusionne cette mise à jour dans l'état global.

## État du graphe

L'état est défini dans [`pipegraph/src/state.py`](../../pipegraph/src/state.py).

| Champ | Description |
| --- | --- |
| `text` | Texte courant, modifié par l'anonymisation et la paraphrase. |
| `original_text` | Copie immuable du texte source, utilisée pour les offsets. |
| `entities` | Entités détectées au format canonique. |
| `config` | Flags et paramètres runtime. |
| `metadata` | Informations complémentaires d'exécution. |
| `errors` | Erreurs collectées sans interrompre nécessairement le pipeline. |
| `privacy_score` | Score de risque LLM, de 0 à 100. |
| `llm_feedback` | Retour structuré de l'audit LLM. |
| `iteration` | Nombre d'itérations de boucle RUPTA. |

## Topologie du graphe

Le graphe compilé est construit par `GraphResources._create_graph()` :

```text
detection
  -> anonymization_pass_1
  -> llm_review
  -> llm_verification
  -> anonymization_pass_2
  -> llm_audit
       |-- si RUPTA continue --> llm_paraphrase -> llm_audit
       |-- sinon -------------> END
```

### Étapes

| Noeud | Rôle |
| --- | --- |
| `detection` | Agrège la détection déterministe et le NER IA. |
| `anonymization_pass_1` | Applique une première anonymisation sur les entités détectées. |
| `llm_review` | Ajoute éventuellement des entités ou signaux détectés par LLM. |
| `llm_verification` | Vérifie ou filtre les entités LLM selon la configuration. |
| `anonymization_pass_2` | Réapplique l'anonymisation après enrichissement des entités. |
| `llm_audit` | Évalue le risque de ré-identification et produit `privacy_score`. |
| `llm_paraphrase` | Réécrit le texte anonymisé si RUPTA estime le risque trop élevé. |

## Routage RUPTA

Le routeur `_rupta_router()` lit :

- `rupta.enabled` dans [`pipegraph/config.json`](../../pipegraph/config.json) ;
- `features.llm_audit` et `features.llm_paraphrase` ;
- les overrides runtime `rupta_enabled`, `llm_audit`, `llm_paraphrase`, `rupta_max_iterations` et `rupta_p_threshold`.

La boucle continue si :

- RUPTA est activé ;
- audit et paraphrase LLM sont activés ;
- `privacy_score > p_threshold` ;
- `iteration < max_iterations`.

Sinon, le graphe termine.

## Gestion des ressources

`GraphResources` centralise l'instanciation des noeuds et la libération des ressources :

- création lazy du graphe compilé via la propriété `graph` ;
- singleton accessible par `GraphResources.get_instance()` ;
- nettoyage via `shutdown()`, notamment des caches de modèles ;
- support du context manager pour les usages longs ou batch.

Pour un usage simple, `create_pipeline_graph()` conserve une API compatible avec les scripts existants.

## Contraintes importantes

- Les offsets des entités sont basés sur `original_text`.
- L'anonymisation repart toujours de `original_text` pour éviter les décalages d'offsets après remplacement.
- Les noeuds LLM sont feature-flagged et doivent pouvoir se dégrader proprement si le provider est indisponible.
- Les évaluations injectent souvent des overrides runtime dans `state.config`; ces valeurs ont généralement priorité sur les fichiers statiques.
