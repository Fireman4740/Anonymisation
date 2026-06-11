# Datasets et sources

Cette page documente les datasets utilisés par le runner officiel `python -m eval.run_pipeline_evaluation`, leur chemin local, leur loader et leur source connue. Les sources externes ne sont indiquées que lorsqu'elles sont explicitement identifiées dans le dépôt ou dans les références publiques du dataset.

Le runner officiel ajoute une metadata de protocole à chaque dataset dans `summary.json` et `datasets/<dataset>/metrics.json`. Cette metadata indique si le run correspond au protocole cible ou s'il utilise un fallback local, par exemple `converted_no_offsets`, `value_search_no_offsets`, `proxy` ou `ner_sanity`.

## Vue synthétique

| Dataset | Chemin local | Loader | Format local | Protocole runner | Source |
| --- | --- | --- | --- | --- | --- |
| TAB | `eval/datasets/TAB/*.jsonl` | `build_docs_from_tab` | JSONL | `legal_text_anonymization`, souvent `converted_no_offsets` localement | Hugging Face `ildpil/text-anonymization-benchmark`, papier TAB, corpus ECHR. |
| RAT-Bench | cache `eval/datasets/RAT-Bench/cache/` | `build_docs_from_ratbench` | JSON téléchargé/cache | `profile_value_search_plus_reidentification_risk` | Hugging Face `imperial-cpg/rat-bench`, GitHub `imperial-aisp/rat-bench`, arXiv `2602.12806`. |
| DB-bio | `eval/datasets/DB-bio/*.jsonl` | `build_docs_from_db_bio` | JSONL | `identity_leakage_plus_utility_proxy` | RUPTA / `UKPLab/acl2025-rupta`, lien Google Drive dans le README local. |
| PersonalReddit | `eval/datasets/PersonalReddit/` | Non intégré au runner officiel actuel | JSONL / archive | Hors runner officiel | RUPTA / `UKPLab/acl2025-rupta`, source originale `eth-sri/llmprivacy`. |
| CoNLL2003 / CleanCoNLL | caches sous `eval/datasets/` | `build_docs_from_conll2003` | Fichiers CoNLL reconstruits | `ner_sanity` | CoNLL-2003 Shared Task, Reuters/annotations CoNLL, CleanCoNLL `flairNLP/CleanCoNLL`. |
| JSON locaux | `eval/datasets/data/*.json` | `build_docs_from_anonymization_dataset` | JSON `examples` | `internal_regression` | Datasets synthétiques/locaux du dépôt. |

## TAB

Chemins :

- `eval/datasets/TAB/train.jsonl`
- `eval/datasets/TAB/dev.jsonl`
- `eval/datasets/TAB/test.jsonl`

Loader :

- [`build_docs_from_tab`](../../eval/core/pipeline.py)

Le loader lit chaque ligne JSONL, récupère `text`, puis construit la vérité terrain en cherchant chaque valeur de `meta.masked_entities` dans le texte. Les spans sont évalués avec le label générique `SENSITIVE`.

Statut dans le runner officiel :

- `official_offsets` si un fichier TAB avec annotations `start/end` et types est disponible ;
- `converted_no_offsets` pour les fichiers locaux actuels basés sur `meta.masked_entities` ;
- `unknown_schema` si le format local ne peut pas être vérifié.

Source à documenter :

- Hugging Face : `ildpil/text-anonymization-benchmark`
- Papier : The Text Anonymization Benchmark (TAB), Computational Linguistics
- Domaine : décisions de la Cour européenne des droits de l'homme (ECHR)

Limites :

- Les annotations locales sont converties en `masked_entities`.
- L'évaluation locale ne reprend pas toute la granularité d'origine de TAB.
- Les offsets sont reconstruits par recherche de sous-chaînes.
- Le runner expose donc strict exact, relaxed overlap et warning de protocole, mais ne doit pas être présenté comme protocole TAB complet si `annotation_status=converted_no_offsets`.

## RAT-Bench

Chemins :

- README local : [`eval/datasets/RAT-Bench/README.md`](../../eval/datasets/RAT-Bench/README.md)
- Cache runtime : `eval/datasets/RAT-Bench/cache/ratbench_<language>.json`

Loader :

- [`eval/core/loaders/ratbench.py`](../../eval/core/loaders/ratbench.py)

Le loader télécharge `imperial-cpg/rat-bench` via `datasets.load_dataset`, puis bascule sur l'API Hugging Face rows si nécessaire. Les profils sont filtrables par :

- `language` : `english`, `mandarin`, `spanish` ;
- `level` : `1`, `2`, `3` ;
- `limit`.

Champs principaux :

- `id`
- `profile`
- `direct_identifiers`
- `indirect_identifiers`
- `features`
- `difficulty`
- `prompt`
- `scenario`
- `text`

Sources :

- Hugging Face : `imperial-cpg/rat-bench`
- GitHub : `imperial-aisp/rat-bench`
- Papier : RAT-Bench, arXiv `2602.12806`

Métriques runner :

- `span_detection` sur les valeurs directes et indirectes retrouvées dans le texte ;
- `anonymization_leakage` par valeurs gold ;
- `ratbench_profile_leakage` par attributs de profil ;
- `ratbench_reid_risk` si OpenRouter est disponible.

Limites :

- RAT-Bench fournit des valeurs d'attributs, pas toujours des offsets caractères.
- Le loader reconstruit les spans par recherche exacte, insensible à la casse ou fuzzy selon le type.
- Les métriques de leak analysis et de risque sont plus alignées avec la philosophie RAT-Bench que la seule métrique span-level.
- Sans `OPENROUTER_API_KEY`, l'axe risque passe en `risk_degraded`, sauf si `--require-risk` est utilisé.

## DB-bio

Chemins :

- `eval/datasets/DB-bio/train.jsonl`
- `eval/datasets/DB-bio/test.jsonl`
- `eval/datasets/DB-bio/val.jsonl`
- variantes `*_sft.jsonl` et `*_dpo.jsonl`
- README local : [`eval/datasets/DB-bio/README.md`](../../eval/datasets/DB-bio/README.md)

Loader :

- [`build_docs_from_db_bio`](../../eval/core/pipeline.py)

Le loader utilise le champ `people` ou, en fallback, `wiki_name` pour construire les spans `PERSON` dans `text`.

Statut dans le runner officiel :

- `span_detection` mesure la couverture des noms de personnes ;
- `anonymization_leakage` vérifie si ces valeurs restent visibles dans le texte anonymisé ;
- `utility_preservation` est actuellement `proxy` si des labels `label`, `l1`, `l2`, `l3` sont présents.

Source :

- RUPTA / `UKPLab/acl2025-rupta`
- Lien Google Drive documenté dans le README local DB-bio

Limites :

- L'évaluation locale cible essentiellement la détection du nom de la personne.
- Les catégories d'occupation sont utiles pour l'utilité downstream dans RUPTA, mais ne sont pas toutes exploitées par le benchmark span-level local.
- Aucun classifieur d'utilité downstream n'est configuré dans le runner officiel; le score d'utilité n'est donc pas inclus comme métrique complète.

## PersonalReddit

Chemins :

- `eval/datasets/PersonalReddit/`
- `eval/datasets/PersonalReddit/Reddit_synthetic/train.jsonl`
- `eval/datasets/PersonalReddit/Reddit_synthetic/test.jsonl`
- README local : [`eval/datasets/PersonalReddit/README.md`](../../eval/datasets/PersonalReddit/README.md)

Loader :

- [`build_docs_from_personalreddit`](../../eval/core/pipeline.py)

Format JSONL :

```json
{
  "response": "Texte Reddit...",
  "feature": "age",
  "hardness": 3,
  "personality": {
    "age": 31, "sex": "male", "occupation": "software engineer", ...
  }
}
```

Source :

- RUPTA / `UKPLab/acl2025-rupta`
- Dataset original : `eth-sri/llmprivacy/tree/main/data/synthetic`

Statut dans ce dépôt :

- Intégré comme dataset standard depuis le fix 2026-05.
- Disponible via `python -m eval run --dataset personalreddit`.
- Profil par défaut : `personalreddit_pii`.

Limites :

- Les spans GT sont construits par recherche case-insensitive des valeurs de `personality` dans `response`.
- De nombreuses valeurs d'attributs (ex. `age=31`) n'apparaissent pas verbatim dans le texte — le recall span-level sera naturellement bas.
- Ce dataset est principalement pertinent pour l'axe **attribute inference risk** (peut l'attaquant deviner les attributs ?), moins pour le span-level classique.

## CoNLL2003 et CleanCoNLL

Chemins et caches :

- loader actif : [`eval/core/loaders/conll2003.py`](../../eval/core/loaders/conll2003.py)
- caches générés sous `eval/datasets/conll2003_cache/` ou `eval/datasets/cleanconll_cache/`

Loader :

- `build_docs_from_conll2003(limit, split, variant)`

Splits :

- `train`
- `dev`
- `test`

Labels :

- `PER`
- `ORG`
- `LOC`
- `MISC`

Sources :

- CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition
- Données anglaises issues du Reuters Corpus, avec annotations CoNLL
- CleanCoNLL depuis `flairNLP/CleanCoNLL`

Statut dans le runner officiel :

- CoNLL2003 est marqué `ner_sanity`.
- Il contribue à la qualité de détection NER, mais pas à l'axe anonymisation ou anti-ré-identification.

Limites :

- Ce dataset est un benchmark NER strict, pas un benchmark d'anonymisation.
- Les prédictions hors scope sont filtrées pour la précision afin de ne pas pénaliser les PII valides non annotées par CoNLL.
- La variante par défaut est `clean`, sauf override `CONLL2003_VARIANT=original`.

## JSON locaux

Chemins :

- `eval/datasets/data/anonymization_dataset.json`
- `eval/datasets/data/hard_quasi_id_dataset.json`
- `eval/datasets/data/max_anonymization_dataset.json`

Loader :

- [`build_docs_from_anonymization_dataset`](../../eval/core/pipeline.py)

Format :

```json
{
  "examples": [
    {
      "id": "ticket_001",
      "langue": "FR",
      "original_text": "...",
      "annotations": []
    }
  ]
}
```

Statut :

- Datasets synthétiques/locaux du dépôt.
- Peuvent être liés au travail Atlas selon les fichiers, mais aucune source externe officielle ne doit être inventée.
- Dans le runner officiel, `anonymization_dataset.json` est traité comme `internal_regression` avec offsets exacts.

## Scope de labels

Le scope est défini dans [`eval/core/datasets.py`](../../eval/core/datasets.py).

| Dataset | Scope |
| --- | --- |
| `conll2003` | `PER`, `ORG`, `LOC`, `MISC` |
| `dbbio` | `PER`, `PERSON` |
| `tab` | Aucun filtrage |
| `anonymization` | Aucun filtrage |
| `ratbench` | Aucun filtrage |
| `personalreddit` | Aucun filtrage |

Quand un scope existe, il s'applique aux prédictions pour la précision. La vérité terrain n'est pas filtrée.
