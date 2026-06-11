# Métriques et rapports

L'évaluation locale est centrée sur des rapports document par document, puis sur des axes de métriques agrégés. Le point d'entrée officiel est :

```bash
python -m eval.run_pipeline_evaluation
```

Ce runner produit un dossier complet par run et conserve la compatibilité avec les anciens rapports document-level.

## Format d'un document évalué

Chaque document produit par `build_report()` contient notamment :

| Champ | Description |
| --- | --- |
| `doc_id` | Identifiant du document. |
| `full_text` | Texte source. |
| `anonymized_text` | Texte produit par PipeGraph. |
| `ground_truth` | Spans attendus. |
| `predictions` | Spans prédits à partir des entités PipeGraph. |
| `precision` | Precision document-level. |
| `recall` | Recall document-level. |
| `exact_label_recall` | Recall avec label exact. |
| `f2` | F-score pondéré recall (β=2). |
| `strict_precision` | Precision strict (offsets + label exacts — standard CoNLL/PII-Bench). |
| `strict_recall` | Recall strict (offsets + label exacts). |
| `strict_f1` | F1 strict (β=1). |
| `strict_f2` | F2 strict (β=2). |
| `error_classification` | Taxonomie : `missed` / `spurious` / `boundary_error` / `type_error`. |
| `bleu_score` | BLEU entre texte original et anonymisé (utilité — standard RAT-Bench). |
| `leaks_count` | Nombre de spans vérité terrain non couverts. |
| `privacy_score` | Score LLM si audit actif. |
| `rupta_iterations` | Nombre d'itérations RUPTA. |
| `llm_feedback` | Feedback d'audit LLM. |
| `leaks` | Liste détaillée des spans non détectés. |

## Évaluation span-level

La fonction centrale est `evaluate_spans()` dans [`eval/core/pipeline.py`](../../eval/core/pipeline.py).

Le runner officiel ajoute des métriques de protocole dans [`eval/core/metrics.py`](../../eval/core/metrics.py) :

| Famille | Description |
| --- | --- |
| `relaxed_overlap` | Logique historique : un span est couvert si les intervalles caractères se chevauchent. |
| `strict_exact_typed` | Match exact `start/end/label`. |
| `strict_exact_untyped` | Match exact `start/end`, sans tenir compte du label. |
| `relaxed_overlap_typed` | Overlap avec correspondance de label. |
| `canonical_strict_exact_typed` | Strict exact après projection dans le label-space canonique. |

### Deux niveaux de matching

#### Matching partiel (défaut pour l'anonymisation)

Une prédiction couvre un span vérité terrain si les intervalles caractères se chevauchent d'au moins 1 caractère. Ce mode est adapté à l'anonymisation : une détection partielle protège partiellement le contenu.

#### Strict entity-level matching (standard de recherche)

Standard CoNLL-2003, PII-Bench (Shen et al., 2025) et TAB (Pilan et al., 2022) : le triplet `(start, end, label)` doit correspondre exactement. Les métriques strictes sont exposées sous les clés `strict_*` dans chaque document et dans les agrégations (`macro_strict_*`, `micro_strict_*`).

### Overlap

### Precision

La precision mesure la proportion de prédictions qui recouvrent au moins un span vérité terrain.

Quand un scope de labels existe, les prédictions hors scope sont ignorées pour la precision. Exemple : sur CoNLL2003, une prédiction `DATE` n'est pas comptée comme faux positif car le dataset n'annote pas les dates.

### Recall

Le recall mesure la proportion de spans vérité terrain couverts par au moins une prédiction.

Le recall utilise toutes les prédictions, même quand un scope de labels existe. Cela permet de reconnaître qu'un span sensible a été couvert, même si le label exact diffère.

### Exact label recall

`exact_label_recall` mesure la couverture avec correspondance de label. C'est une métrique plus stricte que le recall par overlap.

### Asymétrie du scope de labels

> **Intentionnel** : precision utilise les preds filtrées par scope, recall utilise **toutes** les preds. Justification : ne pas pénaliser les détections PII valides que le dataset n'annote pas (ex. une prédiction `DATE` sur CoNLL2003 n'est pas comptée comme FP). L'asymétrie est documentée dans [`eval/core/datasets.py`](../../eval/core/datasets.py).

### Taxonomie d'erreurs

Basée sur Chinchor & Sundheim (1993), confirmée par CoNLL# (Rueda et al., 2024) :

| Type | Description |
| --- | --- |
| `missed` | GT non couvert par aucune prédiction (FN). |
| `spurious` | Prédiction sans overlap GT (FP). |
| `boundary_error` | Overlap mais offsets différents. |
| `type_error` | Offsets exacts, label différent. |

Exposés dans `error_classification` par document et agrégés en `total_missed`, `total_spurious`, `total_boundary_error`, `total_type_error`.

### Métrique d'utilité : BLEU

Le BLEU score mesure la similarité textuelle entre texte original et texte anonymisé (standard RAT-Bench paper §4). Proche de 1.0 = haute utilité, proche de 0.0 = forte réécriture. Calculé via `sacrebleu` ou bigram BLEU en fallback.

### F2

Le score F2 favorise le recall :

```text
F2 = (1 + 2^2) * precision * recall / ((2^2 * precision) + recall)
```

Ce choix reflète le contexte anonymisation : rater une entité sensible est souvent plus grave qu'ajouter une anonymisation superflue.

## Agrégations

Les agrégations sont calculées dans [`eval/core/reporting.py`](../../eval/core/reporting.py).

| Métrique | Description |
| --- | --- |
| `macro_precision` | Moyenne des precisions document-level. |
| `macro_recall` | Moyenne des recalls document-level. |
| `macro_exact_label_recall` | Moyenne des recalls exact-label document-level. |
| `macro_f2` | Moyenne des F2 document-level. |
| `micro_precision` | Precision globale à partir des compteurs TP/FP. |
| `micro_recall` | Recall global à partir des compteurs TP/FN. |
| `micro_exact_label_recall` | Exact label recall global. |
| `micro_f2` | F2 global calculé depuis micro precision et micro recall. |
| `total_predictions` | Nombre total de prédictions retenues. |
| `total_ground_truth` | Nombre total de spans vérité terrain. |
| `total_leaks` | Nombre total de spans non couverts. |
| `macro_bleu` | Moyenne des BLEU scores document-level. |
| `macro_strict_precision` | Moyenne des precisions strictes. |
| `macro_strict_recall` | Moyenne des recalls stricts. |
| `macro_strict_f1` | Moyenne des F1 stricts. |
| `macro_strict_f2` | Moyenne des F2 stricts. |
| `micro_strict_precision` | Precision stricte globale (TP/(TP+FP)). |
| `micro_strict_recall` | Recall strict global (TP/(TP+FN)). |
| `total_missed` | Total erreurs Missed. |
| `total_spurious` | Total erreurs Spurious. |
| `total_boundary_error` | Total erreurs Boundary. |
| `total_type_error` | Total erreurs Type. |

## Métriques par label

`label_metrics()` dans [`eval/core/metrics.py`](../../eval/core/metrics.py) agrège :

- `tp_by_label`
- `fp_by_label`
- `fn_by_label`
- `exact_tp_by_label`
- `exact_fn_by_label`

Le rapport final expose pour chaque type :

| Champ | Description |
| --- | --- |
| `precision`, `recall`, `exact_recall` | Métriques de détection. |
| `f1` | F1 (β=1, standard benchmarks NER). |
| `f2` | F2 (β=2, cohérent avec la métrique globale). |
| `tp`, `fp`, `fn` | Compteurs bruts. |
| `support` | Nombre de GT pour ce label. |

## RAT-Bench : métriques primaires

RAT-Bench évalue principalement la **réduction du risque de ré-identification** (paper arXiv 2602.12806 §4), pas uniquement la détection de spans. Les métriques span-level sont secondaires/diagnostiques.

### R_succ — métrique principale

| Métrique | Description |
| --- | --- |
| `r_succ_rate` | Fraction de documents où R > θ=0.2 (≡ k-anonymity k=5). **Plus bas = meilleur.** |
| `avg_risk` | Risque moyen R sur tous les documents. |
| `macro_bleu` | Utilité moyenne (BLEU original vs. anonymisé). |

Calculés dans [`eval/core/ratbench.py`](../../eval/core/ratbench.py) par `aggregate_ratbench_metrics()`.

Algorithme (paper §4) :
1. Si un identifiant direct est inféré par l'attaquant LLM → R = 1.
2. Sinon → R = 1/k, k = taille de la classe d'équivalence dans la population PUMS.

> **Note implémentation** : notre calcul de k utilise le filtrage direct de la population PUMS (approximation de `correctmatch.individual_uniqueness`, Rocher et al. 2019). Les résultats sont comparables mais pas identiques à l'implémentation officielle.

## RAT-Bench : leak analysis

RAT-Bench ajoute une analyse textuelle des fuites avec `evaluate_text_leaks()` dans [`eval/core/loaders/ratbench.py`](../../eval/core/loaders/ratbench.py).

Cette analyse vérifie si les valeurs d'attributs du profil apparaissent encore dans le texte anonymisé.

Métriques principales :

| Métrique | Description |
| --- | --- |
| `leak_rate` | Part des attributs encore visibles. |
| `direct_leak_rate` | Part des identifiants directs encore visibles. |
| `indirect_leak_rate` | Part des identifiants indirects encore visibles. |
| `n_total_attributes` | Nombre d'attributs vérifiés. |
| `n_leaked` | Nombre d'attributs fuités. |
| `n_protected` | Nombre d'attributs protégés. |

Les agrégations sont exposées comme taux de fuite et taux de protection sous `axes.anonymization_leakage.ratbench_profile_leakage`.

## Fuite des valeurs gold

Tous les datasets évalués par le runner officiel exposent aussi `anonymization_leakage`.

Cette métrique vérifie si les valeurs textuelles des spans vérité terrain restent présentes dans `anonymized_text`.

Champs principaux :

| Champ | Description |
| --- | --- |
| `gold_text_leak_rate` | Part des valeurs gold encore visibles. |
| `gold_text_protection_rate` | `1 - gold_text_leak_rate`. |
| `total_gold_values` | Nombre de valeurs vérifiées. |
| `total_leaked_values` | Nombre de valeurs encore visibles. |
| `per_label` | Taux de fuite par label. |

Cette métrique est stricte et textuelle : elle ne remplace pas le protocole natif d'un benchmark, mais elle détecte les fuites directes évidentes dans la sortie anonymisée.

## RAT-Bench : risque de ré-identification

L'axe risque est exécuté par le runner officiel via [`eval/cli/evaluate_ratbench_risk.py`](../../eval/cli/evaluate_ratbench_risk.py).

Principe :

1. anonymiser le texte ;
2. attaquer le texte anonymisé avec un LLM ;
3. inférer des attributs indirects ;
4. vérifier les fuites directes ;
5. calculer une classe d'équivalence `k` via PUMS ou fallback ;
6. produire un risque `R`.

Pré-requis :

- `OPENROUTER_API_KEY` doit être défini ;
- les dépendances de l'évaluation de risque doivent être installées ;
- utiliser `--skip-risk` pour ignorer cet axe ;
- utiliser `--require-risk` pour faire échouer RAT-Bench si le risque ne peut pas être calculé.

Statuts :

| Statut | Signification |
| --- | --- |
| `risk_full` | OpenRouter disponible, attaque LLM exécutée. |
| `risk_degraded` | Axe non calculé, généralement clé OpenRouter absente. |
| `risk_skipped` | Axe désactivé explicitement par `--skip-risk`. |

Le runner réutilise les textes anonymisés déjà produits par `build_report()` pour éviter de relancer PipeGraph pendant l'attaque de risque.

## Score ARC / ResearchClaw

Le runner officiel produit :

| Champ | Description |
| --- | --- |
| `primary_metric` | Score agrégé destiné au ranking automatique. |
| `primary_metric_status` | `full`, `degraded` ou `error`. |
| `score_components` | Composants utilisés par dataset. |
| `score_weights` | Poids effectifs après retrait des axes non disponibles. |

Règles principales :

- RAT-Bench est `degraded` si `ratbench_reid_risk` n'est pas `risk_full`.
- DB-bio est `degraded` si l'utilité est seulement `proxy`.
- Les runs `full` et `degraded` ne doivent pas être comparés comme équivalents.

## Format officiel des rapports

Le dossier d'un run officiel contient :

```text
artifacts/eval-runs/<run-id>/
├── run_config.json
├── candidate_effective_config.json
├── manifest.json
├── summary.json
├── summary.md
├── datasets/
│   └── <dataset>/
│       ├── documents.jsonl
│       └── metrics.json
└── ratbench/
    └── <dataset>_risk_details.jsonl
```

`summary.json` contient :

| Champ | Description |
| --- | --- |
| `status` | `ok`, `partial` ou `error`. |
| `primary_metric` | Score agrégé. |
| `primary_metric_status` | Qualité du protocole. |
| `datasets` | Résultats par dataset. |
| `aggregate` | Poids, scores pondérés, compteurs d'erreurs. |
| `errors` | Erreurs dataset-level ou fatales. |

`datasets/<dataset>/metrics.json` contient les axes :

- `span_detection`
- `anonymization_leakage`
- `ratbench_reid_risk` si applicable
- `utility_preservation`
- `runtime`

## Format historique des rapports

Les runs détaillés utilisent le format :

```json
{
  "meta": {
    "created_at": "...",
    "pipeline": "pipegraph",
    "run_name": "...",
    "dataset": {
      "name": "..."
    },
    "limit": 50,
    "config": {}
  },
  "data": []
}
```

Compatibilité :

- format canonique `meta + data` ;
- ancien format avec `details` ;
- liste brute de documents.

Le loader de rapports garde cette compatibilité pour Streamlit.

## Emplacements

| Dossier | Contenu |
| --- | --- |
| `artifacts/eval-runs/` | Sorties officielles du runner stable. |
| `eval/evaluation/reports/` | Rapports agrégés et résumés JSON/Markdown. |
| `eval/evaluation/runs/` | Runs document-level utilisés par Streamlit. |
| `eval/datasets/*/cache/` | Datasets téléchargés et mis en cache. |

## Lecture dans Streamlit

L'application Streamlit peut :

- lancer une évaluation ;
- charger un run sauvegardé ;
- charger un rapport historique ;
- comparer des runs ;
- inspecter les documents, prédictions, leaks et feedbacks LLM ;
- afficher des vues spécialisées RAT-Bench et ablations.
