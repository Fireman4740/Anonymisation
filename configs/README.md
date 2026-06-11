# Configs runtime

Fichiers de configuration runtime pour le pipeline (clés `state.config` de
PipeGraph). Chaque baseline/ablation est **une config, pas un fork du code**.

Usage :

```bash
python -m pipegraph anonymize --text "..." --config configs/baselines/no_llm.json
```

```python
from pipegraph import anonymize
result = anonymize("...", config="configs/baselines/regex_only.json")
```

## Baselines (`baselines/`)

| Config | Détection | LLM |
| --- | --- | --- |
| `regex_only.json` | regex + validateurs | aucun |
| `ner_only.json` | NER (GLiNER/Flair/spaCy) | aucun |
| `no_llm.json` | regex + NER | aucun |
| `full_llm.json` | regex + NER | review + verification + audit + paraphrase (RUPTA) |

## Ablations (`ablations/`)

Partent de `full_llm` et retirent un module à la fois :

- `no_review.json` — sans reviewer LLM (détection des fuites manquées)
- `no_verification.json` — sans vérification LLM des entités
- `no_audit.json` — sans audit adversarial (désactive aussi la boucle RUPTA)
- `no_paraphrase.json` — sans paraphrase ciblée (l'audit reste actif, boucle RUPTA off)

## Clés reconnues

Voir `pipegraph/src/state.py` (state.config) et `eval/core/config.py`
(`normalize_runtime_config`). Principales : `enable_deterministic`,
`enable_ai`, `enable_anonymization`, `detection_mode`, `llm_detection`,
`llm_verification`, `llm_audit`, `llm_paraphrase`, `rupta_enabled`,
`disable_llm`, `anon_strategy`, `anon_policy`, `scope_id`, `llm_provider`,
`llm_model`, `llm_mock`.
