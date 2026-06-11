# Ablations

Les ablations passent par la CLI unifiee et les configs JSON de
`configs/evaluation/ablations/`.

## Commandes

```bash
python -m eval ablation --dataset tab \
  --config configs/evaluation/no_llm.json \
  --ablation-config configs/evaluation/ablations/nodes.json

python -m eval ablation --dataset ratbench \
  --config configs/evaluation/no_llm.json \
  --ablation-config configs/evaluation/ablations/ner_ensemble.json \
  --limit 50
```

## Suites

Les suites disponibles sont stockees comme fichiers:

- `nodes.json`
- `ner_presets.json`
- `ner_ensemble.json`
- `ner_threshold.json`
- `ner_vote.json`
- `anon_strategy.json`
- `detection_mode.json`
- `minimal.json`
- `default.json`

## Sorties

Chaque variante ecrit un sous-dossier de run officiel. Le dossier racine
contient:

- `ablation_summary.json`
- `ablation_summary.csv`
- `ablation_report.md`

Les runs de chaque variante gardent la structure standard:
`summary.json`, `summary.md`, `manifest.json`, `datasets/<key>/documents.jsonl`
et `datasets/<key>/metrics.json`.

## Ajouter Une Suite

Ajouter un fichier JSON dans `configs/evaluation/ablations/`:

```json
{
  "ablations": [
    {
      "name": "custom_run",
      "overrides": {
        "enable_ai": true,
        "disable_llm": true
      },
      "runner_overrides": {
        "skip_risk": true
      }
    }
  ]
}
```
