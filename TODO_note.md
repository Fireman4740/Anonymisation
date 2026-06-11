# Journal d'avancement

## Etat actuel

- PipeGraph est le pipeline local principal d'anonymisation.
- L'evaluation passe par la CLI unifiee `python -m eval` (moteur: `eval/run_pipeline_evaluation.py`, API: `eval/api.py`).
- Anciens wrappers `eval/run_full_benchmark.py`, `eval/cli/tab.py`, `eval/cli/ratbench.py` et `eval/datasets/conll2003.py` supprimes (aucune reference code).
- Les sorties officielles sont ecrites dans `artifacts/eval-runs/<run-id>/`.
- Les axes exposes sont `span_detection`, `anonymization_leakage`, `ratbench_reid_risk`, `utility_preservation` et `runtime`.

## Fait

- [x] Pipeline LangGraph PipeGraph.
- [x] Evaluation locale PipeGraph dans Streamlit.
- [x] Runner officiel multi-datasets.
- [x] TAB, DB-bio, RAT-Bench, CoNLL2003 et dataset synthetique local raccordes au runner officiel.
- [x] RAT-Bench: fuite textuelle + risque de re-identification via OpenRouter quand `OPENROUTER_API_KEY` est disponible.
- [x] Documentation evaluation mise a jour.
- [x] Dead code des wrappers d'evaluation nettoye.

## A faire

- [ ] Ajouter un vrai scorer d'utilite DB-bio/RUPTA au lieu du statut `proxy`.
- [ ] Integrer PersonalReddit au runner officiel si le dataset devient utile pour les comparaisons.
- [ ] Ajouter un mode resume/checkpoint pour les evaluations longues si necessaire.
- [ ] Optimiser la detection NER/spaCy/GLiNER et mesurer l'impact via le runner officiel.
- [ ] Revoir la logique de fusion regex + NER sur les erreurs recurrentes observees dans les rapports.
- [ ] Tester une strategie de validation rapide par LLM local/Ollama ou OpenRouter sur les cas ambigus.
