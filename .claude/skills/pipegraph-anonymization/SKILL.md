---
name: pipegraph-anonymization
description: Optimize the local PipeGraph anonymization pipeline through the AutoResearchClaw harness.
trigger-keywords:
  - pipegraph
  - anonymization
  - anonymisation
  - AutoResearchClaw
  - RAT-Bench
applicable-stages:
  - 9
  - 10
  - 12
  - 13
  - 14
enabled: true
---

# PipeGraph Anonymization Harness

Use the local harness in `arc_pipegraph/` to evaluate candidate changes. Do not invent metrics or use external datasets unless the user asks for them.

## Required command shape

Run candidates with:

```bash
.venv/bin/python3 -m arc_pipegraph.evaluate_candidate \
  --candidate arc_pipegraph/candidates/baseline_full_llm.json \
  --datasets tab dbbio anonymization ratbench conll2003 \
  --ratbench-levels 1 \
  --limit 20 \
  --doc-workers 2
```

The last stdout line is JSON and contains `primary_metric`. AutoResearchClaw must use:

- `experiment.metric_key: primary_metric`
- `experiment.metric_direction: maximize`

## Candidate rules

Only edit candidate JSON values in the allowed search space described by `arc_pipegraph/candidate_space.json`.

Never disable LLM or RUPTA. The harness forces:

- `llm_detection`
- `llm_verification`
- `llm_audit`
- `llm_paraphrase`
- `rupta_enabled`

Keep evaluation datasets local under `eval/datasets/`. Preserve existing runs and user edits.
