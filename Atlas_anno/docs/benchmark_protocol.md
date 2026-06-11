# Benchmark Protocol

The MVP uses French support tickets and emails, closed-world candidate pools, three anonymization baselines, one structured attacker and one LLM attacker.

## Difficulty Protocol

Document difficulty (`easy`, `medium`, `hard`) controls the mix of mention-level modes:

- `explicit_easy`: standard surface forms.
- `explicit_hard`: obfuscated but recoverable forms.
- `implicit`: contextual cues only.

Calibration is run on raw generated texts before anonymization. The structured attacker must show decreasing top-1 accuracy from easy to hard, and non-increasing accuracy from `explicit_easy` to `implicit` when the groups are observed.

## Auxiliary Knowledge Protocol

Attack pairs materialize a motivated-intruder setting:

- `none`: no auxiliary attributes.
- `partial`: organization and department.
- `strong`: organization, team, role, and one rare trait when available.

Candidate pools are difficulty-dependent and prefer same-team or same-role decoys before broader organization/global decoys.

## LLM-First Text Generation

In `llm-first` mode, the LLM receives the persona, contextual cues, scenario, and planned mention snippets. The generated JSON is accepted only when all planned snippets are locatable, hard/implicit mentions do not leak canonical values, and unplanned direct identifiers are absent. Failed generations are repaired and then fall back to deterministic text.

