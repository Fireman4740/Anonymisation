# Dataset Card

Atlas_anno produces fully synthetic documents. No real personal data should be included in generated artifacts.

## Generation Modes

Text generation supports three modes:

- `deterministic`: template composition only; fully offline and deterministic.
- `hybrid`: deterministic templates followed by LLM humanization.
- `llm-first`: the LLM drafts the full French document from the persona, scenario, and mention plan; deterministic text remains the fallback.

All LLM stages use checkpointing/cache through the batch runtime and preserve an offline fallback.

## Mention Difficulty

Grounded mentions carry a per-mention difficulty taxonomy:

- `explicit_easy`: direct or humanized surface form.
- `explicit_hard`: recoverable obfuscated surface form, with the canonical value absent verbatim.
- `implicit`: contextual cue that reveals the attribute without stating the canonical value.

Each mention may also include `hardness` and `certainty`. Low-certainty mentions are eligible for human review.

## Attack Pairs

The dataset includes explicit attack pairs in `data/attacks/pairs.jsonl`. Each pair defines:

- the document and target person;
- the auxiliary knowledge level (`none`, `partial`, `strong`);
- the closed-world candidate pool;
- metadata for difficulty and split analysis.

These pairs are the preferred source of truth for re-identification evaluation; legacy document candidate pools remain for compatibility.

