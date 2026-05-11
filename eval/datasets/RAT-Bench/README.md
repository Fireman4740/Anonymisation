# RAT-Bench Dataset

**Source**: [imperial-cpg/rat-bench](https://huggingface.co/datasets/imperial-cpg/rat-bench)

**Paper**: [RAT-Bench: A Comprehensive Benchmark for Evaluating Text Anonymization Tools](https://arxiv.org/pdf/2602.12806v1)

**GitHub**: [imperial-aisp/rat-bench](https://github.com/imperial-aisp/rat-bench)

**Leaderboard**: [HuggingFace Space](https://huggingface.co/spaces/imperial-cpg/rat-bench)

## Description

RAT-Bench is a comprehensive, multilingual benchmark for evaluating text anonymization tools,
with a focus on **re-identification risk**.

## Dataset Structure

Each record (profile) contains:

- `id`: Unique record identifier
- `profile`: All attributes and values leaked in the corresponding text
- `direct_identifiers`: Attributes and values of direct identifiers (name, email, SSN, phone, address, credit card)
- `indirect_identifiers`: Attributes and values of indirect identifiers (race, sex, DOB, occupation, etc.)
- `features`: Names of the leaked attributes
- `difficulty`: Difficulty level of the text (1, 2, or 3)
- `prompt`: Prompt used to generate the text
- `scenario`: Scenario of the text (Medical consultation, Chatbot conversation, Meeting transcript)
- `text`: Generated text containing leaked identifiers

## Languages

- English (300 rows)
- Mandarin
- Spanish

## Levels

- **Level 1**: Direct mentions of PII (easy to detect)
- **Level 2**: Obfuscated PII (spaces, misspellings, indirect references)
- **Level 3**: Deeply embedded PII (implicit, contextual)

## Usage

```bash
python -m eval.run_pipeline_evaluation \
  --datasets ratbench \
  --ratbench-languages english \
  --ratbench-levels 1 \
  --limit 50

python -m eval.run_pipeline_evaluation \
  --datasets ratbench \
  --ratbench-languages english \
  --ratbench-levels 1 2 3 \
  --skip-risk \
  --limit 100
```

`eval/cli/ratbench.py` reste disponible comme wrapper de compatibilité.
