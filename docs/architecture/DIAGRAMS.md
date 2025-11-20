# 🏗️ Architecture Refactorisée - Diagrammes

## Vue d'ensemble du système

```
┌─────────────────────────────────────────────────────────────────┐
│                         API / Entry Point                        │
│                    (main_eval.py, notebooks)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Orchestrator                               │
│                  (orchestrator.py - 300 lignes)                  │
│                                                                   │
│  • Coordonne les services via DI                                │
│  • Applique les remplacements                                   │
│  • Gère le flux L0/L1                                           │
└──────┬────────────┬────────────┬─────────────┬──────────────────┘
       │            │            │             │
       ▼            ▼            ▼             ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│Detection │ │Generaliz.│ │   LLM    │ │GPU Pipeline  │
│ Service  │ │ Service  │ │ Pipeline │ │  (optional)  │
└──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

## Détail des Services

### 1. Detection Service (src/detectors.py)

```
┌────────────────────────────────────────────────────┐
│           DetectionService                         │
├────────────────────────────────────────────────────┤
│                                                    │
│  + detect_regex(text) → List[DetectedEntity]     │
│  + detect_ner(text) → List[DetectedEntity]       │
│  + detect_all(text) → List[DetectedEntity]       │
│                                                    │
├────────────────────────────────────────────────────┤
│  Dependencies:                                     │
│  • text_sanitizer (regex)                         │
│  • ner_ensemble_clean (GLiNER)                    │
│  • ner_gpu_optimizer (optional GPU)               │
└────────────────────────────────────────────────────┘
           │
           ├──► Regex Detection
           │    └─ Email, Phone, IP, IBAN, etc.
           │
           ├──► NER Detection (GLiNER)
           │    ├─ GPU Pipeline (si disponible)
           │    └─ Standard GLiNER (fallback)
           │
           └──► Fusion & Déduplication
                └─ Priorité: regex > ner-gpu > ner
```

### 2. Generalization Service (src/generalizers.py)

```
┌────────────────────────────────────────────────────┐
│         GeneralizationService                      │
├────────────────────────────────────────────────────┤
│                                                    │
│  + generalize_dates(text)                         │
│    → (text, List[Generalization])                 │
│                                                    │
│  + generalize_org_placeholders(text)              │
│    → (text, List[Generalization])                 │
│                                                    │
│  + apply_all(text)                                │
│    → (text, List[Generalization])                 │
│                                                    │
├────────────────────────────────────────────────────┤
│  Policy-Driven:                                    │
│  • date_granularity: month/quarter/year           │
│  • org_policy: generalize/redact                  │
└────────────────────────────────────────────────────┘
           │
           ├──► Date Generalization
           │    ├─ 2024-06-15 → [DATE_2024-06]
           │    ├─ 15 June 1995 → [DATE_1995-Q2]
           │    └─ Supports FR/EN formats
           │
           └──► Org Placeholder Generalization
                ├─ [ORG_ABC] → [ORG]
                └─ Policy-configurable
```

### 3. LLM Pipeline Service (src/llm_pipeline.py)

```
┌────────────────────────────────────────────────────┐
│          LLMPipelineService                        │
├────────────────────────────────────────────────────┤
│                                                    │
│  + detect_and_plan(text, seeds)                   │
│    → LLMDetectionResult                           │
│                                                    │
│  + paraphrase(text)                               │
│    → (text, error)                                │
│                                                    │
│  + audit(text)                                    │
│    → (report, error)                              │
│                                                    │
│  + optimize_with_rupta(...)                       │
│    → (RuptaResult, error)                         │
│                                                    │
├────────────────────────────────────────────────────┤
│  Dependencies:                                     │
│  • llm_reasoner_openrouter                        │
│  • rupta/optimizer                                │
│  • rupta/privacy_evaluator                        │
│  • rupta/utility_evaluator                        │
└────────────────────────────────────────────────────┘
           │
           ├──► LLM Detection
           │    ├─ Entity detection
           │    ├─ Co-reference resolution
           │    └─ Clustering
           │
           ├──► Paraphrase
           │    ├─ Stylometric reduction
           │    └─ Placeholder preservation
           │
           ├──► Audit
           │    ├─ Risk scoring
           │    └─ Recommendations
           │
           └──► RUPTA Optimization
                ├─ Privacy evaluation
                ├─ Utility evaluation
                ├─ Iterative refinement
                └─ Uses rupta/optimizer.py
```

## Flux de Traitement

### Mode L0 (Sans LLM)

```
Text Input
    │
    ▼
┌───────────────┐
│  Detection    │
│   Service     │
├───────────────┤
│ • Regex       │
│ • NER/GPU     │
└───────┬───────┘
        │
        ▼ List[DetectedEntity]
┌───────────────┐
│ Apply         │
│ Replacements  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│Generalization │
│   Service     │
├───────────────┤
│ • Dates       │
│ • Orgs        │
└───────┬───────┘
        │
        ▼
   Anonymized Text
```

### Mode L1 (Avec LLM + RUPTA)

```
Text Input
    │
    ▼
┌───────────────┐
│  Detection    │
│   Service     │
└───────┬───────┘
        │
        ▼ Seeds
┌───────────────┐
│ LLM Pipeline  │
│ .detect_plan()│
└───────┬───────┘
        │
        ▼ Enhanced Entities
┌───────────────┐
│ Apply         │
│ Replacements  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│Generalization │
│   Service     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ LLM Pipeline  │
│ .paraphrase() │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ LLM Pipeline  │
│   .audit()    │
└───────┬───────┘
        │
        ├─ Risk OK? → Output
        │
        ▼ Risk High
┌───────────────┐
│  Hardening    │
│  Loop         │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ RUPTA Enabled?│
└───────┬───────┘
        │ Yes
        ▼
┌───────────────┐
│ LLM Pipeline  │
│.optimize_rupta│
├───────────────┤
│Uses optimizer │
│  .py (unique) │
└───────┬───────┘
        │
        ▼
   Final Anonymized Text
```

## Dépendances des Modules

```
orchestrator.py
    ├── detectors.py
    │   ├── text_sanitizer.py
    │   ├── ner_ensemble_clean.py
    │   │   └── gliner (external)
    │   └── ner_gpu_optimizer.py (optional)
    │       └── torch (external)
    │
    ├── generalizers.py
    │   └── policy.py
    │
    ├── llm_pipeline.py
    │   ├── openrouter_client.py
    │   │   └── config_loader.py
    │   ├── llm_reasoner_openrouter.py
    │   └── rupta/
    │       ├── optimizer.py
    │       ├── privacy_evaluator.py
    │       └── utility_evaluator.py
    │
    ├── utils_pseudo.py
    └── policy.py
```

## Comparaison Avant/Après

### Avant (Monolithique)

```
orchestrator.py (678 lignes)
├─ Regex detection          ┐
├─ NER setup               │
├─ GPU pipeline mgmt       │  Tout mélangé
├─ LLM detection           │  Difficile à tester
├─ Dates generalization    │  État global
├─ Orgs generalization     │
├─ Paraphrase              │
├─ Audit                   │
├─ RUPTA (inline) ❌       │  Dupliqué!
└─ Hardening               ┘
```

### Après (Modulaire)

```
orchestrator.py (300 lignes)
├─ Service creation
├─ Coordination
└─ Result assembly

detectors.py
├─ Regex detection
└─ NER (GPU/standard)

generalizers.py
├─ Date generalization
└─ Org generalization

llm_pipeline.py
├─ LLM detection
├─ Paraphrase
├─ Audit
└─ RUPTA ✅ (appelle optimizer.py)
```

## GPU Pipeline Integration

```
┌─────────────────────────────────────┐
│     Orchestrator                     │
└──────────────┬──────────────────────┘
               │
               ▼
        Create Detection Service
               │
               ├─ GPU Config Available?
               │  └─ Yes → Load GPU Pipeline
               │          from ner_gpu_optimizer
               │
               ▼
┌──────────────────────────────────────┐
│      DetectionService                │
│  ┌────────────────────────────────┐ │
│  │  gpu_pipeline (optional)       │ │
│  │  ┌──────────────────────────┐  │ │
│  │  │ ParallelNERPipeline      │  │ │
│  │  │ • Batch processing       │  │ │
│  │  │ • FP16 (optional)        │  │ │
│  │  │ • Multi-model parallel   │  │ │
│  │  └──────────────────────────┘  │ │
│  └────────────────────────────────┘ │
│                                      │
│  Fallback: Standard GLiNER          │
│  • run_gliner() from ner_ensemble   │
└──────────────────────────────────────┘
```

## Testing Strategy

```
Unit Tests
├── test_detectors.py
│   ├─ test_regex_detection()
│   ├─ test_ner_detection()
│   ├─ test_gpu_fallback()
│   └─ test_deduplication()
│
├── test_generalizers.py
│   ├─ test_date_generalization()
│   ├─ test_org_generalization()
│   └─ test_policy_escalation()
│
├── test_llm_pipeline.py
│   ├─ test_detection_plan()
│   ├─ test_paraphrase()
│   ├─ test_audit()
│   └─ test_rupta_integration()
│
└── test_orchestrator.py
    ├─ test_l0_pipeline()
    ├─ test_l1_pipeline()
    ├─ test_dependency_injection()
    └─ test_regression()

Integration Tests
├── test_end_to_end.py
│   ├─ test_full_l0_flow()
│   ├─ test_full_l1_flow()
│   └─ test_rupta_optimization()
│
└── test_performance.py
    ├─ benchmark_gpu_vs_standard()
    └─ benchmark_services()
```

## Configuration Schema

```yaml
# Exemple de configuration typée (future)
llm:
  provider: "lmstudio"  # ou "openrouter"
  base_url: "http://10.10.153.169:1234/v1"
  models:
    detect: "openai/gpt-oss-20b"
    paraphrase: "openai/gpt-oss-20b"
    audit: "openai/gpt-oss-20b"

ner_gpu:
  enabled: true
  vram_gb: 24
  batch_size: 64
  max_parallel_models: 3
  use_fp16: true
  gliner_preset: "best"

rupta:
  enabled: true
  p_threshold: 10
  max_iterations: 3
  privacy_threshold: null  # ou 11
  utility_threshold: 80

policy_defaults:
  date_granularity: "month"
  org_policy: "categorize"
  risk_threshold: 45
```

---

**Note:** Ces diagrammes représentent l'architecture refactorisée proposée. L'implémentation complète nécessite la migration de l'orchestrateur actuel.
