# PipeGraph — Pipeline d'anonymisation LangGraph

Pipeline modulaire d'anonymisation basé sur `langgraph`. Chaque étape est un node indépendant qui reçoit l'état global, effectue son traitement, et met à jour l'état.

## Architecture

### État du pipeline (`PipelineState`)

| Champ | Description |
|---|---|
| `text` | Texte en cours de traitement |
| `original_text` | Texte brut initial |
| `entities` | Entités détectées |
| `config` | Configuration active (nodes activés) |
| `metadata` | Logs, stats |

### Nodes

1. **DetectionNode** — Détection d'entités (Regex + NER GLiNER/spaCy/Flair)
2. **AnonymizationNode** — Remplacement par placeholders ou généralisation
3. **LLMDetectionNode** — Détection assistée par LLM
4. **LLMAuditNode** — Audit RUPTA (vérification de la qualité d'anonymisation)
5. **LLMParaphraseNode** — Paraphrase pour réduire le risque de ré-identification

## Utilisation

```bash
pip install -r requirements.txt
python main.py
```

## Configuration LLM (`config.json`)

La configuration LLM vit dans `pipegraph/config.json`.

### Provider Ollama (local)

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama3.2:latest",
    "base_url": "http://localhost:11434/v1",
    "timeout_seconds": 90
  }
}
```

**Démarrer Ollama sur Windows** (pour un accès depuis WSL) :

```powershell
# Écoute sur toutes les interfaces — requis pour WSL
$env:OLLAMA_HOST = "http://0.0.0.0:11434"
ollama serve
```

```powershell
# Permanent (toutes nouvelles sessions PowerShell)
setx OLLAMA_HOST "http://0.0.0.0:11434"
```

> L'URL `localhost` dans `base_url` est automatiquement résolue vers l'IP du host Windows en WSL2.

### Provider OpenRouter (cloud)

```json
{
  "llm": {
    "provider": "openrouter"
  },
  "openrouter": {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "google/gemma-3-27b-it",
    "fallback_model": "qwen/qwen3-235b-a22b-2507"
  }
}
```

Variable d'environnement requise : `OPENROUTER_API_KEY`
