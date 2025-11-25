# PipeGraph - Architecture LangGraph pour l'Anonymisation

Ce dossier contient une réécriture modulaire du pipeline d'anonymisation basée sur `langgraph`.

## Architecture

Le pipeline est vu comme un graphe d'états (StateGraph). Chaque étape est un "Node" indépendant qui reçoit l'état global, effectue un travail, et met à jour l'état.

### Flux de Données (State)

L'état (`PipelineState`) contient :
- `text`: Le texte en cours de traitement.
- `original_text`: Le texte brut initial.
- `entities`: Liste des entités détectées.
- `config`: Configuration active (quels nodes activer).
- `metadata`: Infos diverses (logs, stats).

### Nodes (Modules)

1. **DetectionNode** : Identifie les entités sensibles (Regex + NER).
2. **AnonymizationNode** : Remplace les entités par des placeholders ou généralise.
3. **(Futur) LLMNode** : Paraphrase ou audit.

## Utilisation

```bash
pip install -r requirements.txt
python main.py
```
