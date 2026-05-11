# Pipeline PipeGraph

PipeGraph est le pipeline d'anonymisation local du dépôt. Il combine des détecteurs déterministes, des modèles NER, des stratégies d'anonymisation et des noeuds LLM optionnels dans un graphe LangGraph.

## Rôle du pipeline

PipeGraph prend un texte brut en entrée et retourne un état final contenant notamment :

- le texte anonymisé ;
- les entités détectées ;
- les métadonnées d'exécution ;
- les erreurs éventuelles ;
- le score de confidentialité et les retours LLM si les modules RUPTA sont activés.

L'état canonique est défini dans [`pipegraph/src/state.py`](../../pipegraph/src/state.py). Les entités suivent le format `EntityDict` avec les champs principaux `start`, `end`, `type`, `value`, `source` et `score`.

## Flux principal

Le graphe est construit dans [`pipegraph/src/graph.py`](../../pipegraph/src/graph.py). Le flux effectif est :

```text
detection
  -> anonymization_pass_1
  -> llm_review
  -> llm_verification
  -> anonymization_pass_2
  -> llm_audit
  -> llm_paraphrase
  -> llm_audit
  -> END
```

La boucle `llm_audit -> llm_paraphrase -> llm_audit` est conditionnelle. Elle est utilisée par RUPTA quand le score de confidentialité reste au-dessus du seuil configuré et que le nombre maximum d'itérations n'est pas atteint.

## Commandes utiles

Installer les dépendances PipeGraph :

```bash
pip install -r pipegraph/requirements.txt
```

Lancer la démo locale :

```bash
python pipegraph/main.py
```

Lancer les tests PipeGraph :

```bash
pytest pipegraph/tests/
```

## ResearchClaw avec GPU et reprise

Pour lancer AutoResearchClaw sur ce pipeline avec la RTX exposée par WSL, utiliser le lanceur GPU du dépôt :

```bash
./run_researchclaw_gpu.sh
```

Le script vérifie `nvidia-smi`, `torch.cuda.is_available()` et force GLiNER sur CUDA via `NER_FORCE_DEVICE=cuda`.

Si le run s'arrête parce que Codex/ACP atteint une limite ou coupe une requête, ResearchClaw conserve un checkpoint dans le dossier `artifacts/rc-*/checkpoint.json`. Pour reprendre le dernier run au prochain stage non terminé :

```bash
./run_researchclaw_gpu.sh --resume-last
```

Pour reprendre un run précis :

```bash
./run_researchclaw_gpu.sh --resume-run artifacts/rc-20260506-155357-914635
```

Si la session ACP Codex semble bloquée, fermer la session persistante avant la reprise :

```bash
./run_researchclaw_gpu.sh --reset-acp-session --resume-last
```

La reprise native reste disponible :

```bash
./run_researchclaw_gpu.sh --resume --output artifacts/<run-id>
./run_researchclaw_gpu.sh --from-stage RESOURCE_PLANNING --output artifacts/<run-id>
```

## Configuration

Les fichiers de configuration principaux sont :

- [`pipegraph/config.json`](../../pipegraph/config.json) : configuration non secrète PipeGraph (LLM, modèles, features, RUPTA, GPU/NER, sécurité de développement et runtime).
- [`pipegraph/config/patterns_config.yaml`](../../pipegraph/config/patterns_config.yaml) : règles déterministes et patterns utilisés par le détecteur regex/algo.

Les scripts d'évaluation peuvent surcharger une partie de ces paramètres via `state.config`, par exemple `disable_llm`, `gliner_preset`, `ner_min_vote`, `anon_strategy` ou `rupta_enabled`.

## Documentation liée

- [Architecture](architecture.md)
- [Configuration](configuration.md)
- [Composants](components.md)
