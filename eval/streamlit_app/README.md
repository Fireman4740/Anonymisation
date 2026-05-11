# Interface d'évaluation — Anonymization Error Analysis

Application Streamlit pour lancer les benchmarks, visualiser les erreurs du pipeline d'anonymisation et comparer des runs historiques.

## Lancer l'application

Depuis la racine du dépôt :

```bash
# Linux / WSL
streamlit run eval/streamlit_app/app.py --server.headless true --server.port 8501
```

```powershell
# Windows
streamlit run eval/streamlit_app/app.py
```

Ouvrir le navigateur à l'adresse affichée (par défaut `http://localhost:8501`).

## Modes disponibles

| Mode | Description |
|---|---|
| 📊 Lancer Benchmark | Évalue le pipeline sur RAT-Bench, TAB, DB-bio, CoNLL… |
| 🗂️ Historique & Comparaison | Compare des runs sauvegardés |
| 🧩 Études d'Ablation | Isole la contribution de chaque module |

## Ollama (LLM local)

> **Architecture :** Ollama tourne sur Windows, le Streamlit tourne dans WSL. L'IP du host Windows est détectée automatiquement.

### 1) Démarrer Ollama sur Windows

```powershell
# Session courante — nécessaire pour que WSL puisse atteindre Ollama
$env:OLLAMA_HOST = "http://0.0.0.0:11434"
ollama serve
```

Pour rendre le paramètre permanent :

```powershell
setx OLLAMA_HOST "http://0.0.0.0:11434"
# Relancer PowerShell, puis ollama serve
```

> Par défaut Ollama écoute sur `127.0.0.1` (inaccessible depuis WSL).
> `0.0.0.0` expose le serveur sur toutes les interfaces, y compris la passerelle WSL2.

### 2) Dans la sidebar

1. Provider → sélectionner **ollama**
2. La liste de modèles se charge automatiquement
3. Choisir le modèle souhaité et lancer l'évaluation

La résolution WSL2 → IP host Windows est gérée automatiquement (`localhost` → IP gateway).

## Visualisation des erreurs

- **Vert** : Vrai Positif (entité correctement anonymisée)
- **Rouge** : Faux Négatif (entité manquée — critique)
- **Jaune** : Faux Positif (suppression non nécessaire)

## Rapports

Les runs sauvegardés sont écrits dans `eval/evaluation/runs/`. Les rapports officiels ARC/ResearchClaw dans `artifacts/eval-runs/<run-id>/`.
