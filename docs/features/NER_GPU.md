# 🚀 Optimisation NER pour GPU Puissant (24GB VRAM)

## 📋 Vue d'ensemble

Ce document explique l'architecture NER du niveau L0 et les optimisations GPU ajoutées pour tirer pleinement parti d'un GPU de 24GB de VRAM.

## 🏗️ Architecture NER Niveau L0 (Standard)

### Pipeline Actuel (Mode Séquentiel)

```
Texte Original
    |
    v
[Détection Regex] → Patterns évidents (email, téléphone, etc.)
    |
    v
[Split en Phrases] → Découpe du texte en phrases
    |
    v
[Boucle sur chaque phrase]
    |
    +--> [Modèle GLiNER #1] → Inférence séquentielle
    |
    +--> [Modèle GLiNER #2] → Inférence séquentielle
    |
    +--> [Modèle GLiNER #3] → Inférence séquentielle
    |
    v
[Vote/Fusion] → Agrégation des résultats
    |
    v
Liste d'Entités NER
```

### Composants NER Disponibles

#### 1. GLiNER (Recommandé)

**Modèles disponibles** (par preset) :

- `fast` : `gliner_small-v2.1` (~140MB, rapide)
- `balanced` : `gliner_medium-v2.1` (~350MB, équilibré) **← Par défaut**
- `accuracy` : `gliner_large-v2.1` + `gliner_multi-v2.1` (~700MB chacun)
- `pii` : `gliner_multi_pii-v1` (~350MB, spécialisé données personnelles)
- `best` : 4 modèles en ensemble pondéré (~2.5GB total)
  - `EmergentMethods/gliner_medium_news-v2.1` (poids 1.25)
  - `numind/NuNER_Zero-span` (poids 1.20)
  - `urchade/gliner_large-v2.1` (poids 1.10)
  - `urchade/gliner_multi-v2.1` (poids 1.05)

**Variables d'environnement** :

```bash
GLINER_PRESET=best              # Preset à utiliser
GLINER_WEIGHTING=1              # Activer le vote pondéré
GLINER_HALF=1                   # Activer FP16 (CUDA uniquement)
GLINER_ATTENTION=eager          # Forcer attention standard (vs FlashAttention)
```

#### 2. HuggingFace NER (Optionnel)

- Modèle : `Davlan/bert-base-multilingual-cased-ner-hrl`
- Pipeline avec fenêtrage (384 tokens, stride 64)
- Support CUDA + FP16

#### 3. DeepPavlov (Désactivé par défaut en Python 3.11)

- Modèles OntoNotes/BERT
- Peut voter entre plusieurs configs

### Problèmes de Performance (Mode Standard)

❌ **Traitement séquentiel** : Les phrases sont traitées une par une  
❌ **Pas de batching** : Chaque modèle fait une inférence par phrase  
❌ **GPU sous-utilisé** : Un GPU de 24GB peut traiter 10-20x plus en parallèle  
❌ **Overhead Python** : Boucles Python entre chaque inférence  
❌ **Pas de compilation** : Pas d'optimisations PyTorch 2.0

**Exemple** : Sur un texte de 50 phrases avec preset `best` (4 modèles) :
- **Total d'inférences** : 50 phrases × 4 modèles = **200 inférences séquentielles**
- **Temps estimé** : ~25-30 secondes
- **Utilisation GPU** : 15-20% (la majorité du temps est passée en overhead)

---

## ⚡ Optimisations GPU (Nouveau)

### Nouvelles Fonctionnalités

#### 1. **Batching Intelligent**

Au lieu de traiter phrase par phrase, on groupe les phrases en batchs :

```python
# Avant (séquentiel)
for phrase in phrases:
    entities = model.predict(phrase)

# Après (batching)
entities_batch = model.predict_batch(phrases)  # Toutes en une fois
```

**Gain** : 3-5x plus rapide grâce à la parallélisation GPU.

#### 2. **Parallélisation des Modèles**

Les modèles tournent en parallèle via `ThreadPoolExecutor` :

```python
# 3 modèles en parallèle sur le GPU
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(model.predict_batch, texts) for model in models]
    results = [f.result() for f in futures]
```

**Gain** : 2-3x plus rapide (dépend de la VRAM disponible).

#### 3. **Mixed Precision (FP16)**

Utilisation de la demi-précision pour réduire la consommation mémoire :

```python
model.half()  # Passe de FP32 (4 bytes) à FP16 (2 bytes)
```

**Gain** : 
- Mémoire divisée par 2 → Peut charger plus de modèles simultanément
- Inférence 1.5-2x plus rapide sur GPU moderne (Tensor Cores)

#### 4. **Torch Compile (PyTorch 2.0+)**

Compilation JIT du backbone du modèle :

```python
model.token_rep_layer = torch.compile(model.token_rep_layer, mode="reduce-overhead")
```

**Gain** : 10-30% plus rapide après warm-up.

#### 5. **Auto-tuning de Batch Size**

Calcul automatique de la taille de batch optimale selon la VRAM :

| VRAM | Batch Size | Modèles Simultanés |
|------|------------|---------------------|
| 8GB  | 16         | 1-2                 |
| 12GB | 32         | 2                   |
| 16GB | 48         | 2-3                 |
| 24GB | **64**     | **3-4**             |

---

## 📊 Performance Attendue

### Estimation de Speedup

Pour un texte de **50 phrases** avec preset `best` (4 modèles) :

| Métrique | Mode Standard | Mode GPU | Speedup |
|----------|---------------|----------|---------|
| Temps total | ~30s | ~3-5s | **6-10x** |
| GPU Utilisation | 15-20% | 85-95% | **5x** |
| Throughput | 1.6 phrases/s | 10-16 phrases/s | **6-10x** |
| VRAM utilisée | ~2GB | ~8-12GB | - |

**Note** : Le speedup est plus important pour les textes longs (>1000 mots) car l'overhead de parallélisation est amorti.

---

## 🛠️ Configuration

### 1. Dans `config.json`

Ajoutez cette section :

```json
{
  "ner_gpu": {
    "enabled": true,
    "vram_gb": 24,
    "batch_size": 64,
    "max_parallel_models": 3,
    "use_fp16": true,
    "use_torch_compile": false,
    "gliner_preset": "best",
    "prefetch_models": true,
    "optimization_level": "high"
  }
}
```

**Paramètres** :

- `enabled` : Active le mode GPU optimisé
- `vram_gb` : VRAM disponible (pour auto-tuning)
- `batch_size` : Taille des batchs (auto si `optimization_level=high`)
- `max_parallel_models` : Modèles en parallèle (3-4 recommandé pour 24GB)
- `use_fp16` : Activer FP16 (recommandé)
- `use_torch_compile` : Activer torch.compile (PyTorch 2.0+)
- `gliner_preset` : Preset GLiNER (`best` recommandé)
- `prefetch_models` : Précharger les modèles au démarrage
- `optimization_level` : `low`, `medium`, `high` (auto-tune agressif)

### 2. Via Variables d'Environnement

```bash
export NER_GPU_ENABLED=1
export NER_GPU_BATCH_SIZE=64
export NER_GPU_VRAM_GB=24
export NER_GPU_PARALLEL_MODELS=3
export NER_GPU_COMPILE=1  # PyTorch 2.0+
export NER_GPU_DEBUG=1    # Logs détaillés
```

---

## 🚀 Utilisation

### Option 1 : Via l'Orchestrateur (Automatique) ⭐ RECOMMANDÉ

L'orchestrateur détecte automatiquement si le mode GPU est activé et utilise le pipeline optimisé :

```python
from src.orchestrator import anonymize_text, get_ner_mode

# Vérifier le mode actuel
print(f"Mode NER : {get_ner_mode()}")  # 'gpu' ou 'standard'

# Si ner_gpu.enabled=true dans config.json, utilisera le mode GPU automatiquement
result = anonymize_text(
    value="Jean Dupont travaille chez Acme...",
    scope_id="ticket_001",
    secret_salt="my_secret",
    level="L0",  # Niveau L0 utilise le NER
)

print(result["anonymized_text"])
```

**Avantages** :
- ✅ Aucun changement de code nécessaire
- ✅ Détection automatique du mode GPU
- ✅ Fallback automatique vers mode standard en cas d'erreur
- ✅ Pipeline mis en cache (pas de rechargement à chaque appel)

**Fonctions utiles** :

```python
from src.orchestrator import get_ner_mode, reset_ner_pipeline

# Vérifier le mode actuel
mode = get_ner_mode()
print(f"Mode NER : {mode}")  # 'gpu' ou 'standard'

# Forcer le rechargement du pipeline (si config modifiée)
reset_ner_pipeline()
```

### Option 2 : Pipeline Direct (Manuel)

```python
from src.ner_gpu_optimizer import create_optimized_pipeline

# Créer le pipeline optimisé
pipeline = create_optimized_pipeline()

# Prédiction
entities = pipeline.predict("Jean Dupont travaille chez Acme Corp...")

# Résultat : [{'start': 0, 'end': 11, 'entity_group': 'PERSON', 'votes': 3.5}, ...]
```

### Option 3 : Benchmark

Comparer les performances standard vs GPU :

```bash
# Texte court
python scripts/benchmark_ner_gpu.py --text-size short --mode both

# Texte moyen (recommandé)
python scripts/benchmark_ner_gpu.py --text-size medium --mode both

# Texte long
python scripts/benchmark_ner_gpu.py --text-size long --mode both

# Texte personnalisé
python scripts/benchmark_ner_gpu.py --custom-text "Votre texte ici..." --mode both
```

**Sortie** :

```
🏁 BENCHMARK NER - Mode Standard vs GPU Optimisé
================================================================================
Texte : 1247 caractères
Runs : 3

🔵 Benchmark Mode Standard (Séquentiel)
============================================================
Warm-up des modèles...
Run 1/3... 8.45s (42 entités)
Run 2/3... 8.12s (42 entités)
Run 3/3... 8.23s (42 entités)

🟢 Benchmark Mode GPU Optimisé (Parallèle)
============================================================
Configuration GPU :
  - Batch size: 64
  - Modèles parallèles: 3
  - FP16: True
  - Torch compile: False
Création du pipeline optimisé...
Warm-up des modèles...
Run 1/3... 1.23s (44 entités)
Run 2/3... 1.18s (44 entités)
Run 3/3... 1.21s (44 entités)

📊 RÉSULTATS
================================================================================
Métrique                       Standard             GPU Optimisé         Speedup        
-------------------------------------------------------------------------------------
Temps moyen                    8.267s               1.207s               6.85x
Temps min                      8.120s               1.180s
Temps max                      8.450s               1.230s
Entités détectées              42                   44                  
Throughput (docs/s)            0.12                 0.83                

🚀 Speedup : 6.85x plus rapide avec le mode GPU
✅ Amélioration de 585.0% des performances
```

---

## 🔧 Tuning Avancé

### Ajuster le Batch Size

Pour trouver la taille optimale :

```python
from src.ner_gpu_optimizer import auto_tune_batch_size

# Auto-tune selon votre VRAM
batch_size = auto_tune_batch_size(vram_gb=24, model_size="medium")
print(f"Batch size recommandé : {batch_size}")  # → 64
```

### Régler le Nombre de Modèles Parallèles

**Règle empirique** :

- 1 modèle medium ≈ 2-3GB VRAM
- 24GB VRAM → Max 6-8 modèles simultanés
- **Recommandé** : 3-4 modèles (meilleur compromis vitesse/qualité)

### Activer torch.compile (PyTorch 2.0+)

```bash
# Vérifier la version PyTorch
python -c "import torch; print(torch.__version__)"

# Si >= 2.0
export NER_GPU_COMPILE=1
```

**Note** : Le premier run sera plus lent (compilation), mais les suivants seront 10-30% plus rapides.

---

## 📈 Quand Utiliser le Mode GPU ?

### ✅ Utilisez le mode GPU si :

- Vous traitez des textes **longs** (>500 mots)
- Vous traitez **beaucoup de documents** (batch processing)
- Vous voulez la **meilleure qualité** (preset `best` avec 4 modèles)
- Vous avez un **GPU moderne** (RTX 3000+, A100, etc.)

### ❌ N'utilisez PAS le mode GPU si :

- Vous traitez des textes **très courts** (<100 mots) → overhead trop élevé
- Vous n'avez **pas de GPU** ou GPU faible (<8GB VRAM)
- Vous voulez la **latence minimale** pour 1 seul document court

---

## 🐛 Troubleshooting

### Erreur : "CUDA out of memory"

**Solutions** :

1. Réduire `batch_size` dans config.json
2. Réduire `max_parallel_models`
3. Désactiver `use_fp16` (contre-intuitif mais peut aider sur vieux GPU)
4. Utiliser preset `balanced` au lieu de `best`

```bash
# Test avec paramètres conservateurs
NER_GPU_BATCH_SIZE=16 NER_GPU_PARALLEL_MODELS=2 python scripts/benchmark_ner_gpu.py
```

### Erreur : "torch.compile not available"

Vous avez PyTorch < 2.0. Désactivez la compilation :

```json
{
  "ner_gpu": {
    "use_torch_compile": false
  }
}
```

### Le mode GPU est plus lent que le standard

**Causes possibles** :

1. Texte trop court → overhead de parallélisation
2. Premier run → compilation/warm-up
3. GPU partagé avec autre processus

**Solutions** :

1. Activer `prefetch_models: true` pour warm-up initial
2. Tester sur textes plus longs
3. Vérifier `nvidia-smi` pour voir l'utilisation GPU

---

## 📚 Ressources

- **Code** : `src/ner_gpu_optimizer.py`
- **Benchmark** : `scripts/benchmark_ner_gpu.py`
- **Config** : `config.json` (section `ner_gpu`)
- **Variables d'env** : Voir section Configuration

---

## 🎯 Prochaines Améliorations Possibles

1. **Multi-GPU** : Distribuer les modèles sur plusieurs GPUs
2. **Quantization INT8** : Réduire encore la mémoire (via ONNX ou TensorRT)
3. **Pipeline asynchrone** : Traiter plusieurs documents en parallèle
4. **Cache de résultats** : Éviter de ré-analyser les mêmes textes
5. **Flash Attention 2** : Accélérer l'attention (si modèle compatible)

---

**Auteur** : Système d'anonymisation RUPTA  
**Version** : 1.0  
**Date** : Octobre 2025
