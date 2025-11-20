# 🔄 Guide de Migration - Mode NER GPU

## 📋 Vue d'ensemble

Ce guide explique comment migrer vers le mode NER GPU optimisé pour bénéficier d'un speedup de **6-10x** sur les textes longs.

**Bonne nouvelle** : Aucune modification de code n'est nécessaire ! Tout se fait via configuration.

---

## ✅ Prérequis

Avant de migrer, vérifiez que vous avez :

1. **GPU NVIDIA** avec CUDA
   ```bash
   nvidia-smi
   ```

2. **PyTorch avec CUDA**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
   
   Si `False`, installez PyTorch avec CUDA :
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Dépendances NER**
   ```bash
   pip install gliner transformers sentence-transformers
   ```

---

## 🚀 Migration en 3 Étapes

### Étape 1 : Activer le Mode GPU

**Option A : Via Script Automatique (Recommandé)**

```bash
./scripts/setup_gpu_mode.sh
```

Le script va :
- ✅ Vérifier votre GPU et CUDA
- ✅ Configurer automatiquement `config.json`
- ✅ Installer les dépendances manquantes
- ✅ Lancer un benchmark de validation

---

**Option B : Configuration Manuelle**

Éditez `config.json` et ajoutez/modifiez la section `ner_gpu` :

```json
{
  "ner_gpu": {
    "enabled": true,
    "vram_gb": 24,
    "batch_size": 64,
    "max_parallel_models": 3,
    "use_fp16": true,
    "gliner_preset": "best"
  }
}
```

**Paramètres importants** :

- `enabled` : **Mettre à `true`** pour activer le mode GPU
- `vram_gb` : Votre VRAM disponible (8, 12, 16, 24, 40+)
- `batch_size` : Taille des batchs (auto-ajusté si `optimization_level: high`)
- `max_parallel_models` : Nombre de modèles en parallèle (1-4)

---

### Étape 2 : Tester la Configuration

**Test Rapide** :

```bash
python scripts/test_gpu_integration.py
```

Choisissez l'option **1** (Test d'intégration basique).

**Sortie attendue** :

```
🔍 Mode NER détecté : GPU

✅ Le mode GPU est activé dans l'orchestrateur
[orchestrator] Mode NER GPU activé (batch_size=64, models=3)

✅ Anonymisation terminée
⏱️  Temps : 1.234s
🔍 Mode utilisé : GPU
📊 Entités détectées : 12
```

---

**Benchmark Complet** :

```bash
python scripts/benchmark_ner_gpu.py --mode both --text-size medium
```

**Résultat attendu** :

```
🚀 Speedup : 6.85x plus rapide avec le mode GPU
✅ Amélioration de 585.0% des performances
```

---

### Étape 3 : Utilisation dans Votre Code

**Aucun changement de code nécessaire** ! L'orchestrateur détecte automatiquement le mode GPU :

```python
from src.orchestrator import anonymize_text

# Le mode GPU est utilisé automatiquement si enabled=true
result = anonymize_text(
    value="Jean Dupont travaille chez Acme Corporation...",
    scope_id="ticket_001",
    secret_salt="my_secret",
    level="L0",
)

print(result["anonymized_text"])
```

**Vérification du mode** :

```python
from src.orchestrator import get_ner_mode

mode = get_ner_mode()
print(f"Mode NER actuel : {mode}")  # 'gpu' ou 'standard'
```

---

## 🔧 Ajustements Selon Votre GPU

### GPU 8GB (RTX 2060, GTX 1080, etc.)

```json
{
  "ner_gpu": {
    "enabled": true,
    "batch_size": 16,
    "max_parallel_models": 1,
    "gliner_preset": "balanced"
  }
}
```

### GPU 12GB (RTX 3060, RTX 2080 Ti, etc.)

```json
{
  "ner_gpu": {
    "enabled": true,
    "batch_size": 32,
    "max_parallel_models": 2,
    "gliner_preset": "accuracy"
  }
}
```

### GPU 16GB (RTX 4060 Ti, RTX 3070 Ti, etc.)

```json
{
  "ner_gpu": {
    "enabled": true,
    "batch_size": 48,
    "max_parallel_models": 2,
    "gliner_preset": "best"
  }
}
```

### GPU 24GB (RTX 3090, RTX 4090, A5000, etc.)

```json
{
  "ner_gpu": {
    "enabled": true,
    "batch_size": 64,
    "max_parallel_models": 3,
    "gliner_preset": "best",
    "use_fp16": true,
    "optimization_level": "high"
  }
}
```

### GPU 40GB+ (A100, A6000, H100, etc.)

```json
{
  "ner_gpu": {
    "enabled": true,
    "batch_size": 128,
    "max_parallel_models": 4,
    "gliner_preset": "best",
    "use_fp16": true,
    "use_torch_compile": true,
    "optimization_level": "high"
  }
}
```

---

## 🔙 Retour en Arrière (Rollback)

Si vous rencontrez des problèmes, vous pouvez revenir au mode standard :

**Option 1 : Désactiver dans config.json**

```json
{
  "ner_gpu": {
    "enabled": false
  }
}
```

**Option 2 : Variable d'environnement**

```bash
export NER_GPU_ENABLED=0
```

**Option 3 : Supprimer la section**

Supprimez complètement la section `ner_gpu` de `config.json`.

---

## 🐛 Dépannage

### "CUDA out of memory"

**Cause** : Batch size trop grand ou trop de modèles en parallèle.

**Solution** :

```json
{
  "ner_gpu": {
    "batch_size": 16,
    "max_parallel_models": 1
  }
}
```

Puis relancez.

---

### "torch.cuda.is_available() = False"

**Cause** : PyTorch n'est pas installé avec CUDA.

**Solution** :

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

### Le mode GPU est plus lent

**Cause** : Overhead de parallélisation sur textes courts ou premier run (warm-up).

**Solution** :

1. Activer le warm-up :
   ```json
   {
     "ner_gpu": {
       "prefetch_models": true
     }
   }
   ```

2. Tester sur textes plus longs (>500 mots)

---

### "Mode NER détecté : STANDARD" alors que enabled=true

**Causes possibles** :

1. GPU/CUDA non disponible
2. Erreur lors du chargement du pipeline
3. Configuration non rechargée

**Solutions** :

```bash
# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Vérifier les logs
python scripts/test_gpu_integration.py
# Regarder les messages d'erreur

# Forcer le rechargement
python -c "from src.orchestrator import reset_ner_pipeline; reset_ner_pipeline()"
```

---

## 📊 Validation de la Migration

### Checklist de Validation

- [ ] `nvidia-smi` fonctionne
- [ ] `torch.cuda.is_available() = True`
- [ ] `config.json` avec `ner_gpu.enabled=true`
- [ ] `python scripts/test_gpu_integration.py` affiche "Mode GPU"
- [ ] Benchmark montre un speedup > 1x
- [ ] Vos scripts d'anonymisation fonctionnent sans modification

---

### Tests Recommandés

```bash
# 1. Test d'intégration
python scripts/test_gpu_integration.py

# 2. Benchmark
python scripts/benchmark_ner_gpu.py --mode both --text-size medium

# 3. Test sur vos données
python -c "
from src.orchestrator import anonymize_text, get_ner_mode
print(f'Mode: {get_ner_mode()}')
result = anonymize_text('Votre texte de test...', 'scope_001', 'secret', 'L0')
print(f'Entités: {len(result[\"audit\"][\"entities\"])}')
"
```

---

## 📈 Gains Attendus

### Par Type de Texte

| Texte | Taille | Speedup Standard→GPU | GPU Usage |
|-------|--------|----------------------|-----------|
| Ticket court | < 200 mots | **1.3x** | 60% |
| Email moyen | 200-1000 mots | **4-6x** | 85% |
| Rapport long | > 1000 mots | **8-10x** | 95% |

### Par Volume

| Volume | Mode Standard | Mode GPU | Gain |
|--------|---------------|----------|------|
| 10 documents | ~80s | ~12s | **6.7x** |
| 100 documents | ~800s | ~120s | **6.7x** |
| 1000 documents | ~8000s | ~1200s | **6.7x** |

---

## 🎯 Prochaines Étapes

1. **Monitorer** : Surveiller l'utilisation GPU avec `nvidia-smi`
2. **Tuner** : Ajuster `batch_size` et `max_parallel_models` pour votre GPU
3. **Benchmark** : Comparer sur vos propres données
4. **Déployer** : Activer en production si speedup satisfaisant

---

## 📚 Documentation Complète

- **Guide complet** : [`docs/NER_GPU_OPTIMIZATION.md`](NER_GPU_OPTIMIZATION.md)
- **Démarrage rapide** : [`docs/QUICKSTART_NER_GPU.md`](QUICKSTART_NER_GPU.md)
- **Changelog** : [`CHANGELOG_NER_GPU.md`](../CHANGELOG_NER_GPU.md)
- **Tests** : `scripts/test_gpu_integration.py`, `scripts/benchmark_ner_gpu.py`

---

**Besoin d'aide ?**

- 🐛 Consultez la section Troubleshooting ci-dessus
- 📖 Lisez [`docs/NER_GPU_OPTIMIZATION.md`](NER_GPU_OPTIMIZATION.md)
- 🧪 Lancez `python scripts/test_gpu_integration.py`

---

**Version** : 1.0  
**Date** : Octobre 2025
