#!/bin/bash
# Script de démarrage rapide pour le mode NER GPU optimisé

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🚀 Démarrage Rapide - NER GPU Optimisé (24GB VRAM)          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi non trouvé. Avez-vous un GPU NVIDIA installé ?"
    exit 1
fi

echo "📊 Informations GPU :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Vérifier PyTorch CUDA
echo "🔍 Vérification PyTorch + CUDA..."
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "❌ PyTorch ne détecte pas CUDA !"
    echo ""
    echo "Pour installer PyTorch avec CUDA :"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    exit 1
fi

echo "✅ PyTorch + CUDA disponible"

# Afficher version PyTorch
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "   Version PyTorch : $TORCH_VERSION"

# Vérifier VRAM disponible
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
VRAM_GB=$((VRAM_FREE / 1024))
echo "   VRAM disponible : ${VRAM_GB}GB"

if [ "$VRAM_GB" -lt 8 ]; then
    echo "⚠️  Attention : VRAM < 8GB. Le mode GPU peut ne pas fonctionner correctement."
fi

echo ""

# Activer le mode GPU dans config.json
echo "⚙️  Configuration du mode GPU..."

if [ -f "config.json" ]; then
    # Créer une sauvegarde
    cp config.json config.json.bak
    echo "   ✅ Sauvegarde : config.json.bak"
    
    # Activer via Python (plus fiable que sed/jq)
    python << 'PYEOF'
import json
import os

config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Ajouter/mettre à jour la config GPU
if "ner_gpu" not in config:
    config["ner_gpu"] = {}

config["ner_gpu"]["enabled"] = True
config["ner_gpu"]["vram_gb"] = int(os.getenv("VRAM_GB", "24"))
config["ner_gpu"]["batch_size"] = 64
config["ner_gpu"]["max_parallel_models"] = 3
config["ner_gpu"]["use_fp16"] = True
config["ner_gpu"]["use_torch_compile"] = False  # Activer manuellement si PyTorch 2.0+
config["ner_gpu"]["gliner_preset"] = "best"
config["ner_gpu"]["prefetch_models"] = True
config["ner_gpu"]["optimization_level"] = "high"

with open(config_path, "w") as f:
    json.dump(config, f, indent="\t")

print("   ✅ config.json mis à jour (ner_gpu.enabled=true)")
PYEOF
else
    echo "   ❌ config.json non trouvé !"
    exit 1
fi

echo ""

# Installer les dépendances si nécessaire
echo "📦 Vérification des dépendances..."
pip install -q gliner transformers torch sentence-transformers 2>/dev/null
echo "   ✅ Dépendances installées"
echo ""

# Benchmark rapide
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🏁 Benchmark : Mode Standard vs GPU Optimisé                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

read -p "Lancer le benchmark maintenant ? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Lancement du benchmark (texte moyen, 3 runs)..."
    echo ""
    python scripts/benchmark_ner_gpu.py --text-size medium --mode both --n_runs 3
    echo ""
    echo "✅ Benchmark terminé !"
else
    echo ""
    echo "Benchmark annulé. Pour le lancer manuellement :"
    echo "  python scripts/benchmark_ner_gpu.py --text-size medium --mode both"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  📚 Documentation et Commandes Utiles                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📖 Guide complet : docs/NER_GPU_OPTIMIZATION.md"
echo ""
echo "🔧 Commandes utiles :"
echo ""
echo "1️⃣  Benchmark texte court"
echo "    python scripts/benchmark_ner_gpu.py --text-size short --mode both"
echo ""
echo "2️⃣  Benchmark texte long"
echo "    python scripts/benchmark_ner_gpu.py --text-size long --mode both"
echo ""
echo "3️⃣  Désactiver le mode GPU"
echo "    # Dans config.json : \"enabled\": false"
echo "    # Ou : export NER_GPU_ENABLED=0"
echo ""
echo "4️⃣  Ajuster batch size (si OOM)"
echo "    export NER_GPU_BATCH_SIZE=32"
echo "    python scripts/benchmark_ner_gpu.py --mode gpu"
echo ""
echo "5️⃣  Activer torch.compile (PyTorch 2.0+)"
echo "    # Dans config.json : \"use_torch_compile\": true"
echo ""
echo "6️⃣  Monitorer le GPU en temps réel"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "✅ Configuration terminée ! Le mode GPU est maintenant activé."
echo ""
