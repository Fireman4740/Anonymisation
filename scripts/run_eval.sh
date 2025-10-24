#!/bin/bash
# Script pour relancer l'évaluation avec le nouveau baseline

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔄 Relancement Évaluation RUPTA - Nouveau Baseline       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier la clé API
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY non définie !"
    echo "   export OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi

echo "✅ Clé API définie"
echo ""

# Backup anciens résultats
if [ -f "baseline.json" ]; then
    echo "📦 Sauvegarde anciens résultats..."
    mv baseline.json baseline_old.json
    mv rupta.json rupta_old.json 2>/dev/null
    echo "   → baseline_old.json, rupta_old.json"
    echo ""
fi

# Évaluation Baseline
echo "1️⃣  Évaluation Baseline (10 exemples)..."
echo "   Temps estimé : ~3 minutes"
echo ""
python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline --output baseline.json
echo ""

# Évaluation RUPTA  
echo "2️⃣  Évaluation RUPTA (10 exemples)..."
echo "   Temps estimé : ~10 minutes"
echo ""
python eval_rupta_dbbio.py --split test --n_samples 10 --output rupta.json
echo ""

# Comparaison
echo "3️⃣  Comparaison Baseline vs RUPTA..."
echo ""
python compare_baseline_rupta.py --baseline baseline.json --rupta rupta.json --detailed --output rapport_final.md
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Évaluation Terminée !                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Résultats :"
echo "   • baseline.json - Résultats baseline"
echo "   • rupta.json - Résultats RUPTA"
echo "   • rapport_final.md - Rapport comparatif"
echo ""
echo "📖 Lire le rapport :"
echo "   cat rapport_final.md"
echo ""
