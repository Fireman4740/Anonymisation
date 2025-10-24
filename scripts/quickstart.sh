#!/bin/bash
# Script de démarrage rapide pour RUPTA avec Python 3.11

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🚀 Démarrage Rapide - RUPTA avec Python 3.11                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier la version Python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "📌 Version Python détectée : $PYTHON_VERSION"

if [[ $PYTHON_VERSION != 3.11* ]]; then
    echo "⚠️  Attention : Python 3.11 recommandé pour éviter les conflits NER"
fi

echo ""

# Vérifier la clé API
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY non définie !"
    echo ""
    echo "Pour définir votre clé API :"
    echo "  export OPENROUTER_API_KEY=sk-or-v1-..."
    echo ""
    echo "Ou créez un fichier .env :"
    echo "  echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env"
    echo ""
    exit 1
fi

echo "✅ OPENROUTER_API_KEY définie"
echo ""

# Étape 1 : Installer les dépendances minimales
echo "📦 Étape 1 : Vérification des dépendances..."
pip install -q gdown tqdm 2>/dev/null
echo "   ✅ Dépendances installées"
echo ""

# Étape 2 : Télécharger les datasets (si nécessaire)
echo "📂 Étape 2 : Vérification des datasets..."
if [ ! -f "Dataset/evaluation/DB-Bio/test.jsonl" ]; then
    echo "   ⚠️  Dataset DB-Bio manquant"
    echo "   Lancement du téléchargement..."
    python download_datasets.py <<< "1"
    echo ""
else
    echo "   ✅ Dataset DB-Bio présent"
    echo ""
fi

# Étape 3 : Test rapide (1 exemple)
echo "🔬 Étape 3 : Test rapide (1 exemple)..."
echo ""
python eval_rupta_dbbio.py --split test --n_samples 1 --use_baseline --output test_baseline.json
echo ""

# Étape 4 : Proposer suite
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Test réussi ! Prochaines étapes :                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "1️⃣  Évaluation Baseline (10 exemples, ~30 secondes)"
echo "    python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline --output results_baseline.json"
echo ""
echo "2️⃣  Évaluation RUPTA (10 exemples, ~5 minutes)"
echo "    python eval_rupta_dbbio.py --split test --n_samples 10 --output results_rupta.json"
echo ""
echo "3️⃣  Comparaison Baseline vs RUPTA"
echo "    python compare_baseline_rupta.py --baseline results_baseline.json --rupta results_rupta.json --detailed"
echo ""
echo "📚 Documentation :"
echo "   - Guide complet : README_RUPTA.md"
echo "   - Quick start   : QUICKSTART_RUPTA.md"
echo "   - Python 3.11   : PYTHON_311_SOLUTION.md"
echo ""
echo "🎯 Exécuter maintenant ?"
echo "   1) Évaluation baseline (10 exemples)"
echo "   2) Évaluation RUPTA (10 exemples)"  
echo "   3) Les deux + comparaison"
echo "   4) Quitter"
echo ""
read -p "Votre choix (1-4) : " choice

case $choice in
    1)
        echo ""
        echo "🔄 Lancement évaluation baseline..."
        python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline --output results_baseline.json
        ;;
    2)
        echo ""
        echo "🔄 Lancement évaluation RUPTA..."
        python eval_rupta_dbbio.py --split test --n_samples 10 --output results_rupta.json
        ;;
    3)
        echo ""
        echo "🔄 Lancement évaluation complète..."
        python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline --output results_baseline.json
        echo ""
        python eval_rupta_dbbio.py --split test --n_samples 10 --output results_rupta.json
        echo ""
        python compare_baseline_rupta.py --baseline results_baseline.json --rupta results_rupta.json --detailed
        ;;
    4)
        echo "Au revoir ! 👋"
        exit 0
        ;;
    *)
        echo "Choix invalide. Au revoir !"
        exit 1
        ;;
esac

echo ""
echo "✅ Terminé !"
