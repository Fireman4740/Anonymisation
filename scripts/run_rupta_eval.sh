#!/bin/bash
# Script de lancement rapide pour l'évaluation RUPTA

echo "=============================================="
echo "🚀 RUPTA Evaluation Launcher"
echo "=============================================="
echo ""

# Couleurs pour le terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction d'aide
show_help() {
    echo "Usage: ./scripts/run_rupta_eval.sh [command] [policy]"
    echo ""
    echo "Commands:"
    echo "  test          - Test rapide d'intégration (3 cas multilingues)"
    echo "  pilot         - Évaluation pilote (50 échantillons DB-Bio)"
    echo "  dbbio         - Évaluation DB-Bio complète"
    echo "  reddit        - Évaluation PersonalReddit complète"
    echo "  tab           - Évaluation TAB (ECHR) complète"
    echo "  all           - Évaluation de tous les datasets (dbbio, reddit, tab)"
    echo "  compare       - Générer rapport de comparaison"
    echo "  quick         - Test + Pilot + Compare (workflow rapide)"
    echo "  full          - Évaluation complète de tous les datasets"
    echo ""
    echo "Policy (optional, default=L1):"
    echo "  L0            - Baseline sans LLM (NER + regex uniquement)"
    echo "  L1            - Avec LLM + RUPTA (optimisation privacy-utility)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_rupta_eval.sh test"
    echo "  ./scripts/run_rupta_eval.sh tab L1"
    echo "  ./scripts/run_rupta_eval.sh dbbio L0"
    echo "  ./scripts/run_rupta_eval.sh quick"
    echo "  ./scripts/run_rupta_eval.sh all L1"
}

# Créer le dossier results s'il n'existe pas
mkdir -p results

# Policy par défaut
POLICY="${2:-L1}"

# Validation de la policy
if [ "$POLICY" != "L0" ] && [ "$POLICY" != "L1" ]; then
    echo -e "${YELLOW}⚠️  Policy invalide: $POLICY (utilisez L0 ou L1)${NC}"
    POLICY="L1"
fi

echo -e "${BLUE}Policy level: $POLICY${NC}"
echo ""

# Commandes
case "$1" in
    test)
        echo -e "${BLUE}📝 Test d'intégration RUPTA...${NC}"
        python scripts/test_rupta_integration.py
        ;;
    
    pilot)
        echo -e "${BLUE}🧪 Évaluation pilote (50 échantillons DB-Bio)...${NC}"
        python scripts/eval_rupta_pipeline.py \
            --dataset dbbio \
            --split test \
            --n_samples 50 \
            --use_baseline \
            --use_rupta \
            --policy "$POLICY" \
            --output results/pilot_dbbio_${POLICY}.json
        
        echo -e "${GREEN}✅ Résultats sauvegardés: results/pilot_dbbio_${POLICY}.json${NC}"
        ;;
    
    dbbio)
        echo -e "${BLUE}📊 Évaluation DB-Bio complète...${NC}"
        python scripts/eval_rupta_pipeline.py \
            --dataset dbbio \
            --split test \
            --n_samples 0 \
            --use_baseline \
            --use_rupta \
            --policy "$POLICY" \
            --output results/eval_dbbio_full_${POLICY}.json
        
        echo -e "${GREEN}✅ Résultats sauvegardés: results/eval_dbbio_full_${POLICY}.json${NC}"
        ;;
    
    reddit)
        echo -e "${BLUE}📊 Évaluation PersonalReddit complète...${NC}"
        python scripts/eval_rupta_pipeline.py \
            --dataset reddit \
            --split test \
            --n_samples 0 \
            --use_baseline \
            --use_rupta \
            --policy "$POLICY" \
            --output results/eval_reddit_full_${POLICY}.json
        
        echo -e "${GREEN}✅ Résultats sauvegardés: results/eval_reddit_full_${POLICY}.json${NC}"
        ;;
    
    tab)
        echo -e "${BLUE}📊 Évaluation TAB (ECHR) complète...${NC}"
        echo -e "${YELLOW}⚠️  Conversion du dataset TAB si nécessaire...${NC}"
        python scripts/convert_tab_dataset.py
        
        python scripts/eval_rupta_pipeline.py \
            --dataset tab \
            --split test \
            --n_samples 0 \
            --use_baseline \
            --use_rupta \
            --policy "$POLICY" \
            --output results/eval_tab_full_${POLICY}.json
        
        echo -e "${GREEN}✅ Résultats sauvegardés: results/eval_tab_full_${POLICY}.json${NC}"
        ;;
    
    all)
        echo -e "${BLUE}📊 Évaluation de tous les datasets (dbbio, reddit, tab)...${NC}"
        
        # Convertir TAB si nécessaire
        echo -e "${YELLOW}⚠️  Conversion du dataset TAB si nécessaire...${NC}"
        python scripts/convert_tab_dataset.py
        
        python scripts/eval_rupta_pipeline.py \
            --all \
            --split test \
            --n_samples 0 \
            --use_baseline \
            --use_rupta \
            --policy "$POLICY" \
            --output results/eval_all_datasets_${POLICY}.json
        
        echo -e "${GREEN}✅ Résultats sauvegardés: results/eval_all_datasets_${POLICY}.json${NC}"
        ;;
    
    compare)
        echo -e "${BLUE}📈 Génération du rapport de comparaison...${NC}"
        
        # Chercher le dernier fichier de résultats
        LATEST=$(ls -t results/eval_*.json 2>/dev/null | head -1)
        
        if [ -z "$LATEST" ]; then
            echo -e "${YELLOW}⚠️  Aucun fichier de résultats trouvé${NC}"
            echo "Lancez d'abord une évaluation (pilot, dbbio, reddit, ou all)"
            exit 1
        fi
        
        echo "Utilisation de: $LATEST"
        python scripts/compare_baseline_rupta.py \
            --results "$LATEST" \
            --output results/comparison_report.md
        
        echo -e "${GREEN}✅ Rapport généré: results/comparison_report.md${NC}"
        echo ""
        echo -e "${BLUE}📄 Aperçu du rapport:${NC}"
        head -n 30 results/comparison_report.md
        ;;
    
    quick)
        echo -e "${BLUE}⚡ Workflow rapide (Test + Pilot + Compare)${NC}"
        echo ""
        
        echo -e "${BLUE}1/3: Test d'intégration...${NC}"
        python scripts/test_rupta_integration.py
        echo ""
        
        echo -e "${BLUE}2/3: Évaluation pilote...${NC}"
        python scripts/eval_rupta_pipeline.py \
            --dataset dbbio \
            --split test \
            --n_samples 50 \
            --use_baseline \
            --use_rupta \
            --output results/pilot_dbbio.json
        echo ""
        
        echo -e "${BLUE}3/3: Génération du rapport...${NC}"
        python scripts/compare_baseline_rupta.py \
            --results results/pilot_dbbio.json \
            --output results/pilot_report.md
        
        echo ""
        echo -e "${GREEN}✅ Workflow rapide terminé!${NC}"
        echo -e "${BLUE}📄 Fichiers générés:${NC}"
        echo "  - results/pilot_dbbio.json"
        echo "  - results/pilot_report.md"
        ;;
    
    full)
        echo -e "${BLUE}🚀 Évaluation complète (tous datasets: dbbio, reddit, tab)${NC}"
        echo -e "${YELLOW}⚠️  Cela peut prendre 2-3 heures...${NC}"
        read -p "Continuer? (y/n) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Évaluation en cours...${NC}"
            
            # Convertir TAB
            echo "0/3: Conversion TAB..."
            python scripts/convert_tab_dataset.py
            
            # DB-Bio
            echo "1/3: DB-Bio..."
            python scripts/eval_rupta_pipeline.py \
                --dataset dbbio \
                --split test \
                --n_samples 0 \
                --use_baseline \
                --use_rupta \
                --output results/eval_dbbio_full.json
            
            # PersonalReddit
            echo "2/3: PersonalReddit..."
            python scripts/eval_rupta_pipeline.py \
                --dataset reddit \
                --split test \
                --n_samples 0 \
                --use_baseline \
                --use_rupta \
                --output results/eval_reddit_full.json
            
            # TAB
            echo "3/3: TAB (ECHR)..."
            python scripts/eval_rupta_pipeline.py \
                --dataset tab \
                --split test \
                --n_samples 0 \
                --use_baseline \
                --use_rupta \
                --output results/eval_tab_full.json
            
            # Rapports
            echo "Génération des rapports..."
            python scripts/compare_baseline_rupta.py \
                --results results/eval_dbbio_full.json \
                --output results/dbbio_report.md
            
            python scripts/compare_baseline_rupta.py \
                --results results/eval_reddit_full.json \
                --output results/reddit_report.md
            
            python scripts/compare_baseline_rupta.py \
                --results results/eval_tab_full.json \
                --output results/tab_report.md
            
            echo -e "${GREEN}✅ Évaluation complète terminée!${NC}"
            echo -e "${BLUE}📄 Fichiers générés:${NC}"
            echo "  - results/eval_dbbio_full.json + dbbio_report.md"
            echo "  - results/eval_reddit_full.json + reddit_report.md"
            echo "  - results/eval_tab_full.json + tab_report.md"
        fi
        ;;
    
    *)
        show_help
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✨ Terminé!${NC}"
