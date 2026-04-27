#!/usr/bin/env bash
# =============================================================================
# Full Ablation Study — PipeGraph + OpenRouter
# =============================================================================
# Usage:
#   bash run_ablation_full.sh              # run tout
#   bash run_ablation_full.sh --quick      # limit=10 sur chaque config (test rapide)
#   bash run_ablation_full.sh --phase ner  # seulement une phase (nodes|ner|full)
#
# Résultats : eval/evaluation/reports/ablation_*.json
#             eval/evaluation/runs/ablation_*.json (si --save-runs)
# Log       : logs/ablation_<timestamp>.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — modifier ici
# ---------------------------------------------------------------------------
PYTHON="python"                          # ou "python3", "poetry run python", etc.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
# Lire le modèle depuis config.json (openrouter.model → llm.model → default)
EXPORT_MODEL=$($PYTHON -c "
import json, sys
try:
    c = json.load(open('pipegraph/config.json'))
    m = c.get('openrouter', {}).get('model') or c.get('llm', {}).get('model') or 'google/gemma-3-27b-it'
    print(m)
except Exception:
    print('google/gemma-3-27b-it')
" 2>/dev/null || echo "google/gemma-3-27b-it")
LIMIT=300                                 # docs par config (réduire pour test rapide)
LIMIT_NER=30                             # docs pour les suites NER lourdes (ensemble 14 configs)
SAVE_RUNS="--save-runs"                  # "" pour désactiver
LOG_DIR="${PROJECT_ROOT}/logs"

# Phases à exécuter : "nodes ner full"  (retirer celles que vous voulez sauter)
PHASES="nodes ner full"

# ---------------------------------------------------------------------------
# Arguments CLI
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case $arg in
    --quick)
      LIMIT=10
      LIMIT_NER=10
      echo "⚡ Mode --quick : limit=${LIMIT}"
      ;;
    --phase)
      shift
      PHASES="$1"
      ;;
    --phase=*)
      PHASES="${arg#*=}"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOG_DIR}/ablation_${TIMESTAMP}.log"

# Charger les variables du .env dans l'environnement shell
# (nécessaire pour que os.environ.get() trouve OPENROUTER_API_KEY)
ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  echo "  ✅ Variables .env chargées depuis ${ENV_FILE}"
else
  echo "  ⚠️  Fichier .env introuvable : ${ENV_FILE}"
fi

export PYTHONPATH="${PROJECT_ROOT}"
export OPENROUTER_MODEL="${EXPORT_MODEL}"
export PYTHONUNBUFFERED=1   # Force unbuffered stdout — critical for tee visibility

cd "${PROJECT_ROOT}"

# Couleurs
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log() { echo -e "$1" | tee -a "${LOGFILE}"; }
log_section() { log "\n${CYAN}══════════════════════════════════════════════════════${NC}"; log "${CYAN}  $1${NC}"; log "${CYAN}══════════════════════════════════════════════════════${NC}"; }
log_ok()   { log "${GREEN}  ✅ $1${NC}"; }
log_warn() { log "${YELLOW}  ⚠️  $1${NC}"; }
log_err()  { log "${RED}  ❌ $1${NC}"; }

run_ablation() {
  local suite="$1"
  local dataset="$2"
  local limit="$3"
  local extra_args="${4:-}"
  local label="${suite}/${dataset}${extra_args:+ (${extra_args})}"

  log "\n  ▶ ${label}"
  local t0=$SECONDS

  if ${PYTHON} eval/run_ablation.py \
      --suite "${suite}" \
      --dataset "${dataset}" \
      --limit "${limit}" \
      --with-llm \
      ${SAVE_RUNS} \
      ${extra_args} \
      2>&1 | tee -a "${LOGFILE}"; then
    local elapsed=$(( SECONDS - t0 ))
    log_ok "${label} — ${elapsed}s"
    return 0
  else
    log_err "${label} — FAILED (voir ${LOGFILE})"
    return 1
  fi
}

TOTAL=0; PASSED=0; FAILED=0
FAILED_RUNS=()

run_tracked() {
  TOTAL=$(( TOTAL + 1 ))
  if run_ablation "$@"; then
    PASSED=$(( PASSED + 1 ))
  else
    FAILED=$(( FAILED + 1 ))
    FAILED_RUNS+=("$*")
  fi
}

# ---------------------------------------------------------------------------
log_section "Ablation Study PipeGraph — $(date)"
log "  Provider     : OpenRouter"
log "  Model        : ${EXPORT_MODEL}"
log "  Limit        : ${LIMIT} (NER: ${LIMIT_NER})"
log "  Phases       : ${PHASES}"
log "  Log          : ${LOGFILE}"
log "  Project root : ${PROJECT_ROOT}"

# Vérifier que la clé OpenRouter est définie
if grep -q "^OPENROUTER_API_KEY=sk-" "${PROJECT_ROOT}/.env" 2>/dev/null; then
  log_ok "OPENROUTER_API_KEY trouvée dans .env"
else
  log_warn "OPENROUTER_API_KEY introuvable dans .env — les appels LLM échoueront"
fi

T_START=$SECONDS

# =============================================================================
# PHASE 1 — NODES  (contribution de chaque nœud sur tous les datasets)
# Rapide : 5 configs × 6 datasets = 30 runs
# =============================================================================
if [[ "${PHASES}" == *"nodes"* ]]; then
  log_section "PHASE 1 : Suite 'nodes' — tous les datasets"

  run_tracked nodes tab              "${LIMIT}"
  run_tracked nodes dbbio            "${LIMIT}"
  run_tracked nodes anonymization    "${LIMIT}"
  run_tracked nodes ratbench         "${LIMIT}" "--level 1"
  run_tracked nodes ratbench         "${LIMIT}" "--level 2"
  run_tracked nodes ratbench         "${LIMIT}" "--level 3"
fi

# =============================================================================
# PHASE 2 — NER  (presets, ensemble, threshold, vote)
# Lourd : ~40 configs NER × 2 datasets = long
# On fait TAB + RAT-Bench L1 comme datasets représentatifs
# =============================================================================
if [[ "${PHASES}" == *"ner"* ]]; then
  log_section "PHASE 2 : Suites NER — TAB + RAT-Bench L1"

  for dataset in tab ratbench; do
    extra=""
    [[ "${dataset}" == "ratbench" ]] && extra="--level 1"

    run_tracked ner_presets   "${dataset}" "${LIMIT}"     "${extra}"
    run_tracked ner_ensemble  "${dataset}" "${LIMIT_NER}" "${extra}"
    run_tracked ner_threshold "${dataset}" "${LIMIT_NER}" "${extra}"
    run_tracked ner_vote      "${dataset}" "${LIMIT_NER}" "${extra}"
  done
fi

# =============================================================================
# PHASE 3 — FULL  (toutes les suites d'un seul coup sur TAB)
# Utile pour avoir un rapport consolidé unique
# =============================================================================
if [[ "${PHASES}" == *"full"* ]]; then
  log_section "PHASE 3 : Suite 'full' — TAB (rapport consolidé)"
  run_tracked full tab "${LIMIT}"

  log_section "PHASE 3 : Suite 'anon_strategy' — tous les datasets"
  run_tracked anon_strategy tab           "${LIMIT}"
  run_tracked anon_strategy anonymization "${LIMIT}"
  run_tracked anon_strategy ratbench      "${LIMIT}" "--level 1"
fi

# =============================================================================
# Résumé final
# =============================================================================
T_TOTAL=$(( SECONDS - T_START ))
T_MIN=$(( T_TOTAL / 60 ))
T_SEC=$(( T_TOTAL % 60 ))

log_section "Résumé final"
log "  Durée totale : ${T_MIN}m ${T_SEC}s"
log "  Runs total   : ${TOTAL}"
log_ok "Succès : ${PASSED}"
if [[ ${FAILED} -gt 0 ]]; then
  log_err "Échecs  : ${FAILED}"
  for f in "${FAILED_RUNS[@]}"; do
    log_err "    → ${f}"
  done
else
  log_ok "Aucun échec 🎉"
fi

log "\n  📁 Rapports : ${PROJECT_ROOT}/eval/evaluation/reports/"
log "  📁 Runs     : ${PROJECT_ROOT}/eval/evaluation/runs/"
log "  📋 Log      : ${LOGFILE}"
log ""

exit $(( FAILED > 0 ? 1 : 0 ))
