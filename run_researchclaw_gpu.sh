#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_BIN="/home/toor/miniconda3/envs/ano/bin"
NODE_BIN="/home/toor/.nvm/versions/node/v24.13.1/bin"
TOPIC="Improve the PipeGraph anonymization pipeline using arc_pipegraph.evaluate_candidate on local eval datasets"

usage() {
  cat <<'EOF'
Usage:
  ./run_researchclaw_gpu.sh [researchclaw run options]
  ./run_researchclaw_gpu.sh --resume-last [researchclaw run options]
  ./run_researchclaw_gpu.sh --resume-run artifacts/<run-id> [researchclaw run options]

Wrapper options:
  --resume-last          Resume the newest artifacts/rc-*/ checkpoint on GPU.
  --resume-run DIR       Resume a specific run directory on GPU.
  --reset-acp-session    Close the persistent ACP Codex session before launch.
  -h, --help             Show this help.

Native resume equivalents:
  ./run_researchclaw_gpu.sh --resume --output artifacts/<run-id>
  ./run_researchclaw_gpu.sh --from-stage RESOURCE_PLANNING --output artifacts/<run-id>
EOF
}

find_latest_checkpoint_run() {
  "${CONDA_ENV_BIN}/python" - <<'PY'
from pathlib import Path
import sys

runs = [
    path.parent
    for path in Path("artifacts").glob("rc-*/checkpoint.json")
    if path.is_file()
]
if not runs:
    sys.exit("No artifacts/rc-*/checkpoint.json found.")

runs.sort(key=lambda p: (p / "checkpoint.json").stat().st_mtime, reverse=True)
print(runs[0])
PY
}

resume_dir=""
resume_requested=0
reset_acp_session=0
user_set_output=0
run_args=()

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --resume-last)
      resume_requested=1
      shift
      ;;
    --resume-run)
      if (($# < 2)); then
        echo "Missing value for --resume-run" >&2
        exit 2
      fi
      resume_requested=1
      resume_dir="$2"
      shift 2
      ;;
    --reset-acp-session)
      reset_acp_session=1
      shift
      ;;
    -o|--output)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        exit 2
      fi
      user_set_output=1
      run_args+=("$1" "$2")
      shift 2
      ;;
    --output=*)
      user_set_output=1
      run_args+=("$1")
      shift
      ;;
    *)
      run_args+=("$1")
      shift
      ;;
  esac
done

if ((resume_requested)); then
  if ((user_set_output)); then
    echo "--resume-last/--resume-run already choose --output; do not pass --output separately." >&2
    exit 2
  fi
  if [[ -z "${resume_dir}" ]]; then
    resume_dir="$(find_latest_checkpoint_run)"
  fi
  if [[ ! -f "${resume_dir}/checkpoint.json" ]]; then
    echo "No checkpoint.json found in resume directory: ${resume_dir}" >&2
    exit 2
  fi
  run_args=(--resume --output "${resume_dir}" "${run_args[@]}")
fi

export PATH="/usr/lib/wsl/lib:${NODE_BIN}:${CONDA_ENV_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NER_FORCE_DEVICE="${NER_FORCE_DEVICE:-cuda}"
export NER_HALF_PRECISION="${NER_HALF_PRECISION:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

mkdir -p "${MPLCONFIGDIR}"
cd "${ROOT_DIR}"

if ((reset_acp_session)); then
  echo "[acp] closing existing Codex ACP session 'researchclaw' if present"
  acpx --ttl 0 --cwd "${ROOT_DIR}" codex sessions close researchclaw >/dev/null 2>&1 || true
fi

echo "[gpu] nvidia-smi: $(command -v nvidia-smi)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

"${CONDA_ENV_BIN}/python" - <<'PY'
import torch

print(f"[gpu] torch={torch.__version__}")
print(f"[gpu] cuda_available={torch.cuda.is_available()}")
print(f"[gpu] cuda_version={torch.version.cuda}")
print(f"[gpu] device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[gpu] device_name={torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA is not available; aborting ResearchClaw GPU run.")
PY

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "[ollama] API not reachable at http://localhost:11434; OpenRouter can still be used for LLM." >&2
fi

exec researchclaw run \
  --config arc_pipegraph/config.arc.yaml \
  --topic "${TOPIC}" \
  --auto-approve \
  "${run_args[@]}"
