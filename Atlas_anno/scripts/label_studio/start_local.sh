#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

if ! command -v label-studio >/dev/null 2>&1; then
  echo "[atlas] label-studio command not found in PATH" >&2
  exit 1
fi

echo "[atlas] Starting Label Studio from ${ROOT_DIR}"
cd "${ROOT_DIR}"
exec label-studio start "$@"
