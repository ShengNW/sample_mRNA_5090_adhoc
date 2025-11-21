#!/usr/bin/env bash
set -euo pipefail

ORG_ID=${1:? "Usage: sync_to_cpu.sh ORG_ID ORG_NAME"}
ORG_NAME=${2:-}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

export ORG_ID ORG_NAME

SLUG=$(${PYTHON} - <<'PY'
from pathlib import Path
import os, sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "scripts"))
from organ_utils import resolve_organ
oid = int(os.environ["ORG_ID"])
oname = os.environ.get("ORG_NAME")
_, _, slug = resolve_organ(root, oid, oname)
print(slug)
PY
)

CPU_HOST=${CPU_HOST:-}
CPU_PATH=${CPU_PATH:-}
CPU_PORT=${CPU_PORT:-22}

LOG_FILE="${LOG_DIR}/${SLUG}_sync_$(date +%Y%m%d-%H%M).log"

if [[ -z "${CPU_HOST}" || -z "${CPU_PATH}" ]]; then
    echo "[ERROR] CPU_HOST/CPU_PATH not set; export them before syncing." | tee "${LOG_FILE}"
    exit 1
fi

FILES=(
    "${ROOT}/data/raw/predict_eval.${SLUG}.csv"
    "${ROOT}/data/raw/generated_topk.${SLUG}.csv"
    "${ROOT}/data/raw/m1_scored.${SLUG}.csv"
    "${ROOT}/data/raw/m2_cvae_scored.${SLUG}.csv"
    "${ROOT}/data/raw/m2_cgan_scored.${SLUG}.csv"
    "${ROOT}/data/raw/m3_rl_scored.${SLUG}.csv"
)

EXISTING=()
for f in "${FILES[@]}"; do
    if [[ -s "${f}" ]]; then
        EXISTING+=("${f}")
    else
        echo "[WARN] missing ${f}, skip" | tee -a "${LOG_FILE}"
    fi
done

if [[ ${#EXISTING[@]} -eq 0 ]]; then
    echo "[ERROR] nothing to sync for ${SLUG}" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "[INFO] Syncing to ${CPU_HOST}:${CPU_PATH} (port ${CPU_PORT})" | tee -a "${LOG_FILE}"
rsync -av --progress -e "ssh -p ${CPU_PORT}" "${EXISTING[@]}" "${CPU_HOST}:${CPU_PATH}/" | tee -a "${LOG_FILE}"

echo "[OK] Sync complete, log at ${LOG_FILE}"
