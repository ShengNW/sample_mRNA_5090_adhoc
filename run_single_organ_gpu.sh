#!/usr/bin/env bash
set -euo pipefail

ORG_ID=${1:? "Usage: run_single_organ_gpu.sh ORG_ID ORG_NAME"}
ORG_NAME=${2:-}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
export PYTHON PYTHON_BIN=${PYTHON}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

export ORG_ID ORG_NAME
export PATH="/root/miniconda3/bin:${PATH}"

SLUG=$(${PYTHON} - <<'PY'
from pathlib import Path
import os, sys
root = Path(__file__).resolve().parent
sys.path.append(str(root / "scripts"))
from organ_utils import resolve_organ
oid = int(os.environ["ORG_ID"])
oname = os.environ.get("ORG_NAME")
_, _, slug = resolve_organ(root, oid, oname)
print(slug)
PY
)

LOG_FILE="${LOG_DIR}/${SLUG}_gpu_$(date +%Y%m%d-%H%M).log"

(
    set -x
    make organ35_liver_m1_m2_m3 ORG35_ID="${ORG_ID}" ORG35_NAME="${ORG_NAME}" ORG35_SLUG="${SLUG}"
    make organ35_liver_build_tables ORG35_ID="${ORG_ID}" ORG35_NAME="${ORG_NAME}" ORG35_SLUG="${SLUG}"
    make organ35_liver_single_story ORG35_ID="${ORG_ID}" ORG35_NAME="${ORG_NAME}" ORG35_SLUG="${SLUG}"
    bash "${ROOT}/tools/sync_to_cpu.sh" "${ORG_ID}" "${ORG_NAME}"
) 2>&1 | tee "${LOG_FILE}"

echo "[OK] GPU pipeline finished, log saved to ${LOG_FILE}"
