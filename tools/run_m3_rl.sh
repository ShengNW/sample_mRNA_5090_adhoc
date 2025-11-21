#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG=${ORG_SLUG:-organ0}
OUT_DIR=${OUT_DIR:-"${ROOT}/outputs/phase2/m3_rl/${SLUG}"}
ORG_ID=${ORG_ID:-0}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}

${PYTHON_BIN} -m src.gen.train_rl \
    --config "${ROOT}/configs/m3_rl.yaml" \
    --predict-config "${ROOT}/configs/gen_predict.yaml" \
    --out-dir "${OUT_DIR}" \
    --target-organ "${ORG_ID}" \
    "$@"
