#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORG_ID=${ORG_ID:-0}
SLUG=${ORG_SLUG:-organ0}
OUT_DIR=${OUT_DIR:-"${ROOT}/outputs/phase2/m1/${SLUG}"}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}

${PYTHON_BIN} -m src.gen.mutate_search \
    --config "${ROOT}/configs/m1_search.yaml" \
    --predict-config "${ROOT}/configs/gen_predict.yaml" \
    --target-organ "${ORG_ID}" \
    --out-dir "${OUT_DIR}" \
    "$@"
