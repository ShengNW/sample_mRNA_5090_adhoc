#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG=${ORG_SLUG:-organ0}
OUT_DIR=${OUT_DIR:-"${ROOT}/outputs/phase2/m2_cgan/${SLUG}"}
SEED_CSV=${SEED_CSV:-"${ROOT}/outputs/phase2/m1/${SLUG}/m1_topk.csv"}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}

${PYTHON_BIN} -m src.gen.train_cgan \
    --config "${ROOT}/configs/m2_cgan.yaml" \
    --seed_csv "${SEED_CSV}" \
    --out-dir "${OUT_DIR}" \
    "$@"
