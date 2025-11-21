#!/usr/bin/env bash
set -euo pipefail

# Orchestrate M1/M2/M3 generation + sampling + scoring for a single organ.

ORG_ID=${1:? "Usage: run_gpu_models.sh ORG_ID ORG_NAME [ORG_SLUG]"}
ORG_NAME=${2:-}
ORG_SLUG=${3:-}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT}/data/raw"
mkdir -p "${RAW_DIR}"

export ORG_ID ORG_NAME

if [[ -z "${ORG_SLUG}" ]]; then
    ORG_SLUG=$(python3 - <<'PY'
from pathlib import Path
from organ_utils import resolve_organ
import os, sys
root = Path(__file__).resolve().parents[1]
oid = int(os.environ.get("ORG_ID", "0"))
oname = os.environ.get("ORG_NAME")
_, _, slug = resolve_organ(root, oid, oname)
print(slug)
PY
    )
fi

M1_DIR="${ROOT}/outputs/phase2/m1/${ORG_SLUG}"
M2_CVAE_DIR="${ROOT}/outputs/phase2/m2_cvae/${ORG_SLUG}"
M2_CGAN_DIR="${ROOT}/outputs/phase2/m2_cgan/${ORG_SLUG}"
M3_DIR="${ROOT}/outputs/phase2/m3_rl/${ORG_SLUG}"

NUM_SAMPLES=${NUM_SAMPLES:-5000}

find_latest() {
    local pattern=$1
    local latest
    latest=$(ls -1 ${pattern} 2>/dev/null | sort | tail -n 1 || true)
    echo "${latest:-}"
}

score_csv() {
    local input=$1
    local output=$2
    if [[ ! -s "${input}" ]]; then
        echo "[WARN] skip scoring missing ${input}"
        return
    fi
    python3 -m src.side.predict \
        --config "${ROOT}/configs/gen_predict.yaml" \
        --input "${input}" \
        --out "${output}"
}

export ORG_ID ORG_NAME ORG_SLUG

echo "[STEP] M1 mutate-search for ${ORG_SLUG}"
bash "${ROOT}/tools/run_m1.sh" \
    --target-organ "${ORG_ID}" \
    --out-dir "${M1_DIR}"

echo "[STEP] Scoring M1 top-k"
score_csv "${M1_DIR}/m1_topk.csv" "${RAW_DIR}/m1_scored.${ORG_SLUG}.csv"

echo "[STEP] Train M2 cVAE"
bash "${ROOT}/tools/run_m2_cvae.sh" \
    --out-dir "${M2_CVAE_DIR}" \
    --train_csv "${M1_DIR}/m1_topk.csv"

CVAE_CKPT=$(find_latest "${M2_CVAE_DIR}/cvae_epoch*.pt")
if [[ -n "${CVAE_CKPT}" ]]; then
    echo "[STEP] Sample cVAE using ${CVAE_CKPT}"
    python3 "${ROOT}/scripts/sample_cvae.py" \
        --config "${ROOT}/configs/m2_cvae.yaml" \
        --weights "${CVAE_CKPT}" \
        --organ-id "${ORG_ID}" \
        --num-samples "${NUM_SAMPLES}" \
        --out "${M2_CVAE_DIR}/samples.csv"
    score_csv "${M2_CVAE_DIR}/samples.csv" "${RAW_DIR}/m2_cvae_scored.${ORG_SLUG}.csv"
else
    echo "[WARN] No CVAE checkpoint found under ${M2_CVAE_DIR}"
fi

echo "[STEP] Train M2 cGAN"
bash "${ROOT}/tools/run_m2_cgan.sh" \
    --out-dir "${M2_CGAN_DIR}" \
    --seed_csv "${M1_DIR}/m1_topk.csv"

CGAN_CKPT=$(find_latest "${M2_CGAN_DIR}/cganG_epoch*.pt")
if [[ -n "${CGAN_CKPT}" ]]; then
    echo "[STEP] Sample cGAN using ${CGAN_CKPT}"
    python3 "${ROOT}/scripts/sample_cgan.py" \
        --config "${ROOT}/configs/m2_cgan.yaml" \
        --weights "${CGAN_CKPT}" \
        --organ-id "${ORG_ID}" \
        --num-samples "${NUM_SAMPLES}" \
        --out "${M2_CGAN_DIR}/samples.csv"
    score_csv "${M2_CGAN_DIR}/samples.csv" "${RAW_DIR}/m2_cgan_scored.${ORG_SLUG}.csv"
else
    echo "[WARN] No CGAN checkpoint found under ${M2_CGAN_DIR}"
fi

echo "[STEP] Train M3 RL"
bash "${ROOT}/tools/run_m3_rl.sh" \
    --out-dir "${M3_DIR}" \
    --target-organ "${ORG_ID}"

if [[ -s "${M3_DIR}/samples.csv" ]]; then
    echo "[STEP] Score RL samples"
    score_csv "${M3_DIR}/samples.csv" "${RAW_DIR}/m3_rl_scored.${ORG_SLUG}.csv"
else
    echo "[WARN] RL samples.csv missing under ${M3_DIR}"
fi
