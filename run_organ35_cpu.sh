#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2"
TS=$(date +%Y%m%d-%H%M%S)
PY=/root/miniconda3/bin/python
export PATH=/root/meme/bin:/root/miniconda3/bin:/usr/bin:/bin:/usr/sbin:/usr/local/bin
cd "$ROOT"

LOG=logs/run_organ35_cpu_${TS}.log
exec > >(tee -a "$LOG") 2>&1

echo "[INFO] backup outputs"
[ -d figs/outputs ] && mv figs/outputs figs/outputs_${TS}
[ -d data/derived/multi_model ] && mv data/derived/multi_model data/derived/multi_model_${TS}
mkdir -p figs/outputs data/derived/multi_model logs

echo "[INFO] backup defaults and link organ35"
for p in data/raw/predict_eval.csv data/raw/generated_topk.csv \
         data/derived/mfe_predict.csv data/derived/mfe_generated.csv \
         data/derived/rbp_predict.csv data/derived/rbp_generated.csv; do
  mv "$p" "$p.bak" 2>/dev/null || true
done
ln -sf predict_eval.organ35_liver.csv    data/raw/predict_eval.csv
ln -sf generated_topk.organ35_liver.csv  data/raw/generated_topk.csv
ln -sf mfe_predict.organ35_liver.csv     data/derived/mfe_predict.csv
ln -sf mfe_generated.organ35_liver.csv   data/derived/mfe_generated.csv
ln -sf rbp_predict.organ35_liver.csv     data/derived/rbp_predict.csv
ln -sf rbp_generated.organ35_liver.csv   data/derived/rbp_generated.csv

echo "[INFO] run eval/fig2"
$PY scripts/eval_predictor.py --pred data/raw/predict_eval.csv --mfe data/derived/mfe_predict.csv --rbp data/derived/rbp_predict.csv --outdir figs/outputs
$PY scripts/plot_fig2_predictor.py --input data/raw/predict_eval.csv --outdir figs/outputs
$PY scripts/plot_fig2_rbp_multi.py --input data/derived/all_models_rbp_features.csv --outdir figs/outputs || true

echo "[INFO] prepare multi-model features"
$PY scripts/prepare_multi_model_features.py
ln -sf /root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2/data/raw/m2_cvae_scored.organ35_liver.csv data/raw/m2_cvae_scored.csv
ln -sf /root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2/data/raw/m2_cgan_scored.organ35_liver.csv data/raw/m2_cgan_scored.csv
ln -sf /root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2/data/raw/m3_rl_scored.organ35_liver.csv data/raw/m3_rl_scored.csv
$PY scripts/prepare_multi_model_rbp_features.py

echo "[INFO] plot fig3"
$PY scripts/plot_fig3_generation.py --gen data/raw/generated_topk.csv --mfe data/derived/mfe_generated.csv --rbp data/derived/rbp_generated.csv --outdir figs/outputs
$PY scripts/plot_fig3_multi_models.py --outdir figs/outputs

echo "[INFO] final candidates"
$PY scripts/select_final_candidates.py --out-dir data/derived/multi_model

echo "[INFO] rename outputs with organ35_liver"
cd "$ROOT"
for f in data/derived/multi_model/*.csv; do base=$(basename "$f" .csv); mv -f "$f" data/derived/multi_model/${base}.organ35_liver.csv; done
cd figs/outputs
for img in fig2A_scatter fig2B_residual_hist fig2_rbp_hits_per_kb fig2_rbp_hits_total fig2_rbp_hits_vs_score \
           fig3A_gc fig3A_mfe_utr3 fig3A_mfe_utr5 fig3A_rbp_hits_total fig3B_novelty_vs_pred \
           fig3_multi_gc fig3_multi_mfe_utr3 fig3_multi_mfe_utr5 fig3_multi_novelty_vs_score \
           fig_score_pred_all_models fig_score_pred_real_vs_generated; do
  [ -f ${img}.png ] && mv -f ${img}.png ${img}.organ35_liver.png
  [ -f ${img}.svg ] && mv -f ${img}.svg ${img}.organ35_liver.svg
done

cd "$ROOT"
echo "[INFO] restore defaults"
for p in data/raw/predict_eval data/raw/generated_topk data/derived/mfe_predict data/derived/mfe_generated data/derived/rbp_predict data/derived/rbp_generated; do
  [ -f $p.bak ] && mv $p.bak $p
  [ -L $p ] && rm -f $p  # remove symlink if still there
  [ -f $p.bak ] && mv $p.bak $p
  # handle cases where symlink removed; rename .bak back
  [ -f $p.bak ] && mv $p.bak $p
  mv $p.bak $p 2>/dev/null || true
  [ -L $p ] && rm -f $p
  mv $p.bak $p 2>/dev/null || true
  mv $p.bak $p 2>/dev/null || true
done

echo "[OK] organ35 CPU pipeline done. Outputs in figs/outputs (with organ35_liver suffix) and data/derived/multi_model/*.organ35_liver.csv. Backups figs/outputs_${TS}, data/derived/multi_model_${TS}. Log: $LOG"
