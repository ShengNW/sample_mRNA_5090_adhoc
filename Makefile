# 简易工作流
.PHONY: mfe rbp fig2 fig3 all organ35_liver_m1_m2_m3 organ35_liver_build_tables organ35_liver_single_story

PY=python
RBP_MOTIFS=data/external/rbp/Ray2013_rbp_Homo_sapiens.meme

mfe:
	$(PY) -u scripts/calc_mfe.py --input data/raw/predict_eval.csv --output data/derived/mfe_predict.csv
	$(PY) -u scripts/calc_mfe.py --input data/raw/generated_topk.csv --output data/derived/mfe_generated.csv

rbp:
	$(PY) scripts/scan_rbp_fimo.py --input data/raw/predict_eval.csv --motifs $(RBP_MOTIFS) --output data/derived/rbp_predict.csv
	$(PY) scripts/scan_rbp_fimo.py --input data/raw/generated_topk.csv --motifs $(RBP_MOTIFS) --output data/derived/rbp_generated.csv

fig2:
	$(PY) scripts/eval_predictor.py --pred data/raw/predict_eval.csv --mfe data/derived/mfe_predict.csv --rbp data/derived/rbp_predict.csv --outdir figs/outputs
	$(PY) scripts/plot_fig2_predictor.py --input data/raw/predict_eval.csv --outdir figs/outputs

fig3:
	$(PY) scripts/plot_fig3_generation.py --gen data/raw/generated_topk.csv --mfe data/derived/mfe_generated.csv --rbp data/derived/rbp_generated.csv --outdir figs/outputs

all: mfe rbp fig2 fig3

# --- Single-organ GPU pipeline (Liver / organ_id=35) ---
ORG35_ID ?= 35
ORG35_NAME ?= Liver
ORG35_SLUG ?= organ35_liver

organ35_liver_m1_m2_m3:
	@bash scripts/run_gpu_models.sh $(ORG35_ID) "$(ORG35_NAME)" $(ORG35_SLUG)

organ35_liver_build_tables:
	$(PY) scripts/build_single_organ_tables.py --organ-id $(ORG35_ID) --organ-name "$(ORG35_NAME)" --slug $(ORG35_SLUG) --predict-eval data/raw/predict_eval.csv --m1-scored data/raw/m1_scored.$(ORG35_SLUG).csv --m2-cvae-scored data/raw/m2_cvae_scored.$(ORG35_SLUG).csv --m2-cgan-scored data/raw/m2_cgan_scored.$(ORG35_SLUG).csv --m3-rl-scored data/raw/m3_rl_scored.$(ORG35_SLUG).csv --out-dir data/raw

organ35_liver_single_story:
	$(PY) scripts/step1_single_organ_wrapup.py --organ-id $(ORG35_ID) --organ-name "$(ORG35_NAME)" --suffix $(ORG35_SLUG) --predict-eval data/raw/predict_eval.$(ORG35_SLUG).csv --generated-topk data/raw/generated_topk.$(ORG35_SLUG).csv --fig-dir figs/outputs --out-csv-dir outputs/phase2/m1
