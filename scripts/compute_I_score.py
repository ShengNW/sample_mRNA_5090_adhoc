#!/usr/bin/env python3
"""
Compute single-axis I-score:
  I = target_pred - off_target - penalty
Penalty = lambda_novelty*novelty + lambda_rbp*rbp_hits_per_kb + lambda_gc*|gc-0.5|
        + lambda_mfe*mean(|mfe_utr5|, |mfe_utr3|)
If off_target not provided, defaults to 0.
Outputs augmented CSV with I column.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/derived/multi_model_/all_models_features.csv")
    ap.add_argument("--rbp", default="data/derived/all_models_rbp_features.csv")
    ap.add_argument("--target-col", default="score_pred")
    ap.add_argument("--offtarget-col", default=None)
    ap.add_argument("--lambda-novelty", type=float, default=0.2)
    ap.add_argument("--lambda-rbp", type=float, default=0.05)
    ap.add_argument("--lambda-gc", type=float, default=0.5)
    ap.add_argument("--lambda-mfe", type=float, default=0.001)
    ap.add_argument("--out", default="data/derived/multi_model_/all_models_features_with_I.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.features)
    if args.rbp and Path(root / args.rbp).exists():
        rbp = pd.read_csv(root / args.rbp)
        df = df.merge(rbp[["seq_id", "rbp_hits_per_kb"]], on="seq_id", how="left")

    if args.target_col not in df.columns:
        raise RuntimeError(f"target column {args.target_col} not in table")
    target = pd.to_numeric(df[args.target_col], errors="coerce")
    off = pd.Series(0.0, index=df.index)
    if args.offtarget_col and args.offtarget_col in df.columns:
        off = pd.to_numeric(df[args.offtarget_col], errors="coerce").fillna(0.0)

    gc_pen = args.lambda_gc * (df["gc"].fillna(0.0) - 0.5).abs() if "gc" in df.columns else 0.0
    mfe_terms = []
    if "mfe_utr5" in df.columns:
        mfe_terms.append(df["mfe_utr5"].abs())
    if "mfe_utr3" in df.columns:
        mfe_terms.append(df["mfe_utr3"].abs())
    if mfe_terms:
        mfe_pen = args.lambda_mfe * sum(mfe_terms) / len(mfe_terms)
    else:
        mfe_pen = 0.0
    rbp_pen = args.lambda_rbp * df.get("rbp_hits_per_kb", 0.0).fillna(0.0)
    novelty_pen = args.lambda_novelty * df.get("novelty", 0.0).fillna(0.0)

    penalty = gc_pen + mfe_pen + rbp_pen + novelty_pen
    I = target.fillna(0.0) - off.fillna(0.0) - penalty
    df["I_score"] = I
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} with I_score")


if __name__ == "__main__":
    main()
