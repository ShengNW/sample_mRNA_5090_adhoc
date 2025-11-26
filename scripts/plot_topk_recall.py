#!/usr/bin/env python3
"""
Top-K recall curves: fraction of true high-expression samples captured.
Inputs: CSV with y_true and ranking column (score_pred or I_score).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_curve(df: pd.DataFrame, score_col: str, label_col: str, ks: List[int]) -> pd.DataFrame:
    df = df.dropna(subset=[score_col, label_col]).copy()
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["is_high"] = (df[label_col] >= df[label_col].quantile(0.9)).astype(int)
    curves = []
    total_high = df["is_high"].sum()
    if total_high == 0:
        raise RuntimeError("No high-expression samples found for recall computation.")
    for k in ks:
        top = df.iloc[: min(k, len(df))]
        recall = top["is_high"].sum() / total_high
        curves.append({"k": k, "recall": recall})
    return pd.DataFrame(curves)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/predict_eval.csv")
    ap.add_argument("--score-col", default="y_pred")
    ap.add_argument("--label-col", default="y_true")
    ap.add_argument("--ks", default="50,100,200,500,1000")
    ap.add_argument("--outdir", default="figs/outputs")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x]
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.input)
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    curve = compute_curve(df, args.score_col, args.label_col, ks)
    plt.figure(figsize=(6, 4))
    plt.plot(curve["k"], curve["recall"], marker="o")
    plt.xlabel("Top-K")
    plt.ylabel("Recall of high-expression")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_fig = outdir / "fig_topk_recall.png"
    plt.savefig(out_fig, dpi=250)
    plt.savefig(out_fig.with_suffix(".svg"))
    curve.to_csv(outdir / "fig_topk_recall.csv", index=False)
    print(f"[OK] wrote top-K recall plot to {out_fig}")


if __name__ == "__main__":
    main()
