#!/usr/bin/env python3
"""
Plot I-score distributions and feature profiles by I quantiles.
Assumes compute_I_score.py has produced a CSV with I_score column.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    sub = df["I_score"].dropna()
    if sub.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(sub, bins=80, density=True, histtype="step", linewidth=1.5, color="#1f77b4")
    plt.xlabel("I_score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=250)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close()


def plot_box_by_quantile(df: pd.DataFrame, feature: str, out_path: Path, qs: List[float]) -> None:
    if feature not in df.columns:
        print(f"[WARN] missing {feature}")
        return
    df = df.dropna(subset=["I_score", feature])
    if df.empty:
        return
    cuts = df["I_score"].quantile(qs).tolist()
    bins = [-np.inf] + cuts + [np.inf]
    df["I_bin"] = pd.cut(df["I_score"], bins=bins, labels=[f"Q{i+1}" for i in range(len(bins) - 1)])
    data = [df[df["I_bin"] == label][feature] for label in df["I_bin"].cat.categories]
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=df["I_bin"].cat.categories, notch=True)
    plt.ylabel(feature)
    plt.xlabel("I_score bins")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=250)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/derived/multi_model_/all_models_features_with_I.csv")
    ap.add_argument("--outdir", default="figs/outputs")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.input)
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    plot_distribution(df, outdir / "fig_I_distribution")
    for feat in ("gc", "mfe_utr5", "mfe_utr3", "rbp_hits_per_kb"):
        plot_box_by_quantile(df, feat, outdir / f"fig_I_box_{feat}", qs=[0.25, 0.5, 0.75])

    print(f"[OK] wrote I-score plots to {outdir}")


if __name__ == "__main__":
    main()
