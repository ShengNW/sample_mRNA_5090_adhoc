#!/usr/bin/env python3
"""
Feature-level interpretability (Immunity-style):
  - High/low expression groups for RBP/kb, GC, MFE, tRNA distance (if present)
  - Scatter of feature vs expression
Inputs: all_models_features*.csv (+ all_models_rbp_features.csv)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_CANDIDATES = [
    "gc",
    "mfe_utr5",
    "mfe_utr3",
    "rbp_hits_per_kb",
    "rbp_hits_total",
    "trna_distance",
]


def load_features(root: Path, feat_path: Path, rbp_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(feat_path)
    if rbp_path and rbp_path.exists():
        rbp = pd.read_csv(rbp_path)
        df = df.merge(rbp[["seq_id", "rbp_hits_total", "rbp_hits_per_kb"]], on="seq_id", how="left")
    return df


def choose_expr_col(df: pd.DataFrame) -> str:
    for col in ("score_pred", "y_pred", "pred", "reward"):
        if col in df.columns:
            return col
    return "score_pred"


def plot_violin(df: pd.DataFrame, feature: str, label_col: str, out_path: Path) -> None:
    sub = df[[label_col, feature]].dropna()
    if sub.empty:
        print(f"[WARN] no data for {feature}")
        return
    groups = [sub[sub[label_col] == 0][feature], sub[sub[label_col] == 1][feature]]
    labels = ["Low", "High"]
    plt.figure(figsize=(6, 4))
    plt.violinplot(groups, showmeans=True, showextrema=True)
    plt.xticks([1, 2], labels)
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=250)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close()


def plot_scatter(df: pd.DataFrame, feature: str, expr_col: str, out_path: Path, max_points: int = 4000) -> None:
    sub = df[[feature, expr_col]].dropna()
    if sub.empty:
        return
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(sub[feature], sub[expr_col], s=8, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel(expr_col)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=250)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/derived/multi_model_/all_models_features.csv")
    ap.add_argument("--rbp", default="data/derived/all_models_rbp_features.csv")
    ap.add_argument("--outdir", default="figs/outputs")
    ap.add_argument("--organ", default=None, help="Filter organ_name for a specific organ")
    ap.add_argument("--high-q", type=float, default=0.8)
    ap.add_argument("--low-q", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = load_features(root, root / args.features, root / args.rbp)
    if args.organ and "organ_name" in df.columns:
        df = df[df["organ_name"].astype(str) == str(args.organ)]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    expr_col = choose_expr_col(df)
    expr = df[expr_col].dropna()
    if expr.empty:
        raise RuntimeError(f"No expression-like column found (using {expr_col})")
    low_cut = expr.quantile(args.low_q)
    high_cut = expr.quantile(args.high_q)
    df["expr_group"] = np.where(df[expr_col] >= high_cut, 1, np.where(df[expr_col] <= low_cut, 0, np.nan))
    df_hilo = df[~df["expr_group"].isna()].copy()

    for feat in FEATURE_CANDIDATES:
        if feat not in df_hilo.columns:
            continue
        plot_violin(df_hilo, feat, "expr_group", outdir / f"fig_feature_violin_{feat}")
        plot_scatter(df, feat, expr_col, outdir / f"fig_feature_scatter_{feat}")

    print(f"[OK] wrote feature interpretability plots to {outdir}")


if __name__ == "__main__":
    main()
