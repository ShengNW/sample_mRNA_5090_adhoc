#!/usr/bin/env python3
"""
Simple feature importance for "hit" classification (Immunity-style bar plot).
Label: top quantile of score_pred (or provided column).
Models: Logistic Regression (L1) and RandomForest.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "gc",
    "mfe_utr5",
    "mfe_utr3",
    "rbp_hits_per_kb",
    "rbp_hits_total",
    "novelty",
    "trna_distance",
]


def load_table(root: Path, feat_path: Path, rbp_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(feat_path)
    if rbp_path and rbp_path.exists():
        rbp = pd.read_csv(rbp_path)
        df = df.merge(rbp[["seq_id", "rbp_hits_total", "rbp_hits_per_kb"]], on="seq_id", how="left")
    return df


def build_label(df: pd.DataFrame, score_col: str, quantile: float) -> pd.Series:
    thresh = df[score_col].quantile(quantile)
    return (df[score_col] >= thresh).astype(int)


def train_lr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, penalty="l1", solver="liblinear", class_weight="balanced")
    lr.fit(Xs, y)
    return lr.coef_.ravel(), scaler


def train_rf(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=0,
    )
    rf.fit(X, y)
    return rf.feature_importances_


def plot_bar(labels: List[str], importances: List[np.ndarray], names: List[str], out_path: Path) -> None:
    width = 0.35
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    for idx, (imp, name) in enumerate(zip(importances, names)):
        plt.bar(x + idx * width, imp, width=width, label=name, alpha=0.8)
    plt.xticks(x + width / 2, labels, rotation=25, ha="right")
    plt.ylabel("Importance (abs coef or Gini)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=250)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/derived/multi_model_/all_models_features.csv")
    ap.add_argument("--rbp", default="data/derived/all_models_rbp_features.csv")
    ap.add_argument("--score-col", default="score_pred")
    ap.add_argument("--top-quantile", type=float, default=0.9)
    ap.add_argument("--feature-cols", nargs="+", default=None)
    ap.add_argument("--outdir", default="figs/outputs")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = load_table(root, root / args.features, root / args.rbp)
    feat_cols = args.feature_cols or [c for c in DEFAULT_FEATURES if c in df.columns]
    if not feat_cols:
        raise RuntimeError("No usable feature columns found.")

    if args.score_col not in df.columns:
        raise RuntimeError(f"score column {args.score_col} not found")
    df = df.dropna(subset=feat_cols + [args.score_col])
    y = build_label(df, args.score_col, args.top_quantile)
    X = df[feat_cols].values

    # Train models
    lr_coef, _ = train_lr(X, y.values)
    rf_imp = train_rf(X, y.values)

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    plot_bar(
        feat_cols,
        [np.abs(lr_coef), rf_imp],
        ["LR |coef|", "RF Gini"],
        outdir / "fig_feature_importance_hit",
    )
    print(f"[OK] wrote importance plot to {outdir}")


if __name__ == "__main__":
    main()
