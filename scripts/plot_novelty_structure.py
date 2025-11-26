#!/usr/bin/env python3
"""
Distributions of novelty/structure metrics (GEMORNA-style):
  - MIS (max identity vs real reference, approximate)
  - GC, MFE (5'/3'), length
Outputs density + box plots for Real / M1 / M2 / M3 / Random (optional).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_ORDER: List[Tuple[str, str]] = [
    ("Real (eval)", "#1f77b4"),
    ("M1 top-k", "#ff7f0e"),
    ("M2 cVAE", "#2ca02c"),
    ("M2 cGAN", "#d62728"),
    ("M3 RL", "#9467bd"),
    ("Random", "#7f7f7f"),
]


def _simple_identity(a: str, b: str) -> float:
    """Approximate sequence identity by aligning from the end and padding shorter with mismatch."""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    m = min(la, lb)
    matches = sum(1 for i in range(1, m + 1) if a[-i] == b[-i])
    return matches / max(la, lb)


def compute_mis(query: pd.Series, ref: pd.Series, max_ref: int, max_query: int) -> pd.Series:
    ref = ref.dropna().astype(str)
    if len(ref) > max_ref:
        ref = ref.sample(max_ref, random_state=0)
    ref_list = ref.tolist()
    out = []
    q = query.dropna().astype(str)
    if len(q) > max_query:
        q = q.sample(max_query, random_state=0)
    for seq in q.tolist():
        best = 0.0
        for r in ref_list:
            best = max(best, _simple_identity(seq, r))
            if best >= 0.999:
                break
        out.append(best)
    return pd.Series(out, index=q.index)


def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _gc_content(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    gc = sum(1 for ch in seq if ch in ("G", "C"))
    return gc / len(seq)


def _maybe_add_random(df: pd.DataFrame, random_csv: Path | None) -> pd.DataFrame:
    if random_csv is None or not random_csv.exists():
        return df
    rnd = pd.read_csv(random_csv)
    rnd = rnd.copy()
    rnd["source"] = "Random"
    if "gc" not in rnd.columns:
        rnd["gc"] = rnd["utr5"].fillna("").add(rnd["utr3"].fillna("")).apply(_gc_content)
    return pd.concat([df, rnd], ignore_index=True)


def _len_total(row: pd.Series) -> int:
    return len(str(row.get("utr5", ""))) + len(str(row.get("utr3", "")))


def _filter_group(df: pd.DataFrame, organ: str | None) -> pd.DataFrame:
    if organ is None:
        return df
    for col in ("organ_name", "organ"):
        if col in df.columns:
            sub = df[df[col].astype(str) == str(organ)]
            if len(sub):
                return sub
    return df


def _plot_density_box(
    df: pd.DataFrame,
    col: str,
    xlabel: str,
    out_prefix: Path,
    bins: int = 80,
) -> None:
    sub = df[["source", col]].dropna()
    if sub.empty:
        print(f"[WARN] missing data for {col}")
        return
    rng = (sub[col].min(), sub[col].max())
    plt.figure(figsize=(7, 4))
    for label, color in GROUP_ORDER:
        vals = sub[sub["source"] == label][col]
        if vals.empty:
            continue
        plt.hist(
            vals,
            bins=bins,
            range=rng,
            density=True,
            histtype="step",
            linewidth=1.4,
            label=label,
            color=color,
        )
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=250)
    plt.savefig(out_prefix.with_suffix(".svg"))
    plt.close()

    plt.figure(figsize=(7, 4))
    data = [sub[sub["source"] == label][col] for label, _ in GROUP_ORDER if not sub[sub["source"] == label][col].empty]
    labels = [label for label, _ in GROUP_ORDER if not sub[sub["source"] == label][col].empty]
    if not data:
        print(f"[WARN] no box data for {col}")
        return
    plt.boxplot(data, labels=labels, notch=True, patch_artist=False)
    plt.ylabel(xlabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.stem + "_box.png"), dpi=250)
    plt.savefig(out_prefix.with_name(out_prefix.stem + "_box.svg"))
    plt.close()


def summarize(df: pd.DataFrame, col: str) -> None:
    print(f"\n=== {col} ===")
    for label, _ in GROUP_ORDER:
        vals = df[df["source"] == label][col].dropna()
        if vals.empty:
            print(f"{label}: n=0")
            continue
        print(
            f"{label}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}, "
            f"q10={vals.quantile(0.10):.4f}, q50={vals.quantile(0.50):.4f}, q90={vals.quantile(0.90):.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/derived/multi_model_/all_models_features.csv")
    ap.add_argument("--ref", default="data/raw/predict_eval.csv", help="Reference real sequences for MIS")
    ap.add_argument("--random-csv", default=None, help="Optional random baseline CSV with utr5/utr3 columns")
    ap.add_argument("--outdir", default="figs/outputs", help="Output directory for figures")
    ap.add_argument("--organ", default=None, help="Optional organ name to filter (matches organ_name)")
    ap.add_argument("--mis-max-ref", type=int, default=3000, help="Sample size for reference sequences")
    ap.add_argument("--mis-max-rows", type=int, default=3000, help="Sample size for MIS query sequences")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    feat_path = root / args.features
    df = _load_features(feat_path)
    df = _filter_group(df, args.organ)

    ref_df = pd.read_csv(root / args.ref)
    if args.organ and "organ_name" in ref_df.columns:
        ref_df = ref_df[ref_df["organ_name"].astype(str) == str(args.organ)]
    ref_seq = ref_df["utr5"].astype(str) + ref_df["utr3"].astype(str)

    df["seq_concat"] = df["utr5"].fillna("").astype(str) + df["utr3"].fillna("").astype(str)
    df = _maybe_add_random(df, root / args.random_csv if args.random_csv else None)

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    mis = compute_mis(
        df["seq_concat"],
        ref_seq,
        max_ref=args.mis_max_ref,
        max_query=args.mis_max_rows,
    )
    df.loc[mis.index, "mis_identity"] = mis

    df["len_total"] = df.apply(_len_total, axis=1)

    _plot_density_box(df, "mis_identity", "Max identity vs real (approx.)", outdir / "fig_novelty_mis")
    _plot_density_box(df, "gc", "GC content", outdir / "fig_novelty_gc")
    if "mfe_utr5" in df.columns:
        _plot_density_box(df, "mfe_utr5", "MFE 5' UTR", outdir / "fig_novelty_mfe_utr5")
    if "mfe_utr3" in df.columns:
        _plot_density_box(df, "mfe_utr3", "MFE 3' UTR", outdir / "fig_novelty_mfe_utr3")
    _plot_density_box(df, "len_total", "Length (5'+3')", outdir / "fig_novelty_len")

    summarize(df, "mis_identity")
    summarize(df, "gc")
    if "mfe_utr5" in df.columns:
        summarize(df, "mfe_utr5")
    if "mfe_utr3" in df.columns:
        summarize(df, "mfe_utr3")
    summarize(df, "len_total")


if __name__ == "__main__":
    main()
