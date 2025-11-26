#!/usr/bin/env python3
"""
Plot 2D projection of embeddings (TSNE).
Input: CSV from extract_embeddings.py (emb_* columns).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/derived/multi_model_/embeddings.csv")
    ap.add_argument("--outdir", default="figs/outputs")
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.input)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("No emb_* columns found.")
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows)

    X = df[emb_cols].values.astype(np.float32)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="random", random_state=0)
    emb2d = tsne.fit_transform(X)
    df["tsne1"] = emb2d[:, 0]
    df["tsne2"] = emb2d[:, 1]

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        plt.scatter(sub["tsne1"], sub["tsne2"], s=8, alpha=0.6, label=str(src))
    plt.legend(markerscale=1.5, fontsize=8)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    out_fig = outdir / "fig_embedding_tsne.png"
    plt.savefig(out_fig, dpi=250)
    plt.savefig(out_fig.with_suffix(".svg"))
    plt.close()

    df_out = root / "figs" / "outputs" / "embeddings_tsne.csv"
    df.to_csv(df_out, index=False)
    print(f"[OK] wrote {out_fig} and {df_out}")


if __name__ == "__main__":
    main()
