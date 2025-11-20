#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from organ_utils import resolve_organ, suffix_path


def gc_content(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    gc = sum(1 for ch in seq if ch in {"G", "C"})
    return gc / len(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build organ-specific predict_eval and generated_topk tables")
    ap.add_argument("--organ-id", type=int, required=True)
    ap.add_argument("--organ-name", default=None)
    ap.add_argument("--slug", default=None, help="Override slug, otherwise computed from manifest")
    ap.add_argument("--predict-eval", default="data/raw/predict_eval.csv")
    ap.add_argument("--m1-scored", required=True, help="Path to scored M1 CSV (with pred column)")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--topk", type=int, default=None, help="Optional cap on generated rows")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    organ_id, organ_name, computed_slug = resolve_organ(root, args.organ_id, args.organ_name)
    slug = args.slug or computed_slug
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_path = root / args.predict_eval
    df_real = pd.read_csv(predict_path)
    if "organ_id" not in df_real.columns:
        raise SystemExit(f"predict_eval file missing organ_id column: {predict_path}")
    mask = df_real["organ_id"].astype(str) == organ_name
    filtered_real = df_real[mask].copy()
    if filtered_real.empty:
        raise SystemExit(f"No rows found for organ {organ_name} in {predict_path}")
    real_out = suffix_path(out_dir / Path(predict_path).name, slug)
    filtered_real.to_csv(real_out, index=False)
    print(f"[build] Wrote real table -> {real_out} ({len(filtered_real)} rows)")

    real_pairs = set(zip(filtered_real["utr5"].astype(str), filtered_real["utr3"].astype(str)))

    m1_path = root / args.m1_scored
    df_gen = pd.read_csv(m1_path)
    if "pred" not in df_gen.columns:
        raise SystemExit(f"m1 scored file lacks pred column: {m1_path}")
    if args.topk:
        df_gen = df_gen.head(args.topk)
    seq_ids = [f"m1_{slug}_{i:06d}" for i in range(len(df_gen))]
    df_gen = df_gen.copy()
    df_gen.insert(0, "seq_id", seq_ids)
    df_gen["score_pred"] = pd.to_numeric(df_gen["pred"], errors="coerce")
    df_gen["organ_id_target"] = organ_name
    df_gen["gc"] = [gc_content((u5 or "") + (u3 or "")) for u5, u3 in zip(df_gen["utr5"], df_gen["utr3"])]
    df_gen["novelty"] = [0 if (str(u5), str(u3)) in real_pairs else 1 for u5, u3 in zip(df_gen["utr5"], df_gen["utr3"])]
    gen_out = suffix_path(out_dir / "generated_topk.csv", slug)
    df_gen[["seq_id", "utr5", "utr3", "organ_id_target", "score_pred", "gc", "novelty"]].to_csv(gen_out, index=False)
    print(f"[build] Wrote generated table -> {gen_out} ({len(df_gen)} rows)")


if __name__ == "__main__":
    main()
