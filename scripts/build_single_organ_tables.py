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

def _score_column(df: pd.DataFrame, name: str) -> pd.Series:
    for col in ["score_pred", "pred", "y_pred", "reward"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                print(f"[{name}] using column '{col}' as score_pred")
                return s
    raise SystemExit(f"{name} missing score column; available: {list(df.columns)}")


def _filter_organ(df: pd.DataFrame, organ_id: int, organ_name: str, name: str) -> pd.DataFrame:
    for col in ["organ_id_target", "organ_id", "organ"]:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            mask = series.astype(int) == organ_id
        else:
            mask = series.astype(str) == str(organ_name)
        sub = df[mask].copy()
        print(f"[{name}] filtered by {col}, rows={len(sub)}")
        return sub
    print(f"[WARN] {name} no organ column matched; using all rows={len(df)}")
    return df.copy()


def _prepare_generated(df: pd.DataFrame, organ_id: int, organ_name: str, slug: str, label: str, real_pairs: set[tuple]) -> pd.DataFrame:
    df = _filter_organ(df, organ_id, organ_name, label)
    if df.empty:
        return df
    score = _score_column(df, label)
    df = df.copy()
    df["_score_pred"] = score
    df["organ_id_target"] = organ_id
    df["seq_id"] = [f"{label}_{slug}_{i:06d}" for i in range(len(df))]
    df["gc"] = [gc_content((u5 or "") + (u3 or "")) for u5, u3 in zip(df["utr5"], df["utr3"])]
    df["novelty"] = [0 if (str(u5), str(u3)) in real_pairs else 1 for u5, u3 in zip(df["utr5"], df["utr3"])]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build organ-specific predict_eval and generated_topk tables")
    ap.add_argument("--organ-id", type=int, required=True)
    ap.add_argument("--organ-name", default=None)
    ap.add_argument("--slug", default=None, help="Override slug, otherwise computed from manifest")
    ap.add_argument("--predict-eval", default="data/raw/predict_eval.csv")
    ap.add_argument("--m1-scored", required=True, help="Path to scored M1 CSV (with pred column)")
    ap.add_argument("--m2-cvae-scored", default=None)
    ap.add_argument("--m2-cgan-scored", default=None)
    ap.add_argument("--m3-rl-scored", default=None)
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
    filtered_real = _filter_organ(df_real, organ_id, organ_name, "predict_eval")
    if filtered_real.empty:
        raise SystemExit(f"No rows found for organ {organ_name} in {predict_path}")
    real_out = suffix_path(out_dir / Path(predict_path).name, slug)
    filtered_real.to_csv(real_out, index=False)
    print(f"[build] Wrote real table -> {real_out} ({len(filtered_real)} rows)")

    real_pairs = set(zip(filtered_real["utr5"].astype(str), filtered_real["utr3"].astype(str)))

    m1_path = root / args.m1_scored
    sources = {
        "m1": m1_path,
        "m2_cvae": root / args.m2_cvae_scored if args.m2_cvae_scored else None,
        "m2_cgan": root / args.m2_cgan_scored if args.m2_cgan_scored else None,
        "m3_rl": root / args.m3_rl_scored if args.m3_rl_scored else None,
    }

    generated = []
    for label, path in sources.items():
        if path is None:
            continue
        if not path.exists():
            print(f"[WARN] {label}: missing {path}, skip")
            continue
        df = pd.read_csv(path)
        df = _prepare_generated(df, organ_id, organ_name, slug, label, real_pairs)
        if df.empty:
            print(f"[WARN] {label}: empty after filtering, skip")
            continue
        generated.append(df)

    if not generated:
        raise SystemExit("No generated datasets found; ensure scored CSVs exist.")

    gen_df = pd.concat(generated, ignore_index=True)
    if args.topk:
        gen_df = gen_df.sort_values("_score_pred", ascending=False).head(args.topk)
    gen_df = gen_df.rename(columns={"_score_pred": "score_pred"})

    gen_out = suffix_path(out_dir / "generated_topk.csv", slug)
    gen_df[["seq_id", "utr5", "utr3", "organ_id_target", "score_pred", "gc", "novelty"]].to_csv(gen_out, index=False)
    print(f"[build] Wrote generated table -> {gen_out} ({len(gen_df)} rows)")


if __name__ == "__main__":
    main()
