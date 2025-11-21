#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from organ_utils import resolve_organ, suffix_path


def get_score(series_df: pd.DataFrame, name: str) -> pd.Series:
    """
    尝试在 DataFrame 里找到预测得分列，并转成 float。
    优先级：score_pred > y_pred > score
    """
    for col in ["score_pred", "y_pred", "score"]:
        if col in series_df.columns:
            s = pd.to_numeric(series_df[col], errors="coerce")
            if s.notna().any():
                print(f"[{name}] 使用列 '{col}' 作为 score_pred")
                return s
    raise RuntimeError(
        f"{name} 表里找不到 score 列（期望列名之一：score_pred / y_pred / score）。"
        f" 实际列: {list(series_df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ-id", type=int, default=0)
    ap.add_argument("--organ-name", default=None)
    ap.add_argument("--suffix", default=None, help="Suffix to append to outputs (fallback to slug)")
    ap.add_argument("--predict-eval", default="data/raw/predict_eval.csv")
    ap.add_argument("--generated-topk", default="data/raw/generated_topk.csv")
    ap.add_argument("--fig-dir", default="figs/outputs")
    ap.add_argument("--out-csv-dir", default="outputs/phase2/m1")
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    organ_id, organ_name, slug = resolve_organ(root, args.organ_id, args.organ_name)
    suffix = args.suffix or slug

    raw_dir = root / "data" / "raw"
    out_fig_dir = root / args.fig_dir
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir = root / args.out_csv_dir
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    df_real = pd.read_csv(root / args.predict_eval)
    if "organ_id" in df_real.columns:
        col = df_real["organ_id"]
        if pd.api.types.is_numeric_dtype(col):
            df_real = df_real[col.astype(int) == organ_id]
        else:
            df_real = df_real[col.astype(str).isin({str(organ_id), organ_name})]
    df_gen = pd.read_csv(root / args.generated_topk)
    if "organ_id_target" in df_gen.columns:
        col = df_gen["organ_id_target"]
        if pd.api.types.is_numeric_dtype(col):
            df_gen = df_gen[col.astype(int) == organ_id]
        else:
            df_gen = df_gen[col.astype(str).isin({str(organ_id), organ_name})]

    real_scores = get_score(df_real, "predict_eval")
    gen_scores = get_score(df_gen, "generated_topk")

    # 2）画真实 vs 生成的 score_pred 直方图
    plt.figure()
    bins = 40

    plt.hist(
        real_scores.dropna(),
        bins=bins,
        density=True,
        alpha=0.5,
        label="Real (predict_eval)",
    )
    plt.hist(
        gen_scores.dropna(),
        bins=bins,
        density=True,
        alpha=0.5,
        label="Generated (generated_topk)",
    )

    plt.xlabel("score_pred")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    fig_path = suffix_path(out_fig_dir / "fig_score_pred_real_vs_generated.png", suffix)
    plt.savefig(fig_path, dpi=200)
    print(f"[OK] 保存图像到 {fig_path}")

    # 顺便在终端打印一点统计量，方便你写总结
    def summarize(name, s: pd.Series):
        s = s.dropna()
        print(
            f"[{name}] n={len(s)}, "
            f"mean={s.mean():.4f}, std={s.std():.4f}, "
            f"q10={s.quantile(0.1):.4f}, "
            f"q50={s.quantile(0.5):.4f}, "
            f"q90={s.quantile(0.9):.4f}"
        )

    summarize("Real", real_scores)
    summarize("Generated", gen_scores)

    # 3）从 generated_topk 里导出 top-N
    N = args.topn
    df_gen_sorted = df_gen.copy()
    df_gen_sorted["_score_for_sort"] = gen_scores
    df_gen_sorted = df_gen_sorted.sort_values("_score_for_sort", ascending=False)
    topN = df_gen_sorted.head(N).drop(columns=["_score_for_sort"])

    top_path = suffix_path(out_csv_dir / "generated_top20_single_organ.csv", suffix)
    topN.to_csv(top_path, index=False)
    print(f"[OK] 保存 top-{N} 结果到 {top_path}")


if __name__ == "__main__":
    main()
