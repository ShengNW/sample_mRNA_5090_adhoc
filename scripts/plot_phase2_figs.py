#!/usr/bin/env python3
"""
Generate figures for Fig3C–H and Fig2E/F using prepared scoring outputs.
Outputs: PNG/SVG + Excel per figure into figs/final and figs/final/fig_data.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs" / "final"
DATA_DIR = FIG_DIR / "fig_data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------
def load_vocab() -> Dict[int, str]:
    manifest = json.loads((ROOT / "data/processed/seq_cnn_v1_rbp_trna/manifest.json").read_text())
    return {int(k): v for k, v in manifest["organ_vocab"].items()}


def save_excel(df_map: Dict[str, pd.DataFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import openpyxl  # type: ignore  # noqa: F401

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for sheet, df in df_map.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
    except ImportError:
        # Fallback: write separate CSV files when Excel writer is unavailable
        for sheet, df in df_map.items():
            df.to_csv(out_path.with_suffix(f".{sheet}.csv"), index=False)


# ---------- Fig3C: RL trajectory ----------
def plot_rl_trajectory(traj_files: Dict[int, Path], vocab: Dict[int, str]) -> None:
    plt.figure(figsize=(8, 5))
    all_rows = []
    for oid, path in traj_files.items():
        df = pd.read_csv(path)
        plt.plot(df["step"], df["mean_reward"], label=f"{vocab.get(oid, oid)} mean")
        plt.plot(df["step"], df["max_reward"], linestyle="--", label=f"{vocab.get(oid, oid)} max")
        df.insert(0, "organ_id", oid)
        all_rows.append(df)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("RL optimization trajectory")
    out_png = FIG_DIR / "fig3C_rl_traj.png"
    out_svg = FIG_DIR / "fig3C_rl_traj.svg"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    if all_rows:
        save_excel({"traj": pd.concat(all_rows, ignore_index=True)}, DATA_DIR / "fig3C.xlsx")


# ---------- Fig3D: natural vs generated ----------
def _ecdf(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.sort(series.to_numpy())
    y = np.arange(1, len(vals) + 1) / len(vals)
    return vals, y


def plot_distribution_shift(natural_file: Path, generated_file: Path, organ_name: str, prefix: str) -> None:
    nat = pd.read_csv(natural_file)
    gen = pd.read_csv(generated_file)
    plt.figure(figsize=(6, 4))
    bins = np.linspace(
        min(nat["pred"].min(), gen["pred"].min()), max(nat["pred"].max(), gen["pred"].max()), 40
    )
    plt.hist(nat["pred"], bins=bins, density=True, alpha=0.45, label="Natural", color="#888888")
    plt.hist(gen["pred"], bins=bins, density=True, alpha=0.45, label="Generated", color="#c23b22")
    plt.xlabel("Predicted expression")
    plt.ylabel("Density")
    plt.title(f"Distribution shift: {organ_name}")
    plt.legend()
    out_png = FIG_DIR / f"{prefix}_dist.png"
    out_svg = FIG_DIR / f"{prefix}_dist.svg"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    # ECDF overlay
    xv_nat, y_nat = _ecdf(nat["pred"])
    xv_gen, y_gen = _ecdf(gen["pred"])
    save_excel(
        {
            "hist_nat_gen": pd.DataFrame(
                {
                    "bin_center": 0.5 * (bins[1:] + bins[:-1]),
                    "natural_density": np.histogram(nat["pred"], bins=bins, density=True)[0],
                    "generated_density": np.histogram(gen["pred"], bins=bins, density=True)[0],
                }
            ),
            "ecdf_natural": pd.DataFrame({"x": xv_nat, "y": y_nat}),
            "ecdf_generated": pd.DataFrame({"x": xv_gen, "y": y_gen}),
        },
        DATA_DIR / f"{prefix}.xlsx",
    )


# ---------- Fig3E: Off-target matrix ----------
def plot_offtarget(
    off_file: Path, target_file: Path, vocab: Dict[int, str], organ_id: int, prefix: str, topk: int = 30
) -> None:
    off = pd.read_csv(off_file)
    target = pd.read_csv(target_file)[["seq_id", "pred"]]
    merged = target.merge(off, on="seq_id")
    merged = merged.sort_values("pred", ascending=False).head(topk)
    organ_cols = [c for c in merged.columns if c.startswith("organ_")]
    mat = merged[organ_cols].to_numpy()
    mat_norm = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-6)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mat_norm, aspect="auto", cmap="RdBu_r")
    plt.colorbar(im, label="Row z-score")
    plt.xlabel("Organ")
    plt.ylabel("Candidates (top pred)")
    plt.title(f"Off-target heatmap ({vocab.get(organ_id, organ_id)})")
    plt.xticks(
        ticks=np.arange(len(organ_cols)),
        labels=[vocab.get(int(c.split("_")[1]), c) for c in organ_cols],
        rotation=90,
        fontsize=6,
    )
    plt.tight_layout()
    out_png = FIG_DIR / f"{prefix}_offtarget.png"
    out_svg = FIG_DIR / f"{prefix}_offtarget.svg"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    save_excel({"offtarget": merged}, DATA_DIR / f"{prefix}_offtarget.xlsx")


# ---------- Fig3F: Diversity/novelty ----------
def compute_kmer_novelty(seqs: pd.Series, natural: pd.Series, k: int = 6) -> List[float]:
    nat_kmers = set()
    for seq in natural.str.upper():
        nat_kmers.update(seq[i : i + k] for i in range(len(seq) - k + 1))
    scores = []
    for seq in seqs.str.upper():
        kmers = set(seq[i : i + k] for i in range(len(seq) - k + 1))
        overlap = len(kmers & nat_kmers)
        total = len(kmers | nat_kmers)
        scores.append(1.0 - overlap / total if total else 0.0)
    return scores


def plot_diversity(generated_file: Path, natural_file: Path, organ_name: str, prefix: str) -> None:
    gen = pd.read_csv(generated_file)
    nat = pd.read_csv(natural_file)
    novelty = compute_kmer_novelty(gen["utr3"], nat["utr3"], k=6)
    gen = gen.copy()
    gen["novelty"] = novelty
    plt.figure(figsize=(6, 4))
    plt.hist(gen["novelty"], bins=30, color="#6a5acd", alpha=0.8)
    plt.xlabel("Novelty (1 - Jaccard of 6-mers vs natural)")
    plt.ylabel("Count")
    plt.title(f"Sequence novelty: {organ_name}")
    plt.tight_layout()
    out_png = FIG_DIR / f"{prefix}_novelty.png"
    out_svg = FIG_DIR / f"{prefix}_novelty.svg"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    save_excel({"generated_novelty": gen}, DATA_DIR / f"{prefix}_novelty.xlsx")


# ---------- Fig3G/H: Top1 profile & structure ----------
def select_top1(target_file: Path) -> pd.Series:
    df = pd.read_csv(target_file)
    return df.sort_values("pred", ascending=False).iloc[0]


def plot_top1_profile(
    off_file: Path, target_file: Path, vocab: Dict[int, str], organ_id: int, prefix: str
) -> Path:
    top = select_top1(target_file)
    off = pd.read_csv(off_file)
    row = off[off["seq_id"] == top["seq_id"]]
    if row.empty:
        raise ValueError("Top1 seq_id not found in off-target matrix")
    organ_cols = [c for c in row.columns if c.startswith("organ_")]
    vals = row.iloc[0][organ_cols].to_numpy(dtype=float)
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(vals)), vals, color="#1f77b4")
    plt.xticks(
        ticks=np.arange(len(vals)),
        labels=[vocab.get(int(c.split("_")[1]), c) for c in organ_cols],
        rotation=90,
        fontsize=6,
    )
    plt.ylabel("Predicted expression")
    plt.title(f"Top1 profile: {top['seq_id']} ({vocab.get(organ_id, organ_id)})")
    plt.tight_layout()
    out_png = FIG_DIR / f"{prefix}_top1_profile.png"
    out_svg = FIG_DIR / f"{prefix}_top1_profile.svg"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    fasta_path = FIG_DIR / f"{prefix}_top1.fa"
    seq_id = str(top["seq_id"]).replace("\\n", " ").replace("\n", " ").split()[0]
    raw_seq = str(top["utr3"]).replace("\\n", "").replace("\n", "").upper()
    raw_seq = raw_seq.replace("T", "U")
    seq_clean = "".join(ch for ch in raw_seq if ch in {"A", "C", "G", "U"})
    with open(fasta_path, "w") as f:
        f.write(f">{seq_id}\\n{seq_clean}\\n")
    save_excel(
        {
            "top1_profile": pd.DataFrame(
                {
                    "organ": [vocab.get(int(c.split('_')[1]), c) for c in organ_cols],
                    "pred": vals,
                }
            )
        },
        DATA_DIR / f"{prefix}_top1.xlsx",
    )
    return fasta_path


def run_rnafold(fasta_path: Path) -> Tuple[str, float]:
    cmd = ["/root/miniconda3/bin/RNAfold", "--noPS", str(fasta_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lines = [ln.strip() for ln in res.stdout.decode(errors="ignore").splitlines() if ln.strip()]
    structure_line = None
    for ln in reversed(lines):
        if "(" in ln and ")" in ln:
            structure_line = ln
            break
    if structure_line is None:
        raise RuntimeError("Unexpected RNAfold output")
    struct, energy = structure_line.split(" (")
    energy = float(energy.strip("()"))
    return struct.strip(), energy


def _pairs_from_dotbracket(db: str) -> List[Tuple[int, int]]:
    stack = []
    pairs = []
    for i, ch in enumerate(db):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))
    return pairs


def plot_structure(fasta_path: Path, prefix: str) -> None:
    seq = "".join(line.strip() for line in fasta_path.read_text().splitlines()[1:])
    try:
        dotb, energy = run_rnafold(fasta_path)
    except Exception as exc:  # pragma: no cover - graceful degradation
        note = f"RNAfold failed: {exc}"
        (FIG_DIR / f"{prefix}_structure.txt").write_text(note)
        save_excel({"structure_error": pd.DataFrame({"error": [str(exc)]})}, DATA_DIR / f"{prefix}_structure.xlsx")
        return
    pairs = _pairs_from_dotbracket(dotb)
    plt.figure(figsize=(8, 4))
    for i, j in pairs:
        mid = (i + j) / 2
        radius = (j - i) / 2
        theta = np.linspace(0, math.pi, 50)
        x = mid + radius * np.cos(theta)
        y = radius * np.sin(theta)
        plt.plot(x, y, color="#4c72b0", linewidth=1)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlim(0, len(seq))
    plt.ylim(0, None)
    plt.xlabel("Position")
    plt.ylabel("Base-pair arc")
    plt.title(f"RNAfold structure {prefix} (MFE={energy:.2f})")
    plt.tight_layout()
    out_png = FIG_DIR / f"{prefix}_structure.png"
    out_svg = FIG_DIR / f"{prefix}_structure.svg"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    save_excel({"structure": pd.DataFrame({"dot_bracket": [dotb], "mfe": [energy]})}, DATA_DIR / f"{prefix}_structure.xlsx")


# ---------- Fig2E: Saliency/mutscan ----------
def plot_saliency_heatmaps(csv_liver: Path, csv_adipose: Path, prefix: str) -> None:
    liv = pd.read_csv(csv_liver, header=None).to_numpy()
    adi = pd.read_csv(csv_adipose, header=None).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    im0 = axes[0].imshow(liv, aspect="auto", cmap="bwr")
    axes[0].set_title("Liver context")
    im1 = axes[1].imshow(adi, aspect="auto", cmap="bwr")
    axes[1].set_title("Adipose context")
    for ax in axes:
        ax.set_xlabel("Position")
    axes[0].set_ylabel("Mutations")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label="Delta score")
    plt.tight_layout()
    out_png = FIG_DIR / f"{prefix}_saliency.png"
    out_svg = FIG_DIR / f"{prefix}_saliency.svg"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    save_excel({"liver_ctx": pd.DataFrame(liv), "adipose_ctx": pd.DataFrame(adi)}, DATA_DIR / f"{prefix}_saliency.xlsx")


# ---------- Fig2F: Motif via FIMO ----------
def build_hot_windows(csv_file: Path, out_fa: Path, window: int = 12, top_n: int = 20) -> None:
    arr = pd.read_csv(csv_file, header=None).to_numpy()
    score = np.abs(arr).sum(axis=0)
    top_idx = np.argsort(score)[::-1][:top_n]
    seq = pd.read_csv(ROOT / "outputs/figure_inputs/mutscan_topseq.csv").iloc[0]["utr5"]
    seq = str(seq).replace("\\n", "").replace("\n", "").upper()
    seq = seq.replace("T", "U")
    half = window // 2
    with open(out_fa, "w") as f:
        for i, idx in enumerate(top_idx):
            start = max(0, idx - half)
            end = min(len(seq), idx + half)
            f.write(f">hot_{i}_{idx}\\n{seq[start:end]}\\n")


def run_fimo(fa: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/root/meme/bin/fimo",
        "--oc",
        str(out_dir),
        str(ROOT / "data/external/rbp/Ray2013_rbp_Homo_sapiens.meme"),
        str(fa),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_dir / "fimo.tsv"
    except subprocess.CalledProcessError as exc:
        (out_dir / "fimo_error.txt").write_text(exc.stderr.decode() if exc.stderr else str(exc))
        return out_dir / "fimo.tsv"


def plot_motif_hits(fimo_tsv: Path, prefix: str) -> None:
    if not fimo_tsv.exists():
        return
    df = pd.read_csv(fimo_tsv, sep="\\t")
    if df.empty:
        return
    top = df[df["q-value"] <= 0.05].copy()
    if top.empty:
        top = df.nsmallest(20, "q-value")
    counts = top["motif_id"].value_counts().head(15)
    plt.figure(figsize=(6, 4))
    counts.iloc[::-1].plot(kind="barh", color="#f28e2b")
    plt.xlabel("Hit count (q<=0.05)")
    plt.ylabel("Motif")
    plt.title("Motif enrichment")
    plt.tight_layout()
    out_png = FIG_DIR / f"{prefix}_motif.png"
    out_svg = FIG_DIR / f"{prefix}_motif.svg"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    save_excel({"fimo_hits": top}, DATA_DIR / f"{prefix}_motif.xlsx")


def main():
    vocab = load_vocab()
    # Fig3C
    traj_files = {
        35: ROOT / "outputs/phase2/m3_rl/organ35_liver/traj.csv",
        0: ROOT / "outputs/phase2/m3_rl/organ0_adipose/traj.csv",
    }
    plot_rl_trajectory(traj_files, vocab)

    # Fig3D
    plot_distribution_shift(
        ROOT / "outputs/figure_inputs/natural_liver_target.csv",
        ROOT / "outputs/figure_inputs/liver_target.csv",
        "Liver",
        "fig3D_liver",
    )
    plot_distribution_shift(
        ROOT / "outputs/figure_inputs/natural_adipose_target.csv",
        ROOT / "outputs/figure_inputs/adipose_target.csv",
        "Adipose - Subcutaneous",
        "fig3D_adipose",
    )

    # Fig3E
    plot_offtarget(
        ROOT / "outputs/figure_inputs/liver_offtarget.csv",
        ROOT / "outputs/figure_inputs/liver_target.csv",
        vocab,
        35,
        "fig3E_liver",
    )
    plot_offtarget(
        ROOT / "outputs/figure_inputs/adipose_offtarget.csv",
        ROOT / "outputs/figure_inputs/adipose_target.csv",
        vocab,
        0,
        "fig3E_adipose",
    )

    # Fig3F
    plot_diversity(
        ROOT / "outputs/figure_inputs/liver_target.csv",
        ROOT / "outputs/figure_inputs/natural_liver_target.csv",
        "Liver",
        "fig3F_liver",
    )
    plot_diversity(
        ROOT / "outputs/figure_inputs/adipose_target.csv",
        ROOT / "outputs/figure_inputs/natural_adipose_target.csv",
        "Adipose - Subcutaneous",
        "fig3F_adipose",
    )

    # Fig3G/H
    fasta_liver = plot_top1_profile(
        ROOT / "outputs/figure_inputs/liver_offtarget.csv",
        ROOT / "outputs/figure_inputs/liver_target.csv",
        vocab,
        35,
        "fig3G_liver",
    )
    plot_structure(fasta_liver, "fig3H_liver")

    fasta_adipose = plot_top1_profile(
        ROOT / "outputs/figure_inputs/adipose_offtarget.csv",
        ROOT / "outputs/figure_inputs/adipose_target.csv",
        vocab,
        0,
        "fig3G_adipose",
    )
    plot_structure(fasta_adipose, "fig3H_adipose")

    # Fig2E
    plot_saliency_heatmaps(
        ROOT / "figs/outputs/mutscan/mutscan_liver_00000_liverctx_utr5.csv",
        ROOT / "figs/outputs/mutscan/mutscan_liver_00000_adiposectx_utr5.csv",
        "fig2E",
    )

    # Fig2F: use liver context heatmap to derive windows
    hot_fa = ROOT / "outputs/analysis/hot_windows.fa"
    fimo_out = ROOT / "outputs/analysis/fimo_hits"
    hot_fa.parent.mkdir(parents=True, exist_ok=True)
    build_hot_windows(ROOT / "figs/outputs/mutscan/mutscan_liver_00000_liverctx_utr5.csv", hot_fa)
    fimo_tsv = run_fimo(hot_fa, fimo_out)
    plot_motif_hits(fimo_tsv, "fig2F")


if __name__ == "__main__":
    main()
