#!/usr/bin/env python3
"""
Generate Fig3H structure plots for liver/adipose top1 sequences using RNAfold.
Outputs: figs/final/fig3H_{liver,adipose}_structure.{png,svg} and fig_data CSV.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs" / "final"
DATA_DIR = FIG_DIR / "fig_data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_top1(target_file: Path, max_len: int = 800) -> Tuple[str, str]:
    """Return (seq_id, seq) cleaned to ACGU only, optionally truncated for RNAfold stability."""
    import pandas as pd

    df = pd.read_csv(target_file)
    top = df.sort_values("pred", ascending=False).iloc[0]
    seq_id = str(top["seq_id"]).replace("\\n", " ").replace("\n", " ").split()[0]
    seq = str(top["utr3"]).replace("\\n", "").replace("\n", "").upper().replace("T", "U")
    seq = "".join(ch for ch in seq if ch in {"A", "C", "G", "U"})
    if max_len and len(seq) > max_len:
        seq = seq[:max_len]
    return seq_id, seq


def run_rnafold_inline(seq_id: str, seq: str) -> Tuple[str, float]:
    """Call RNAfold with FASTA on stdin, capture dot-bracket and MFE."""
    fasta = f">{seq_id}\\n{seq}\\n"
    cmd = ["/root/miniconda3/bin/RNAfold", "--noPS"]
    res = subprocess.run(cmd, input=fasta.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lines = [ln.strip() for ln in res.stdout.decode(errors="ignore").splitlines() if ln.strip()]
    structure_line = None
    for ln in reversed(lines):
        if "(" in ln and ")" in ln:
            structure_line = ln
            break
    if structure_line is None:
        raise RuntimeError(f"Unexpected RNAfold output: {lines}")
    struct, energy = structure_line.split(" (")
    energy = float(energy.strip("()"))
    return struct.strip(), energy


def pairs_from_dotbracket(db: str):
    stack = []
    pairs = []
    for i, ch in enumerate(db):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))
    return pairs


def plot_structure(seq_id: str, db: str, mfe: float, prefix: str) -> None:
    pairs = pairs_from_dotbracket(db)
    plt.figure(figsize=(8, 4))
    for i, j in pairs:
        mid = (i + j) / 2
        radius = (j - i) / 2
        theta = [k * math.pi / 49 for k in range(50)]
        x = [mid + radius * math.cos(t) for t in theta]
        y = [radius * math.sin(t) for t in theta]
        plt.plot(x, y, color="#4c72b0", linewidth=0.8)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlim(0, len(db))
    plt.ylim(0, None)
    plt.xlabel("Position")
    plt.ylabel("Base-pair arc")
    plt.title(f"{prefix} ({seq_id}) MFE={mfe:.2f}")
    plt.tight_layout()
    out_png = FIG_DIR / f"{prefix}.png"
    out_svg = FIG_DIR / f"{prefix}.svg"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seq_id": [seq_id], "dot_bracket": [db], "mfe": [mfe]}).to_csv(
        DATA_DIR / f"{prefix}.structure.csv", index=False
    )


def main():
    cases = [
        ("liver", ROOT / "outputs/figure_inputs/liver_target.csv"),
        ("adipose", ROOT / "outputs/figure_inputs/adipose_target.csv"),
    ]
    for organ, path in cases:
        seq_id, seq = load_top1(path, max_len=1200)
        db, mfe = run_rnafold_inline(seq_id, seq)
        prefix = f"fig3H_{organ}_structure"
        plot_structure(seq_id, db, mfe, prefix)
        # also write fasta for reference
        fasta_path = FIG_DIR / f"{prefix}.fa"
        fasta_path.write_text(f">{seq_id}\\n{seq}\\n")
        print(f"[OK] {organ}: mfe={mfe:.2f}, len={len(seq)} -> {prefix}")


if __name__ == "__main__":
    main()
