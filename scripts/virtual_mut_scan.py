#!/usr/bin/env python3
"""
Virtual mutation scan heatmaps (PISTE-style) for representative UTRs.
- Loads phase1 predictor (configs/gen_predict.yaml)
- For selected sequences, mutates last N bases of 5'/3' UTRs (default 200)
- Outputs per-base delta heatmaps and CSVs

Example:
  python scripts/virtual_mut_scan.py \
    --input data/derived/multi_model_/all_models_features.csv \
    --outdir figs/outputs/mutscan \
    --num-seqs 4 --max-len-utr5 200 --max-len-utr3 200
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.side.predict import PairDataset, _build_alphabet, _one_hot_encode, load_phase1_model


def _select_score_col(df: pd.DataFrame) -> str | None:
    for col in ["score_pred", "y_pred", "pred", "reward"]:
        if col in df.columns:
            return col
    return None


def _trim_indices(seq: str, max_len: int) -> Tuple[str, int]:
    """Return trimmed seq (last max_len bases) and starting offset in original."""
    if len(seq) <= max_len:
        return seq, 0
    return seq[-max_len:], len(seq) - max_len


def _mutations(seq: str, offset: int, alphabet: Sequence[str]) -> List[Tuple[int, str]]:
    muts = []
    for i, ch in enumerate(seq):
        for base in alphabet:
            if base == ch:
                continue
            muts.append((offset + i, base))
    return muts


class MutDataset(Dataset):
    def __init__(
        self,
        rows: List[Tuple[str, str, int]],
        L5: int,
        L3: int,
        manifest: Dict,
    ) -> None:
        self.rows = rows
        self.L5 = L5
        self.L3 = L3
        self.alphabet = _build_alphabet(manifest)
        self.mapping = {ch: idx for idx, ch in enumerate(self.alphabet)}
        self.seq_channels = len(self.alphabet)
        utr5_shape = manifest["shapes"]["utr5"]
        utr3_shape = manifest["shapes"]["utr3"]
        self.total_c5 = int(utr5_shape[0])
        self.total_c3 = int(utr3_shape[0])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s5, s3, organ = self.rows[idx]
        enc5 = _one_hot_encode(s5, self.L5, self.alphabet, self.mapping)
        enc3 = _one_hot_encode(s3, self.L3, self.alphabet, self.mapping)
        x5 = np.zeros((self.total_c5, self.L5), dtype=np.float32)
        x5[: self.seq_channels] = enc5
        x3 = np.zeros((self.total_c3, self.L3), dtype=np.float32)
        x3[: self.seq_channels] = enc3
        return {
            "utr5": torch.from_numpy(x5),
            "utr3": torch.from_numpy(x3),
            "organ_id": torch.tensor(int(organ), dtype=torch.long),
        }


def predict_scores(
    model,
    manifest: Dict,
    seq_pairs: List[Tuple[str, str, int]],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    L5 = int(manifest["shapes"]["utr5"][1])
    L3 = int(manifest["shapes"]["utr3"][1])
    ds = MutDataset(seq_pairs, L5, L3, manifest)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            utr5 = batch["utr5"].to(device)
            utr3 = batch["utr3"].to(device)
            organ = batch["organ_id"].to(device)
            y = model(utr5, utr3, organ)
            preds.append(y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_mut_scan(row: pd.Series, alphabet: Sequence[str], max_len5: int, max_len3: int, model, manifest, device, outdir: Path) -> None:
    seq_id = row.get("seq_id", f"seq_{row.name}")
    utr5 = str(row["utr5"])
    utr3 = str(row["utr3"])
    organ = int(row["organ_id"])

    # Baseline
    base_pred = predict_scores(model, manifest, [(utr5, utr3, organ)], device)[0]

    def mutate_and_score(utr: str, max_len: int, is_utr5: bool):
        trimmed, offset = _trim_indices(utr, max_len)
        muts = _mutations(trimmed, offset, alphabet)
        seqs = []
        for pos, base in muts:
            s_list = list(utr)
            s_list[pos] = base
            mut_seq = "".join(s_list)
            if is_utr5:
                seqs.append((mut_seq, utr3, organ))
            else:
                seqs.append((utr5, mut_seq, organ))
        if not seqs:
            return None, None
        preds = predict_scores(model, manifest, seqs, device)
        deltas = preds - base_pred
        mat = np.full((len(trimmed), len(alphabet)), np.nan, dtype=np.float32)
        idx = 0
        for pos, base in muts:
            local_pos = pos - offset
            base_idx = alphabet.index(base)
            mat[local_pos, base_idx] = deltas[idx]
            idx += 1
        return mat, trimmed

    mat5, trim5 = mutate_and_score(utr5, max_len5, True)
    mat3, trim3 = mutate_and_score(utr3, max_len3, False)

    def save_heatmap(mat: np.ndarray | None, trimmed: str | None, title: str, fname: str):
        if mat is None:
            return
        plt.figure(figsize=(12, 4))
        plt.imshow(mat.T, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))
        plt.colorbar(label="Δ score")
        plt.yticks(ticks=range(len(alphabet)), labels=alphabet)
        plt.xlabel("Position (trimmed region)")
        plt.title(f"{title} | baseline={base_pred:.4f}")
        plt.tight_layout()
        plt.savefig(outdir / f"{fname}.png", dpi=300)
        plt.savefig(outdir / f"{fname}.svg")
        plt.close()
        np.savetxt(outdir / f"{fname}.csv", mat, delimiter=",")

    save_heatmap(mat5, trim5, f"{seq_id} 5' UTR", f"mutscan_{seq_id}_utr5")
    save_heatmap(mat3, trim3, f"{seq_id} 3' UTR", f"mutscan_{seq_id}_utr3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_predict.yaml")
    ap.add_argument("--input", default="data/derived/multi_model_/all_models_features.csv")
    ap.add_argument("--outdir", default="figs/outputs/mutscan")
    ap.add_argument("--num-seqs", type=int, default=4, help="Number of sequences to scan (sorted by score if available)")
    ap.add_argument("--max-len-utr5", type=int, default=200)
    ap.add_argument("--max-len-utr3", type=int, default=200)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Path(args.config)
    cfg = cfg if cfg.is_absolute() else root / cfg
    import yaml

    cfg_data = yaml.safe_load(cfg.read_text())
    manifest_path = Path(cfg_data["dataset_dir"])
    if not manifest_path.is_absolute():
        manifest_path = (root / manifest_path).resolve()
    manifest = manifest_path / "manifest.json"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, man = load_phase1_model(str(manifest), cfg_data["phase1_checkpoint"], device)

    df = pd.read_csv(root / args.input)
    score_col = _select_score_col(df)
    if score_col:
        df = df.sort_values(score_col, ascending=False)
    df = df.head(args.num_seqs).copy()

    alphabet = [ch for ch in _build_alphabet(man) if ch in ("A", "C", "G", "T")]
    for _, row in df.iterrows():
        run_mut_scan(row, alphabet, args.max_len_utr5, args.max_len_utr3, model, man, device, outdir)

    print(f"[OK] wrote mutation scan heatmaps to {outdir}")


if __name__ == "__main__":
    main()
