#!/usr/bin/env python3
"""
Extract latent embeddings from the phase1 predictor for a sampled set of sequences.
Outputs CSV with seq_id/source/organ_id + embedding columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


class EmbDataset(Dataset):
    def __init__(
        self,
        rows: List[Tuple[str, str, int, str, str]],
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
        utr5, utr3, organ, seq_id, source = self.rows[idx]
        enc5 = _one_hot_encode(utr5, self.L5, self.alphabet, self.mapping)
        enc3 = _one_hot_encode(utr3, self.L3, self.alphabet, self.mapping)
        x5 = np.zeros((self.total_c5, self.L5), dtype=np.float32)
        x5[: self.seq_channels] = enc5
        x3 = np.zeros((self.total_c3, self.L3), dtype=np.float32)
        x3[: self.seq_channels] = enc3
        return {
            "utr5": torch.from_numpy(x5),
            "utr3": torch.from_numpy(x3),
            "organ_id": torch.tensor(int(organ), dtype=torch.long),
            "seq_id": seq_id,
            "source": source,
        }


def extract_embeddings(model, manifest: Dict, seq_rows, device: torch.device, batch_size: int = 256):
    L5 = int(manifest["shapes"]["utr5"][1])
    L3 = int(manifest["shapes"]["utr3"][1])
    ds = EmbDataset(seq_rows, L5, L3, manifest)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    records = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            utr5 = batch["utr5"].to(device)
            utr3 = batch["utr3"].to(device)
            organ = batch["organ_id"].to(device)
            cond = model.tissue_embedding(organ)
            feat5 = model.branch5(utr5, cond)
            feat3 = model.branch3(utr3, cond)
            fused = torch.cat([feat5, feat3], dim=-1).detach().cpu().numpy()
            for i in range(fused.shape[0]):
                records.append(
                    {
                        "seq_id": batch["seq_id"][i],
                        "source": batch["source"][i],
                        "organ_id": int(batch["organ_id"][i]),
                        **{f"emb_{j}": fused[i, j] for j in range(fused.shape[1])},
                    }
                )
    return pd.DataFrame(records)


def _invert_vocab(manifest_path: Path) -> Dict[str, int]:
    import json

    data = json.loads(manifest_path.read_text())
    vocab = data.get("organ_vocab", {})
    inv = {}
    for k, v in vocab.items():
        inv[str(v)] = int(k)
    return inv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_predict.yaml")
    ap.add_argument("--input", default="data/derived/multi_model_/all_models_features.csv")
    ap.add_argument("--out", default="data/derived/multi_model_/embeddings.csv")
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--sources", default=None, help="Comma-separated sources to include (e.g., Real (eval),M1 top-k)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default auto)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    manifest_dir = Path(cfg["dataset_dir"])
    if not manifest_dir.is_absolute():
        manifest_dir = (root / manifest_dir).resolve()
    manifest = manifest_dir / "manifest.json"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, man = load_phase1_model(str(manifest), cfg["phase1_checkpoint"], device)
    organ_inv = _invert_vocab(manifest)

    df = pd.read_csv(root / args.input)
    if args.sources:
        keep = set(s.strip() for s in args.sources.split(",") if s.strip())
        if "source" in df.columns:
            df = df[df["source"].isin(keep)]
    score_col = _select_score_col(df)
    if score_col:
        df = df.sort_values(score_col, ascending=False)
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows)

    rows = []
    for _, r in df.iterrows():
        organ_val = r.get("organ_id", None)
        if pd.isna(organ_val):
            organ_val = r.get("organ_name", "")
        try:
            organ_int = int(organ_val)
        except Exception:
            organ_int = organ_inv.get(str(organ_val), 0)
        rows.append((str(r["utr5"]), str(r["utr3"]), int(organ_int), r.get("seq_id", f"seq_{_}"), r.get("source", "")))

    emb_df = extract_embeddings(model, man, rows, device)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(out_path, index=False)
    print(f"[OK] wrote embeddings to {out_path} (n={len(emb_df)})")


if __name__ == "__main__":
    main()
