#!/usr/bin/env python3
"""
Score sequences with the phase1 predictor.
Features:
- Single-organ scoring (target) -> CSV with seq_id, organ_id, pred
- Optional sweep across all organs for off-target matrix -> NPZ + wide CSV

Inputs must contain columns: utr5, utr3. If seq_id is missing, it will be
generated. If organ_id is not provided in the file, pass --organ-id or
--organ-name. When --sweep-offtarget is enabled, organ in the input is ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.side.predict import PairDataset, load_phase1_model


def _load_manifest(cfg_path: Path) -> Dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    manifest = cfg.get("dataset_dir")
    if not manifest:
        raise ValueError("configs/gen_predict.yaml missing dataset_dir")
    manifest = Path(manifest)
    if not manifest.is_absolute():
        manifest = (cfg_path.parents[1] / manifest).resolve()
    manifest = manifest / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    return json.loads(manifest.read_text()), cfg


def _resolve_organ_id(name: str, vocab: Dict[int, str]) -> int:
    name = name.strip().lower()
    for k, v in vocab.items():
        if str(v).strip().lower() == name:
            return int(k)
    raise ValueError(f"organ_name '{name}' not found in manifest organ_vocab")


def _infer_rows(df: pd.DataFrame, organ_id: int | None, organ_name: str | None, vocab: Dict[int, str]) -> List[tuple[str, str, int]]:
    if "organ_id" in df.columns and organ_id is None and organ_name is None:
        org_ids = df["organ_id"].astype(int).tolist()
    else:
        if organ_id is None and organ_name:
            organ_id = _resolve_organ_id(organ_name, vocab)
        if organ_id is None:
            raise ValueError("organ_id not provided and organ_id column missing")
        org_ids = [int(organ_id)] * len(df)
    rows = list(zip(df["utr5"].tolist(), df["utr3"].tolist(), org_ids))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_predict.yaml")
    ap.add_argument("--input", required=True, help="CSV/TSV with utr5,utr3[,seq_id][,organ_id]")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (without extension)")
    ap.add_argument("--organ-id", type=int, default=None, help="Target organ id (overrides column)")
    ap.add_argument("--organ-name", type=str, default=None, help="Target organ name (overrides column)")
    ap.add_argument("--sweep-offtarget", action="store_true", help="Score across all organs and write NPZ+CSV")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    man, cfg = _load_manifest(cfg_path)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = args.batch_size or int(cfg.get("batch_size", 512))
    num_workers = args.num_workers or int(cfg.get("num_workers", 0))
    model, manifest = load_phase1_model(str((root / cfg["dataset_dir"]).resolve() / "manifest.json"), cfg["phase1_checkpoint"], device)
    L5 = int(manifest["shapes"]["utr5"][1])
    L3 = int(manifest["shapes"]["utr3"][1])
    vocab = {int(k): str(v) for k, v in manifest.get("organ_vocab", {}).items()}
    num_organs = int(manifest.get("num_organs", 0) or len(vocab))

    sep = "," if args.input.endswith(".csv") else "\t"
    df = pd.read_csv(root / args.input, sep=sep)
    if "seq_id" not in df.columns:
        df["seq_id"] = [f"seq_{i:05d}" for i in range(len(df))]

    rows = _infer_rows(df, args.organ_id, args.organ_name, vocab)
    ds = PairDataset(rows, L5, L3, manifest)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            utr5 = batch["utr5"].to(device)
            utr3 = batch["utr3"].to(device)
            organ = batch["organ_id"].to(device)
            y = model(utr5, utr3, organ)
            preds.append(y.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).reshape(-1)
    target_out = Path(args.out_prefix + "_target.csv")
    out_df = pd.DataFrame(
        {
            "seq_id": df["seq_id"],
            "utr5": df["utr5"],
            "utr3": df["utr3"],
            "organ_id": [r[2] for r in rows],
            "pred": pred,
        }
    )
    target_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(target_out, index=False)
    print(f"[OK] wrote target scores: {target_out} ({len(out_df)} rows)")

    if args.sweep_offtarget:
        seq_ids = df["seq_id"].tolist()
        preds_mat = np.zeros((len(df), num_organs), dtype=np.float32)
        for oid in range(num_organs):
            sweep_rows = list(zip(df["utr5"].tolist(), df["utr3"].tolist(), [oid] * len(df)))
            sweep_ds = PairDataset(sweep_rows, L5, L3, manifest)
            sweep_loader = DataLoader(
                sweep_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )
            part = []
            with torch.no_grad():
                for batch in sweep_loader:
                    utr5 = batch["utr5"].to(device)
                    utr3 = batch["utr3"].to(device)
                    organ = batch["organ_id"].to(device)
                    y = model(utr5, utr3, organ)
                    part.append(y.detach().cpu().numpy())
            preds_mat[:, oid] = np.concatenate(part, axis=0).reshape(-1)
            print(f"[sweep] organ {oid}/{num_organs-1} done")

        npz_path = Path(args.out_prefix + "_offtarget.npz")
        np.savez_compressed(npz_path, seq_id=np.array(seq_ids), organ_ids=np.arange(num_organs), preds=preds_mat)
        wide = pd.DataFrame(preds_mat, columns=[f"organ_{i}" for i in range(num_organs)])
        wide.insert(0, "seq_id", seq_ids)
        wide_csv = Path(args.out_prefix + "_offtarget.csv")
        wide.to_csv(wide_csv, index=False)
        print(f"[OK] wrote off-target matrix: {npz_path} and {wide_csv}")


if __name__ == "__main__":
    main()
