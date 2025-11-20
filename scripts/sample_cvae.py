#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.gen.models.cvae import CVAE, CVAEConfig, VOCAB


def sample_sequences(model: CVAE, organ_id: int, total: int, batch: int, temperature: float) -> list[str]:
    device = next(model.parameters()).device
    latent_dim = model.cfg.latent_dim
    organ = torch.full((batch,), organ_id, dtype=torch.long, device=device)
    outputs: list[str] = []
    while len(outputs) < total:
        cur = min(batch, total - len(outputs))
        organ_batch = organ[:cur]
        z = torch.randn(cur, latent_dim, device=device)
        logits = model.dec(z, organ_batch)
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        tokens = torch.distributions.Categorical(probs=probs).sample()
        for row in tokens.cpu().tolist():
            seq = "".join(VOCAB[idx] for idx in row)
            outputs.append(seq)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_cvae.yaml")
    ap.add_argument("--weights", required=True, help="Path to CVAE checkpoint")
    ap.add_argument("--organ-id", type=int, required=True)
    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    mcfg = CVAEConfig(
        max_len_utr5=int(cfg["arch"]["max_len_utr5"]),
        max_len_utr3=int(cfg["arch"]["max_len_utr3"]),
        latent_dim=int(cfg["arch"]["latent_dim"]),
        hidden=int(cfg["arch"]["hidden"]),
        num_layers=int(cfg["arch"]["num_layers"]),
        num_organs=int(cfg.get("arch", {}).get("num_organs", cfg.get("train", {}).get("num_organs", 32)))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(mcfg).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    seqs = sample_sequences(model, args.organ_id, args.num_samples, args.batch_size, args.temperature)
    L5 = mcfg.max_len_utr5
    data = {
        "utr5": [s[:L5] for s in seqs],
        "utr3": [s[L5:] for s in seqs],
        "organ_id": [args.organ_id] * len(seqs),
    }
    df = pd.DataFrame(data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[CVAE] Wrote {len(df)} samples to {out_path}")


if __name__ == "__main__":
    main()
