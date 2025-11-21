#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

# Allow running as a script without installing the package.
sys.path.append(str(Path(__file__).resolve().parents[1]))
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
    ap.add_argument("--num-organs", type=int, default=None, help="Override num organs for conditional embedding")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.weights, map_location=device)
    state_num_org = None
    if "enc.cond.weight" in state:
        state_num_org = state["enc.cond.weight"].shape[0]

    num_org_cfg = (cfg.get("arch", {}) or {}).get("num_organs") or (cfg.get("train", {}) or {}).get("num_organs")
    num_organs = args.num_organs or num_org_cfg or state_num_org or 32
    if state_num_org and num_organs != state_num_org:
        print(f"[INFO] overriding num_organs={num_organs} -> {state_num_org} to match checkpoint")
        num_organs = state_num_org

    mcfg = CVAEConfig(
        max_len_utr5=int(cfg["arch"]["max_len_utr5"]),
        max_len_utr3=int(cfg["arch"]["max_len_utr3"]),
        latent_dim=int(cfg["arch"]["latent_dim"]),
        hidden=int(cfg["arch"]["hidden"]),
        num_layers=int(cfg["arch"]["num_layers"]),
        num_organs=int(num_organs),
    )
    model = CVAE(mcfg).to(device)
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
