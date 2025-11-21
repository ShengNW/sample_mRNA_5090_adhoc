#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.gen.models.cgan import Generator, VOCAB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_cgan.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--organ-id", type=int, required=True)
    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tau", type=float, default=1.0, help="Gumbel-softmax temperature")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-organs", type=int, default=None, help="Override num organs for conditional embedding")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    L5 = int(cfg["arch"]["max_len_utr5"])
    L3 = int(cfg["arch"]["max_len_utr3"])
    hidden = int(cfg["arch"]["hidden"])
    num_layers = int(cfg["arch"]["num_layers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.weights, map_location=device)
    state_num_org = state.get("cond.weight", None)
    state_num_org = state_num_org.shape[0] if state_num_org is not None else None
    num_org_cfg = (cfg.get("arch", {}) or {}).get("num_organs")
    num_organs = args.num_organs or num_org_cfg or state_num_org or 64
    if state_num_org and num_organs != state_num_org:
        print(f"[INFO] overriding num_organs={num_organs} -> {state_num_org} to match checkpoint")
        num_organs = state_num_org

    G = Generator(L5, L3, hidden=hidden, num_layers=num_layers, num_organs=int(num_organs)).to(device)
    G.load_state_dict(state, strict=True)
    G.eval()

    samples = []
    organ = torch.full((args.batch_size,), args.organ_id, dtype=torch.long, device=device)
    while len(samples) < args.num_samples:
        cur = min(args.batch_size, args.num_samples - len(samples))
        organ_batch = organ[:cur]
        z = torch.randn(cur, hidden, device=device)
        with torch.no_grad():
            probs = G(z, organ_batch, tau=args.tau)
        tokens = torch.argmax(probs, dim=-1)
        for row in tokens.cpu().tolist():
            seq = "".join(VOCAB[idx] for idx in row)
            samples.append(seq)
    L = L5
    df = pd.DataFrame({
        "utr5": [s[:L] for s in samples],
        "utr3": [s[L:] for s in samples],
        "organ_id": [args.organ_id] * len(samples),
    })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[CGAN] Wrote {len(df)} samples to {out_path}")


if __name__ == "__main__":
    main()
