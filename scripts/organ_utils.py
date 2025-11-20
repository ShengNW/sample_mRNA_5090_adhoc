#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import yaml


def _load_predict_cfg(root: Path) -> dict:
    cfg_path = root / "configs" / "gen_predict.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing predictor config: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def load_manifest(root: Path) -> dict:
    cfg = _load_predict_cfg(root)
    dataset_dir = cfg.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("configs/gen_predict.yaml missing dataset_dir")
    manifest = Path(dataset_dir)
    if not manifest.is_absolute():
        manifest = (root / manifest).resolve()
    manifest = manifest / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    return json.loads(manifest.read_text())


def organ_vocab(root: Path) -> Dict[int, str]:
    man = load_manifest(root)
    vocab = man.get("organ_vocab") or {}
    out: Dict[int, str] = {}
    for key, val in vocab.items():
        try:
            out[int(key)] = str(val)
        except (TypeError, ValueError):
            continue
    return out


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "organ"


def resolve_organ(root: Path, organ_id: int, organ_name: str | None = None) -> Tuple[int, str, str]:
    vocab = organ_vocab(root)
    if organ_id not in vocab:
        raise ValueError(f"organ_id={organ_id} not found in manifest")
    resolved = vocab[organ_id]
    if organ_name and organ_name.strip() and organ_name.strip() != resolved:
        raise ValueError(f"organ_id {organ_id} maps to '{resolved}' not '{organ_name}'")
    slug = f"organ{organ_id}_{slugify(resolved)}"
    return organ_id, resolved, slug


def suffix_path(path: Path, slug: str | None) -> Path:
    if not slug:
        return path
    return path.with_name(f"{path.stem}.{slug}{path.suffix}")
