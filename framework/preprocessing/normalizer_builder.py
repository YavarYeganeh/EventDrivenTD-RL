from __future__ import annotations

from pathlib import Path

import torch

from encoder.normalizer import NormalizationStats


def build_standard_normalizer(x: torch.Tensor, eps: float = 1e-8) -> NormalizationStats:
    return NormalizationStats(mean=x.mean(dim=0), std=x.std(dim=0), eps=eps)


def save_normalizer(stats: NormalizationStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": stats.mean, "std": stats.std, "eps": stats.eps}, path)


def load_normalizer(path: str | Path) -> NormalizationStats:
    data = torch.load(path, map_location="cpu")
    return NormalizationStats(mean=data["mean"], std=data["std"], eps=float(data.get("eps", 1e-8)))
