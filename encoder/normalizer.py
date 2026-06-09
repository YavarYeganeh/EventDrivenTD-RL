from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8


class StandardNormalizer:
    def __init__(self, stats: NormalizationStats) -> None:
        self.stats = stats

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.stats.mean.to(x.device)) / (self.stats.std.to(x.device) + self.stats.eps)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.stats.std.to(x.device) + self.stats.eps) + self.stats.mean.to(x.device)


class IdentityNormalizer:
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x
