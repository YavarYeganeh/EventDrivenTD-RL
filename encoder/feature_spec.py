from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FeatureSpec:
    """Domain-neutral feature dimensions and named indices."""

    state_dim: int
    action_dim: int
    state_action_dim: int
    state_features: Sequence[str] | None = None
    action_features: Sequence[str] | None = None
    feature_index: Mapping[str, int] | None = None
    reward_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_features", tuple(self.state_features or ()))
        object.__setattr__(self, "action_features", tuple(self.action_features or ()))
        object.__setattr__(self, "feature_index", dict(self.feature_index or {}))
        object.__setattr__(self, "reward_names", tuple(self.reward_names or ()))

    def index(self, name: str) -> int:
        try:
            return int(self.feature_index[name])
        except KeyError as exc:
            raise KeyError(f"Unknown feature name: {name!r}") from exc
