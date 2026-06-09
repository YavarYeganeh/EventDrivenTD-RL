from __future__ import annotations

from typing import Any, Protocol

import torch

from encoder.feature_spec import FeatureSpec


class Encoder(Protocol):
    """Generic encoder interface for event-driven systems."""

    feature_spec: FeatureSpec

    def encode_state(self, system_state: Any) -> torch.Tensor:
        ...

    def encode_action(self, system_state: Any, action: Any) -> torch.Tensor:
        ...

    def encode_state_action(self, system_state: Any, action: Any) -> torch.Tensor:
        ...

    def encode_candidates(self, system_state: Any, decision_context: Any) -> torch.Tensor:
        ...

    def terminal_state_action(self, state: torch.Tensor) -> torch.Tensor:
        ...
