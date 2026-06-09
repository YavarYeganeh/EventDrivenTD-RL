from __future__ import annotations

from typing import Any, Mapping

import torch

from encoder.feature_spec import FeatureSpec


class DictEncoder:
    """
    Minimal dictionary-based encoder.

    This works as a simple example and test utility. Real systems should supply
    their own adapter-specific encoder.
    """

    def __init__(self, feature_spec: FeatureSpec) -> None:
        self.feature_spec = feature_spec

    def encode_state(self, system_state: Mapping[str, Any]) -> torch.Tensor:
        values = [float(system_state.get(name, 0.0)) for name in self.feature_spec.state_features]
        return torch.tensor(values, dtype=torch.float32)

    def encode_action(self, system_state: Mapping[str, Any], action: Mapping[str, Any]) -> torch.Tensor:
        del system_state
        values = [float(action.get(name, 0.0)) for name in self.feature_spec.action_features]
        return torch.tensor(values, dtype=torch.float32)

    def encode_state_action(self, system_state: Mapping[str, Any], action: Mapping[str, Any]) -> torch.Tensor:
        state = self.encode_state(system_state)
        action_tensor = self.encode_action(system_state, action)
        return torch.cat([state, action_tensor], dim=0)

    def encode_candidates(self, system_state: Mapping[str, Any], decision_context: Mapping[str, Any]) -> torch.Tensor:
        actions = decision_context.get("candidate_actions", [])
        encoded = [self.encode_state_action(system_state, action) for action in actions]
        if not encoded:
            return torch.empty(0, self.feature_spec.state_action_dim, dtype=torch.float32)
        return torch.stack(encoded, dim=0)

    def terminal_state_action(self, state: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(self.feature_spec.action_dim, dtype=state.dtype, device=state.device)
        return torch.cat([state, zeros], dim=0)
