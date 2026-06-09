"""
Domain-neutral compatibility encoder module.

Current agents and algorithms still import ``framework.encoder``. This module keeps
those imports working while moving all real encoding decisions into explicit
FeatureSpec/Encoder objects supplied by a system adapter.

No simulator-specific feature groups are defined here. Dimensions and optional
feature indices are configured by environment variables or by constructing a
FeatureSpec in the adapter.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Any

import torch

from encoder.feature_spec import FeatureSpec
from encoder.generic_encoder import DictEncoder


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Generic dimensions
# ---------------------------------------------------------------------------
# These names are kept because existing algorithms import d_s_a and d_s.
# New code should prefer FeatureSpec.state_action_dim and FeatureSpec.state_dim.

d_s_a = _env_int("RL_STATE_ACTION_DIM", 1)
d_s = _env_int("RL_STATE_DIM", d_s_a)
d_action = max(d_s_a - d_s, 0)

state_action_dim = d_s_a
state_dim = d_s
action_dim = d_action


# ---------------------------------------------------------------------------
# Optional generic feature indices
# ---------------------------------------------------------------------------
# Existing heuristic/constraint agents used a domain-specific hardcoded feature.
# In the standalone package this is only a generic optional constraint index.

constraint_feature_id = _env_int("RL_CONSTRAINT_FEATURE_INDEX", -1)
priority_feature_id = _env_int("RL_PRIORITY_FEATURE_INDEX", -1)
processing_time_feature_id = _env_int("RL_PROCESSING_TIME_FEATURE_INDEX", -1)

# Compatibility aliases used by current agents during migration. They are not
# tied to any concrete simulator meaning.
constraint_id = constraint_feature_id
constraint_enc_id = constraint_feature_id
pt_id = processing_time_feature_id


# Optional generic feature-step dictionaries for adapters that want named
# offsets. They start empty and are not populated with system-specific data.
state_feature_steps: dict[str, int] = {}
action_feature_steps: dict[str, int] = {}
feature_steps: dict[str, int] = {}

batch_max_scalar = 1.0


def make_feature_spec(
    state_features: Iterable[str] = (),
    action_features: Iterable[str] = (),
    feature_index: Mapping[str, int] | None = None,
    reward_names: Iterable[str] = (),
) -> FeatureSpec:
    """Create a domain-neutral feature specification."""

    state_features = tuple(state_features)
    action_features = tuple(action_features)

    return FeatureSpec(
        state_dim=d_s,
        action_dim=d_action,
        state_action_dim=d_s_a,
        state_features=state_features,
        action_features=action_features,
        feature_index=dict(feature_index or {}),
        reward_names=tuple(reward_names),
    )


def sa_batchable(items: Any) -> torch.Tensor:
    """Convert rows of state-action data into a float tensor."""

    if isinstance(items, torch.Tensor):
        return items.float()

    if items is None:
        return torch.empty(0, d_s_a, dtype=torch.float32)

    if isinstance(items, (list, tuple)) and len(items) == 0:
        return torch.empty(0, d_s_a, dtype=torch.float32)

    if isinstance(items, (list, tuple)):
        return torch.stack(
            [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in items]
        ).float()

    return torch.as_tensor(items, dtype=torch.float32)


def extract_s_1_started(next_state_action_parent: Any) -> torch.Tensor:
    """
    Compatibility placeholder for next-state extraction.

    System adapters should replace this with logic that constructs the constraintect
    next state-action tensor for their event semantics.
    """

    if isinstance(next_state_action_parent, torch.Tensor):
        return next_state_action_parent.float()

    return torch.as_tensor(next_state_action_parent, dtype=torch.float32)


def retrieve_processing_time(state_action: Any, system_context: Any = None) -> torch.Tensor:
    """Retrieve an optional processing-time-like feature if configured."""

    del system_context
    tensor = state_action if isinstance(state_action, torch.Tensor) else torch.as_tensor(state_action)

    if processing_time_feature_id < 0:
        return torch.zeros(tensor.shape[:-1], dtype=tensor.dtype, device=tensor.device)

    return tensor[..., processing_time_feature_id]


# Compatibility name used by framework.reward.
def retrieve_ptime(state_action: Any, system_context: Any = None) -> torch.Tensor:
    return retrieve_processing_time(state_action, system_context)


def system_summary_vector(system_context: Any = None) -> torch.Tensor:
    """Optional adapter-provided summary features; empty by default."""

    if system_context is not None and hasattr(system_context, "system_summary_vector"):
        value = system_context.system_summary_vector()
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    return torch.empty(0)


def resource_status_vector(system_context: Any = None, scaling: bool = True) -> torch.Tensor:
    """Optional adapter-provided resource-status features; empty by default."""

    if system_context is not None and hasattr(system_context, "resource_status_vector"):
        value = system_context.resource_status_vector(scaling=scaling)
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    return torch.empty(0)


def resource_count_vector(system_context: Any = None) -> torch.Tensor:
    """Optional adapter-provided resource count/capacity features; empty by default."""

    if system_context is not None and hasattr(system_context, "resource_count_vector"):
        value = system_context.resource_count_vector()
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    return torch.empty(0)


__all__ = [
    "FeatureSpec",
    "DictEncoder",
    "d_s_a",
    "d_s",
    "d_action",
    "state_action_dim",
    "state_dim",
    "action_dim",
    "constraint_feature_id",
    "priority_feature_id",
    "processing_time_feature_id",
    "constraint_id",
    "constraint_enc_id",
    "pt_id",
    "state_feature_steps",
    "action_feature_steps",
    "feature_steps",
    "batch_max_scalar",
    "make_feature_spec",
    "sa_batchable",
    "extract_s_1_started",
    "retrieve_processing_time",
    "retrieve_ptime",
    "system_summary_vector",
    "resource_status_vector",
    "resource_count_vector",
]
