from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch


@dataclass(frozen=True)
class RewardResult:
    """
    Generic reward output.

    event_reward:
        Reward associated with the current decision/transition.

    group_reward:
        Optional aggregate reward associated with a larger window, segment,
        episode section, or performance interval.

    elements:
        Individual weighted reward components, kept for logging/debugging.
    """

    event_reward: torch.Tensor
    elements: torch.Tensor
    group_reward: torch.Tensor | None = None


class RewardModel(Protocol):
    """Generic reward model interface."""

    reward_names: Sequence[str]

    def __call__(self, experience: Mapping[str, Any], **kwargs: Any) -> RewardResult:
        ...


class GroupRewardModel(Protocol):
    """Generic reward interface for window/segment/episode-level rewards."""

    reward_names: Sequence[str]

    def __call__(
        self,
        start_record: Mapping[str, Any],
        end_record: Mapping[str, Any],
        **kwargs: Any,
    ) -> RewardResult:
        ...
