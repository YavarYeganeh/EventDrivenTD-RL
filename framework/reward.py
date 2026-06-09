"""
Compatibility layer for generic reward models.

New code should prefer imports from the top-level `rewards` package:

    from rewards import WeightedRewardModel, WindowDeltaRewardModel

This module is kept so existing code using `framework.reward` can continue
importing reward classes while the project is migrated.
"""

from __future__ import annotations

from typing import Any, Mapping

from rewards.generic_reward import (
    Aggregator,
    DeltaRewardComponent,
    RewardComponent,
    SegmentAggregator,
    Segment_Aggregator,
    WeightedRewardModel,
    WindowDeltaRewardModel,
    extract_experience,
)


# Older code may call `extract(...)`. In the generic package, extraction is
# record-based: pass a dictionary-like transition/experience record.
def extract(
    record: Mapping[str, Any],
    aggregate_func: WeightedRewardModel | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return extract_experience(
        record=record,
        reward_model=aggregate_func,
        **kwargs,
    )


__all__ = [
    "Aggregator",
    "DeltaRewardComponent",
    "RewardComponent",
    "SegmentAggregator",
    "Segment_Aggregator",
    "WeightedRewardModel",
    "WindowDeltaRewardModel",
    "extract",
    "extract_experience",
]
