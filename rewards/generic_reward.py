from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from rewards.interfaces import RewardResult


Transform = Callable[[float], float]


def identity(x: float) -> float:
    return x


@dataclass(frozen=True)
class RewardComponent:
    """
    One generic reward component.

    name:
        Human-readable name used for logging.

    key:
        Field name to read from the experience record.

    weight:
        Multiplier applied to the transformed value.

    objective:
        Use "minimize" for costs/durations and "maximize" for benefits.

    default:
        Value used when the key is missing.

    transform:
        Optional value transform, for example a normalizer or log transform.
    """

    name: str
    key: str
    weight: float = 1.0
    objective: str = "minimize"
    default: float = 0.0
    transform: Transform = identity

    def value(self, record: Mapping[str, Any]) -> float:
        raw = float(record.get(self.key, self.default))
        transformed = float(self.transform(raw))

        if self.objective == "minimize":
            return -self.weight * transformed

        if self.objective == "maximize":
            return self.weight * transformed

        raise ValueError(
            f"Unknown reward objective {self.objective!r}. "
            "Use 'minimize' or 'maximize'."
        )


@dataclass(frozen=True)
class DeltaRewardComponent:
    """
    Reward component based on the difference between two records.

    The component computes:

        end_record[key] - start_record[key]

    and then applies the same minimize/maximize convention as RewardComponent.
    """

    name: str
    key: str
    weight: float = 1.0
    objective: str = "maximize"
    default: float = 0.0
    transform: Transform = identity

    def value(
        self,
        start_record: Mapping[str, Any],
        end_record: Mapping[str, Any],
    ) -> float:
        start_value = float(start_record.get(self.key, self.default))
        end_value = float(end_record.get(self.key, self.default))
        delta = float(self.transform(end_value - start_value))

        if self.objective == "minimize":
            return -self.weight * delta

        if self.objective == "maximize":
            return self.weight * delta

        raise ValueError(
            f"Unknown reward objective {self.objective!r}. "
            "Use 'minimize' or 'maximize'."
        )


class WeightedRewardModel:
    """
    Generic weighted reward model for one transition/experience.

    This class intentionally knows nothing about any specific application.
    It only expects dictionary-like experience records.
    """

    def __init__(
        self,
        event_components: Sequence[RewardComponent] | None = None,
        group_components: Sequence[RewardComponent] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.event_components = tuple(event_components or default_event_components())
        self.group_components = tuple(group_components or default_group_components())
        self.dtype = dtype
        self.reward_names = tuple(
            component.name
            for component in (*self.event_components, *self.group_components)
        )

    def __call__(
        self,
        experience: Mapping[str, Any],
        *,
        include_group_reward: bool = True,
        **_: Any,
    ) -> RewardResult:
        event_values = [component.value(experience) for component in self.event_components]
        group_values = [component.value(experience) for component in self.group_components]

        event_reward = torch.tensor(sum(event_values), dtype=self.dtype)
        elements = torch.tensor(event_values + group_values, dtype=self.dtype).view(1, -1)

        group_reward = None
        if include_group_reward:
            group_reward = torch.tensor(sum(group_values), dtype=self.dtype)

        return RewardResult(
            event_reward=event_reward,
            group_reward=group_reward,
            elements=elements,
        )

    def aggregate(
        self,
        experience: Mapping[str, Any],
        *,
        include_group_reward: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compatibility helper for older training code.

        Returns:
            (event_reward, elements) when include_group_reward=False
            (event_reward, group_reward, elements) otherwise
        """

        result = self(
            experience,
            include_group_reward=include_group_reward,
            **kwargs,
        )

        if include_group_reward and result.group_reward is not None:
            return result.event_reward, result.group_reward, result.elements

        return result.event_reward, result.elements


class WindowDeltaRewardModel:
    """
    Generic aggregate reward between two records.

    Useful for rewards computed over a time window, segment, episode section,
    or any other pair of boundary records.
    """

    def __init__(
        self,
        components: Sequence[DeltaRewardComponent] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.components = tuple(components or default_delta_components())
        self.dtype = dtype
        self.reward_names = tuple(component.name for component in self.components)

    def __call__(
        self,
        start_record: Mapping[str, Any],
        end_record: Mapping[str, Any],
        **_: Any,
    ) -> RewardResult:
        values = [component.value(start_record, end_record) for component in self.components]
        total = torch.tensor(sum(values), dtype=self.dtype)
        elements = torch.tensor(values, dtype=self.dtype).view(1, -1)

        return RewardResult(
            event_reward=torch.tensor(0.0, dtype=self.dtype),
            group_reward=total,
            elements=elements,
        )

    def aggregate(
        self,
        start_record: Mapping[str, Any],
        end_record: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self(start_record, end_record, **kwargs)

        if result.group_reward is None:
            raise RuntimeError("WindowDeltaRewardModel did not produce a group reward.")

        return result.group_reward, result.elements


def default_event_components() -> tuple[RewardComponent, ...]:
    """
    Six-component default layout with generic names.

    The first three are transition-level components. The last three are
    aggregate/system-level placeholders. This keeps the same broad shape as
    many existing RL logging pipelines while avoiding domain-specific fields.
    """

    return (
        RewardComponent(
            name="service_time_term",
            key="service_time",
            objective="minimize",
        ),
        RewardComponent(
            name="waiting_time_term",
            key="waiting_time",
            objective="minimize",
        ),
        RewardComponent(
            name="constraint_violation_term",
            key="constraint_violation",
            objective="minimize",
        ),
    )


def default_group_components() -> tuple[RewardComponent, ...]:
    return (
        RewardComponent(
            name="throughput_term",
            key="throughput",
            objective="maximize",
        ),
        RewardComponent(
            name="utilization_term",
            key="utilization",
            objective="maximize",
        ),
        RewardComponent(
            name="completion_term",
            key="completion_count",
            objective="maximize",
        ),
    )


def default_delta_components() -> tuple[DeltaRewardComponent, ...]:
    return (
        DeltaRewardComponent(
            name="throughput_delta_term",
            key="throughput",
            objective="maximize",
        ),
        DeltaRewardComponent(
            name="utilization_delta_term",
            key="utilization",
            objective="maximize",
        ),
        DeltaRewardComponent(
            name="completion_delta_term",
            key="completion_count",
            objective="maximize",
        ),
    )


def extract_experience(
    record: Mapping[str, Any],
    reward_model: WeightedRewardModel | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generic experience extraction helper.

    Expected record keys are intentionally generic, for example:

        state_action
        next_state_action
        candidate_state_actions
        action_log_prob
        service_time
        waiting_time
        constraint_violation
        throughput
        utilization
        completion_count
        done
        time

    Unknown keys are preserved.
    """

    model = reward_model or WeightedRewardModel()
    output = dict(record)
    result = model(output, **kwargs)

    output["event_reward"] = result.event_reward
    output["group_reward"] = result.group_reward
    output["reward_elements"] = result.elements

    return output


# Compatibility aliases used by the existing training scripts.
Aggregator = WeightedRewardModel
SegmentAggregator = WindowDeltaRewardModel
Segment_Aggregator = WindowDeltaRewardModel
