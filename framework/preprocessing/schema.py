from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DatasetSchema:
    """Generic offline/online transition schema."""

    state_action_key: str = "state_action"
    next_state_action_key: str = "next_state_action"
    reward_key: str = "reward"
    done_key: str = "done"
    time_key: str = "time"
    decision_id_key: str = "decision_id"
    reward_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_names", tuple(self.reward_names or ()))
