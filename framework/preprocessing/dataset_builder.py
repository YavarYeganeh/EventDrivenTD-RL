from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from encoder.interfaces import Encoder
from preprocessing.schema import DatasetSchema


class DatasetBuilder:
    """Convert raw domain-neutral event records to tensors using an encoder."""

    def __init__(self, encoder: Encoder, schema: DatasetSchema) -> None:
        self.encoder = encoder
        self.schema = schema

    def build_from_records(self, records: Iterable[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        state_actions = []
        next_state_actions = []
        rewards = []
        dones = []
        times = []
        decision_ids = []

        for record in records:
            system_state = record["state"]
            action = record["action"]
            next_system_state = record["next_state"]
            next_action = record.get("next_action")

            state_actions.append(self.encoder.encode_state_action(system_state, action))

            if next_action is None:
                next_state = self.encoder.encode_state(next_system_state)
                next_state_actions.append(self.encoder.terminal_state_action(next_state))
            else:
                next_state_actions.append(self.encoder.encode_state_action(next_system_state, next_action))

            rewards.append(float(record.get("reward", 0.0)))
            dones.append(bool(record.get("done", False)))
            times.append(float(record.get("time", 0.0)))
            decision_ids.append(int(record.get("decision_id", -1)))

        return {
            self.schema.state_action_key: torch.stack(state_actions),
            self.schema.next_state_action_key: torch.stack(next_state_actions),
            self.schema.reward_key: torch.tensor(rewards, dtype=torch.float32),
            self.schema.done_key: torch.tensor(dones, dtype=torch.bool),
            self.schema.time_key: torch.tensor(times, dtype=torch.float32),
            self.schema.decision_id_key: torch.tensor(decision_ids, dtype=torch.long),
        }

    def save(self, dataset: dict[str, torch.Tensor], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output_path)
