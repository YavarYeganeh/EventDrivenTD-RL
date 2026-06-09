from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """
    Domain-neutral simulation settings.

    Keep this generic. Do not add domain-specific fields such as fab, factory,
    machine, lot, tool, product, corrosion, or WIP here. Such information belongs
    in a user-provided system adapter.
    """

    training_mode: bool = True

    # Initial run before policy-controlled training/evaluation begins.
    warmup_time: float = 0.0

    # Additional initial run to fill an experience store before training starts.
    prefill_time: float = 0.0

    # Evaluation horizon.
    horizon_time: float = 1000.0

    # Simulation time advanced per training interaction.
    step_time: float = 1.0

    # Polling interval for controller commands.
    command_poll_seconds: float = 0.05

    scenario_id: Optional[int | str] = None
    random_seed: Optional[int] = None

    load_policy: bool = False
    policy_path: str = ""

    max_empty_intervals: int = 1000

    # Optional output locations used by adapters.
    results_dir: str = ""
    tensor_dir: str = ""


def default_config() -> SimulationConfig:
    return SimulationConfig()
