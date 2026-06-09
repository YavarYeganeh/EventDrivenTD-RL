from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    import simpy  # type: ignore
except ModuleNotFoundError:  # lets the placeholder run even before installing extras
    simpy = None  # type: ignore

from sim.config import SimulationConfig
from sim.interfaces import ControlChannel, ExperienceStore
from sim.runtime import set_env


class SimpleClockEnvironment:
    """Tiny fallback environment used only when SimPy is not installed."""

    def __init__(self) -> None:
        self.now = 0.0

    def run(self, until: float) -> None:
        if until > self.now:
            self.now = float(until)


@dataclass
class SimPyEventDrivenSystem:
    """
    Generic SimPy-backed event-driven system base class.

    Subclass this in a system adapter. The core RL code should never need to
    know the domain-specific model internals. If SimPy is not installed, this
    class falls back to SimpleClockEnvironment for placeholder/demo use only.
    """

    config: SimulationConfig
    control: ControlChannel
    experiences: ExperienceStore
    env: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.env is None:
            if simpy is None:
                self.env = SimpleClockEnvironment()
            else:
                self.env = simpy.Environment()

        set_env(self.env)

    def now(self) -> float:
        return float(self.env.now)

    def run_until(self, target_time: float) -> None:
        self.env.run(until=target_time)

    def enable_policy_initialization(self) -> None:
        # Override in adapter if a policy/decision module needs explicit setup.
        pass

    def request_policy_sync(self) -> None:
        # Override in adapter if policy parameters are loaded/synced externally.
        pass

    def close(self) -> None:
        pass
