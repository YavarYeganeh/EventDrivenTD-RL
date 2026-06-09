from __future__ import annotations

from typing import Any, Optional


ENV: Optional[Any] = None


def set_env(env: Any) -> None:
    """
    Register the active event-driven environment.

    This compatibility hook supports legacy code that expects a module-level
    ENV object and a run_env(sim_time) function. New code should prefer the
    EventDrivenSystem.run_until(...) adapter method.
    """

    global ENV
    ENV = env



def get_env() -> Any:
    """Return the registered environment, or raise a clear error."""

    if ENV is None:
        raise RuntimeError(
            "No simulation environment has been registered. "
            "Create a SimPyEventDrivenSystem or call sim.runtime.set_env(env) first."
        )

    return ENV



def run_env(sim_time: float) -> None:
    """
    Legacy-compatible simulation runner.

    Equivalent to the original project helper:

        def run_env(sim_time):
            ENV.run(until=sim_time)

    Kept as a thin compatibility layer. The disentangled worker calls
    system.run_until(sim_time) instead.
    """

    get_env().run(until=sim_time)
