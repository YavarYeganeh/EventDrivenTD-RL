from __future__ import annotations

import importlib
from typing import Any

from sim.config import SimulationConfig
from sim.interfaces import EventDrivenSystem


def load_system(
    import_path: str,
    *,
    config: SimulationConfig | None = None,
    cli_args: Any = None,
    unknown_args: list[str] | None = None,
) -> EventDrivenSystem:
    """
    Load a user-provided system builder.

    Example:
        python simulate.py --system examples.simple_queue.system:build_system

    The builder can have one of these signatures:
        build_system(config=config, cli_args=args, unknown_args=unknown)
        build_system(config=config, cli_args=args)
        build_system(config=config)
        build_system()
    """

    module_name, function_name = import_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    builder = getattr(module, function_name)

    attempts = [
        lambda: builder(config=config, cli_args=cli_args, unknown_args=unknown_args or []),
        lambda: builder(config=config, cli_args=cli_args),
        lambda: builder(config=config),
        lambda: builder(),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            system = attempt()
            break
        except TypeError as exc:
            last_error = exc
    else:
        raise TypeError(f"Could not call system builder {import_path}: {last_error}")

    if not isinstance(system, EventDrivenSystem):
        raise TypeError(
            "Loaded system does not implement sim.interfaces.EventDrivenSystem."
        )

    return system
