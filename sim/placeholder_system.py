from __future__ import annotations

from typing import Any


from sim.config import SimulationConfig, default_config
from sim.in_memory import InMemoryControlChannel, InMemoryExperienceStore
from sim.shared_memory import SharedMemoryControlChannel, SharedMemoryExperienceStore
from sim.simpy_system import SimPyEventDrivenSystem


class PlaceholderSimPySystem(SimPyEventDrivenSystem):
    """
    Minimal placeholder system.

    This is intentionally not a real simulator. Replace it with a domain adapter
    such as examples.simple_queue.system:build_system.
    """

    def run_until(self, target_time: float) -> None:
        # SimPy raises if target_time <= now. The worker normally advances time,
        # but this guard makes the placeholder safer.
        if target_time > self.env.now:
            self.env.run(until=target_time)

        # For in-memory demos only, record a completed placeholder experience.
        # Shared-memory training still needs a real adapter to write tensors and
        # reward fields expected by the current sampler.
        add_completed = getattr(self.experiences, "add_completed_experience", None)
        if callable(add_completed):
            add_completed()

    def enable_policy_initialization(self) -> None:
        print("Placeholder system: policy initialization requested.")

    def request_policy_sync(self) -> None:
        print("Placeholder system: policy sync requested.")


def _load_current_experience_dtype() -> Any:
    try:
        from framework.replay_buffer import experience_dtype

        return experience_dtype
    except Exception:
        from sim.shared_memory import GENERIC_EXPERIENCE_DTYPE

        return GENERIC_EXPERIENCE_DTYPE


def build_system(
    config: SimulationConfig | None = None,
    cli_args: Any = None,
    unknown_args: list[str] | None = None,
) -> PlaceholderSimPySystem:
    del unknown_args

    config = config or default_config()

    if cli_args is not None and getattr(cli_args, "cmd_shm", ""):
        control = SharedMemoryControlChannel(name=cli_args.cmd_shm, owner=False)
    else:
        control = InMemoryControlChannel()

    if cli_args is not None and getattr(cli_args, "rb_shm", ""):
        dtype = _load_current_experience_dtype()
        rb_cap = int(getattr(cli_args, "rb_cap", 0) or 1)
        experiences = SharedMemoryExperienceStore(
            name=cli_args.rb_shm,
            dtype=dtype,
            shape=(rb_cap,),
            owner=False,
            control=control if isinstance(control, SharedMemoryControlChannel) else None,
        )
    else:
        experiences = InMemoryExperienceStore(control)

    return PlaceholderSimPySystem(
        config=config,
        control=control,
        experiences=experiences,
        env=None,
    )
