from __future__ import annotations

from sim.commands import CMD_WAITING


class InMemoryControlChannel:
    """Small in-memory command channel for tests and examples."""

    def __init__(self, initial_command: int = CMD_WAITING) -> None:
        self.command = initial_command
        self.position = 0

    def get_command(self) -> int:
        return int(self.command)

    def set_command(self, command: int) -> None:
        self.command = int(command)

    def get_position(self) -> int:
        return int(self.position)

    def increment_position(self, amount: int = 1) -> None:
        self.position += int(amount)

    def close(self) -> None:
        pass


class InMemoryExperienceStore:
    """Small in-memory experience store for tests and examples."""

    def __init__(self, control: InMemoryControlChannel) -> None:
        self.control = control
        self.completed_positions: list[int] = []

    def add_completed_experience(self) -> None:
        self.control.increment_position()
        self.completed_positions.append(self.control.get_position())

    def has_completed_experience(self, start_position: int) -> bool:
        return any(position > start_position for position in self.completed_positions)

    def close(self) -> None:
        pass
