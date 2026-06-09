from __future__ import annotations

from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from sim.commands import CMD_WAITING


DEFAULT_COMMAND_DTYPE = np.dtype([
    ("cmd", np.int32),
    ("position", np.int32),
])

GENERIC_EXPERIENCE_DTYPE = np.dtype([
    ("active", np.bool_),
    ("completed", np.bool_),
])


class SharedMemoryControlChannel:
    """
    Shared-memory implementation of the generic control protocol.

    This is compatible with the current framework.replay_buffer.cmd_dtype shape:
        [("cmd", int32), ("position", int32)]
    """

    def __init__(
        self,
        name: str,
        dtype: np.dtype = DEFAULT_COMMAND_DTYPE,
        shape: tuple[int, ...] = (1,),
        owner: bool = False,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.owner = owner

        nbytes = int(np.prod(shape)) * dtype.itemsize

        self.shm = shared_memory.SharedMemory(
            name=name,
            create=owner,
            size=nbytes if owner else 0,
        )
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

        if owner:
            self.array[0]["cmd"] = CMD_WAITING
            self.array[0]["position"] = 0

    def get_command(self) -> int:
        return int(self.array[0]["cmd"])

    def set_command(self, command: int) -> None:
        self.array[0]["cmd"] = int(command)

    def get_position(self) -> int:
        return int(self.array[0]["position"])

    def close(self) -> None:
        self.shm.close()
        if self.owner:
            self.shm.unlink()


class SharedMemoryExperienceStore:
    """
    Shared-memory implementation of the generic experience protocol.

    This class deliberately only looks for generic completion fields. For
    compatibility with the current replay buffer it also accepts the existing
    uppercase field name "Completed".
    """

    def __init__(
        self,
        name: str,
        dtype: np.dtype,
        shape: tuple[int, ...],
        owner: bool = False,
        control: Optional[SharedMemoryControlChannel] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.owner = owner
        self.control = control

        nbytes = int(np.prod(shape)) * dtype.itemsize

        self.shm = shared_memory.SharedMemory(
            name=name,
            create=owner,
            size=nbytes if owner else 0,
        )
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

    def _completion_field(self) -> str:
        names = self.array.dtype.names or ()
        if "completed" in names:
            return "completed"
        if "Completed" in names:
            return "Completed"
        raise KeyError(
            "Experience store dtype must include either 'completed' or 'Completed'."
        )

    def has_completed_experience(self, start_position: int) -> bool:
        # If attached to a control channel, a position change is the most direct
        # signal that new data has been written.
        if self.control is not None and self.control.get_position() != start_position:
            return True

        names = self.array.dtype.names or ()
        completed_field = self._completion_field()

        completed = self.array[completed_field]
        if "active" in names:
            return bool((self.array["active"] & completed).any())
        return bool(completed.any())

    def close(self) -> None:
        self.shm.close()
        if self.owner:
            self.shm.unlink()
