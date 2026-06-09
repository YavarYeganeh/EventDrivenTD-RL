"""
Generic utility functions for the event-driven RL package.

This module keeps the style and most of the reusable logic from the original
``framework.utils`` while removing application-specific concepts.  It does not
assume any particular event-driven system, resource naming scheme, KPI layout,
or feature encoding beyond optional generic feature indices supplied by the
encoder compatibility layer.

Including:
    - set_seed
    - open_pickle
    - load_data_memory
    - sample_experiences_read_memory
    - unpack_data
    - sa_batchable
    - run_envs
    - ExploreScheduler
    - split_into_subspaces
    - Logger
    - enc_normalization_tensor
    - find_selected_ids_s_a
    - feature_normalization_tensor
    - log_metrics
    - save_metrics_to_csv
    - has_cached_state_action
"""

from __future__ import annotations

import gzip
import math
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional for pure RL training
    pd = None

try:
    from framework.encoder import (
        batch_max_scalar,
        processing_time_feature_id,
        d_s, d_s_a
    )
except Exception:  # pragma: no cover - keeps this file usable outside framework package
    batch_max_scalar = 1.0
    processing_time_feature_id = -1
    d_s = 0

try:
    from sim.commands import CMD_RUN_STEP, CMD_WAITING
except Exception:  # pragma: no cover - fallback for standalone utility use
    CMD_RUN_STEP = 1
    CMD_WAITING = 2


# ---------------------------------------------------------------------------
# Generic experience keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceKeys:
    """
    Field names used by generic offline/online records.

    The default keys are domain-neutral.  Compatibility fallback keys are used
    by helper functions so older state-action datasets can still be read during
    migration.
    """

    state_action: str = "state_action"
    next_state_action: str = "next_state_action"
    candidate_state_actions: str = "candidate_state_actions"
    reward: str = "reward"
    reward_components: str = "reward_components"
    group_reward: str = "group_reward"
    nested: str = "nested"
    dtype: str = "dtype"
    start_time: str = "start_time"
    end_time: str = "end_time"


DEFAULT_KEYS = ExperienceKeys()

# Only tensor-role aliases; these are not tied to any specific application.
LEGACY_STATE_ACTION_KEYS = ("state_action", "s_a_0")
LEGACY_NEXT_STATE_ACTION_KEYS = ("next_state_action", "s_a_1")
LEGACY_CANDIDATE_KEYS = ("candidate_state_actions", "s_a_0_all")
LEGACY_END_TIME_KEYS = ("end_time", "out_time")
LEGACY_START_TIME_KEYS = ("start_time", "queue_time")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _first_present(record: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return default


def _ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _view_state_action(value: Any, s_a_dim: int) -> torch.Tensor:
    tensor = _to_tensor(value)
    return tensor.view(-1, s_a_dim)


def has_cached_state_action(entity: Any, attr_name: str = "_s_a_0") -> bool:
    """
    Return True when an object has a cached state-action attribute.

    ``attr_name`` defaults to the historical cache name for compatibility.
    New adapters should pass their own cache attribute name if they use one.
    """

    return hasattr(entity, attr_name)


# Backward-compatible name used in older system adapters.
def existing_s_a_0(entity: Any) -> bool:
    return has_cached_state_action(entity)


# ---------------------------------------------------------------------------
# Reproducibility and persistence
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms are useful for research reproducibility.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def open_pickle(file_path: str | os.PathLike[str]) -> Any:
    """Open a pickle file, supporting both plain and gzip-compressed files."""

    file_path = str(file_path)

    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rb") as f:
            return pickle.load(f)

    with open(file_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Generic offline data loading
# ---------------------------------------------------------------------------


def _iter_source_dirs(
    root_dir: str | os.PathLike[str],
    *,
    source_prefix: str = "source_",
) -> Iterable[tuple[int | str, Path]]:
    """
    Yield ``(source_id, source_path)`` pairs from a root directory.

    Generic default folder form:
        source_0/
        source_1/

    For older data layouts, call with the old prefix from outside this core
    utility instead of hardcoding it here.
    """

    root = Path(root_dir)

    for source_folder in os.listdir(root):
        source_path = root / source_folder

        if not source_path.is_dir():
            continue

        if not source_folder.startswith(source_prefix):
            print(
                "Data Loader: [Skipped] "
                f"'{source_folder}' does not match expected prefix "
                f"'{source_prefix}'."
            )
            continue

        suffix = source_folder[len(source_prefix):]

        try:
            source_id: int | str = int(suffix)
        except ValueError:
            source_id = suffix

        yield source_id, source_path


def _find_experience_dir(
    source_path: Path,
    *,
    experience_dir_name: str = "experiences",
) -> Optional[Path]:
    for dirpath, _, _ in os.walk(source_path):
        path = Path(dirpath)
        if path.name == experience_dir_name:
            return path
    return None


def _record_end_time(record: Mapping[str, Any], end_time_key: str) -> Any:
    return _first_present(record, (end_time_key, *LEGACY_END_TIME_KEYS))


def load_data_memory(
    root_dir: str | os.PathLike[str],
    meta_data: Mapping[Any, Mapping[int, Mapping[str, Any]]],
    *,
    source_prefix: str = "source_",
    experience_dir_name: str = "experiences",
    end_time_key: str = "end_time",
) -> dict[Any, dict[int, Any]]:
    """
    Load offline experience records into memory.

    The function is layout-configurable rather than tied to one simulator.  It
    validates record ordering against metadata using a configurable end-time key.
    """

    data: dict[Any, dict[int, Any]] = {}

    for source_id, source_path in _iter_source_dirs(root_dir, source_prefix=source_prefix):
        print(f"Data Loader: Processing source: {source_id}")

        experience_path = _find_experience_dir(
            source_path,
            experience_dir_name=experience_dir_name,
        )

        data[source_id] = {}

        if experience_path is None:
            print(f"Data Loader: No '{experience_dir_name}' directory found in {source_path}")
            continue

        print(f"Data Loader: Processing experience directory: {experience_path}")

        exp_id = 0

        for file_name in sorted(os.listdir(experience_path)):
            if not (file_name.endswith(".pkl") or file_name.endswith(".pkl.gz")):
                continue

            records = open_pickle(experience_path / file_name)

            for record in records:
                meta_record = meta_data[source_id][exp_id]
                expected_end = _record_end_time(meta_record, end_time_key)
                observed_end = _record_end_time(record, end_time_key)

                if expected_end == observed_end:
                    data[source_id][exp_id] = record
                else:
                    raise ValueError(
                        "Data Loader: Mismatch in end time for "
                        f"source {source_id}, exp_id {exp_id}: "
                        f"{expected_end} != {observed_end}"
                    )

                exp_id += 1

    return data


def sample_experiences_read_memory(
    meta_data: Mapping[Any, Mapping[int, Mapping[str, Any]]],
    full_experiences: Mapping[Any, Mapping[int, Mapping[str, Any]]],
    aggregate: Callable[[Mapping[str, Any]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    len_exps: Mapping[Any, int],
    num_sampled_seeds: int = 5,
    s_a_dim: int = d_s_a,
    *,
    keys: ExperienceKeys = DEFAULT_KEYS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], list[int]]:
    """
    Sample grouped offline experiences already loaded in memory.

    The sampler keeps the original return contract used by existing algorithms:
        state_action, next_state_action, reward, reward_components,
        group_sizes, state_action_sizes, next_state_action_sizes
    """

    group_sizes: list[int] = []
    s0_sizes: list[int] = []
    s1_sizes: list[int] = []

    available_sources = list(len_exps.keys())
    k = min(num_sampled_seeds, len(available_sources))
    sampled_sources = random.sample(available_sources, k)

    state_actions: list[torch.Tensor] = []
    next_state_actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    reward_components: list[torch.Tensor] = []

    for source_id in sampled_sources:
        exp_id = random.randint(0, len_exps[source_id] - 1)

        _, group_reward, _ = aggregate(full_experiences[source_id][exp_id])

        nested_ids = meta_data[source_id][exp_id].get(keys.nested, [])
        group_ids = [exp_id] + list(nested_ids)
        group_sizes.append(len(group_ids))

        for gid in group_ids:
            exp = full_experiences[source_id][gid]

            s0 = _first_present(exp, LEGACY_STATE_ACTION_KEYS)
            s1 = _first_present(exp, LEGACY_NEXT_STATE_ACTION_KEYS)

            if s0 is None or s1 is None:
                raise KeyError(
                    "Experience record must contain state-action and "
                    "next-state-action tensors."
                )

            s0 = _view_state_action(s0, s_a_dim)
            s1 = _view_state_action(s1, s_a_dim)

            event_reward, _, components = aggregate(exp)

            state_actions.append(s0)
            next_state_actions.append(s1)
            rewards.append((event_reward + group_reward).view(1, 1))
            reward_components.append(components)

            s0_sizes.append(s0.shape[0])
            s1_sizes.append(s1.shape[0])

    return (
        torch.cat(state_actions, dim=0),
        torch.cat(next_state_actions, dim=0),
        torch.cat(rewards, dim=0),
        torch.cat(reward_components, dim=0),
        group_sizes,
        s0_sizes,
        s1_sizes,
    )


def unpack_data(
    root_dir: str | os.PathLike[str],
    meta_data: MutableMapping[Any, MutableMapping[int, MutableMapping[str, Any]]],
    store_dir: str | os.PathLike[str] = "./_buffer",
    *,
    source_prefix: str = "source_",
    experience_dir_name: str = "experiences",
    end_time_key: str = "end_time",
) -> MutableMapping[Any, MutableMapping[int, MutableMapping[str, Any]]]:
    """
    Convert grouped pickle records into torch files and attach paths to metadata.

    This keeps the original lazy-loading logic, but source folder naming,
    experience directory naming, and end-time fields are configurable.
    """

    store_dir = Path(store_dir)

    if store_dir.exists():
        print(f"[clean] Removing existing directory: {store_dir}")
        shutil.rmtree(store_dir)

    _ensure_dir(store_dir)
    print(f"Data Loader: Loading data into {store_dir}.")

    for source_id, source_path in _iter_source_dirs(root_dir, source_prefix=source_prefix):
        print(f"Data Loader: Processing source: {source_id}")

        experience_path = _find_experience_dir(
            source_path,
            experience_dir_name=experience_dir_name,
        )

        if experience_path is None:
            print(f"Data Loader: No '{experience_dir_name}' directory found in {source_path}")
            continue

        print(f"Data Loader: Processing experience directory: {experience_path}")

        exp_id = 0

        for file_name in sorted(os.listdir(experience_path)):
            if not (file_name.endswith(".pkl") or file_name.endswith(".pkl.gz")):
                continue

            segment_id = Path(file_name).name.split(".")[0]
            records = open_pickle(experience_path / file_name)

            tensor_path = store_dir / f"{source_id}_{segment_id}.pt"
            torch.save(records, tensor_path)

            for local_index, record in enumerate(records):
                meta_record = meta_data[source_id][exp_id]
                expected_end = _record_end_time(meta_record, end_time_key)
                observed_end = _record_end_time(record, end_time_key)

                if expected_end != observed_end:
                    raise ValueError(
                        "Data Loader: Mismatch in end time for "
                        f"source {source_id}, exp_id {exp_id}: "
                        f"{expected_end} != {observed_end}"
                    )

                meta_record["tensor_path"] = str(tensor_path)
                meta_record["tensor_index"] = local_index
                exp_id += 1

    return meta_data


# ---------------------------------------------------------------------------
# Feature and action helpers
# ---------------------------------------------------------------------------


def sa_batchable(
    state_action: torch.Tensor,
    batchable_feature_id: Optional[int] = None,
    batch_size_feature_id: Optional[int] = None,
    batch_size_scale: float = batch_max_scalar,
) -> tuple[bool, int]:
    """
    Generic batchability helper.

    No feature semantics are hardcoded.  If no batchability feature is supplied,
    the action is treated as a single item.
    """

    if batchable_feature_id is None or batchable_feature_id < 0:
        return False, 1

    batchable = bool(state_action[0, batchable_feature_id])

    if batchable and batch_size_feature_id is not None and batch_size_feature_id >= 0:
        max_batch_size = batch_size_scale * state_action[0, batch_size_feature_id]
    else:
        max_batch_size = 1

    return batchable, int(max_batch_size)


def feature_normalization_tensor(
    x: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    feature_id: Optional[int] = None,
    *,
    apply_log1p: bool = False,
) -> torch.Tensor:
    """
    Normalize one configured feature column without assuming what it means.
    """

    if feature_id is None or feature_id < 0:
        warnings.warn(
            "Feature normalization requested, but no valid feature_id was supplied. "
            "Returning the input unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
        return x

    x_new = x.clone()
    values = x_new[:, feature_id]

    if apply_log1p:
        values = torch.log1p(values)

    x_new[:, feature_id] = transform(values)
    return x_new


def enc_normalization_tensor(x: torch.Tensor, tr: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Compatibility wrapper used by existing training scripts.

    New code should call ``feature_normalization_tensor`` with an explicit
    feature id and transform policy.
    """

    return feature_normalization_tensor(
        x,
        tr,
        feature_id=processing_time_feature_id,
        apply_log1p=True,
    )


def find_selected_ids_s_a(
    candidate_state_actions: torch.Tensor,
    selected_state_actions: torch.Tensor,
    atol: float = 1e-8,
    state_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Match selected state-action rows to candidate state-action rows.

    By default, the state prefix is ignored and only the action part is matched,
    preserving the original behavior.
    """

    state_dim = d_s if state_dim is None else state_dim

    eq = torch.isclose(
        selected_state_actions[:, state_dim:].unsqueeze(1),
        candidate_state_actions[:, state_dim:].unsqueeze(0),
        atol=atol,
    ).all(dim=2)

    if not eq.any(dim=1).all():
        warnings.warn(
            "Some selected state-action rows were not found in the candidate set.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eq.float().argmax(dim=1)


# ---------------------------------------------------------------------------
# Generic metric logging
# ---------------------------------------------------------------------------


def log_metrics(
    logger_obj: Any,
    time_value: float,
    metrics: Mapping[str, Any],
    saving_interval: int = 5,
    *,
    rows_attr: str = "metric_rows",
    counter_attr: str = "log_counter",
    csv_path_attr: str = "metrics_csv_path",
) -> None:
    """
    Append generic time-indexed metrics and periodically flush them to CSV.

    ``logger_obj`` can be any object that stores the given attributes.
    """

    if not hasattr(logger_obj, counter_attr):
        setattr(logger_obj, counter_attr, 0)

    if not hasattr(logger_obj, rows_attr):
        setattr(logger_obj, rows_attr, [])

    setattr(logger_obj, counter_attr, getattr(logger_obj, counter_attr) + 1)

    row = {"time": time_value}
    row.update(dict(metrics))
    getattr(logger_obj, rows_attr).append(row)

    if getattr(logger_obj, counter_attr) % saving_interval == 0:
        save_metrics_to_csv(
            logger_obj,
            rows_attr=rows_attr,
            csv_path_attr=csv_path_attr,
        )


def save_metrics_to_csv(
    logger_obj: Any,
    *,
    rows_attr: str = "metric_rows",
    csv_path_attr: str = "metrics_csv_path",
) -> None:
    rows = getattr(logger_obj, rows_attr, [])

    if not rows:
        return

    if not hasattr(logger_obj, csv_path_attr):
        raise AttributeError(f"Metric logger object is missing '{csv_path_attr}'.")

    path = getattr(logger_obj, csv_path_attr)
    _ensure_dir(Path(path).parent)

    if pd is None:
        import csv

        write_header = not os.path.exists(path)
        fieldnames = list(rows[0].keys())

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    else:
        df_new = pd.DataFrame(rows)
        df_new.to_csv(
            path,
            mode="a" if os.path.exists(path) else "w",
            header=not os.path.exists(path),
            index=False,
        )

    rows.clear()


# Compatibility wrapper for older code that called log_kpis(...).
def log_kpis(
    logger_obj: Any,
    metric_a: float,
    metric_b: float,
    metric_c: float,
    saving_interval: int = 5,
    *,
    metric_names: Sequence[str] = ("metric_a", "metric_b", "metric_c"),
) -> None:
    env = getattr(logger_obj, "_env", None)
    time_value = float(getattr(env, "now", 0.0))

    log_metrics(
        logger_obj,
        time_value=time_value,
        metrics={
            metric_names[0]: metric_a,
            metric_names[1]: metric_b,
            metric_names[2]: metric_c,
        },
        saving_interval=saving_interval,
        rows_attr="kpi_data",
        csv_path_attr="kpi_csv_path",
    )


def _save_kpis_to_csv(logger_obj: Any) -> None:
    save_metrics_to_csv(
        logger_obj,
        rows_attr="kpi_data",
        csv_path_attr="kpi_csv_path",
    )


# ---------------------------------------------------------------------------
# Multi-worker command helpers
# ---------------------------------------------------------------------------


def _command_value(slot: Any) -> int:
    value = slot["cmd"]

    try:
        value = value[0]
    except Exception:
        pass

    try:
        value = value.item()
    except AttributeError:
        pass

    return int(value)


def all_idle(cmd: Mapping[Any, Any], idle_command: int = CMD_WAITING) -> bool:
    """Return True if all command slots are in the idle/waiting state."""

    return all(_command_value(slot) == idle_command for slot in list(cmd.values()))


def all_idle_other(cmd_dict: Mapping[Any, Any]) -> bool:
    """Backward-compatible alias for all_idle."""

    return all_idle(cmd_dict)


def run_envs(
    cmd: MutableMapping[Any, Any],
    poll_interval: float = 0.05,
    timeout: float = 6 * 3600,
    *,
    run_command: int = CMD_RUN_STEP,
    idle_command: int = CMD_WAITING,
) -> None:
    """
    Wait until all workers are idle, command them to run, then wait again.
    """

    start = time.time()
    warned = False

    while not all_idle(cmd, idle_command=idle_command):
        if timeout and (time.time() - start) > timeout and not warned:
            print(
                "Warning [Training] = Waiting longer than timeout for all "
                "workers to become idle before run."
            )
            warned = True
        time.sleep(poll_interval)

    for slot in cmd.values():
        slot["cmd"] = run_command

    start = time.time()
    warned = False

    while not all_idle(cmd, idle_command=idle_command):
        if timeout and (time.time() - start) > timeout and not warned:
            print(
                "Warning [Training] = Waiting longer than timeout for all "
                "workers to become idle after run."
            )
            warned = True
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Exploration schedule and experiment utilities
# ---------------------------------------------------------------------------


class ExploreScheduler:
    """
    Generic exploration-value scheduler by absolute iteration.

    At iteration 1, it returns ``explore_start``.  At and after
    ``max_explore_it``, it returns ``explore_end``.  Between those points, it
    interpolates using linear, cosine, exponential, or a custom schedule.
    """

    def __init__(
        self,
        explore_start: float = 3.0,
        explore_end: float = 1.0,
        max_explore_it: int = 10_000,
        mode: str = "cosine",
        min_value: float = 1e-6,
        custom_interp: Optional[Callable[[float], float]] = None,
    ) -> None:
        assert max_explore_it >= 1, "max_explore_it must be >= 1"

        self.explore_start = float(explore_start)
        self.explore_end = float(explore_end)
        self.max_explore_it = int(max_explore_it)
        self.mode = mode.lower()
        self.min_value = float(min_value)
        self.custom_interp = custom_interp
        self._it = 0

        if self.mode not in {"linear", "cosine", "exp"} and custom_interp is None:
            raise ValueError(
                "mode must be 'linear', 'cosine', or 'exp' unless "
                "custom_interp is provided."
            )

    def _interp_value(self, t: float) -> float:
        if self.custom_interp is not None:
            weight = float(self.custom_interp(t))
            return self.explore_start + (self.explore_end - self.explore_start) * weight

        if self.mode == "linear":
            weight = t
            return self.explore_start + (self.explore_end - self.explore_start) * weight

        if self.mode == "cosine":
            weight = 0.5 * (1.0 - math.cos(math.pi * t))
            return self.explore_start + (self.explore_end - self.explore_start) * weight

        start = max(self.explore_start, self.min_value)
        end = max(self.explore_end, self.min_value)
        return math.exp((1 - t) * math.log(start) + t * math.log(end))

    def at(self, it: int) -> float:
        """Exploration value at absolute iteration number, using 1-based indexing."""

        i = max(1, int(it))

        if i >= self.max_explore_it:
            return max(self.explore_end, self.min_value)

        denom = max(1, self.max_explore_it - 1)
        t = (i - 1) / denom
        value = self._interp_value(t)
        return max(float(value), self.min_value)

    __call__ = at

    def step(self, k: int = 1) -> float:
        """Advance the internal iteration counter and return the new value."""

        self._it = max(1, self._it + int(k))
        return self.at(self._it)

    @property
    def current_it(self) -> int:
        return max(1, self._it)

    def reset(self, it: int = 1) -> None:
        self._it = max(1, int(it))


def split_into_subspaces(scenarios: Iterable[Any], num_subspaces: int) -> list[list[Any]]:
    """Shuffle items and split them into near-equal subspaces."""

    scenarios = list(scenarios)
    random.shuffle(scenarios)

    if num_subspaces <= 0:
        raise ValueError("num_subspaces must be positive.")

    count = len(scenarios)
    base = count // num_subspaces
    extra = count % num_subspaces

    subspaces: list[list[Any]] = []
    idx = 0

    for i in range(num_subspaces):
        size = base + (1 if i < extra else 0)
        subspaces.append(scenarios[idx: idx + size])
        idx += size

    return subspaces


class Logger:
    """Redirect stdout to both console and a file."""

    def __init__(self, filename: str = "training_log.txt") -> None:
        self.console = sys.stdout
        _ensure_dir(Path(filename).parent)
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.console.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.console.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "ExperienceKeys",
    "DEFAULT_KEYS",
    "has_cached_state_action",
    "existing_s_a_0",
    "set_seed",
    "open_pickle",
    "load_data_memory",
    "sample_experiences_read_memory",
    "unpack_data",
    "sa_batchable",
    "feature_normalization_tensor",
    "enc_normalization_tensor",
    "find_selected_ids_s_a",
    "log_metrics",
    "save_metrics_to_csv",
    "log_kpis",
    "_save_kpis_to_csv",
    "all_idle",
    "all_idle_other",
    "run_envs",
    "ExploreScheduler",
    "split_into_subspaces",
    "Logger",
]
