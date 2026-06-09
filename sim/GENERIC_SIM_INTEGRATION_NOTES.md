# Generic simulation integration scaffold

This branch replaces the old peripheral `sim.config` dependency at the `simulate.py` boundary.

## What is integrated

- `simulate.py` is now a thin entry point:
  - `from sim.worker import main`
- New generic `sim/` package:
  - `commands.py`: command IDs compatible with the original protocol
  - `config.py`: domain-neutral simulation settings
  - `interfaces.py`: `EventDrivenSystem`, `ControlChannel`, `ExperienceStore`
  - `worker.py`: generic replacement for the old simulation loop
  - `loader.py`: loads a user-provided system adapter via `--system module:function`
  - `simpy_system.py`: base class for SimPy-backed adapters
  - `shared_memory.py`: adapter for the current shared-memory command/replay buffers
  - `in_memory.py`: simple test channels
  - `placeholder_system.py`: non-domain placeholder, not a real simulator
- `train_online.py` now has `--system` and passes it to `simulate.py`.
- Generic top-level `encoder/` package was added.
- Generic top-level `preprocessing/` package was added.
- Compatibility modules were added:
  - `framework/encoder.py`
  - `framework/preprocessing/meta_generator.py`

## What is not yet a real simulator

The default `sim.placeholder_system:build_system` only exists to make the generic boundary importable. It does not generate real state-action tensors, rewards, or complete domain experiences. For training, replace it with your own adapter:

```bash
python train_online.py --system my_system.simpy_adapter:build_system
```

or directly:

```bash
python simulate.py --system my_system.simpy_adapter:build_system --train ...
```

## Adapter contract

Your adapter should return an object that implements `sim.interfaces.EventDrivenSystem`:

- `now()`
- `run_until(target_time)`
- `enable_policy_initialization()`
- `request_policy_sync()`
- `close()`
- attributes: `config`, `control`, `experiences`

Use `sim.simpy_system.SimPyEventDrivenSystem` as the base class if your system uses SimPy.


## Legacy `run_env` compatibility

A thin compatibility shim is provided in `sim/runtime.py`:

```python
from sim.runtime import run_env

run_env(sim_time)
```

It is equivalent to the old helper:

```python
def run_env(sim_time):
    ENV.run(until=sim_time)
```

New generic code should still prefer `system.run_until(sim_time)`. The shim exists so older code can be migrated incrementally without reintroducing a domain-specific `sim.config`.
