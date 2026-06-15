
<h2 align="center">
    Event-driven Reinforcement Learning for Long-horizon Control in Asynchronous, Discrete-event Systems
</h2>

<p align="center">
  <img src="img/new_layout.png" alt="EventDrivenTD-RL modular architecture" width="900"/>
</p>

<p align="center">
  <em>
    Modular event-driven RL architecture for policy interaction, representation, reward computation,
    replay, and training in asynchronous discrete-event systems.
  </em>
</p>

`EventDrivenTD-RL` is a modular reinforcement learning framework for systems where decisions/actions are triggered by events rather than fixed time steps. It is designed for long-horizon control problems with delayed feedback, adaptive candidate sets, overlapping events, and system-level objectives.

The framework was developed in the context of event-driven reinforcement learning for semiconductor fabrication, but the public repository is intended to provide the **generic RL framework only**. It does **not** include proprietary industrial data, private simulator internals, confidential scenario files, or application-specific database schemas.

## Associated Paper

This repository accompanies the research framework described in:

**Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication**

The paper studies reinforcement learning for long-horizon control in event-driven systems, using semiconductor fabrication as a challenging representative application. The code in this repository is intended to provide the reusable RL framework independent of proprietary data and simulator internals.

<p align="center">
    <a href="https://arxiv.org/abs/2606.10705">📄 Read the Paper (arXiv:2606.10705)</a>
</p>

## Overview

Many real-world control systems are naturally event-driven:

- Manufacturing and production systems;
- Semiconductor fabrications
- Logistics and supply chains;
- Healthcare operations;
- Telecommunications and network control;
- Resource-constrained service systems;
- SimPy and discrete-event simulation environments;
- Digital-twin-based control systems.

In such systems:

- Dctions occur at irregular decision times;
- Actions may have variable duration;
- Multiple events may overlap;
- Feasible action sets may change at every decision;
- Local actions may influence delayed global objectives;
- Rewards may be sparse, delayed, or computed over time windows;
- One-step Markov transitions may be insufficient for credit assignment.

`EventDrivenTD-RL` addresses these issues by separating the reusable RL core from the application-specific environment through a modular adapter structure.

## Key Features

- Generic event-driven simulation interface
- SimPy-compatible adapter structure
- Candidate-set action selection with variable action spaces
- Event-level and group-level reward support
- Event-group temporal-difference learning
- Offline training from stored experience
- Online training through simulator interaction
- Generic encoder interfaces
- Generic reward-model interfaces
- Replay-buffer and sampler utilities
- Multiple model-free RL backbones
- Modular agent factory
- Generic placeholder system for testing package wiring
- Compatibility wrappers for migration from application-specific systems
- Efficient GPU-based policy optimization
- Parallel simulation-worker support for online training
- Simultaneous simulator interaction and trainer updates through shared-memory coordination

The package separates:

```text
environment interaction
state/action encoding
reward construction
experience storage
experience sampling
event grouping
temporal-difference aggregation
policy optimization
logging and evaluation
```

## Training Modes

`EventDrivenTD-RL` supports two complementary training modes: **offline training** and **online training**. Both use the same modular RL core, including the encoder, reward, replay, sampler, and algorithm interfaces, but differ in how experience is generated and consumed.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="img/12_offline_training.png" alt="Offline Training pipeline" width="100%"/>
      <br/>
      <strong>Offline training</strong>
    </td>
    <td align="center" width="50%">
      <img src="img/13_online_training.png" alt="Online Training pipeline" width="100%"/>
      <br/>
      <strong>Online training</strong>
    </td>
  </tr>
</table>

In **offline training**, the policy is learned from a fixed dataset of previously collected experience. The simulator or environment does not need to run during gradient updates: stored transitions are loaded, sampled through the replay/sampler interface, and used to train the agent efficiently. This mode is useful for pretraining, conservative offline RL, checkpoint selection, and learning when direct exploration is costly, risky, or unavailable.

In **online training**, the policy is improved through interaction with an event-driven simulator or system adapter. The environment generates new experience while the training process updates the agent from recent or replayed data. This enables policy synchronization, simulator-based fine-tuning, adaptive exploration, and continued improvement after offline pretraining. The design is efficient because of parallel simulations can run on CPU while neural-network optimization runs on GPU.

Together, the two modes support a practical workflow for long-horizon event-driven control: learn an initial policy from stored experience, then refine it through controlled online interaction.

| Mode | Experience source | Main use | Efficiency benefit |
|---|---|---|---|
| Offline | Fixed replay/logged data | Pretraining, offline policy optimization, conservative RL, model selection | No simulator interaction needed during training, CPU data loading and GPU training |
| Online | New simulator/system interaction | Fine-tuning, online policy optimization, exploration, policy improvement, safe initialization before live interaction | CPU simulations and GPU training can be pipelined |

## Supported RL Backbones

The framework includes wrappers and agents for several reinforcement learning families in the different modes adopted by the proposed Aggregating Temporal Difference method:

| Algorithm family | Description |
|---|---|
| DQL / DDQN | Value-based learning with target networks |
| CQL | Conservative Q-learning for offline RL |
| IQL | Implicit Q-learning with value learning and advantage-weighted policy extraction |
| PPO-style methods | Clipped policy-gradient updates with event-driven advantage construction |
| SAC-style methods | Entropy-regularized actor-critic learning |
| Weighted TD | Weighted temporal-difference aggregation for event groups |
| Heuristic agents | Random, FIFO-style, and minimum-feature / SPT-style baselines |

*More algorithms can be integrated.*

<!-- ## Simulator Integration

A key motivation of the package is efficient training. When integrated with a SimPy-compatible simulator, multiple simulation workers can run in parallel on CPU while the trainer performs batched neural-network updates on GPU. This enables simultaneous environment interaction, replay generation, policy optimization, and checkpoint synchronization, which is especially useful for long-horizon event-driven systems where simulation and learning can otherwise become computational bottlenecks.

`EventDrivenTD-RL` provides the reinforcement-learning framework, but it is not a standalone simulator.  
To run meaningful experiments, the framework must be integrated with an external event-driven simulator or environment.

The preferred integration style is a **SimPy-compatible simulator**, since the framework is designed around asynchronous event execution, variable decision times, and event-driven interaction. Other discrete-event simulators can also be used if they expose the required adapter interface.

The simulator is responsible for system evolution, event execution, and candidate generation. The RL framework is responsible for encoding, reward construction, replay storage, sampling, policy optimization, checkpointing, and training coordination.

In online mode, the framework supports simultaneous simulation and training. Simulation workers can run on CPU, while the learning process performs neural-network optimization on GPU. Experience and synchronization signals can be exchanged through shared-memory structures, enabling efficient interaction between the simulator and trainer.

```text
Event-driven simulator
        |
        v
System adapter
        |
        v
Encoder + reward model
        |
        v
Replay buffer and sampler
        |
        v
RL algorithm and agent
        |
        v
Updated policy
```

This design keeps simulator-specific logic outside the RL core and allows the same framework to be reused across different event-driven systems.

--- -->


## Simulator Integration

`EventDrivenTD-RL` provides the reinforcement-learning framework, but it is not a standalone simulator. To run meaningful experiments, it must be integrated with an external event-driven simulator or environment, preferably a **SimPy-compatible simulator**.

The package follows an **ISM-style integration concept**: the simulator handles system dynamics, event execution, and candidate generation, while the RL framework handles state/action encoding, reward construction, replay storage, sampling, policy optimization, and checkpoint synchronization.

New simulators should be connected through the provided **system and action interfaces** rather than by modifying the RL core. A simulator adapter should expose the current state, feasible candidate actions, action execution, and generated experience records.

A key motivation is efficient training. In online mode, multiple simulation workers can run in parallel on CPU while the trainer performs batched neural-network updates on GPU. This enables simultaneous simulator interaction, replay generation, policy optimization, and policy synchronization, which is especially useful for long-horizon event-driven systems.

```text
Event-driven simulator
        |
        v
System / action interface
        |
        v
Encoder + reward model
        |
        v
Replay buffer and sampler
        |
        v
RL algorithm and agent
        |
        v
Updated policy
```

This design keeps simulator-specific logic outside the RL core and allows the same framework to be reused across different event-driven systems.

Conceptually, a simulator adapter should provide:
```text
def get_state():
    ...

def get_candidate_actions():
    ...

def apply_action(action):
    ...

def collect_experience():
    ...
```

---

## Code Structure

The repository is organized around three main executable blocks:

```text
simulate.py
train_offline.py
train_online.py
```

These scripts form the main user-facing interface of the framework.

```text
.
├── simulate.py
├── train_offline.py
├── train_online.py
│
├── framework/
│   ├── agent.py
│   ├── encoder.py
│   ├── replay_buffer.py
│   ├── sampler.py
│   ├── reward.py
│   ├── utils.py
│   ├── agents/
│   │   ├── random.py
│   │   ├── fifo.py
│   │   ├── spt.py
│   │   ├── dql.py
│   │   ├── cql.py
│   │   ├── iql.py
│   │   ├── ac_generic.py
│   │   └── ppo_v.py
│   └── algorithms/
│       ├── offline/
│       └── online/
│
├── sim/*
│
├── encoder/*
│
├── rewards/*
│
└── preprocessing/*
```

### Main Blocks

| Block | Purpose |
|---|---|
| `simulate.py` | Starts the simulation interface and connects the RL policy to the event-driven environment. |
| `train_offline.py` | Trains an agent from stored experience without active simulator interaction. |
| `train_online.py` | Trains an agent while interacting with one or more simulation workers. |

### Framework Core

The `framework/` directory contains the reusable RL core.

| Module | Purpose |
|---|---|
| `framework/agent.py` | Agent construction, policy loading, and checkpoint utilities. |
| `framework/replay_buffer.py` | Experience storage for event-driven transitions. |
| `framework/sampler.py` | Sampling and temporal aggregation of event-driven experience. |
| `framework/reward.py` | Reward aggregation for event-level and group-level feedback. |
| `framework/encoder.py` | Bridge between simulator-specific encodings and the RL core. |
| `framework/utils.py` | Shared utilities for logging, seeding, exploration, and process coordination. |

### Agents and Algorithms

The repository already includes several agents and algorithm implementations under:

```text
framework/agents/
framework/algorithms/offline/
framework/algorithms/online/
```

These modules are intentionally extensible. New agents, offline algorithms, online algorithms, samplers, encoders, or reward models can be added without changing the main training scripts, as long as they follow the same basic interfaces used by the existing implementations.

```text
new agent          -> framework/agents/
new offline method -> framework/algorithms/offline/
new online method  -> framework/algorithms/online/
new sampler        -> framework/sampler.py or a sampler module
new reward model   -> framework/reward.py or rewards/
new simulator      -> sim/ or an external simulator adapter
```

---


## Requirements

Main dependencies:

```txt
torch>=2.4.1
numpy>=2.2.1
pandas>=2.2.3
simpy>=4.1.1
```

### Environment Setup

Using `venv`:

```bash
git clone https://github.com/YavarYeganeh/EventDrivenTD-RL.git
cd EventDrivenTD-RL

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Using Conda:

```bash
git clone https://github.com/YavarYeganeh/EventDrivenTD-RL.git
cd EventDrivenTD-RL

conda create -n eventdriven-td-rl python=3.10 -y
conda activate eventdriven-td-rl

python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
git clone https://github.com/YavarYeganeh/EventDrivenTD-RL.git
cd EventDrivenTD-RL

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---


## Example Command-Line Usage

The three main entry points can be used through command-line arguments.  
The examples below are intentionally generic; exact arguments may depend on the simulator adapter, algorithm, sampler, and experiment configuration.

---

### Simulation

The simulation entry point connects the RL policy to an event-driven simulator or runs a simulation worker during online training.

```bash
python simulate.py --system <simulator_adapter> --agent <agent_name> --load_path <checkpoint_path>
```

**Usage: `simulate.py [-h]`**

- `--system` : Path to the simulator adapter or system builder.
- `--agent` : Agent or policy type to use during simulation.
- `--load_path` : Path to a saved checkpoint.
- `--train` : Run the simulator as a training worker.
- `--horizon_time` : Simulation horizon or stopping time.
- `--output_dir` : Directory for logs, generated experience, or evaluation results.
- `--precision` : Numerical precision used by the policy.

Example:

```bash
python simulate.py \
  --system my_project.my_system:build_system \
  --agent dql \
  --load_path ./results/checkpoints/agent.pt
```

---

### Offline Training

Offline training learns from stored experience without running the simulator during gradient updates.

```bash
python train_offline.py --algorithm <algorithm_name> --data_path <dataset_path> --num_iterations <steps>
```

**Usage: `train_offline.py [-h]`**

- `--algorithm` : Offline RL algorithm to train.
- `--data_path` : Path to stored experience or replay data.
- `--meta_data_path` : Optional path to precomputed metadata.
- `--num_iterations` : Number of training iterations.
- `--seed` : Random seed for reproducibility.
- `--device` : Training device, such as CPU or CUDA.
- `--dtype` : Numerical precision for training.
- `--td_agg` : Temporal-difference aggregation mode.
- `--gamma` : Discount factor or bootstrapping coefficient.
- `--lr` : Learning rate.
- `--save_interval` : Checkpoint saving interval.

Example:

```bash
python train_offline.py \
  --algorithm dql \
  --data_path ./data/offline \
  --num_iterations 100000 \
  --device cuda
```

---

### Online Training

Online training improves a policy through simulator interaction. Simulation workers generate new event-driven experience while the trainer updates the policy.

```bash
python train_online.py --algorithm <algorithm_name> --train_scenarios <source_ids> --num_iterations <steps>
```

**Usage: `train_online.py [-h]`**

- `--algorithm` : Online RL algorithm to train.
- `--train_scenarios` : List of simulator sources or training instances.
- `--use_pretrained` : Start from a pretrained checkpoint.
- `--load_path` : Path to the pretrained checkpoint.
- `--num_iterations` : Number of online interaction/training iterations.
- `--seed` : Random seed for reproducibility.
- `--rb_cap` : Replay-buffer capacity.
- `--sampler` : Sampling strategy for online replay.
- `--td_agg` : Temporal-difference aggregation mode.
- `--sim_interval_time` : Amount of simulator time advanced between training updates.
- `--replay` : Number of replay passes per interaction step.
- `--grad_steps` : Number of gradient updates per sampled batch.
- `--precision` : Numerical precision for training and policy synchronization.

Example:

```bash
python train_online.py \
  --algorithm sac \
  --use_pretrained \
  --load_path ./results/offline/checkpoints/agent.pt \
  --num_iterations 15000 \
  --precision float32
```

---

### Typical Workflow

For instance, a complete workflow may look like:

```bash
# 1. Verify simulator integration
python simulate.py \
  --system my_project.my_system:build_system \
  --agent random

# 2. Train an initial policy offline
python train_offline.py \
  --algorithm dql \
  --data_path ./data/offline \
  --num_iterations 100000

# 3. Fine-tune the policy online
python train_online.py \
  --algorithm sac \
  --use_pretrained \
  --load_path ./results/offline/checkpoints/agent.pt \
  --num_iterations 15000
```

In practice, users should adapt these commands to their simulator, dataset, reward model, and training configuration.

## Citation

Yeganeh, Y., Shekari, M., Frigerio, N., Pagano, D., & Matta, A. (2026). *Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication*. *arXiv preprint arXiv:2606.10705*.

```
@article{yeganeh2026event,
  title={Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication},
  author={Yeganeh, Yavar and Shekari, Mahsa and Frigerio, Nicla and Pagano, Daniele and Matta, Andrea},
  journal={arXiv preprint arXiv:2606.10705},
  year={2026},
  url={https://arxiv.org/abs/2606.10705}
}
```

## Contributing

Contributions are welcome. Feel free to contribute by improving the code, adding examples, extending documentation, fixing bugs, or proposing new event-driven RL components.

For bugs, questions, feature requests, or documentation problems, please open an issue on the GitHub repository.

When contributing, please keep the repository generic and avoid adding proprietary data, private simulator details, confidential scenario identifiers, or application-specific schemas.