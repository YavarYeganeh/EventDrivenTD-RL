
<h2 align="center">
    Event-driven Reinforcement Learning for Long-horizon Control in Asynchronous, Discrete-event Systems
</h2>

`EventDrivenTD-RL` is a modular reinforcement learning framework for systems where decisions are triggered by events rather than fixed time steps. It is designed for long-horizon control problems with delayed feedback, adaptive candidate sets, overlapping events, and system-level objectives.

The framework was developed in the context of event-driven reinforcement learning for semiconductor fabrication, but the public repository is intended to provide the **generic RL framework only**. It does **not** include proprietary industrial data, private simulator internals, confidential scenario files, or application-specific database schemas.

## Associated Paper

This repository accompanies the research framework described in:

**Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication**

The paper studies reinforcement learning for long-horizon control in event-driven systems, using semiconductor fabrication as a challenging representative application. The code in this repository is intended to provide the reusable RL framework independent of proprietary data and simulator internals.

## Overview

Many real-world control systems are naturally event-driven:

- manufacturing and production systems;
- logistics and supply chains;
- healthcare operations;
- telecommunications and network control;
- resource-constrained service systems;
- SimPy and discrete-event simulation environments;
- digital-twin-based control systems.

In such systems:

- actions occur at irregular decision times;
- actions may have variable duration;
- multiple events may overlap;
- feasible action sets may change at every decision;
- local actions may influence delayed global objectives;
- rewards may be sparse, delayed, or computed over time windows;
- one-step Markov transitions may be insufficient for credit assignment.

`EventDrivenTD-RL` addresses these issues by separating the reusable RL core from the application-specific environment through a modular adapter structure.

## Main Design Principles

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

*-- Documentation will be added soon!*

## Citation

Yeganeh, Y., Shekari, M., Frigerio, N., Pagano, D., & Matta, A. (2026). *Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication*. *arXiv preprint arXiv:2606.10705*.

```
@article{yeganeh2026eventdrivenrl,
  title={Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication},
  author={Yeganeh, Yavar and Shekari, Mahsa and Frigerio, Nicla and Pagano, Daniele and Matta, Andrea},
  journal={arXiv preprint arXiv:2606.10705}, 
  year={2026},
  url={https://arxiv.org/abs/2606.10705}
}
```