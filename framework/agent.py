
import torch


def instantiate_agent(agent_type: str, *args, **kwargs):

    """
    Instantiate an agent based on the specified type.

    Args:
        agent_type (str): The type of agent to instantiate.
        *args: Positional arguments to pass to the agent constructor.
        **kwargs: Keyword arguments to pass to the agent constructor.

    Constraint (Corr):
        Push: Soft new_probs = probs + push -> could be new_probs = probs * push for sharper push

    Returns:
        An instance of the specified agent type.

    Raises:
        ValueError: If the specified agent type is not recognized.
    """

    agent_type = agent_type.lower()
    
    if agent_type == 'random':
        from framework.agents.random import Random_Agent
        return Random_Agent(*args, **kwargs)
    
    elif agent_type == 'fifo':
        from framework.agents.fifo import FIFO_Agent
        return FIFO_Agent(*args, **kwargs)
    
    elif agent_type == 'spt':
        from framework.agents.spt import SPT_Agent
        return SPT_Agent(*args, **kwargs)
    
    elif agent_type == 'base':
        from framework.agents.dql import Base_Agent
        return Base_Agent(*args, **kwargs)
    
    elif agent_type == 'ddqn':
        from framework.agents.dql import DDQN_Agent
        return DDQN_Agent(*args, **kwargs)
    
    elif agent_type == 'dql':
        from framework.agents.dql import DDQN_Agent
        return DDQN_Agent(*args, **kwargs)
    
    elif agent_type == 'iql':
        from framework.agents.iql import IQL_Agent
        return IQL_Agent(*args, **kwargs)
    
    elif agent_type == 'cql_ac':
        from framework.agents.cql import CQL_Agent
        return CQL_Agent(*args, **kwargs, build_policy=True)
    
    elif agent_type == 'cql_q':
        from framework.agents.cql import CQL_Agent
        return CQL_Agent(*args, **kwargs, build_policy=False)
    
    elif agent_type == 'ppo':
        from framework.agents.ac_generic import AC_Agent
        return AC_Agent(*args, **kwargs, agent_name='ppo')

    elif agent_type == 'ppo_q':
        from framework.agents.ac_generic import AC_Agent
        return AC_Agent(*args, **kwargs, agent_name='ppo_q', build_value=False)
    
    elif agent_type == 'ppo_q_v':
        from framework.agents.ac_generic import AC_Agent
        return AC_Agent(*args, **kwargs, agent_name='ppo_q_v', build_value=True)

    elif agent_type == 'ppo_v':
        from framework.agents.ac_generic import PPO_V_Agent
        return PPO_V_Agent(*args, **kwargs)

    elif agent_type == 'sac':
        from framework.agents.ac_generic import AC_Agent
        return AC_Agent(*args, **kwargs, agent_name='sac')
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def load_chpt_mixed(agent, checkpoint_path, config=None, verbose=True):

    """
    flexible checkpoint loader.

    rules
    -----
    1) If checkpoint algorithm matches this agent -> try strict full load.
    2) Otherwise, try component-wise loading:
         - q_net1 / q_net2
         - q_net
         - policy_net (alias: pi_net)
         - v_net
    3) If policy is missing in checkpoint but current model has policy_net,
       initialize policy_net from q_net1 if available, else q_net.

    Notes
    -----
    - Only loads keys with matching names AND matching tensor shapes.
    - compatible across AC/CQL/IQL/DDQN/PPO style agents.
    - If target_net exists and q_net was loaded, sync target_net <- q_net.
    - Cross-Q fallback is taken from checkpoint substates, not from already-loaded agent nets.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # support both raw state_dict checkpoints and wrapped checkpoints
    ckpt_algo = checkpoint.get("algorithm", None)
    ckpt_state = checkpoint.get("model_state_dict", checkpoint)

    # -------- 1) strict load if algorithm matches --------
    if _algorithm_matches(agent, ckpt_algo):

        try:
            agent.load_state_dict(ckpt_state, strict=True)

            if verbose:
                print(f"[load_chpt_mixed] strict full load succeeded (algorithm={ckpt_algo})")

            _maybe_sync_target(agent, verbose)

            if config and getattr(config, "policy_sync", False):
                config.policy_sync = False

            _restore_explore_value(agent, checkpoint)
            return
        except Exception as e:
            if verbose:
                print(f"[load_chpt_mixed] strict load failed, falling back to mixed load: {e}")

    # -------- 2) mixed/component-wise load --------
    loaded_any = False
    q1_loaded = False
    q2_loaded = False
    q_loaded = False
    policy_loaded = False

    # pre-extract checkpoint substates once
    ckpt_q1 = _extract_substate(ckpt_state, ["q_net1"])
    ckpt_q2 = _extract_substate(ckpt_state, ["q_net2"])
    ckpt_q = _extract_substate(ckpt_state, ["q_net"])
    ckpt_pi = _extract_substate(ckpt_state, ["policy_net", "pi_net"])
    ckpt_v = _extract_substate(ckpt_state, ["v_net", "value_net"])

    # twin Q loaders: direct match first
    if hasattr(agent, "q_net1") and agent.q_net1 is not None:

        loaded = _safe_load_module(agent.q_net1, ckpt_q1, "q_net1", verbose=verbose)
        q1_loaded = loaded > 0
        loaded_any |= q1_loaded

    if hasattr(agent, "q_net2") and agent.q_net2 is not None:

        loaded = _safe_load_module(agent.q_net2, ckpt_q2, "q_net2", verbose=verbose)
        q2_loaded = loaded > 0
        loaded_any |= q2_loaded

    # single Q loader: direct match first
    if hasattr(agent, "q_net") and agent.q_net is not None:
        loaded = _safe_load_module(agent.q_net, ckpt_q, "q_net", verbose=verbose)
        q_loaded = loaded > 0
        loaded_any |= q_loaded

    # policy loader: support both policy_net and pi_net names
    if hasattr(agent, "policy_net") and agent.policy_net is not None:
        loaded = _safe_load_module(agent.policy_net, ckpt_pi, "policy_net", verbose=verbose)
        policy_loaded = loaded > 0
        loaded_any |= policy_loaded

    # value loader
    if hasattr(agent, "v_net") and agent.v_net is not None:
        loaded = _safe_load_module(agent.v_net, ckpt_v, "v_net", verbose=verbose)
        loaded_any |= (loaded > 0)

    # -------- 2.5) cross-init missing Q nets from checkpoint substates --------

    # q_net <- ckpt.q_net1 or ckpt.q_net2
    if hasattr(agent, "q_net") and agent.q_net is not None and not q_loaded:
        loaded = 0

        if ckpt_q1 is not None:
            loaded = _safe_load_module(agent.q_net, ckpt_q1, "q_net<-q_net1", verbose=verbose)

        if loaded == 0 and ckpt_q2 is not None:
            loaded = _safe_load_module(agent.q_net, ckpt_q2, "q_net<-q_net2", verbose=verbose)

        q_loaded = loaded > 0
        loaded_any |= q_loaded

    # q_net1 <- ckpt.q_net or ckpt.q_net2
    if hasattr(agent, "q_net1") and agent.q_net1 is not None and not q1_loaded:
        loaded = 0

        if ckpt_q is not None:
            loaded = _safe_load_module(agent.q_net1, ckpt_q, "q_net1<-q_net", verbose=verbose)

        q1_loaded = loaded > 0
        loaded_any |= q1_loaded

    # q_net2 <- ckpt.q_net or ckpt.q_net1
    if hasattr(agent, "q_net2") and agent.q_net2 is not None and not q2_loaded:
        loaded = 0

        if ckpt_q is not None:
            loaded = _safe_load_module(agent.q_net2, ckpt_q, "q_net2<-q_net", verbose=verbose)

        q2_loaded = loaded > 0
        loaded_any |= q2_loaded

    # -------- 3) if policy missing -> init from loaded q modules --------
    if hasattr(agent, "policy_net") and agent.policy_net is not None and not policy_loaded:
        copied = 0

        if hasattr(agent, "q_net1") and agent.q_net1 is not None:
            copied = _copy_module_weights(
                agent.policy_net, agent.q_net1, "policy_net", "q_net1", verbose=verbose
            )

        if copied == 0 and hasattr(agent, "q_net") and agent.q_net is not None:
            copied = _copy_module_weights(
                agent.policy_net, agent.q_net, "policy_net", "q_net", verbose=verbose
            )

        if copied == 0 and hasattr(agent, "q_net2") and agent.q_net2 is not None:
            copied = _copy_module_weights(
                agent.policy_net, agent.q_net2, "policy_net", "q_net2", verbose=verbose
            )

        if copied == 0 and verbose:
            print("[load_chpt_mixed] policy_net could not be initialized from Q network")

    _maybe_sync_target(agent, verbose)

    if config and getattr(config, "policy_sync", False):
        config.policy_sync = False

    _restore_explore_value(agent, checkpoint)

    if verbose:
        if loaded_any:
            print(f"[load_chpt_mixed] mixed load completed from {checkpoint_path}")
        else:
            print(f"[load_chpt_mixed] no compatible tensors found in {checkpoint_path}")


# -------- helpers for load --------

def _safe_load_module(module, incoming_substate, module_name="module", verbose=False):

    """
    load only matching keys/shapes into a submodule.
    returns number of loaded tensors.
    """

    if module is None or incoming_substate is None:
        return 0

    current = module.state_dict()
    filtered = {}

    for k, v in incoming_substate.items():
        if k in current and current[k].shape == v.shape:
            filtered[k] = v

    if filtered:
        current.update(filtered)
        module.load_state_dict(current, strict=False)

    if verbose:
        print(f"[load_chpt_mixed] {module_name}: loaded {len(filtered)} tensors")

    return len(filtered)


def _extract_substate(state_dict, prefix_list):
    
    """
    extract submodule state dict by prefix, stripping prefix.
    example:
        q_net1.0.weight -> 0.weight
    """
    for prefix in prefix_list:
        sub = {}
        prefix_dot = prefix + "."
        for k, v in state_dict.items():
            if k.startswith(prefix_dot):
                sub[k[len(prefix_dot):]] = v
        if sub:
            return sub
    return None


def _copy_module_weights(dst_module, src_module, dst_name="dst", src_name="src", verbose=False):

    """
    copy matching tensors from src_module to dst_module.
    """

    if dst_module is None or src_module is None:
        return 0

    dst_sd = dst_module.state_dict()
    src_sd = src_module.state_dict()
    copied = {}

    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            copied[k] = v.clone()

    if copied:
        dst_sd.update(copied)
        dst_module.load_state_dict(dst_sd, strict=False)

    if verbose:
        print(f"[load_chpt_mixed] init {dst_name} from {src_name}: copied {len(copied)} tensors")

    return len(copied)


def _maybe_sync_target(agent, verbose=False):

    if hasattr(agent, "target_net") and hasattr(agent, "q_net") and agent.target_net is not None and agent.q_net is not None:
        try:
            agent.target_net.load_state_dict(agent.q_net.state_dict())
            if verbose:
                print("[load_chpt_mixed] target_net synced from q_net")
        except Exception as e:
            if verbose:
                print(f"[load_chpt_mixed] target sync skipped: {e}")


def _restore_explore_value(agent, checkpoint):

    if getattr(agent, "explore_mode", False):
        temp = checkpoint.get("future_explore_value", None)
        agent.explore_value = 1.0 if temp is None else float(temp)
        agent.explore_value = max(agent.explore_value, 1e-6)


def _algorithm_matches(agent, ckpt_algo):
    # current agent identifier(s)
    current_names = set()

    if hasattr(agent, "agent_name"):
        current_names.add(str(agent.agent_name).lower())

    cls_name = agent.__class__.__name__.lower()
    current_names.add(cls_name)

    # helpful aliases for your current codebase
    alias_map = {
        "ac_agent": {"ppo", "sac"},
        "cql_agent": {"cql", "cql_ac", "cql_q"},
        "iql_agent": {"iql", "iql_twin_q", "iql_polyak"},
        "ddqn_agent": {"dql", "ddqn"},
    }

    for key, vals in alias_map.items():
        if key in cls_name:
            current_names.update(vals)

    return ckpt_algo is not None and str(ckpt_algo).lower() in current_names




