
"""

Points:
- Commands:
    {
        0: Trainer → Sim = Initialize and run until INIT_TIME with the policy
        1: Trainer → Sim = Run for INTERVAL_TIME with the policy to obtain at least one new experience, then trainer update the policy and require policy synchronization.
        2: Sim → Trainer = Task completed, waiting for the next command
       -1: Trainer → Sim = Finish the simulation and close
    }

To do:
    - episodic behavior and random init
    - Faster chpt load during training

"""

import copy
import importlib
import os
import sys
import shutil
import torch
import argparse
from datetime import datetime
import numpy as np
from multiprocessing import shared_memory
import subprocess


from framework.utils import (
    Logger,
    run_envs,
    set_seed,
    ExploreScheduler,
    split_into_subspaces,  
)
from framework.replay_buffer import cmd_dtype, experience_dtype 
from framework.reward import Segment_Aggregator


""" Default Settings """

default_scenarios = [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29]


""" Option Settings """

parser = argparse.ArgumentParser()

# training settings
parser.add_argument('--algorithm', type=str, default='dql', help="Online RL algorithm to use, e.g., wtd, dql, ppo, sac")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_scenarios', type=int, nargs='+', default=default_scenarios) # list of scenario ids to train on
parser.add_argument('--use_pretrained', action='store_true', help="To use a pretrained agent model to continue training.")
parser.add_argument('--reset_q_head', action='store_true', help="To reset the Q-head when using a pretrained agent model. If not set, the full model is loaded.")
parser.add_argument('--freeze_q_backbone', action='store_true', help="To freeze the Q-backbone when using a pretrained agent model.")
parser.add_argument('--unfreeze_iteration', type=int, default=100, help="The iteration to unfreeze the Q-backbone if --freeze_q_backbone is set.")
parser.add_argument('--load_path',
                    type=str,
                    default='',
                    help="Path to the checkpoint to load the agent from")
parser.add_argument('--ignore_sim_duration', action='store_true', help="Ignore the limit simulation interaction and use specified num_interactions.") 
parser.add_argument('--num_iterations', type=int, default=15_000) # training iteration; not gradient iteration
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--rb_cap', type=int, default=100_000) 
parser.add_argument('--latent_dim', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--grad_clip_maxnorm', type=float, default=5.0)
parser.add_argument('--save_interval', type=int, default=500)
parser.add_argument('--valid_interval', type=int, default=500)
parser.add_argument("--sim_interval_time", type=int, help="Interval time to run sim to obtain new experiences", default=60) # 60 mins is faster
parser.add_argument(
    '--precision',
    type=str,
    default='float32',
    choices=['float32', 'float64', 'float16', 'bfloat16'],
    help="Floating point precision to use (bfloat16 is only for torch)"
)

# sampler and td settings
parser.add_argument('--sampler', type=str, default='segment', choices=['latest', 'segment'], help="Sampler type to obtain training data")
parser.add_argument('--num_scenario_subsampling', type=int, default=20,
                    help="Number of scenario subsampling steps (going over all) for gradient steps.") # paying attention to gpu memory usage # 20 only merge one batch with 2 scenarios
parser.add_argument('--td_agg', type=str, default='event', choices=['event', 'time_weighted'], help="TD aggregation method for the TD error calculation.")
parser.add_argument('--parent_threshold', type=float, default=120, help="This threshold is used to find the last valid parent experience for training from the replay buffer. Only with --td_agg time_weighted.")
parser.add_argument('--segment_length', type=float, default=480, help="This is the length of each segment used for aggregation of events for training from the replay buffer.") # better shorter for better backprop of the rsystem
parser.add_argument('--replay', type=int, default=3, help="More than one is how many times sampling from replay buffer. (off-policy)")
parser.add_argument('--grad_steps', type=int, default=1, help="Number of gradient updates per sampled data")

# explore parameters
parser.add_argument("--explore_start", type=float, default=1.0)
parser.add_argument("--explore_end", type=float, default=1.0)
parser.add_argument("--explore_max_it_ratio", type=float, default=0.5)
parser.add_argument("--explore_mode", type=str, default="cosine", choices=["cosine", "linear", "exp"])

# reward aggregation parameters
parser.add_argument('--no_constraint_term', action='store_true', help="To remove the constraint term in reward aggregation.")
parser.add_argument('--r_alpha', type=float, default=0) # removing shaping elements 
parser.add_argument('--r_beta', type=float, default=0) # removing shaping elements 
parser.add_argument('--r_zeta', type=float, default=1)
parser.add_argument('--r_gamma', type=float, default=1)
parser.add_argument('--r_delta', type=float, default=1)
parser.add_argument('--r_phi', type=float, default=0) # removing number of move outs

# shared settings
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lr_actor', type=float, default=5e-5)
parser.add_argument('--lr_critic', type=float, default=None)
parser.add_argument('--target_tau', type=float, default=0.005)
parser.add_argument('--target_sync_every', type=int, default=500)

# PPO-style hyperparameters
parser.add_argument('--ppo_clip_eps', type=float, default=0.2)
parser.add_argument('--ppo_clip_log_ratio', type=float, default=7) # clips ratio to ~1000 to prevent loss exploding
parser.add_argument('--ppo_entropy_coef', type=float, default=0.01) 
# parser.add_argument('--ppo_adv_clip', type=float, default=0.05)

# sac hyperparameters
parser.add_argument('--no_auto_alpha', action='store_true')
parser.add_argument('--alpha_init', type=float, default=0.1)
parser.add_argument('--lr_alpha', type=float, default=None)
parser.add_argument('--target_entropy', type=float, default=None)
parser.add_argument('--target_entropy_multiplier', type=float, default=0.98) # 0.98 keep it close to the uniform while 0.5 incentivize sparse policy and search: sqrt(|A|))

args = parser.parse_args()

constraint_term = not args.no_constraint_term
print(f"Using constraint term: {constraint_term} in reward aggregation!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

precision = args.precision  # fp32 is faster / safer than fp64 (but less accurate)
# set torch precision
if precision == "float64":
    dtype = torch.float64
elif precision == "float16":
    dtype = torch.float16
elif precision == "bfloat16":
    dtype = torch.bfloat16
else:
    dtype = torch.float32


""" Results Directory """

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# signature
values = [

    # algorithm
    args.algorithm,

    # initialization / training mode
    f"pre{int(args.use_pretrained)}",
    f"rq{int(args.reset_q_head)}",

    # core experiment identity
    f"s{args.seed}",
    f"sub{args.num_scenario_subsampling}",
    f"rep{args.replay}",
    f"gs{args.grad_steps}",

    # RL setup
    f"td{args.td_agg}",
    f"seg{args.segment_length}"
    f"samp{args.sampler}",
    f"g{args.gamma}",

    # optimization
    f"lr{args.lr}",

    # model size
    f"h{args.latent_dim}",

    # reward aggregation
    f"ra{args.r_alpha}",
    f"rb{args.r_beta}",
    f"rz{args.r_zeta}",
    f"rg{args.r_gamma}",
    f"rd{args.r_delta}",
    f"rp{args.r_phi}",
]

suffix = '_'.join(map(str, values))
suffix = suffix.replace('.', '-')

results_dir = os.path.join('./results/online_training/',timestamp, suffix)
checkpoint_dir = os.path.join(results_dir, "checkpoints/")
mm_dir = os.path.join(results_dir, "mm_tensors/")
checkpoint_path = os.path.join(checkpoint_dir, "agent.pt")

os.makedirs(checkpoint_dir, exist_ok=True)


""" CSV for recording training statistics (e.g., loss, rewards)"""

stats_path = os.path.join(results_dir, "train_stats.csv")


""" Reproducibility
    (Environment variable for reproducibility: $ export CUBLAS_WORKSPACE_CONFIG=:4096:8)
"""

torch.use_deterministic_algorithms(True)
set_seed(args.seed)


""" Logging """

sys.stdout = Logger(f'{results_dir}/output_log.txt')


""" Algorithm Instantiation (includes agent, optimizers, and trainer) """

algorithm_module = importlib.import_module(f"framework.algorithms.online.{args.algorithm}")
algorithm = algorithm_module.Algorithm(args, device, dtype, stats_path)

if args.algorithm in {'ppo', 'ppo_q', 'ppo_q_v'}:
    on_policy = True
else:
    on_policy = False

if args.algorithm in {'ppo', 'ppo_q', 'ppo_q_v', 'sac'}:
    with_candidate_info = True
else:
    with_candidate_info = False


""" Number of Iterations """

if args.ignore_sim_duration:
    num_iterations = args.num_iterations
else:
    num_iterations = (48000)//(args.sim_interval_time) 
num_gradients = args.grad_steps * args.replay * num_iterations * args.num_scenario_subsampling
print(
    f"Constrain sim duration: {not args.ignore_sim_duration} | "
    f"Num iterations: {num_iterations} | "
    f"Num gradient steps: {num_gradients}"
)


""" Exploration Instantiation """

max_explore_grad_it = int(num_gradients * args.explore_max_it_ratio)
explore = ExploreScheduler(
    explore_start = args.explore_start,
    explore_end   = args.explore_end,
    max_explore_it = max_explore_grad_it,
    mode = args.explore_mode
)
future_explore_value = explore(it=1)


""" Simulation Agent Instantiation """

if args.use_pretrained:
    
    assert os.path.isfile(args.load_path), f"Checkpoint file {args.load_path} not found!"
    
    algorithm.agent.load_chpt(args.load_path)
    print(f"Copied checkpoint from {args.load_path}, to be trained and maintained at {checkpoint_path}.")

    if args.reset_q_head:
        algorithm.agent.reset_q_head()
        print(f"Q-head reset after loading pretrained model.")

    if args.algorithm in ["dql", "wtd"]:
        algorithm.agent.update_target_network()
        print(f"Target network synced.")
    else: 
        algorithm.target_agent = copy.deepcopy(algorithm.agent).to(device)
        algorithm.target_agent.eval()
        for p in algorithm.target_agent.parameters():
            p.requires_grad_(False)

    if args.freeze_q_backbone:
        algorithm.agent.freeze_q_backbone()
        print(f"Q-backbone frozen for the first {args.unfreeze_iteration} iterations.")

    shutil.copy(args.load_path, checkpoint_path) # since the current approach is (generally) off-policy the replay buffer pre-fill can be with full chpt (incl. Q-head)

else:

    algorithm.agent.save_chpt(checkpoint_path, future_explore_value=future_explore_value) # save initial model
    print(f"Copied checkpoint into {checkpoint_path}, to be trained and maintained there.")


""" TD Method """

print("Warning: The length of window in the simulations should be appropriate with the TD aggregation method.")

if args.td_agg == "time_weighted":
    assert args.sampler == "latest", "The TD time_weighted aggregation is needs the latest_experiences sampler. Please set --sampler to 'latest'."
    assert args.algorithm == "wtd", "The TD time_weighted aggregation is designed for Weighted TD (based on DQL) in this version. Please set --algorithm to wtd."

elif args.td_agg == "event":
    assert args.sampler == "segment", "The TD event-based aggregation is needs the segment_experiences sampler. Please set --sampler to 'segment'."

else:
    raise ValueError(f"Unsupported TD aggregation method: {args.td_agg}")
    

""" Sampler Instantiation """

if args.sampler == "latest":
    print("Using latest_experiences sampler to obtain training data from the replay buffer.")
    from framework.sampler import latest_experiences
    sampler = latest_experiences

elif args.sampler == "segment":
    print("Using segment_experiences sampler to obtain training data from the replay buffer.")
    from framework.sampler import segment_experiences
    sampler = segment_experiences
    segment_reward = Segment_Aggregator(gamma=args.r_gamma, delta=args.r_delta, phi=args.r_phi)


""" Simulation scenarios instantiation """

proc = {}

shm_cmd = {}
shm_cmd_size = int(3*cmd_dtype.itemsize) # 3 times the capacity to avoid overflow
cmd = {}

shm_rb = {}
shm_rb_size = int(1.5*args.rb_cap*experience_dtype.itemsize) # 1.5 times the capacity to avoid overflow
rb = {}

for i in args.train_scenarios:

    shm_cmd[i] = shared_memory.SharedMemory(create=True, size=shm_cmd_size, name= timestamp + "_" + suffix + "_cmd_" + str(i)) 
    cmd[i] = np.ndarray((1,), dtype= cmd_dtype, buffer=shm_cmd[i].buf)
    cmd[i]["cmd"] = 0 # init command to 0 # will int sims
    cmd[i]["position"] = 0 # init position in the replay buffer
    
    shm_rb[i] = shared_memory.SharedMemory(create=True, size=shm_rb_size, name= timestamp + "_" + suffix + "_rb_" + str(i)) 
    rb[i] = np.ndarray((args.rb_cap,), dtype=experience_dtype, buffer=shm_rb[i].buf)
    rb[i]["active"] = False # init all slots to inactive

    argv = [
        sys.executable, "simulate.py",
        "--scenario_id", str(i),
        "--train",
        "--agent", str(args.algorithm),
        "--load_agent",
        "--load_path", str(checkpoint_path),
        "--train_results_dir", results_dir, 
        "--precision", precision,
        "--sim_interval_time", str(args.sim_interval_time),
        "--cmd_shm", str(shm_cmd[i].name),
        "--rb_shm", str(shm_rb[i].name),
        "--rb_cap", str(args.rb_cap),
        "--r_alpha", str(args.r_alpha),
        "--r_beta", str(args.r_beta),
        "--r_zeta", str(args.r_zeta),
        "--r_gamma", str(args.r_gamma),
        "--r_delta", str(args.r_delta),
        "--r_phi", str(args.r_phi),
        "--mm_dir", str(mm_dir),
    ]
    
    if args.no_constraint_term:
        argv.append("--no_constraint_term")
    
    if args.sampler == "segment": # for faster sim
        argv.append("--skip_r_system")

    proc[i] = subprocess.Popen(argv)    
    
    print(f"Started simulation process for scenario {i}, with PID {proc[i].pid}") 

 
""" Training Loop """

start_time = datetime.now()
grad_it = 1 # number of gradient steps is different than training iteration, which is interacting with the environment obtaining new data
for train_it in range(1, num_iterations + 1):

    # ---- running all environments to obtain new experiences and data for policy update ----------
    run_envs(cmd) 

    """ Start of the gradient sub-loops """

    #---- randomly split scenarios into subspaces ----------
    subspaces = split_into_subspaces(scenarios=args.train_scenarios, num_subspaces=args.num_scenario_subsampling) 
    
    for scenario_list in subspaces:

        for i in range(args.replay):
 
            # ----------------  latest vs replay  -------------------
            if i==0:
                latest = True
            else:
                latest = False # -> sample replay (except for time_weighted, which resample latest)

            # ----------------  TD method  -------------------
            if args.td_agg == "time_weighted":

                # ---- obtaining training data experiences ----------
                with torch.no_grad():
                    data = sampler(rb, cmd, scenario_list=scenario_list, threshold=args.parent_threshold)

                # ------- unpack ---------
                s_a_0, s_a_1, rewards, reward_elements, group_sizes, s0_sizes, s1_sizes, weights = data

                # ---- move (single .to call avoids extra GPU copy) ----------
                s_a_0 = s_a_0.to(device=device, dtype=dtype, non_blocking=True)
                s_a_1 = s_a_1.to(device=device, dtype=dtype, non_blocking=True)
                rewards = rewards.to(device=device, dtype=dtype, non_blocking=True)
                weights = weights.to(device=device, dtype=dtype, non_blocking=True)
                # for reward_elements it's not necessary to be transferred to the gpu as it's only recorded

                # ----------------  data tuple for training step  -------------------
                data = (s_a_0, s_a_1, rewards, group_sizes, s0_sizes, s1_sizes, weights)

            # ----------------  TD method  -------------------
            elif args.td_agg == "event":

                # ---- obtaining training data experiences ----------
                with torch.no_grad():
                    data = sampler(rb, segment_reward=segment_reward, scenario_list=scenario_list, length=args.segment_length, gamma=args.gamma, latest=latest, with_candidate_info=with_candidate_info)

                # ------- unpack ---------
                s_a_0, log_prob_0, s_a_0_all, s_a_1, r_event, r_group, reward_elements, group_sizes, s0_sizes, s0_all_sizes, s1_sizes, weights = data

                # ---- move (single .to call avoids extra GPU copy) ----------
                s_a_0 = s_a_0.to(device=device, dtype=dtype, non_blocking=True)
                s_a_1 = s_a_1.to(device=device, dtype=dtype, non_blocking=True)
                r_event = r_event.to(device=device, dtype=dtype, non_blocking=True)
                r_group = r_group.to(device=device, dtype=dtype, non_blocking=True)
                weights = weights.to(device=device, dtype=dtype, non_blocking=True) # gamma here
                # for reward_elements it's not necessary to be transferred to the gpu as it's only recorded
                rewards = r_event + r_group # total rewards for stats 
                if with_candidate_info:
                    log_prob_0 = log_prob_0.to(device=device, dtype=dtype, non_blocking=True)
                    s_a_0_all = s_a_0_all.to(device=device, dtype=dtype, non_blocking=True)

                # ----------------  data tuple for training step  -------------------
                data = (s_a_0, log_prob_0, s_a_0_all, s_a_1, r_event, r_group, group_sizes, s0_sizes, s0_all_sizes, s1_sizes, weights)      
            
            # ----------------  TD method  -------------------
            else: 
                raise NotImplementedError("The TD aggregation is unknown.")

            # ----------------  gradient loop -------------------------
            for j in range(args.grad_steps):
        
                # ----------------  train step  -------------------
                stats = algorithm.train_step(data, grad_it)

                # ----------------  recoding history  -------------------------
                algorithm.record_stats(stats, rewards, reward_elements, grad_it)

                # ----------------  updating grad it  -------------------------
                grad_it +=1
            # ----------------  end gradient loop -------------------------

        # ---------------- update exploration value --------------------------
        future_explore_temp = explore(it=grad_it) # it+1 for future

        # ---------------- save the shared checkpoint for the online sims --------------------------
        algorithm.agent.save_chpt(checkpoint_path, future_explore_value=future_explore_temp) 

        # -----------------  checkpointing  -------------------------
        if (grad_it-1) % args.save_interval == 0: # grad_it is more than 1
            elapsed_time = (datetime.now() - start_time).total_seconds() / 60  # in minutes
            # could also be with agent.save()
            torch.save({
                "algorithm": args.algorithm,
                "iteration": grad_it,
                "model_state_dict": algorithm.agent.state_dict(),
                "optimizer_state_dict": algorithm.optimizer.state_dict() if hasattr(algorithm, 'optimizer') else None,
                "loss": stats['loss'],
                "training_time_min": round(elapsed_time, 3),
                "precision": str(precision),
            }, os.path.join(checkpoint_dir, f"ckpt_{grad_it}.pt"))
            print(f"[Iter {grad_it}] checkpoint saved (loss={stats['loss']:.4g})")

        # ---------------- unfreeze q-backbone if applicable ----------------
        if args.freeze_q_backbone and grad_it >= args.unfreeze_iteration:
            algorithm.agent.unfreeze_q_backbone()
            args.freeze_q_backbone = False # to avoid repeating unfreezing in the later iterations
            print(f"Q-backbone unfrozen at iteration {grad_it}.")

        # ---------------- cleanup: drop references held by python on gpu ---------------- 
        del data, s_a_0, s_a_1 

        # only sometimes
        if device.type == "cuda" and grad_it % 100 == 0:
            torch.cuda.empty_cache()

# ---------------------------------  end of training  --------------------------------
print("Training finished!")

# -------- final checkpoint -----------
elapsed_time = (datetime.now() - start_time).total_seconds() / 60  # in minutes
# could also be with agent.save()
torch.save({
    "algorithm": args.algorithm,
    "iteration": grad_it,
    "model_state_dict": algorithm.agent.state_dict(),
    "optimizer_state_dict": algorithm.optimizer.state_dict() if hasattr(algorithm, 'optimizer') else None,
    "loss": stats['loss'],
    "training_time_min": round(elapsed_time, 3),
    "precision": str(precision),
}, os.path.join(checkpoint_dir, f"ckpt_final_{grad_it}.pt"))
print(f"Final checkpoint saved (loss={stats['loss']:.4g})")
print(f"Total training time: {elapsed_time:.2f} minutes.")

# ---------------------------------  ending simulations  --------------------------------

# signal to have the sims closed
for i in args.train_scenarios:

    cmd[i]["cmd"] = -1

# close the shared memory in the main process
for i in args.train_scenarios:
    
    shm_cmd[i].close()
    shm_cmd[i].unlink()

    shm_rb[i].close()
    shm_rb[i].unlink()






