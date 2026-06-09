
import os, gzip, pickle
from networkx import constraint
import torch
import argparse
from datetime import datetime
import importlib

from framework.reward import Aggregator, _load_norm_params
from framework.preprocessing.meta_generator import generate_metadata
from framework.utils import set_seed, enc_normalization_tensor


""" Settings """
parser = argparse.ArgumentParser()

# training settings
parser.add_argument('--algorithm', type=str, default='dql', help="Offline RL algorithm to use, e.g., dql, iql, iql_twin_q, iql_polyak, cql_ac, cql_q")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_iterations', type=int, default=100_000)
parser.add_argument('--num_sampled_seeds', type=int, default=3) # 7
parser.add_argument('--gamma', type=float, default=0.99) # 0.999 unstable (e.g., iql tau=0.90-95)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--latent_dim', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--grad_clip_maxnorm', type=float, default=5.0)
parser.add_argument('--save_interval', type=int, default=1_000)
parser.add_argument('--valid_interval', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default='float32') # better memory management
parser.add_argument('--enc_normalization', action='store_true', help="To the normalization of the encoder features (i.e., processing time) as by default it will be first log1p transformed then normalized to z(0,1))")
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--meta_data_path', type=str, default=None)

# sampler
parser.add_argument('--td_agg', type=str, default='event', choices=['event', 'action', 'stack'])
parser.add_argument('--resample', type=int, default=20,
                    help='Number of resampling steps (default: 20)')
parser.add_argument('--avg_reward', action='store_true', help= "Inside of td taking average only for action agg method.")

# reward aggregation parameters
parser.add_argument('--no_constraint_term', action='store_true', help="To remove the constraint term in reward aggregation.")
parser.add_argument('--r_alpha', type=float, default=1)
parser.add_argument('--r_beta', type=float, default=1)
parser.add_argument('--r_zeta', type=float, default=0)
parser.add_argument('--r_gamma', type=float, default=0)
parser.add_argument('--r_delta', type=float, default=0)
parser.add_argument('--r_phi', type=float, default=0)

# dql settings
parser.add_argument('--target_sync_every', type=int, default=500)

# iql settings
parser.add_argument('--iql_tau', type=float, default=0.95,
                    help='IQL expectile (higher = more like Q-Learning; e.g., 0.70–0.95)')
parser.add_argument('--iql_beta', type=float, default=10.0,
                    help='IQL inverse temperature (higher = more max Q extraction by advantage-weighting; 1–3 common)')
parser.add_argument('--iql_exp_adv_max', type=float, default=100.0,
                    help='Clip for exp(beta*adv) to avoid blow-ups (e.g., 25–100)')
parser.add_argument('--iql_alpha', type=float, default=0.005,
                    help='IQL target network update rate (Polyak averaging coefficient, e.g., 0.005–0.01)')
parser.add_argument('--q_coef', type=float, default=1.0, help='Weight for Q loss')
parser.add_argument('--v_coef', type=float, default=1.0, help='Weight for V (expectile) loss')
parser.add_argument('--pi_coef', type=float, default=1.0,
                    help='Weight for policy loss')

# cql settings
parser.add_argument('--cql_alpha', type=float, default=1.0,
                    help='CQL conservative weight (higher = more conservative Q estimates)')
parser.add_argument('--cql_temp', type=float, default=1.0,
                    help='CQL temperature for log-sum-exp (controls softness of conservative penalty)')
parser.add_argument('--ent_coef', type=float, default=0.0,
                    help='Entropy regularization coefficient (set >0 for SAC-style entropy in CQL)')
parser.add_argument('--target_entropy', type=float, default=None,
                    help='Target entropy for automatic entropy tuning (default: -action_dim)')
parser.add_argument('--target_tau', type=float, default=0.005,
                    help='Soft target update rate (Polyak averaging coefficient)')
parser.add_argument('--lr_actor', type=float, default=1e-5,
                    help='Learning rate for actor')
parser.add_argument('--lr_critic', type=float, default=1e-5,
                    help='Learning rate for critic')
parser.add_argument('--lr_alpha', type=float, default=1e-5,
                    help='Learning rate for entropy coefficient')
parser.add_argument('--no_auto_alpha', action='store_true',
                    help='Disable automatic entropy coefficient tuning')
parser.add_argument('--alpha_init', type=float, default=0.1,
                    help='Initial entropy coefficient (used if auto_alpha=True)')

args = parser.parse_args()

constraint_term = not args.no_constraint_term
print(f"Using constraint term: {constraint_term} in reward aggregation!")

device   = torch.device(args.device)   # or torch.device("cpu")

# fp32 is faster / safer than fp64 (but less accurate)
if args.dtype == "float64":
    dtype = torch.float64
elif args.dtype == "float16":
    dtype = torch.float16
elif args.dtype == "bfloat16":
    dtype = torch.bfloat16
else:
    dtype = torch.float32

data_path            = args.data_path
meta_data_path       = args.meta_data_path


""" Results Directory """
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
values = [
    args.algorithm,
    args.seed, # at least sampler has seed
    args.num_iterations,
    args.num_sampled_seeds,
    args.gamma,
    args.lr,
    args.latent_dim,
    args.dropout,
    args.r_alpha,
    args.r_beta,
    args.r_zeta,
    args.r_gamma,
    args.r_delta,
    args.r_phi,
]
suffix = '_'.join(map(str, values))
suffix = suffix.replace('.', '-')
results_dir = os.path.join('./results/offline_training/',timestamp, suffix)
checkpoint_dir = os.path.join(results_dir, "checkpoints/")

os.makedirs(checkpoint_dir, exist_ok=True)


""" CSV for recording training statistics (e.g., loss, rewards)"""

stats_path = os.path.join(results_dir, "train_stats.csv")


""" Sampler """

if args.td_agg == "event":
    print("[Sampler] Using EVENT-level TD aggregation.")
    print("Warning: this is the default TD aggregation method, require a special loss calculation with .mean().pow(2) rather than typical .pow(2).mean().")
    from framework.sampler import sample_exp_offline_events as sampler

elif args.td_agg == "action":
    print("[Sampler] Using action sequences for TD aggregation.")
    from framework.sampler import sample_exp_offline as sampler

elif args.td_agg == "stack":
    raise NotImplementedError("Stack-level TD aggregation requires typical loss of typical .pow(2).mean() which is not implemented yet. Please use event or action level for now.")

else:
    raise ValueError(f"Unknown td_agg mode: {args.td_agg}")

if args.algorithm in ['cql_ac', 'cql_q']:
    add_s_a_0_all = True  # for CQL, we need all s_a_0 (options) to compute the conservative penalty
else:    
    add_s_a_0_all = False


""" Reproducibility
Environment variable for reproducibility: $ export CUBLAS_WORKSPACE_CONFIG=:4096:8
"""
torch.use_deterministic_algorithms(True)
set_seed(args.seed)


""" Loading Metadata """
if meta_data_path is not None:
     with gzip.open(meta_data_path, "rb") as f:
        meta_data = pickle.load(f)
else:
    meta_data = generate_metadata(data_path, save=True, output_path=os.path.join(data_path,"metadata.pkl.gz"))

# full_experiences = load_data(data_path, meta_data) # lazy loading (memory mapping)
len_exps         = {s: len(meta_data[s]) for s in meta_data}


""" Algorithm Instantiation (includes agent, optimizers, and trainer) """
algorithm_module = importlib.import_module(f"framework.algorithms.offline.{args.algorithm}")
algorithm = algorithm_module.Algorithm(args, device, dtype, stats_path)


""" Aggregator Instantiation """
agg = Aggregator(
    alpha=args.r_alpha,
    beta=args.r_beta,
    zeta=args.r_zeta,
    gamma=args.r_gamma,
    delta=args.r_delta,
    phi=args.r_phi,
    constraint_term=constraint_term,
    dtype=dtype)


""" Preprocessing """
if args.enc_normalization:
    pt_params = _load_norm_params(enc_params_only=True) 
    pt_normalizer = _load_norm_params()["process_time_enc"]
    print("Preprocessing s_a data need preprocessing (pt) -> enc_normalization was True!")
else:
    print("Preprocessing: s_a data do not need preprocessing (pt) -> enc_normalization was False!")
print("Warning: It's important to have the encodings similar preprocessing during the training and evaluation.")



""" Training loop """
start_time = datetime.now()
for it in range(1, args.num_iterations + 1):

    # ----------------  data sampling -------------------

    if add_s_a_0_all:

        with torch.no_grad():
            (s_a_0_all, s_a_0, s_a_1, rewards, reward_elements,
            group_sizes,  # renamed: this is the per-seed sizes list
            s0_all_sizes, s0_sizes, s1_sizes, gammas) = sampler(
                meta_data, agg, len_exps=len_exps, num_sampled_seeds=args.num_sampled_seeds, gamma=args.gamma, resample=args.resample, avg_reward=args.avg_reward,add_s_a_0_all=True) 

        # ---- preprocessing (single .to call avoids extra GPU copy) ----------
        if args.enc_normalization:
            s_a_0_all = enc_normalization_tensor(s_a_0_all, pt_normalizer).to(device=device, dtype=dtype, non_blocking=True)
            s_a_0 = enc_normalization_tensor(s_a_0, pt_normalizer).to(device=device, dtype=dtype, non_blocking=True)
            s_a_1 = enc_normalization_tensor(s_a_1, pt_normalizer).to(device=device, dtype=dtype, non_blocking=True)
        else:
            s_a_0_all = s_a_0_all.to(device=device, dtype=dtype, non_blocking=True)
            s_a_0 = s_a_0.to(device=device, dtype=dtype, non_blocking=True)
            s_a_1 = s_a_1.to(device=device, dtype=dtype, non_blocking=True)
        rewards = rewards.to(device=device, dtype=dtype, non_blocking=True)
        gammas = gammas.to(device=device, dtype=dtype, non_blocking=True)

        # ----------------  data tuple for training step  -------------------
        data = s_a_0_all, s_a_0, s_a_1, rewards, group_sizes, s0_all_sizes, s0_sizes, s1_sizes, gammas 

    else:

        with torch.no_grad():
            (s_a_0, s_a_1, rewards, reward_elements,
            group_sizes,  # renamed: this IS the per-seed sizes list
            s0_sizes, s1_sizes, gammas) = sampler(
                meta_data, agg, len_exps=len_exps, num_sampled_seeds=args.num_sampled_seeds, gamma=args.gamma, resample=args.resample, avg_reward=args.avg_reward,add_s_a_0_all=False) 

        # ---- preprocessing (single .to call avoids extra GPU copy) ----------
        if args.enc_normalization:
            s_a_0 = enc_normalization_tensor(s_a_0, pt_normalizer).to(device=device, dtype=dtype, non_blocking=True)
            s_a_1 = enc_normalization_tensor(s_a_1, pt_normalizer).to(device=device, dtype=dtype, non_blocking=True)
        else:
            s_a_0 = s_a_0.to(device=device, dtype=dtype, non_blocking=True)
            s_a_1 = s_a_1.to(device=device, dtype=dtype, non_blocking=True)
        rewards = rewards.to(device=device, dtype=dtype, non_blocking=True)
        gammas = gammas.to(device=device, dtype=dtype, non_blocking=True)

        # ----------------  data tuple for training step  -------------------
        data = s_a_0, s_a_1, rewards, group_sizes, s0_sizes, s1_sizes, gammas 

    # ----------------  train step  -------------------
    stats = algorithm.train_step(data, it)

    # ----------------  Recoding history  -------------------------
    algorithm.record_stats(stats, rewards, reward_elements, it)
    
    # -----------------  checkpointing  -------------------------
    if it % args.save_interval == 0:
        elapsed_time = (datetime.now() - start_time).total_seconds() / 60  # in minutes
        # could also be with agent.save()
        torch.save({
            "iteration": it,
            "model_state_dict": algorithm.agent.state_dict(),
            "optimizer_state_dict": algorithm.optimizer.state_dict() if hasattr(algorithm, 'optimizer') else None,
            "loss": stats['loss'],
            "training_time_min": round(elapsed_time, 3),
            "precision": str(dtype),
        }, os.path.join(checkpoint_dir, f"ckpt_{it}.pt"))
        print(f"[Iter {it}] checkpoint saved (loss={stats['loss']:.4g})")

    # ---------------- cleanup: drop references held by python ---------------- 
    del data, s_a_0, s_a_1, rewards, gammas

    if add_s_a_0_all:
        del s_a_0_all

    # only sometimes
    if it % 100 == 0:
        torch.cuda.empty_cache()

# ---------------------------------  end of training  --------------------------------
print("Training finished!")

# -------- final checkpoint -----------
elapsed_time = (datetime.now() - start_time).total_seconds() / 60  # in minutes
# could also be with agent.save()
torch.save({
    "algorithm": args.algorithm,
    "iteration": it,
    "model_state_dict": algorithm.agent.state_dict(),
    "optimizer_state_dict": algorithm.optimizer.state_dict() if hasattr(algorithm, 'optimizer') else None,
    "loss": stats['loss'],
    "training_time_min": round(elapsed_time, 3),
    "precision": str(dtype),
}, os.path.join(checkpoint_dir, f"ckpt_final_{it}.pt"))
print(f"Final checkpoint saved (loss={stats['loss']:.4g})")
print(f"Total training time: {elapsed_time:.2f} minutes.")




