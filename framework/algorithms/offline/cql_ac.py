
import csv
import copy
import math
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_percentage_error

from framework.agent import instantiate_agent
from framework.encoder import d_s_a, d_s
from framework.utils import sa_batchable


@torch.no_grad()
def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


class Algorithm:

    """
    CQL + discrete SAC (candidate-set / slate-style discrete actions),
    written to follow the exact structure of the provided CQL-Q code:
      - data unpacking
      - per-machine aggregation (handles batch machines)
      - per-group aggregation via group_sizes
      - CQL penalty helper (logsumexp over all candidates per event)
      - by default this is c multiplier of entropy is 1 and therefore maximum entropy case
    """

    def __init__(self, args, device, dtype, stats_path):

        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats_path = stats_path

        # --- agent with critics + policy head ---
        # agent which="pi" returns logits per (s,a_candidate)
        self.agent = instantiate_agent(
            agent_type="cql_ac",  # <- build_policy=True for the SAC-capable version (needs policy head)
            input_dim=d_s_a,
            state_dim=d_s,
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            precision=dtype,
            device=device,
        ).to(device)

        # --- target critic (stability) ---
        self.target_agent = copy.deepcopy(self.agent).to(device)
        self.target_agent.eval()
        for p in self.target_agent.parameters():
            p.requires_grad_(False)

        # --- SAC entropy temperature (alpha) ---
        no_auto_alpha = getattr(args, "no_auto_alpha", False)
        self.auto_alpha = not no_auto_alpha
        alpha_init = float(getattr(args, "alpha_init", 0.1))
        self.log_alpha = torch.tensor(
            math.log(max(alpha_init, 1e-8)),
            device=device,
            requires_grad=True,
            dtype=torch.float32,
        )
        self.alpha = float(alpha_init)

        # --- optimizers ---
        # split actor vs critic params by name, but keep safe fallbacks.
        lr = float(getattr(args, "lr", 1e-5))
        lr_actor = float(getattr(args, "lr_actor", lr))
        lr_critic = float(getattr(args, "lr_critic", lr))
        lr_alpha = float(getattr(args, "lr_alpha", lr))

        actor_params = list(self.agent.policy_net.parameters()) if self.agent.build_policy else []
        critic_params = list(self.agent.q_net1.parameters()) + list(self.agent.q_net2.parameters())

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # if user provides fixed target entropy, we use it.
        # otherwise we will use per-event target entropy = -log(|A_candidates|)
        self.target_entropy = getattr(args, "target_entropy", None)

        # --- CQL hyperparams ---
        self.cql_alpha = float(getattr(args, "cql_alpha", 1.0))
        self.cql_temp = float(getattr(args, "cql_temp", 1.0))
        self.target_tau = float(getattr(args, "target_tau", 0.005))

        self.init_stats(stats_path)

        print(
            f"Initialized CQL-SAC-discrete with "
            f"cql_alpha={self.cql_alpha}, cql_temp={self.cql_temp}, target_tau={self.target_tau}, "
            f"auto_alpha={self.auto_alpha}, alpha_init={alpha_init}"
        )


    def init_stats(self, stats_path):

        with open(stats_path, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "iteration",
                    "avg_loss", "avg_q_value", "avg_target_value", "avg_mape", "avg_reward",
                    "avg_r_pros", "avg_r_idle", "avg_r_constraint", "avg_r_throughput", "avg_r_wl", "avg_r_end",
                    "avg_loss_td", "avg_loss_cql",
                    "avg_loss_actor", "avg_loss_alpha", "avg_alpha",
                ]
            )

        self.valid_losses = []
        self.q_history = []
        self.t_history = []
        self.mape_history = []
        self.reward_history = []
        self.reward_elements_history = []

        self.loss_td_hist = []
        self.loss_cql_hist = []
        self.loss_actor_hist = []
        self.loss_alpha_hist = []
        self.alpha_hist = []


    def record_stats(self, stats, rewards, reward_elements, it):

        self.valid_losses.append(stats["loss"])
        self.q_history.append(stats["q_value"])
        self.t_history.append(stats["target_value"])
        self.mape_history.append(stats["mape"])
        self.reward_history.append(rewards.mean().item())
        self.reward_elements_history.append(reward_elements.mean(dim=0))

        self.loss_td_hist.append(stats.get("loss_td", np.nan))
        self.loss_cql_hist.append(stats.get("loss_cql", np.nan))
        self.loss_actor_hist.append(stats.get("loss_actor", np.nan))
        self.loss_alpha_hist.append(stats.get("loss_alpha", np.nan))
        self.alpha_hist.append(stats.get("alpha", np.nan))

        if it % self.args.valid_interval == 0:

            avg_loss = np.mean(self.valid_losses)
            avg_q = np.mean(self.q_history)
            avg_t = np.mean(self.t_history)
            avg_mape = np.mean(self.mape_history)
            avg_reward = np.mean(self.reward_history)

            avg_r_elements = torch.stack(self.reward_elements_history, dim=0).mean(dim=0).tolist()
            avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end = avg_r_elements

            avg_loss_td = np.mean(self.loss_td_hist)
            avg_loss_cql = np.mean(self.loss_cql_hist)
            avg_loss_actor = np.mean(self.loss_actor_hist)
            avg_loss_alpha = np.mean(self.loss_alpha_hist)
            avg_alpha = np.mean(self.alpha_hist)

            self.valid_losses = []
            self.q_history = []
            self.t_history = []
            self.mape_history = []
            self.reward_history = []
            self.reward_elements_history = []
            self.loss_td_hist = []
            self.loss_cql_hist = []
            self.loss_actor_hist = []
            self.loss_alpha_hist = []
            self.alpha_hist = []

            with open(self.stats_path, mode="a", newline="") as file:

                writer = csv.writer(file)
                writer.writerow(
                    [
                        it, avg_loss, avg_q, avg_t, avg_mape, avg_reward,
                        avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end,
                        avg_loss_td, avg_loss_cql,
                        avg_loss_actor, avg_loss_alpha, avg_alpha,
                    ]
                )

            print(f"[Iter {it}] avg loss over last {self.args.valid_interval} iters: {avg_loss:.4f}")


    def _cql_penalty_per_event(self, q_all, q_taken, s0_all_sizes, s0_sizes):

        """
        q_all:   (sum(s0_all_sizes),)   Q(s, a_candidate) for all candidates per event
        q_taken: (sum(s0_sizes),)       Q(s, a_data) for dataset-taken actions per event
        returns: (num_events,)
        """

        temp = max(self.cql_temp, 1e-6)
        penalties = []
        start_all = 0
        start_taken = 0

        for all_sz, taken_sz in zip(s0_all_sizes, s0_sizes):
            end_all = start_all + int(all_sz)
            end_taken = start_taken + int(taken_sz)

            lse = torch.logsumexp(q_all[start_all:end_all] / temp, dim=0) * temp
            # max() for batch-machines (multiple taken actions)
            penalties.append(lse - q_taken[start_taken:end_taken].max())

            start_all = end_all
            start_taken = end_taken

        return torch.stack(penalties, dim=0)


    def _slice_policy(self, logits_slice):

        """
        logits_slice: (K,) logits for a single event's candidate set
        returns probs (K,), log_probs (K,)
        """

        log_probs = F.log_softmax(logits_slice, dim=0)
        probs = log_probs.exp()
        return probs, log_probs


    def _event_target_entropy(self, num_candidates: int):
        if self.target_entropy is not None:
            return float(self.target_entropy)
        # default: log(|A|) (variable candidate count)
        return math.log(max(int(num_candidates), 1))


    def train_step(self, data, it):

        """
         Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        1) Critic_loss = MSE(Q, Q_target)
        2) Actor_loss = α * log_pi(a|s) - Q(s,a)
        
        """
        
        args = self.args
        device = self.device
        agent = self.agent
        target_agent = self.target_agent

        # -------- data unpacking --------
        s_a_0_all, s_a_0, s_a_1, rewards, group_sizes, s0_all_sizes, s0_sizes, s1_sizes, gammas = data

        agent.train()

        rewards = rewards.view(-1).to(device)
        gammas = gammas.view(-1).to(device)

        # =========================
        # 1) Critic update (SAC + CQL)
        # =========================

        # Q(s, a_data)
        q1_taken = agent(s_a_0, which="q1").view(-1)
        q2_taken = agent(s_a_0, which="q2").view(-1)

        # Q(s, a_all) for CQL penalty
        q1_all = agent(s_a_0_all, which="q1").view(-1)
        q2_all = agent(s_a_0_all, which="q2").view(-1)

        # CQL penalty per event
        cql1 = self._cql_penalty_per_event(q1_all, q1_taken, s0_all_sizes, s0_sizes)
        cql2 = self._cql_penalty_per_event(q2_all, q2_taken, s0_all_sizes, s0_sizes)
        cql_per_event = 0.5 * (cql1 + cql2)

        # Next-state target critics on all next candidates
        with torch.no_grad():
            q1_next_all = target_agent(s_a_1, which="q1").view(-1)
            q2_next_all = target_agent(s_a_1, which="q2").view(-1)
            qmin_next_all = torch.min(q1_next_all, q2_next_all).view(-1)

        # Next-state policy logits on all next candidates (online actor)
        # (This is what BY571’s discrete SAC does in the fixed-action setting; here it's per-slice.)
        pi_next_logits_all = agent(s_a_1, which="pi").view(-1)

        # -------- per-machine aggregation (handles batch machines) --------
        q1_list, q2_list, qmin_list, t_list = [], [], [], []
        start0 = 0
        start1 = 0

        alpha_t = float(self.log_alpha.exp().detach().cpu().item())
        self.alpha = alpha_t  # keep readable

        for sz0, sz1 in zip(s0_sizes, s1_sizes):
            sz0 = int(sz0)
            sz1 = int(sz1)

            end0 = start0 + sz0
            end1 = start1 + sz1

            # mean Q over dataset actions for batch machines
            q1_list.append(q1_taken[start0:end0].mean())
            q2_list.append(q2_taken[start0:end0].mean())
            qmin_list.append(torch.min(q1_taken[start0:end0], q2_taken[start0:end0]).mean())

            # policy distribution over next candidate set
            logits_slice = pi_next_logits_all[start1:end1].view(-1)
            probs, log_probs = self._slice_policy(logits_slice)

            # SAC soft value terms: Q - alpha * log pi
            q_slice = qmin_next_all[start1:end1].view(-1)
            soft_q = q_slice - alpha_t * log_probs

            # batch-machine selection logic
            batchable, k = sa_batchable(s_a_0[start0:end0])
            if batchable:
                if sz1 > k:
                    idx = probs.topk(k=k).indices
                else:
                    idx = torch.arange(sz1, device=device)
                t_list.append(soft_q[idx].mean())
            else:
                # standard discrete SAC expectation over actions
                t_list.append((probs * soft_q).sum())

            start0, start1 = end0, end1

        q1_value = torch.stack(q1_list).unsqueeze(1).to(device)
        q2_value = torch.stack(q2_list).unsqueeze(1).to(device)
        qmin_value = torch.stack(qmin_list).unsqueeze(1).to(device)
        t_value = torch.stack(t_list).unsqueeze(1).to(device)

        target = (rewards + gammas * t_value).detach()

        q1_td = q1_value - target
        q2_td = q2_value - target

        # -------- per-group aggregation --------
        loss_list, td_list, cql_list = [], [], []
        q_s, t_s, cursor = [], [], 0

        for gs in group_sizes:

            gs = int(gs)
            sl = slice(cursor, cursor + gs)

            """
            TD error of nested events with different group sizes
            -
            Note on the special loss calculation:.mean().pow(2) rather than typical .pow(2).mean(): we take the mean over the group slice first and then square, rather than the more typical approach of squaring individual TD errors and then averaging. This is because in our system-level calculation, rewards of all nested events are averaged together to form the system-level reward, and accordingly we want to compute the TD error on that average reward rather than on individual event rewards. This way, the loss reflects the error in predicting the average return for the entire group of nested events, which is more aligned with our overall objective.
            Doesn't impact gs=1 -> .pow(2).mean()
            """
            td_g = 0.5 * (q1_td[sl].mean().pow(2) + q2_td[sl].mean().pow(2))

            cql_g = cql_per_event[sl].mean()
            loss_g = td_g + self.cql_alpha * cql_g

            loss_list.append(loss_g)
            td_list.append(td_g)
            cql_list.append(cql_g)

            q_s.append(qmin_value[sl].mean())
            t_s.append(target[sl].mean())

            cursor += gs

        critic_loss = torch.stack(loss_list).mean()
        loss_td = torch.stack(td_list).mean()
        loss_cql = torch.stack(cql_list).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip_maxnorm)
        self.critic_optimizer.step()

        # soft-update target  
        soft_update(self.agent, self.target_agent, self.target_tau)

        # =========================
        # 2) Actor + Alpha update (discrete SAC)
        # =========================
        # 2) Recompute (fresh) on s_a_0_all for current policy improvement.
        
        pi_logits_all = agent(s_a_0_all, which="pi").view(-1)

        with torch.no_grad():
            q1_all_new = agent(s_a_0_all, which="q1").view(-1)
            q2_all_new = agent(s_a_0_all, which="q2").view(-1)
            qmin_all_new = torch.min(q1_all_new, q2_all_new).view(-1)

        actor_losses = []
        log_action_pis = []
        cursor_all = 0
        cursor_taken = 0

        # we need per-event slicing over "all candidates" sizes (s0_all_sizes),
        # and batchability info from the taken-action slice (s0_sizes).
        for all_sz, taken_sz in zip(s0_all_sizes, s0_sizes):
            
            all_sz = int(all_sz)
            taken_sz = int(taken_sz)

            sl_all = slice(cursor_all, cursor_all + all_sz)
            sl_taken = slice(cursor_taken, cursor_taken + taken_sz)

            logits_slice = pi_logits_all[sl_all].view(-1)
            probs, log_probs = self._slice_policy(logits_slice)

            q_slice = qmin_all_new[sl_all].view(-1)

            batchable, k = sa_batchable(s_a_0[sl_taken])
            if batchable:
                if all_sz > k:
                    idx = probs.topk(k=k).indices
                else:
                    idx = torch.arange(all_sz, device=device)
                # deterministic top-k proxy for batch machines
                actor_losses.append((alpha_t * log_probs[idx] - q_slice[idx]).mean())
                log_action_pis.append(log_probs[idx].mean())
            else:
                # standard discrete SAC actor objective
                actor_losses.append((probs * (alpha_t * log_probs - q_slice)).sum())
                log_action_pis.append((probs * log_probs).sum())

            cursor_all += all_sz
            cursor_taken += taken_sz

        #  per-group aggregation style for the actor term too
        actor_loss_list = []
        alpha_loss_list = []

        cursor = 0
        for gs in group_sizes:
            gs = int(gs)
            sl = slice(cursor, cursor + gs)

            actor_g = torch.stack(actor_losses[sl]).mean()
            actor_loss_list.append(actor_g)

            # alpha loss (auto temperature)
            if self.auto_alpha:
                # per-event target entropy (variable candidate count)
                # we approximate by using log(|A|) unless user overrides
                # and apply it to the expected log pi.
                te = []
                for i in range(cursor, cursor + gs):
                    # candidate count for event i is s0_all_sizes[i]
                    te.append(self._event_target_entropy(int(s0_all_sizes[i])))
                target_entropy_g = torch.tensor(te, device=device, dtype=torch.float32).mean()

                log_pi_g = torch.stack(log_action_pis[sl]).mean()
                alpha_loss_g = -(self.log_alpha.exp() * (log_pi_g + target_entropy_g).detach())
                alpha_loss_list.append(alpha_loss_g)

            cursor += gs

        actor_loss = torch.stack(actor_loss_list).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip_maxnorm)
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = torch.stack(alpha_loss_list).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_t = float(self.log_alpha.exp().detach().cpu().item())
            self.alpha = alpha_t
            alpha_loss_item = float(alpha_loss.detach().cpu().item())
        else:
            alpha_loss_item = 0.0

        q_s = torch.stack(q_s).detach().cpu().view(-1)
        t_s = torch.stack(t_s).detach().cpu().view(-1)

        # mape can (in rare cases) explode if targets close to 0; keeping metric anyway.
        mape = mean_absolute_percentage_error(q_s.numpy(), t_s.numpy())
        q_mean = q_s.mean().item()
        t_mean = t_s.mean().item()

        total_loss = float(critic_loss.detach().cpu().item())

        return {
            "loss": total_loss,
            "q_value": q_mean,
            "target_value": t_mean,
            "mape": mape,
            "loss_td": float(loss_td.detach().cpu().item()),
            "loss_cql": float(loss_cql.detach().cpu().item()),
            "loss_actor": float(actor_loss.detach().cpu().item()),
            "loss_alpha": alpha_loss_item,
            "alpha": float(self.alpha),
        }
    


