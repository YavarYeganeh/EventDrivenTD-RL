

import csv
import copy
import math
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_percentage_error

from framework.agent import instantiate_agent
from framework.encoder import d_s_a, d_s


@torch.no_grad()
def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


class Algorithm:

    """
    SAC-style actor update with twin Q critics in the exact implementation style of PPO:
      - explicit data unpacking
      - per-machine aggregation (handles batch machines)
      - per-group aggregation via group_sizes
      - twin critics + clipped min(Q1,Q2) for the soft value / target side
      - target critic soft updates for stability
      - automatic entropy-temperature tuning

    with:
    require log_prob_0 is provided by the dataloader,
    although SAC itself does not directly use old action log-probabilities.
    """

    def __init__(self, args, device, dtype, stats_path):

        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats_path = stats_path

        self.agent = instantiate_agent(
            agent_type="sac",
            input_dim=d_s_a,
            state_dim=d_s,
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            precision=dtype,
            device=device,
        ).to(device)

        # target agent
        self.target_agent = copy.deepcopy(self.agent).to(device)
        self.target_agent.eval()
        for p in self.target_agent.parameters():
            p.requires_grad_(False)

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

        lr = float(getattr(args, "lr", 5e-5))
        lr_actor  = float(self.args.lr_actor)  if getattr(self.args, "lr_actor",  None) is not None else lr
        lr_critic = float(self.args.lr_critic) if getattr(self.args, "lr_critic", None) is not None else lr
        lr_alpha  = float(self.args.lr_alpha)  if getattr(self.args, "lr_alpha",  None) is not None else lr

        actor_params = list(self.agent.policy_net.parameters())
        critic_params = list(self.agent.q_net1.parameters()) + list(self.agent.q_net2.parameters())

        # optimizers
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # hyperparameters
        self.target_entropy = getattr(args, "target_entropy", None)
        self.target_entropy_c = getattr(args, "target_entropy_multiplier", 0.98) # 0.98 keep it close to the uniform while 0.5 incentivize sparse policy and search: sqrt(|A|) 
        self.target_tau = float(getattr(args, "target_tau", 0.005))

        self.init_stats(stats_path)

        print(
            f"Initialized SAC (Twin Q) with target_tau={self.target_tau}, "
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
                    "avg_loss_td", "avg_loss_actor", "avg_loss_alpha", "avg_alpha", "avg_entropy",
                ]
            )

        self.valid_losses = []
        self.q_history = []
        self.t_history = []
        self.mape_history = []
        self.reward_history = []
        self.reward_elements_history = []

        self.loss_td_hist = []
        self.loss_actor_hist = []
        self.loss_alpha_hist = []
        self.alpha_hist = []
        self.entropy_hist = []


    def record_stats(self, stats, rewards, reward_elements, it):

        self.valid_losses.append(stats["loss"])
        self.q_history.append(stats["q_value"])
        self.t_history.append(stats["target_value"])
        self.mape_history.append(stats["mape"])
        self.reward_history.append(rewards.mean().item())
        self.reward_elements_history.append(reward_elements.mean(dim=0))

        self.loss_td_hist.append(stats.get("loss_td", np.nan))
        self.loss_actor_hist.append(stats.get("loss_actor", np.nan))
        self.loss_alpha_hist.append(stats.get("loss_alpha", np.nan))
        self.alpha_hist.append(stats.get("alpha", np.nan))
        self.entropy_hist.append(stats.get("entropy", np.nan))

        if it % self.args.valid_interval == 0:

            avg_loss = np.mean(self.valid_losses)
            avg_q = np.mean(self.q_history)
            avg_t = np.mean(self.t_history)
            avg_mape = np.mean(self.mape_history)
            avg_reward = np.mean(self.reward_history)

            avg_r_elements = torch.stack(self.reward_elements_history, dim=0).mean(dim=0).tolist()
            avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end = avg_r_elements

            avg_loss_td = np.mean(self.loss_td_hist)
            avg_loss_actor = np.mean(self.loss_actor_hist)
            avg_loss_alpha = np.mean(self.loss_alpha_hist)
            avg_alpha = np.mean(self.alpha_hist)
            avg_entropy = np.mean(self.entropy_hist)

            self.valid_losses = []
            self.q_history = []
            self.t_history = []
            self.mape_history = []
            self.reward_history = []
            self.reward_elements_history = []
            self.loss_td_hist = []
            self.loss_actor_hist = []
            self.loss_alpha_hist = []
            self.alpha_hist = []
            self.entropy_hist = []

            with open(self.stats_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        it, avg_loss, avg_q, avg_t, avg_mape, avg_reward,
                        avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end,
                        avg_loss_td, avg_loss_actor, avg_loss_alpha, avg_alpha, avg_entropy,
                    ]
                )

            print(f"[Iter {it}] avg loss over last {self.args.valid_interval} iters: {avg_loss:.4f}")


    def _slice_policy(self, logits_slice):

        """
        logits_slice: (K,) logits for a single event's candidate set
        returns probs (K,), log_probs (K,)
        """

        log_probs = F.log_softmax(logits_slice, dim=0)
        probs = log_probs.exp()
        return probs, log_probs


    def _event_target_entropy(self, num_candidates: int):

        """
        default: log(|A|) -> c * log(|A|) (variable candidate count)
        could be annealed self.target_entropy_c = max(0.0, initial_c * (1.0 - it / anneal_steps)) 
        """

        if self.target_entropy is not None:
            return float(self.target_entropy)
        
        c = self.target_entropy_c
        t_ent = math.log(max(int(num_candidates), 1))
        
        return c * t_ent


    def train_step(self, data, it):

        args = self.args
        device = self.device
        agent = self.agent
        target_agent = self.target_agent

        # ----------------  data unpacking -------------------
        s_a_0, log_prob_0, s_a_0_all, s_a_1, r_event, r_group, group_sizes, s0_sizes, s0_all_sizes, s1_sizes, weights = data
        
        # ----------------  necessary prep for the update loop -------------------
        rewards = r_event + r_group
        gammas = weights

        # ----------------  forward passes  -------------------
        agent.train()

        # =========================
        # 1) Twin-Q critic update
        # =========================
        q1_taken = agent(s_a_0, which="q1").view(-1)
        q2_taken = agent(s_a_0, which="q2").view(-1)

        # target
        with torch.no_grad():
            q1_next_all = target_agent(s_a_1, which="q1").view(-1)
            q2_next_all = target_agent(s_a_1, which="q2").view(-1)
            qmin_next_all = torch.min(q1_next_all, q2_next_all).view(-1)

        # next-state policy logits on all next candidates (online actor); it's grad isn't necessary
        with torch.no_grad():
            pi_next_logits_all = agent(s_a_1, which="pi").view(-1)

        # ----------------  per-machine aggregation (also handles batch machines) -----------
        q1_list, q2_list, qmin_list, t_list = [], [], [], []
        start0 = 0
        start1 = 0

        alpha_t = float(self.log_alpha.exp().detach().cpu().item())
        self.alpha = alpha_t

        for sz0, sz1 in zip(s0_sizes, s1_sizes):

            sz0 = int(sz0)
            sz1 = int(sz1)
            end0 = start0 + sz0
            end1 = start1 + sz1

            # mean Q over for the case it's a batch machine
            q1_list.append(q1_taken[start0:end0].mean())
            q2_list.append(q2_taken[start0:end0].mean())
            qmin_list.append(torch.min(q1_taken[start0:end0], q2_taken[start0:end0]).mean())

            # policy distribution over next candidate set
            logits_slice = pi_next_logits_all[start1:end1].view(-1)
            probs, log_probs = self._slice_policy(logits_slice)

            # soft value: E_pi[ Q - alpha * log pi ]
            slice_q = qmin_next_all[start1:end1].view(-1)
            soft_q = slice_q - alpha_t * log_probs

            t_list.append((probs * soft_q).sum())

            start0, start1 = end0, end1

        q1_value = torch.stack(q1_list).unsqueeze(1).to(device)
        q2_value = torch.stack(q2_list).unsqueeze(1).to(device)
        qmin_value = torch.stack(qmin_list).unsqueeze(1).to(device)
        t_value = torch.stack(t_list).unsqueeze(1).to(device)

        # reward contribution
        td1_r_g = q1_value - r_event - gammas * t_value
        td2_r_g = q2_value - r_event - gammas * t_value

        # -------- per-group aggregation --------
        td_list, q_s, t_s, cursor = [], [], [], 0
        for gs in group_sizes:
            gs = int(gs)
            sl = slice(cursor, cursor + gs)

            """
            TD error of nested events with different group sizes
            -
            Note on the special loss calculation:.mean().pow(2) rather than typical .pow(2).mean(): we take the mean over the group slice first and then square, rather than the more typical approach of squaring individual TD errors and then averaging. This is because in our systen-level calculation, rewards of all nested events are averaged together to form the systen-level reward, and accordingly we want to compute the TD error on that average reward rather than on individual event rewards. This way, the loss reflects the error in predicting the average return for the entire group of nested events, which is more aligned with our overall objective.
            Doesn't impact gs=1 -> .pow(2).mean()
            Equal to having: Assuming event q_value = r_group_contribution + r_event + gammas * t_value
            """
            shared_r_group = r_group[sl].mean()

            td1 = td1_r_g[sl].mean() - shared_r_group
            td2 = td2_r_g[sl].mean() - shared_r_group

            td_g = 0.5 * (td1.pow(2) + td2.pow(2))
            td_list.append(td_g)

            q_s.append(qmin_value[sl])
            t_s.append((rewards + gammas * t_value)[sl])
            cursor += gs

        critic_loss = torch.stack(td_list).mean()
        loss_td = critic_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.q_net1.parameters()) + list(agent.q_net2.parameters()), args.grad_clip_maxnorm)
        self.critic_optimizer.step()

        #------------- critic update -------------------
        soft_update(self.agent, self.target_agent, self.target_tau)

        # =========================
        # 2) SAC actor + alpha update
        # =========================

        # -------- current logits --------
        pi_curr_logits = agent(s_a_0_all, which="pi").view(-1)

        with torch.no_grad():
            q1_all = agent(s_a_0_all, which="q1").view(-1)
            q2_all = agent(s_a_0_all, which="q2").view(-1)
            qmin_all = torch.min(q1_all, q2_all).view(-1)

        # -------- actor and alpha terms --------
        start = 0
        start_a = 0
        event_actor_losses = []
        event_entropies = []

        for sz0, sz0_a in zip(s0_sizes, s0_all_sizes):

            sz0, sz0_a = int(sz0), int(sz0_a)
            end = start + sz0
            end_a = start_a + sz0_a

            logits_slice = pi_curr_logits[start_a:end_a]
            probs, log_probs = self._slice_policy(logits_slice)
            q_slice = qmin_all[start_a:end_a]

            actor_loss_i = (probs * (alpha_t * log_probs - q_slice)).sum()
            entropy = - (probs * log_probs).sum()

            event_actor_losses.append(actor_loss_i)
            event_entropies.append(entropy.detach())

            start = end
            start_a = end_a

        # -------- actor / alpha group aggregation --------
        actor_loss_list = []
        alpha_loss_list = []
        ent_list = []
        cursor = 0
        for gs in group_sizes:
            gs = int(gs)
            sl = slice(cursor, cursor + gs)

            actor_loss_list.append(torch.stack(event_actor_losses[sl]).mean())
            ent_list.append(torch.stack(event_entropies[sl]).mean())

            if self.auto_alpha:
                te = []
                for i in range(cursor, cursor + gs):
                    te.append(self._event_target_entropy(int(s0_all_sizes[i])))
                target_entropy_g = torch.tensor(te, device=device, dtype=torch.float32)

                log_pi_g = - torch.stack(event_entropies[sl]) # negative as entropy is it's negative
                alpha_loss_sl = - (self.log_alpha * (log_pi_g + target_entropy_g).detach()) # self.log_alpha surrogate for stable grad instead of log_alpha.exp() = alpha
                alpha_loss_list.append(alpha_loss_sl.mean())

            cursor += gs

        actor_loss = torch.stack(actor_loss_list).mean()
        entropy_mean = torch.stack(ent_list).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), args.grad_clip_maxnorm)
        self.actor_optimizer.step()

        if self.auto_alpha:

            alpha_loss = torch.stack(alpha_loss_list).mean()
            
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_item = float(alpha_loss.detach().cpu().item())
        
            alpha_t = float(self.log_alpha.exp().detach().cpu().item())
            self.alpha = alpha_t
        
        else:
        
            alpha_loss_item = 0.0

        q_s = torch.cat(q_s, dim=0).detach().cpu().view(-1)
        t_s = torch.cat(t_s, dim=0).detach().cpu().view(-1)
        mape = mean_absolute_percentage_error(q_s.numpy(), t_s.numpy())

        total_loss = critic_loss + actor_loss
        if self.auto_alpha:
            total_loss = total_loss + alpha_loss.detach()

        return {
            "loss": float(total_loss.detach().cpu().item()),
            "q_value": float(q_s.mean().item()),
            "target_value": float(t_s.mean().item()),
            "mape": mape,
            "loss_td": float(loss_td.detach().cpu().item()),
            "loss_actor": float(actor_loss.detach().cpu().item()),
            "loss_alpha": alpha_loss_item,
            "alpha": float(self.alpha),
            "entropy": float(entropy_mean.detach().cpu().item()),
        }





