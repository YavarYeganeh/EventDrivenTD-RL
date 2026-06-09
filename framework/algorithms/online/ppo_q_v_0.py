
import csv
import copy
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_percentage_error

from framework.agent import instantiate_agent
from framework.encoder import d_s_a, d_s
from framework.utils import sa_batchable, find_selected_ids_s_a


@torch.no_grad()
def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


class Algorithm:

    """
    PPO-style clipped actor update with twin Q critics and of a V network:
      - needs Q as well as V for the specific aggregation method
      - explicit data unpacking
      - per-machine aggregation (handles batch machines)
      - per-group aggregation via group_sizes
      - twin critics + clipped min(Q1,Q2) for the baseline / target side
      - target critic soft updates for stability

    with:
    require old_log_probs is expected to align with s_a_0 (the dataset-taken actions) as well as s_a_0_all
    Therefore the clipped PPO ratio is computed over the taken-action slice rather
    than over a separate all-candidate current-state support.
    this version calculates importance sampling with the exact regulated policies 
    """

    def __init__(self, args, device, dtype, stats_path):

        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats_path = stats_path

        self.agent = instantiate_agent(
            agent_type="ppo",
            input_dim=d_s_a,
            state_dim=d_s,
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            precision=dtype,
            device=device,
            build_value=True,
        ).to(device)

        
        # target agent
        self.target_agent = copy.deepcopy(self.agent).to(device)
        self.target_agent.eval()
        for p in self.target_agent.parameters():
            p.requires_grad_(False)

        lr = float(getattr(args, "lr", 1e-5))
        lr_actor  = float(self.args.lr_actor)  if getattr(self.args, "lr_actor",  None) is not None else lr
        lr_critic = float(self.args.lr_critic) if getattr(self.args, "lr_critic", None) is not None else lr

        actor_params = list(self.agent.policy_net.parameters()) 
        critic_params = list(self.agent.q_net1.parameters()) + list(self.agent.q_net2.parameters())
        v_params = list(self.agent.v_net.parameters())

        # optimizers
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr_critic)
        self.v_optimizer = torch.optim.Adam(v_params, lr=lr_critic)

        # hyperparameters
        self.clip_eps = float(getattr(args, "ppo_clip_eps", 0.2))
        self.clip_log_ratio = float(getattr(args, "ppo_clip_log_ratio", 7)) # for safety
        self.entropy_coef = float(getattr(args, "ppo_entropy_coef", 0.01))
        self.adv_clip = float(getattr(args, "ppo_adv_clip", 0.05))
        self.target_tau = float(getattr(args, "target_tau", 0.005))

        self.init_stats(stats_path)

        print(
            f"Initialized PPO (Twin Q + V) with clip_eps={self.clip_eps}, entropy_coef={self.entropy_coef}, "
            f"target_tau={self.target_tau}"
        )


    def init_stats(self, stats_path):

        with open(stats_path, mode="w", newline="") as f:

            w = csv.writer(f)
            w.writerow(
                [
                    "iteration",
                    "avg_loss", "avg_q_value", "avg_target_value", "avg_mape", "avg_reward",
                    "avg_r_pros", "avg_r_idle", "avg_r_constraint", "avg_r_throughput", "avg_r_wl", "avg_r_end",
                    "avg_loss_td", "avg_loss_actor", "avg_loss_v", "avg_advantage", "avg_entropy", "avg_ratio",
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
        self.loss_v_hist = []
        self.adv_hist = []
        self.entropy_hist = []
        self.ratio_hist = []


    def record_stats(self, stats, rewards, reward_elements, it):

        self.valid_losses.append(stats["loss"])
        self.q_history.append(stats["q_value"])
        self.t_history.append(stats["target_value"])
        self.mape_history.append(stats["mape"])
        self.reward_history.append(rewards.mean().item())
        self.reward_elements_history.append(reward_elements.mean(dim=0))

        self.loss_td_hist.append(stats.get("loss_td", np.nan))
        self.loss_actor_hist.append(stats.get("loss_actor", np.nan))
        self.loss_v_hist.append(stats.get("loss_v", np.nan))
        self.adv_hist.append(stats.get("advantage", np.nan))
        self.entropy_hist.append(stats.get("entropy", np.nan))
        self.ratio_hist.append(stats.get("ratio", np.nan))

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
            avg_loss_v = np.mean(self.loss_v_hist)
            avg_advantage = np.mean(self.adv_hist)
            avg_entropy = np.mean(self.entropy_hist)
            avg_ratio = np.mean(self.ratio_hist)

            self.valid_losses = []
            self.q_history = []
            self.t_history = []
            self.mape_history = []
            self.reward_history = []
            self.reward_elements_history = []
            self.loss_td_hist = []
            self.loss_actor_hist = []
            self.loss_v_hist = []
            self.adv_hist = []
            self.entropy_hist = []
            self.ratio_hist = []

            with open(self.stats_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        it, avg_loss, avg_q, avg_t, avg_mape, avg_reward,
                        avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end,
                        avg_loss_td, avg_loss_actor, avg_loss_v, avg_advantage, avg_entropy, avg_ratio,
                    ]
                )

            print(f"[Iter {it}] avg loss over last {self.args.valid_interval} iters: {avg_loss:.4f}")


    def _slice_policy(self, logits_slice, constraint_regularization=False, s_a_all=None):

        """
        logits_slice: (K,) logits for a single event's candidate set
        constraint_regularization if requested
        returns probs (K,), log_probs (K,)
        """

        probs = F.softmax(logits_slice, dim=0)
        
        if constraint_regularization:
            probs = self.agent.constraint_regularization(probs, s_a_all) # regularization for importance sampling

        log_probs = torch.log(probs + 1e-6) # some constraints can make some probs 0 (in case)

        return probs, log_probs
    

    def train_step(self, data, it):

        args = self.args
        device = self.device
        agent = self.agent
        target_agent = self.target_agent

        # ----------------  data unpacking -------------------
        s_a_0, log_prob_0, s_a_0_all, s_a_1, r_event, r_group, group_sizes, s0_sizes, s0_all_sizes, s1_sizes, weights = data

        # ----------------  necessary prep for the update loop -------------------
        rewards = r_event + r_group
        gammas = weights # could be gae (lambda*gamma)

        # ----------------  forward passes  -------------------
        agent.train()

        # =========================
        # 2) Twin-Q critic update
        # =========================
        q1_taken = agent(s_a_0, which="q1").view(-1)
        q2_taken = agent(s_a_0, which="q2").view(-1)

        # V on s and s'
        s0 = s_a_0[:, :d_s]
        s1 = s_a_1[:, :d_s]
        v0_logits = agent(s0, which="v")      # (n0,)
        v1_logits = agent(s1, which="v")      # (n1,)

        # target
        with torch.no_grad():
            q1_next_all = target_agent(s_a_1, which="q1").view(-1)
            q2_next_all = target_agent(s_a_1, which="q2").view(-1)
            qmin_next_all = torch.min(q1_next_all, q2_next_all).view(-1)

        # Next-state policy logits on all next candidates (online actor)
        pi_next_logits_all = agent(s_a_1, which="pi").view(-1)

        # ----------------  per-machine aggregation (also handles batch machines) -----------
        q1_list, q2_list, qmin_list, t_list, v0_list, v_loss_list = [], [], [], [], [], []
        start0 = 0
        start1 = 0
 
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

            # ----- choose action indices inside this slice -----
            slice_len = end1 - start1
            slice_q = qmin_next_all[start1:end1].view(-1)

            batchable, k = sa_batchable(s_a_0[start0:end0])
            if batchable and slice_len > k:
                act_idx = probs.topk(k=k).indices
            elif batchable:
                act_idx = torch.arange(slice_len, device=device)
            else:
                act_idx = probs.argmax().unsqueeze(0)

            t_list.append(slice_q[act_idx].mean())
            start0, start1 = end0, end1

            # ---------- V loss  ---------- 
            # s is shared and first/mean elements are ok
            # LV = E[ L_2( v(s) - r - gamma*V(s') ) ]
            v0_logits_slice = v0_logits[start0:end0].mean()
            v1_logits_slice = v1_logits[start1:end1].mean() 
            r_slice = rewards[start0:end0].mean()
            gammas_slice = gammas[start0:end0].mean()
            v_loss = F.mse_loss(v0_logits_slice, r_slice + gammas_slice * v1_logits_slice.detach())
    
            v0_list.append(v0_logits_slice)
            v_loss_list.append(v_loss)


        q1_value = torch.stack(q1_list).unsqueeze(1).to(device)
        q2_value = torch.stack(q2_list).unsqueeze(1).to(device)
        qmin_value = torch.stack(qmin_list).unsqueeze(1).to(device)
        t_value = torch.stack(t_list).unsqueeze(1).to(device)
        v0 = torch.stack(v0_list).unsqueeze(1).to(device)
        v_loss = torch.stack(v_loss_list).unsqueeze(1).to(device)

        # reward contribution
        td1_r_g = q1_value - r_event - gammas * t_value 
        td2_r_g = q2_value - r_event - gammas * t_value 

        # -------- per-group aggregation --------
        td_list, v_loss_list, q_s, t_s, cursor = [], [], [], [], 0
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

            # mean aggregation
            td1 = td1_r_g[sl].mean() - shared_r_group
            td2 = td2_r_g[sl].mean() - shared_r_group 

            td_g = 0.5 * (td1.pow(2) + td2.pow(2))
            td_list.append(td_g)

            q_s.append(qmin_value[sl])
            t_s.append((rewards + gammas * t_value)[sl])
            
            # V loss per group
            v_loss_g = v_loss[sl].mean()
            v_loss_list.append(v_loss_g)
            
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
        # 2) PPO actor update using twin-Q advantage (The order of update is synchronous here critic and actor)
        # =========================

        # -------- advantage --------

        """
        r_empirical = (qmin_value - gammas * t_value).detach() # r_empirical vs pred
        # adv_per_event = r_empirical - r_group #  using stop-grad clipped double-Q target minus current clipped double-Q baseline

               r_avg_v = v
        adv_per_event = r_empirical - r_avg_v.detach() 

        adv_per_event = qmin_value - t_value # using clipped double-Q target minus current clipped double-Q baseline (no explicit reward term, just TD error style advantage)
        
        """
        adv_per_event = qmin_value - v0.detach() # using clipped double-Q target minus current V baseline (no explicit reward term, just TD error style advantage)
 
        
        # -------- normalize and clip advantages --------
        adv_per_event = (adv_per_event - adv_per_event.mean()) / (adv_per_event.std() + 1e-8)
        if self.adv_clip > 0.0:
            adv_per_event = adv_per_event.clamp(-self.adv_clip, self.adv_clip)

        # -------- current logits --------
        pi_curr_logits = agent(s_a_0_all, which="pi").view(-1)

        # -------- ratio and advantages --------
        start = 0
        start_a = 0
        event_actor_losses = []
        event_advantages = []
        event_entropies = []
        event_ratios = []

        for i, (sz0, sz0_a) in enumerate(zip(s0_sizes, s0_all_sizes)):

            sz0, sz0_a = int(sz0), int(sz0_a)
            end = start + sz0
            end_a = start_a + sz0_a

            # 1. Slice current log probs (regulated) for this specific experience 
            s_a_0_all_slice = s_a_0_all[start_a:end_a]
            curr_logits_slice = pi_curr_logits[start_a:end_a]
            curr_probs_slice, curr_log_probs_slice = self._slice_policy(curr_logits_slice, constraint_regularization=True, s_a_all=s_a_0_all_slice) 

            # 2. Map taken actions back to the full candidate set
            # note: s_a_0 is the 'taken' action tensor
            selected_ids = find_selected_ids_s_a(s_a_0_all[start_a:end_a], s_a_0[start:end]) 
            
            # 3. Calculate Ratio
            curr_log_selected = curr_log_probs_slice[selected_ids].sum()
            old_logp_selected = log_prob_0[i] # torch.log(probs + 1e-6).sum(), # sum for the case batch/multi select actions
            log_ratio = curr_log_selected - old_logp_selected.detach()
            log_ratio_safe = torch.clamp(log_ratio, -self.clip_log_ratio, self.clip_log_ratio) # clip ratio numerical safety preventing exploding
            ratio = torch.exp(log_ratio_safe)

            # 4. Entropy Calculation
            # Entropy = -sum(p * log_p)
            entropy = -(curr_probs_slice * curr_log_probs_slice).sum() # already normalized

            # 5. PPO Clipping
            adv_i = adv_per_event[i]
            surr1 = ratio * adv_i
            surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_i

            # 6. Loss assembly
            actor_loss_i = -torch.min(surr1, surr2) - self.entropy_coef * entropy
            
            # Collect results
            event_actor_losses.append(actor_loss_i)
            event_advantages.append(adv_i)
            event_entropies.append(entropy.detach())
            event_ratios.append(ratio.detach())

            # Update offsets for next iteration
            start = end
            start_a = end_a

        # -------- actor loss --------
        actor_loss_list = []
        adv_list = []
        ent_list = []
        ratio_list = []
        cursor = 0
        for gs in group_sizes:
            gs = int(gs)
            sl = slice(cursor, cursor + gs)
            actor_loss_list.append(torch.stack(event_actor_losses[sl]).mean())
            adv_list.append(torch.stack(event_advantages[sl]).mean())
            ent_list.append(torch.stack(event_entropies[sl]).mean())
            ratio_list.append(torch.stack(event_ratios[sl]).mean())
            cursor += gs

        actor_loss = torch.stack(actor_loss_list).mean()
        adv_mean = torch.stack(adv_list).mean()
        entropy_mean = torch.stack(ent_list).mean()
        ratio_mean = torch.stack(ratio_list).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), args.grad_clip_maxnorm)
        self.actor_optimizer.step()

        # =========================
        # 3)  Value update
        # =========================
        v_loss = torch.stack(v_loss_list).mean()

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.v_net.parameters(), args.grad_clip_maxnorm)
        self.v_optimizer.step()

        # =========================
        # 3)  stats calculation and recording
        # =========================
        q_s = torch.cat(q_s, dim=0).detach().cpu().view(-1)
        t_s = torch.cat(t_s, dim=0).detach().cpu().view(-1)
        mape = mean_absolute_percentage_error(q_s.numpy(), t_s.numpy())

        return {
            "loss": float((critic_loss + actor_loss).detach().cpu().item()),
            "q_value": float(q_s.mean().item()),
            "target_value": float(t_s.mean().item()),
            "mape": mape,
            "loss_td": float(loss_td.detach().cpu().item()),
            "loss_actor": float(actor_loss.detach().cpu().item()),
            "loss_v": float(v_loss.detach().cpu().item()),
            "advantage": float(adv_mean.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "ratio" :  float(ratio_mean.detach().cpu().item()),
        }

