import csv
import copy
import numpy as np
from scipy import stats
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

    def __init__(self, args, device, dtype, stats_path):

        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats_path = stats_path

        self.agent = instantiate_agent(
            agent_type = "cql_q", # build_policy=False for the Q-learning variant doesn't need policy head
            input_dim=d_s_a,
            state_dim=d_s,
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            precision=dtype,
            device=device,
        ).to(device)

        # target critic ( for stability)
        self.target_agent = copy.deepcopy(self.agent).to(device)
        self.target_agent.eval()
        for p in self.target_agent.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.lr)

        # CQL hyperparams
        self.cql_alpha = float(getattr(args, "cql_alpha", 1.0))
        self.cql_temp = float(getattr(args, "cql_temp", 1.0))
        self.target_tau = float(getattr(args, "target_tau", 0.005))

        self.init_stats(stats_path)

        print(f"Initialized CQL-Q (H, entropy) with cql_alpha={self.cql_alpha}, cql_temp={self.cql_temp}, target_tau={self.target_tau}")


    def init_stats(self, stats_path):
        with open(stats_path, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "iteration",
                "avg_loss", "avg_q_value", "avg_target_value", "avg_mape", "avg_reward",
                'avg_r_pros', 'avg_r_idle', 'avg_r_constraint', 'avg_r_throughput', 'avg_r_wl', 'avg_r_end',
                "avg_loss_td", "avg_loss_cql"
            ])

        self.valid_losses = []
        self.q_history = []
        self.t_history = []
        self.mape_history = []
        self.reward_history = []
        self.reward_elements_history = []

        # cql-specific stats
        self.loss_td_hist = []
        self.loss_cql_hist = []


    def record_stats(self, stats, rewards, reward_elements, it):

        # ----------------  Recording history  -------------------------
        self.valid_losses.append(stats['loss'])
        self.q_history.append(stats['q_value'])
        self.t_history.append(stats['target_value'])
        self.mape_history.append(stats['mape'])
        self.reward_history.append(rewards.mean().item())
        self.reward_elements_history.append(reward_elements.mean(dim=0))

        # cql breakdown
        self.loss_td_hist.append(stats.get("loss_td", np.nan))
        self.loss_cql_hist.append(stats.get("loss_cql", np.nan))

        # every valid_interval iterations: compute and save avg stats
        if it % self.args.valid_interval == 0:

            avg_loss = np.mean(self.valid_losses)
            avg_q = np.mean(self.q_history)
            avg_t = np.mean(self.t_history)
            avg_mape = np.mean(self.mape_history)
            avg_reward = np.mean(self.reward_history)

            avg_r_elements = (
                torch.stack(self.reward_elements_history, dim=0)
                .mean(dim=0)
                .tolist()
            )
            avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end = avg_r_elements

            avg_loss_td = np.mean(self.loss_td_hist)
            avg_loss_cql = np.mean(self.loss_cql_hist)

            # reset histories
            self.valid_losses = []
            self.q_history = []
            self.t_history = []
            self.mape_history = []
            self.reward_history = []
            self.reward_elements_history = []

            self.loss_td_hist = []
            self.loss_cql_hist = []

            # append to CSV
            with open(self.stats_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    it, avg_loss, avg_q, avg_t, avg_mape, avg_reward,
                    avg_r_pros, avg_r_idle, avg_r_constraint,
                    avg_r_throughput, avg_r_wl, avg_r_end,
                    avg_loss_td, avg_loss_cql
                ])

            print(f"[Iter {it}] avg loss over last {self.args.valid_interval} iters: {avg_loss:.4f}")


    def _cql_penalty_per_event(self, q_all, q_taken, s0_all_sizes, s0_sizes):

        """
        q_all:   (sum(s0_all_sizes),)
        q_taken: (sum(s0_sizes),)
        returns: (num_events,)
        """

        temp = max(self.cql_temp, 1e-6)
        penalties = []
        start_all = 0
        start_taken = 0

        for i, (all_sz, taken_sz) in enumerate(zip(s0_all_sizes, s0_sizes)):
        
            end_all = start_all + int(all_sz)
            end_taken = start_taken + int(taken_sz)
        
            lse = torch.logsumexp(q_all[start_all:end_all] / temp, dim=0) * temp
            penalties.append(lse - q_taken[start_taken:end_taken].mean()) # for the case of batch machines (max would be less conservative than mean)
        
            start_all = end_all
            start_taken = end_taken

        return torch.stack(penalties, dim=0)


    def _next_max_per_event(self, q_next_all, s1_sizes):

        """
        q_next_all: (sum(s1_sizes),)
        returns:    (num_events,)
        """

        nxt = []
        start = 0

        for sz in s1_sizes:

            end = start + int(sz)
            nxt.append(q_next_all[start:end].max())
            start = end

        return torch.stack(nxt, dim=0)


    def train_step(self, data, it):

        args = self.args
        device = self.device
        agent = self.agent

        # -------- data unpacking --------
        s_a_0_all, s_a_0, s_a_1, rewards, group_sizes, s0_all_sizes, s0_sizes, s1_sizes, gammas = data

        agent.train()

        # ensure shapes are 1D (num_events,)
        rewards = rewards.view(-1).to(device)
        gammas = gammas.view(-1).to(device)

        # Q(s,a_data) (behavior-taken or action)
        q1_taken = agent(s_a_0, which="q1")
        q2_taken = agent(s_a_0, which="q2")
        qmin_taken = torch.min(q1_taken, q2_taken)

        # Q(s,a_all) (all candidates for CQL penalty)
        q1_all = agent(s_a_0_all, which="q1")
        q2_all = agent(s_a_0_all, which="q2")

        # CQL penalty per event (apply to each critic then average)
        cql1 = self._cql_penalty_per_event(q1_all, q1_taken, s0_all_sizes, s0_sizes)
        cql2 = self._cql_penalty_per_event(q2_all, q2_taken, s0_all_sizes, s0_sizes)
        cql_per_event = 0.5 * (cql1 + cql2)

        # for target: r + gamma * max_a Q_target(s',a)
        with torch.no_grad():
            q1_next_all = self.target_agent(s_a_1, which="q1")
            q2_next_all = self.target_agent(s_a_1, which="q2")
            qmin_next_all = torch.min(q1_next_all, q2_next_all)

        # ----------------  per-machine aggregation (also handles batch machines) -----------
        q1_list, q2_list, t_list = [], [], []
        start0 = start1 = 0

        for sz0, sz1 in zip(s0_sizes, s1_sizes):
        
            end0, end1 = start0 + sz0, start1 + sz1

            # mean Q over for the case it's a batch machine
            q1_list.append(q1_taken[start0:end0].mean())
            q2_list.append(q2_taken[start0:end0].mean())

            # ----- choose action indices inside this slice -----
            slice_len = end1 - start1
            slice_logits = qmin_next_all[start1:end1].view(-1)

            batchable, k = sa_batchable(s_a_0[start0:end0])
            if batchable and slice_len > k:
                act_idx = slice_logits.topk(k=k).indices  # 0 … slice_len-1
            elif batchable:  # take all
                act_idx = torch.arange(slice_len, device=device)
            else:  # greedy 1-best
                act_idx = slice_logits.argmax().unsqueeze(0)

            # gather target-net values for those actions
            t_list.append(qmin_next_all[act_idx].mean())

            start0, start1 = end0, end1

        # stack lists → tensors [num_machines, 1]
        q1_value = torch.stack(q1_list).unsqueeze(1).to(device)
        q2_value = torch.stack(q2_list).unsqueeze(1).to(device)
        t_value = torch.stack(t_list).unsqueeze(1).to(device)

        target = rewards + gammas * t_value  # Bellman target (num_events/num machines,)

        # TD loss per event
        q1_td = q1_value - target
        q2_td = q2_value - target

        # -------- per-seed aggregation --------
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

            q_s.append(qmin_taken[sl].mean())
            t_s.append(target[sl].mean())

            cursor += gs

        loss = torch.stack(loss_list).mean()
        loss_td = torch.stack(td_list).mean()
        loss_cql = torch.stack(cql_list).mean()

        # stats
        q_s = torch.stack(q_s).detach().cpu()
        t_s = torch.stack(t_s).detach().cpu()
        mape = mean_absolute_percentage_error(q_s.numpy(), t_s.numpy())
        q_mean = q_s.mean().item()
        t_mean = t_s.mean().item()

        # optimize
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip_maxnorm)
        self.optimizer.step()

        # target update
        soft_update(self.agent, self.target_agent, self.target_tau)

        return {
            "loss": loss.item(),
            "q_value": q_mean,
            "target_value": t_mean,
            "mape": mape,
            "loss_td": loss_td.item(),
            "loss_cql": loss_cql.item(),
        }
    
