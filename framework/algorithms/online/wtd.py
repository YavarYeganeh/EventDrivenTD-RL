

import torch
import torch.nn.functional as F
import csv
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from framework.encoder import d_s_a
from framework.agent import instantiate_agent 
from framework.utils import sa_batchable


class Algorithm:

    """
    Online TD algorithm wrapper, written in the same style as  the DQL Algorithm working with time weighted TD error:
      - init agent + optimizer
      - one train_step(data, it)
      - stats init/record helpers

    IMPORTANT:
      - This version uses `weights` exactly like  the loop:
          td = ((q_slice - gamma*t_slice)*weights_slice).sum()
          loss = (td - reward_group).pow(2)
      - rewards are averaged per seed-group (single reward scheme)
    """

    def __init__(self, args, device, dtype, stats_path):

        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats_path = stats_path

        self.agent = instantiate_agent(
            agent_type="dql",
            input_dim=d_s_a,
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            precision=dtype,
            device=device,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.lr)

        self.gamma = float(getattr(args, "gamma", 0.99))
        self.init_stats(stats_path)
        print(f"Initialized Weighted TD Algorithm based DQL with gamma={self.gamma} and stats path: {stats_path}")


    def init_stats(self, stats_path):

        with open(stats_path, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "iteration",
                "avg_loss", "avg_q_value", "avg_target_value", "avg_mape", "avg_reward",
                "avg_r_pros", "avg_r_idle", "avg_r_constraint", "avg_r_throughput", "avg_r_wl", "avg_r_end",
            ])

        self.valid_losses = []
        self.q_history = []
        self.t_history = []
        self.mape_history = []
        self.reward_history = []
        self.reward_elements_history = []


    def record_stats(self, stats, rewards, reward_elements, it):

        self.valid_losses.append(stats["loss"])
        self.q_history.append(stats["q_value"])
        self.t_history.append(stats["target_value"])
        self.mape_history.append(stats["mape"])
        self.reward_history.append(rewards.mean().item())
        self.reward_elements_history.append(reward_elements.mean(dim=0).detach().cpu())

        if it % self.args.valid_interval == 0:
            avg_loss = float(np.mean(self.valid_losses))
            avg_q = float(np.mean(self.q_history))
            avg_t = float(np.mean(self.t_history))
            avg_mape = float(np.mean(self.mape_history))
            avg_reward = float(np.mean(self.reward_history))

            avg_r_elements = torch.stack(self.reward_elements_history, dim=0).mean(dim=0).tolist()
            avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end = avg_r_elements

            # reset
            self.valid_losses = []
            self.q_history = []
            self.t_history = []
            self.mape_history = []
            self.reward_history = []
            self.reward_elements_history = []

            with open(self.stats_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    it, avg_loss, avg_q, avg_t, avg_mape, avg_reward,
                    avg_r_pros, avg_r_idle, avg_r_constraint, avg_r_throughput, avg_r_wl, avg_r_end
                ])

            print(f"[Iter {it}] avg loss over last {self.args.valid_interval} iters: {avg_loss:.4f}")


    def train_step(self, data, it):

        """
        data:
          s_a_0, s_a_1, rewards, reward_elements, group_sizes, s0_sizes, s1_sizes, weights
        """
        args = self.args
        device = self.device
        agent = self.agent
        optimizer = self.optimizer

        s_a_0, s_a_1, rewards, group_sizes, s0_sizes, s1_sizes, weights = data

        # ---- forward passes ----
        agent.train()
        q_logits = agent(s_a_0, target=False)  # (sum s0_sizes,)

        agent.eval()
        with torch.no_grad():
            act_logits = agent(s_a_1, target=False)  # action selector logits
            tgt_logits = agent(s_a_1, target=True)   # target Q values (sum s1_sizes,)

        # ---- per-machine aggregation ----
        q_list, t_list = [], []
        start0 = 0
        start1 = 0

        for sz0, sz1 in zip(s0_sizes, s1_sizes):
            sz0 = int(sz0)
            sz1 = int(sz1)
            end0 = start0 + sz0
            end1 = start1 + sz1

            # mean Q for current (batch-machine safe)
            q_list.append(q_logits[start0:end0].mean())

            # choose indices in next-slice for target bootstrap
            slice_len = end1 - start1
            slice_logits = act_logits[start1:end1].view(-1)
            slice_tvalues = tgt_logits[start1:end1].view(-1)

            # if  have sa_batchable, call it here instead
            batchable, k = sa_batchable(s_a_0[start0:end0], batch_max=getattr(args, "batch_max", 1))

            if batchable and slice_len > k:
                act_idx = slice_logits.topk(k=k).indices
            elif batchable:
                act_idx = torch.arange(slice_len, device=device)
            else:
                act_idx = slice_logits.argmax().unsqueeze(0)

            t_list.append(slice_tvalues[act_idx].mean())

            start0, start1 = end0, end1

        q_value = torch.stack(q_list).unsqueeze(1).to(device)  # (num_machines,1)
        t_value = torch.stack(t_list).unsqueeze(1).to(device)  # (num_machines,1)

        dis_t_value = self.gamma * t_value

        # ---- per-seed loss (weighted TD; reward averaged per group) ----
        losses, q_s, t_s = [], [], []
        cursor = 0

        for gs in group_sizes:
            gs = int(gs)
            sl = slice(cursor, cursor + gs)

            q_slice = q_value[sl]                 # (gs,1)
            t_dis_slice = dis_t_value[sl]         # (gs,1)
            weights_slice = weights[sl]           # (gs,1) or (gs,)
            weights_slice = weights_slice.view(-1, 1)

            reward_group = rewards[sl].mean()     # scalar

            # weighted TD error ( the exact pattern)
            td = ((q_slice - t_dis_slice) * weights_slice).sum()
            losses.append((td - reward_group).pow(2))

            q_s.append(q_slice)
            t_s.append(t_dis_slice + reward_group)  # for logging: add reward back

            cursor += gs

        loss = torch.stack(losses).mean()

        q_s = torch.cat(q_s, dim=0).detach().cpu().view(-1)
        t_s = torch.cat(t_s, dim=0).detach().cpu().view(-1)
        mape = mean_absolute_percentage_error(q_s.numpy(), t_s.numpy())

        q_mean = q_s.mean().item()
        t_mean = t_s.mean().item()

        # ---- optimize ----
        agent.train()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip_maxnorm)
        optimizer.step()

        stats = {
            "loss": loss.item(),
            "q_value": q_mean,
            "target_value": t_mean,
            "mape": mape,
        }
        return stats
    



