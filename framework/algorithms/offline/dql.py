import torch
import csv
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


from framework.agent import instantiate_agent
from framework.utils import sa_batchable
from framework.encoder import d_s_a


class Algorithm:
   
    """
    Wraps:
      - agent + optimizer instantiation
      - one training step 
    """

    def __init__(self, args, device, dtype, stats_path):
        
        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats_path = stats_path

        self.agent = instantiate_agent(
            agent_type = "dql",
            input_dim=d_s_a,
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            precision=dtype,
            device=device,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.lr)

        self.init_stats(self.stats_path)


    def init_stats(self, stats_path):

        # initialize CSV header
        with open(stats_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['iteration', 'avg_loss', 'avg_q_value', 'avg_target_value', 'avg_mape', 'avg_reward', 'avg_r_pros', 'avg_r_idle', 'avg_r_constraint', 'avg_r_throughput', 'avg_r_wl', 'avg_r_end'])

        """ Statistics History """
        self.valid_losses = []
        # self.avg_loss_records = []
        self.q_history = []
        self.t_history = []
        self.mape_history = []
        self.reward_history = []
        self.reward_elements_history = []


    def record_stats(self, stats, rewards, reward_elements, it):
     
        # ----------------  Recording history  -------------------------
        self.valid_losses.append(stats['loss'])
        self.q_history.append(stats['q_value'])
        self.t_history.append(stats['target_value'])
        self.mape_history.append(stats['mape'])
        self.reward_history.append(rewards.mean().item())
        self.reward_elements_history.append(reward_elements.mean(dim=0))

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

            # reset histories
            self.valid_losses = []
            self.q_history = []
            self.t_history = []
            self.mape_history = []
            self.reward_history = []
            self.reward_elements_history = []

            # append to CSV
            with open(self.stats_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    it, avg_loss, avg_q, avg_t, avg_mape, avg_reward,
                    avg_r_pros, avg_r_idle, avg_r_constraint,
                    avg_r_throughput, avg_r_wl, avg_r_end
                ])

            print(f"[Iter {it}] avg loss over last {self.args.valid_interval} iters: {avg_loss:.4f}")


    def train_step(self, data, it):

        args = self.args
        device = self.device
        agent = self.agent
        optimizer = self.optimizer

        # ----------------  data unpacking -------------------
        s_a_0, s_a_1, rewards, group_sizes, s0_sizes, s1_sizes, gammas = data

        # ----------------  forward passes  -------------------
        agent.train()
        q_logits = agent(s_a_0, target=False)  # online net
        act_logits = agent(s_a_1, target=False)

        agent.eval()
        with torch.no_grad():
            tgt_logits = agent(s_a_1, target=True)  # target net (no grad)

        # ----------------  per-machine aggregation (also handles batch machines) -----------
        q_list, t_list = [], []
        start0 = start1 = 0
        for sz0, sz1 in zip(s0_sizes, s1_sizes):
            end0, end1 = start0 + sz0, start1 + sz1

            # mean Q over for the case it's a batch machine
            q_list.append(q_logits[start0:end0].mean())

            # ----- choose action indices inside this slice -----
            slice_len = end1 - start1
            slice_logits = act_logits[start1:end1].view(-1)
            slice_tvalues = tgt_logits[start1:end1].view(-1)

            batchable, k = sa_batchable(s_a_0[start0:end0])
            if batchable and slice_len > k:
                act_idx = slice_logits.topk(k=k).indices  # 0 … slice_len-1
            elif batchable:  # take all
                act_idx = torch.arange(slice_len, device=device)
            else:  # greedy 1-best
                act_idx = slice_logits.argmax().unsqueeze(0)

            # gather target-net values for those actions
            t_list.append(slice_tvalues[act_idx].mean())

            start0, start1 = end0, end1

        # stack lists → tensors [num_machines, 1]
        q_value = torch.stack(q_list).unsqueeze(1).to(device)
        t_value = torch.stack(t_list).unsqueeze(1).to(device)

        target = rewards + gammas * t_value  # Bellman target

        # ----------------  per-seed loss  ---------------------
        losses, q_s, t_s, cursor = [], [], [], 0
        for gs in group_sizes:

            q_slice = q_value[cursor : cursor + gs]
            t_slice = target[cursor : cursor + gs]

            """
            TD error of nested events with different group sizes
            -
            Note on the special loss calculation:.mean().pow(2) rather than typical .pow(2).mean(): we take the mean over the group slice first and then square, rather than the more typical approach of squaring individual TD errors and then averaging. This is because in our system-level calculation, rewards of all nested events are averaged together to form the system-level reward, and accordingly we want to compute the TD error on that average reward rather than on individual event rewards. This way, the loss reflects the error in predicting the average return for the entire group of nested events, which is more aligned with our overall objective.
            Doesn't impact gs=1 -> .pow(2).mean()
            """
            losses.append((q_slice - t_slice).mean().pow(2)) # special loss 

            q_s.append(q_slice.mean())
            t_s.append(t_slice.mean())
            cursor += gs

        loss = torch.stack(losses).mean()

        q_s = torch.stack(q_s).detach().cpu()
        t_s = torch.stack(t_s).detach().cpu()
        mape = mean_absolute_percentage_error(q_s.numpy(), t_s.numpy())

        q_s = q_s.mean()
        t_s = t_s.mean()

        # ----------------  optimize  --------------------------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip_maxnorm)
        optimizer.step()

        # ----------------  target sync & ckpt  ----------------
        if it % args.target_sync_every == 0:
            agent.update_target_network()
            print(f"[Iter {it}] target network synced")

        # -----------------  stats dict  --------------------------
        stats = {
            "loss": loss.item(),
            "q_value": q_s.item(),
            "target_value": t_s.item(),
            "mape": mape,
        }

        return stats