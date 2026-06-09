
import torch
import torch.nn.functional as F
import csv
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from framework.agent import instantiate_agent
from framework.encoder import d_s_a, d_s


# exponential moving average update for target networks
@torch.no_grad()
def soft_update(source: torch.nn.Module, target: torch.nn.Module, alpha: float):
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - alpha).add_(alpha * p.data)


class Algorithm:

    """
    Wraps:
      - agent + optimizer instantiation
      - one training step
      - stat helpers
    """

    def __init__(self, args, device, dtype, stats_path):

        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats_path = stats_path

        self.agent = instantiate_agent(
            agent_type = "iql",
            input_dim=d_s_a,
            state_dim=d_s,
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            precision=dtype,
            device=device,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.lr)

        # IQL hyperparams (defaults if not in args)
        self.tau = float(getattr(args, "iql_tau", 0.95))                    # expectile 
        self.beta = float(getattr(args, "iql_beta", 10.0))                  # inverse temperature
        self.exp_adv_max = float(getattr(args, "iql_exp_adv_max", 100.0))  # clip exp(beta*adv)
        self.alpha = float(getattr(args, "iql_alpha", 0.005))                 # target network update rate

        # loss weights (defaults to 1.0 if not in args)
        self.q_coef = float(getattr(args, "q_coef", 1.0))
        self.v_coef = float(getattr(args, "v_coef", 1.0))
        self.pi_coef = float(getattr(args, "pi_coef", 1.0))

        print(f"Initialized IQL with tau={self.tau}, beta={self.beta}, exp_adv_max={self.exp_adv_max}, q_coef={self.q_coef}, v_coef={self.v_coef}, pi_coef={self.pi_coef}")

        self.init_stats(stats_path)


    def init_stats(self, stats_path):

        # initialize CSV header
        with open(stats_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'iteration',
                'avg_loss', 'avg_q_value', 'avg_target_value', 'avg_mape', 'avg_reward',
                'avg_r_pros', 'avg_r_idle', 'avg_r_constraint', 'avg_r_throughput', 'avg_r_wl', 'avg_r_end',
                'avg_loss_q', 'avg_loss_v', 'avg_loss_pi'
            ])

        """ Statistics History """
        self.valid_losses = []
        self.q_history = []
        self.t_history = []
        self.mape_history = []
        self.reward_history = []
        self.reward_elements_history = []

        # iql-specific breakdown
        self.loss_q_history = []
        self.loss_v_history = []
        self.loss_pi_history = []


    def record_stats(self, stats, rewards, reward_elements, it):

        # ----------------  Recording history  -------------------------
        self.valid_losses.append(stats['loss'])
        self.q_history.append(stats['q_value'])
        self.t_history.append(stats['target_value'])
        self.mape_history.append(stats['mape'])
        self.reward_history.append(rewards.mean().item())
        self.reward_elements_history.append(reward_elements.mean(dim=0))

        # iql breakdown (safe if present)
        self.loss_q_history.append(stats.get('loss_q', np.nan))
        self.loss_v_history.append(stats.get('loss_v', np.nan))
        self.loss_pi_history.append(stats.get('loss_pi', np.nan))

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

            avg_loss_q = np.mean(self.loss_q_history)
            avg_loss_v = np.mean(self.loss_v_history)
            avg_loss_pi = np.mean(self.loss_pi_history)

            # reset histories
            self.valid_losses = []
            self.q_history = []
            self.t_history = []
            self.mape_history = []
            self.reward_history = []
            self.reward_elements_history = []

            self.loss_q_history = []
            self.loss_v_history = []
            self.loss_pi_history = []

            # append to CSV
            with open(self.stats_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    it, avg_loss, avg_q, avg_t, avg_mape, avg_reward,
                    avg_r_pros, avg_r_idle, avg_r_constraint,
                    avg_r_throughput, avg_r_wl, avg_r_end,
                    avg_loss_q, avg_loss_v, avg_loss_pi
                ])

            print(f"[Iter {it}] avg loss over last {self.args.valid_interval} iters: {avg_loss:.4f}")


    @staticmethod
    def _expectile_loss(u, tau: float):
        # L^tau_2(u) = |tau - 1(u < 0)| * u^2
        w = torch.where(
            u >= 0,
            torch.tensor(tau, device=u.device, dtype=u.dtype),
            torch.tensor(1.0 - tau, device=u.device, dtype=u.dtype),
        )
        return w * (u ** 2)


    def train_step(self, data, it):

        args = self.args
        device = self.device
        agent = self.agent
        optimizer = self.optimizer

        # ----------------  data unpacking -------------------
        s_a_0, s_a_1, rewards, group_sizes, s0_sizes, s1_sizes, gammas = data

        # ----------------  forward passes  -------------------
        agent.train()

        # q value (s,a)
        q1_logits = agent(s_a_0, which="q1")  # (n0,)

        # target q value (s,a) for CQL penalty
        with torch.no_grad():
            q2_logits = agent(s_a_0, which="q2")  # (n0,)

        # V on s and s'
        s0 = s_a_0[:, :d_s]
        s1 = s_a_1[:, :d_s]
        v0_logits = agent(s0, which="v")      # (n0,)
        v1_logits = agent(s1, which="v")      # (n1,)

        # policy logits on (s,a) candidates (used for AWR extraction)
        pi_logits = agent(s_a_0, which="pi")  # (n0,)

        # ----------------  per-machine aggregation -----------
        q1_list, q2_list, t_list = [], [], []
        vloss_list, piloss_list = [], []

        start0 = start1 = 0
        for sz0, sz1 in zip(s0_sizes, s1_sizes):
            end0, end1 = start0 + sz0, start1 + sz1

            # mean Q over for the case it's a batch machine
            q1_list.append(q1_logits[start0:end0].mean())
            # q2_list.append(q2_logits[start0:end0].mean())

            # V(s') as the bootstrapped target value (state repeats across actions -> mean ok)
            t_list.append(v1_logits[start1:end1].mean())

            # ---------- V loss: expectile regression ----------
            # LV = E[ L^tau_2( Qmin(s,a) - V(s) ) ]
            u = q2_logits[start0:end0].detach() - v0_logits[start0:end0]
            vloss_list.append(self._expectile_loss(u, self.tau).mean())

            # ---------- Policy loss: AWR extraction ----------
            # Lpi = E[ exp(beta*(Qmin - V)) * log pi(a|s) ]
            # minimize negative weighted log-likelihood over slice actions
            slice_pi = pi_logits[start0:end0].view(-1)
            log_probs = F.log_softmax(slice_pi, dim=0)

            adv = (q2_logits[start0:end0].detach() - v0_logits[start0:end0].detach())
            weights = torch.exp(self.beta * adv).clamp(max=self.exp_adv_max)

            piloss_list.append(-(weights * log_probs).mean())

            start0, start1 = end0, end1

        # stack lists → tensors [num_events, 1]
        q1_value = torch.stack(q1_list).unsqueeze(1).to(device)
        # q2_value = torch.stack(q2_list).unsqueeze(1).to(device)
        v1_value = torch.stack(t_list).unsqueeze(1).to(device)

        v_loss_m = torch.stack(vloss_list).unsqueeze(1).to(device)
        pi_loss_m = torch.stack(piloss_list).unsqueeze(1).to(device)

        # IQL TD target: r + gamma * V(s')
        target = rewards + gammas * v1_value.detach()

        # ----------------  per-seed loss  ---------------------
        q_losses, v_losses, pi_losses = [], [], []
        q_s, t_s, cursor = [], [], 0

        for gs in group_sizes:

            q1_slice = q1_value[cursor: cursor + gs]
            # q2_slice = q2_value[cursor: cursor + gs]
            t_slice = target[cursor: cursor + gs]

            """
            TD error of nested events with different group sizes
            -
            Note on the special loss calculation:.mean().pow(2) rather than typical .pow(2).mean(): we take the mean over the group slice first and then square, rather than the more typical approach of squaring individual TD errors and then averaging. This is because in our system-level calculation, rewards of all nested events are averaged together to form the system-level reward, and accordingly we want to compute the TD error on that average reward rather than on individual event rewards. This way, the loss reflects the error in predicting the average return for the entire group of nested events, which is more aligned with our overall objective.
            Doesn't impact gs=1 -> .pow(2).mean() 0r (q1_slice - t_slice).pow(2).mean(
            """
            q_losses.append(
                (q1_slice - t_slice).mean().pow(2) 
            )

            v_losses.append(v_loss_m[cursor: cursor + gs].mean())
            pi_losses.append(pi_loss_m[cursor: cursor + gs].mean())

            # q_s.append(0.5 * (q1_slice.mean() + q2_slice.mean()))
            q_s.append(q1_slice.mean())
            t_s.append(t_slice.mean())
            cursor += gs

        loss_q = torch.stack(q_losses).mean()
        loss_v = torch.stack(v_losses).mean()
        loss_pi = torch.stack(pi_losses).mean()

        loss = self.q_coef * loss_q + self.v_coef * loss_v + self.pi_coef * loss_pi

        q_s = torch.stack(q_s).detach().cpu()
        t_s = torch.stack(t_s).detach().cpu()
        mape = mean_absolute_percentage_error(q_s.numpy(), t_s.numpy())

        q_s = q_s.mean()
        t_s = t_s.mean()

        # ---------  target network update  ----------------------
        soft_update(agent.q_net1, agent.q_net2, self.alpha)

        # ----------------  optimize  --------------------------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip_maxnorm)
        optimizer.step()

        # -----------------  stats dict  --------------------------
        stats = {
            "loss": loss.item(),
            "q_value": q_s.item(),
            "target_value": t_s.item(),
            "mape": mape,
            "loss_q": loss_q.item(),
            "loss_v": loss_v.item(),
            "loss_pi": loss_pi.item(),
        }

        return stats














