import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

Number = Union[int, float]

from framework.encoder import constraint_id, d_s_a, d_s
from framework.agent import load_chpt_mixed


class PPO_V_Agent(nn.Module):

    """
    PPO agent (V-based) for candidate-set discrete actions (s_a rows are candidates).
    Similar helper/constraint methods to IQL_Agent.

    - policy network on (s,a) candidates -> logits
    - value network on s only -> V(s)

    forward(which):
      - 'pi' : policy logits on (s,a)
      - 'v'  : V(s) on s

    Selection:
      - based on policy logits
      - explore_mode toggles stochastic sampling vs greedy
      - constraint regularization same as IQL (needs constraint_id)
    """

    def __init__(

        self,
        input_dim=d_s_a,
        state_dim=d_s,
        hidden_dim=512,
        dropout=0.1,
        precision=torch.float32,
        checkpoint_path=None,
        device="cpu",
        constraint_regularization=True,
        explore_mode=False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.precision = precision
        self.device = device
        self.constraint_regularization = constraint_regularization

        self.input_dim = input_dim
        self.state_dim = state_dim

        self.policy_net = self._build_network(self.input_dim, hidden_dim, dropout, out_dim=1)
        self.v_net = self._build_network(self.state_dim, hidden_dim, dropout, out_dim=1)

        self.explore_mode = explore_mode
        self.explore_value = 1.0

        if checkpoint_path is not None:
            self.load_chpt(checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")

        self.eval()
        self.to(device=self.device)

        print("PPO Agent initialized -- V + Policy.")

    def _build_network(self, input_dim, hidden_dim, dropout, out_dim=1):

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 // 2, dtype=self.precision),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 // 2, out_dim, dtype=self.precision),
        )

    def forward(self, x, which="pi"):

        if which == "pi":
            return self.policy_net(x).squeeze(-1)

        if which == "v":
            return self.v_net(x).squeeze(-1)
        raise ValueError(f"Unknown head '{which}'. Use one of: pi, v.")

    def forward_v(self, s):

        return self.forward(s, which="v")

    def forward_v_from_sa(self, s_a):

        s = s_a[:, :self.state_dim]
        return self.forward_v(s)

    def forward_policy(self, s_a):

        return self.forward(s_a, which="pi")

    # -------- constraint helpers (same as IQL) --------
    def _regulate_constraint_strict(self, constraint):

        mask = (constraint >= 1.0).float()
        regulated_probs = mask / (mask.sum() + 1e-12)
        return regulated_probs

    def _regulate_constraint_push(self, probs, constraint):

        push = 1.0 + (constraint >= 2/3).float()
        new_probs = probs + push
        regulated_probs = new_probs / (new_probs.sum() + 1e-12)
        return regulated_probs

    # -------- selection helpers (same as IQL) --------
    def select(self, s_a, batch_list=None, batch=False, batch_max=None):

        """
        Select using policy logits.
        """
        n, d = s_a.shape

        scores = self.forward_policy(s_a)

        T = self.explore_value if self.explore_mode else 1.0
        T = max(float(T), 1e-6)

        logits = (scores - scores.max()) / T
        raw_probs = torch.softmax(logits, dim=0)

        if batch and n <= batch_max:
            idx = torch.arange(n, device=s_a.device)
            return idx.tolist(), s_a, raw_probs[idx.tolist()]

        if self.constraint_regularization:
            constraint = s_a[:, constraint_id]
            max_constraint = torch.max(constraint)
            constraint_strict = (max_constraint >= 1.0)
            constraint_push = (max_constraint > 2/3)
            if constraint_strict:
                probs = self._regulate_constraint_strict(constraint)
            elif constraint_push:
                probs = self._regulate_constraint_push(raw_probs, constraint)
            else:
                probs = raw_probs
                
        else:
            probs = raw_probs

        if self.explore_mode:
            if batch:
                idx = torch.multinomial(probs, batch_max, replacement=False)
                return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
            idx = torch.multinomial(probs, 1)
            return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]

        if batch:
            _, idx = torch.topk(probs, batch_max, largest=True, sorted=False)
            return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
        idx = torch.argmax(probs)
        return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]


    def select_stochastic(self, s_a, batch=False, batch_max=None):

        n, d = s_a.shape

        if batch and n <= batch_max:
            idx = torch.arange(n, device=s_a.device)
            return idx.tolist(), s_a

        scores = self.forward_policy(s_a)
        raw_probs = F.softmax(scores, dim=0)

        if self.constraint_regularization:
            probs = raw_probs
        else:
            probs = raw_probs

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), s_a[idx].view(-1, d)
        idx = torch.multinomial(probs, 1)
        return idx.item(), s_a[idx].view(-1, d)

    def select_from_logit(self, logits, batch=False, batch_max=None):

        n = logits.shape[0]

        if batch and n <= batch_max:
            idx = torch.arange(n, device=logits.device)
            return idx.tolist(), logits

        probs = F.softmax(logits.squeeze(), dim=0)

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), logits[idx].view(-1, 1)
        idx = torch.multinomial(probs, 1)
        return idx.item(), logits[idx].view(-1, 1)


    # ---- checkpointing / head management ----
    def load_chpt(self, checkpoint_path, config=None):

        load_chpt_mixed(self, checkpoint_path, config=config, verbose=False)


    def load_chpt_no_mix(self, checkpoint_path, config=None):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict["model_state_dict"])

        if config and getattr(config, "policy_sync", False):
            config.policy_sync = False

        if self.explore_mode:
            temp = state_dict.get("future_explore_value", None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)


    def load_chpt_wo_policy_head(self, checkpoint_path):
        """
        Load checkpoint but discard policy head.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model_state_dict"]
        self.load_state_dict(model_state)

        self.policy_net[-1].reset_parameters()

        if self.explore_mode:
            temp = checkpoint.get("future_explore_value", None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)


    def reset_policy_head(self):
        self.policy_net[-1].reset_parameters()


    def freeze_policy_backbone(self):
        for param in self.policy_net.parameters():
            param.requires_grad = False
        for param in self.policy_net[-1].parameters():
            param.requires_grad = True

    def unfreeze_policy_backbone(self):
        for param in self.policy_net.parameters():
            param.requires_grad = True

    def reset_v_head(self):
        self.v_net[-1].reset_parameters()

    def freeze_v_backbone(self):
        for param in self.v_net.parameters():
            param.requires_grad = False
        for param in self.v_net[-1].parameters():
            param.requires_grad = True

    def unfreeze_v_backbone(self):
        for param in self.v_net.parameters():
            param.requires_grad = True

    def save_chpt(self, checkpoint_path, loss=None, iteration=None, optimizer_state_dict=None, future_explore_value=None):
        torch.save({
            "algorithm": "ppo_v",
            "iteration": iteration,
            "loss": loss,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "future_explore_value": future_explore_value,
        }, checkpoint_path)