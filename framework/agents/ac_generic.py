
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Union

from framework.encoder import constraint_id, d_s_a, d_s
from framework.agent import load_chpt_mixed

Number = Union[int, float]


class AC_Agent(nn.Module):
  
    """
    AC agent (tailored).
      - twin Q networks (q_net1, q_net2)  (clipped double Q)
      - policy network (policy_net) for the actor-critic variants (ppo/sac/...)

    action selection:
      - can select by policy logits (use_policy=True) or by min(Q1,Q2) 
      - supports explore_mode (softmax temp sampling)
      - supports constraint hard inductive bias 

    notes:
      - target network is typically created/managed in the algorithm wrapper (like in PPO/SAC/CQL).
    """

    def __init__(
        self,
        agent_name='ac',
        input_dim=d_s_a,
        state_dim=d_s,
        hidden_dim=1024,
        dropout=0.1,
        precision=torch.float32,
        checkpoint_path=None,
        device="cpu",
        constraint_regularization=True,
        explore_mode=False,
        build_value=False,
        *args,
        **kwargs,
    ): 
        super().__init__()

        self.agent_name = agent_name
        self.precision = precision
        self.device = device
        self.constraint_regularization = constraint_regularization

        # dims
        self.input_dim = input_dim
        self.state_dim = state_dim

        # twin Q networks (clipped double Q)
        self.q_net1 = self._build_network(self.input_dim, hidden_dim, dropout, out_dim=1)
        self.q_net2 = self._build_network(self.input_dim, hidden_dim, dropout, out_dim=1)

        # policy network (for actor-critic)
        self.policy_net = self._build_network(self.input_dim, hidden_dim, dropout, out_dim=1)

        # value head (optional, for PPO-style V-based variants)
        if build_value:
            self.v_net = self._build_network(self.state_dim, hidden_dim, dropout, out_dim=1)

        # explore flags
        self.explore_mode = explore_mode
        self.explore_value = 1.0  # temperature; 1.0 means no explore smoothing
    
        # load at init
        if checkpoint_path is not None:
            self.load_chpt(checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")

        self.eval()
        self.to(device=self.device)

        print(f"{self.agent_name} Agent is initialized -- Twin Q + Policy")
        print("This agent version assumes inputs scaled and pre-processed, therefore not clamping during forward pass!")
        print("Selection can be based on policy logits or on min(Q1,Q2) if requested.")


    def _build_network(self, input_dim, hidden_dim, dropout, out_dim=1):
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, dtype=self.precision),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, out_dim, dtype=self.precision),
        )


    def forward(self, x, which="pi"):

        """
        Caution: assumes inputs scaled and pre-processed therefore not clamping here.

        which:
          - 'q1', 'q2' : individual Q
          - 'q_min'    : min(Q1,Q2)
          - 'pi'       : policy logits
          - 'v'        : value head (requires build_value=True)
        """

        if which == "q1":
            return self.q_net1(x).squeeze(-1)

        if which == "q2":
            return self.q_net2(x).squeeze(-1)

        if which == "q_min":
            q1 = self.forward(x, which="q1")
            q2 = self.forward(x, which="q2")
            return torch.min(q1, q2)

        if which == "pi":
            return self.policy_net(x).squeeze(-1)
        
        if which == "v":
            if not hasattr(self, "v_net"):
                raise ValueError("Value head not built. Initialize agent with build_value=True to use value head.")
            return self.v_net(x).squeeze(-1)

        raise ValueError(f"Unknown head '{which}'. Use one of: q1, q2, q_min, pi, v (requires build_value=True).")


    # ---- convenience forwards ----
    def forward_qmin(self, s_a):

        """ min(Q1,Q2) on (s,a): expects tensor of shape (n, input_dim) """

        return self.forward(s_a, which="q_min")


    def forward_policy(self, s_a):

        """ policy logits/score for each candidate: expects tensor of shape (n, input_dim) """

        return self.forward(s_a, which="pi")


    # ---- selection ----
    def select(self, s_a, batch_list=None, batch=False, batch_max=None, use_policy=True):

        """
        Select from candidate set s_a: shape (n, input_dim).

        - use_policy=True  : select according to policy logits (actor-critic CQL inference)
        - use_policy=False : select according to min(Q1,Q2) values (Q-learning style inference)

        Behavior:
          - explore_mode=False -> greedy (argmax over probs)
          - explore_mode=True  -> sample from probs
        """

        n, d = s_a.shape

        # compute logits/scores
        if use_policy:
            scores = self.forward_policy(s_a)          # (n,)
        else:
            scores = self.forward(s_a, which="q_min")  # (n,)

        # temperature (smoothing)
        T = self.explore_value if self.explore_mode else 1.0
        T = max(float(T), 1e-6)

        logits = (scores - scores.max()) / T
        raw_probs = torch.softmax(logits, dim=0)

        # early return
        if batch and (batch_max is not None) and n <= batch_max:
            idx = torch.arange(n, device=s_a.device)
            return idx.tolist(), s_a, raw_probs[idx.tolist()]

        # constraint regularization
        probs = self.constraint_regularization(raw_probs, s_a)

        # selection
        if self.explore_mode:  # stochastic
            if batch:
                if batch_max is None:
                    raise ValueError("batch_max must be specified when batch=True.")
                idx = torch.multinomial(probs, batch_max, replacement=False)
                return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
            else:
                idx = torch.multinomial(probs, 1)
                return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]

        else:  # greedy
            if batch:
                if batch_max is None:
                    raise ValueError("batch_max must be specified when batch=True.")
                _, idx = torch.topk(probs, batch_max, largest=True, sorted=False)
                return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
            else:
                idx = torch.argmax(probs)
                return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]


    def select_stochastic(self, s_a, batch=False, batch_max=None, use_policy=False):

        """
        Stochastic selection (always samples) from:
          - policy logits if use_policy=True
          - min(Q1,Q2) if use_policy=False
        """
        
        n, d = s_a.shape

        if batch and (batch_max is not None) and n <= batch_max:
            idx = torch.arange(n, device=s_a.device)
            return idx.tolist(), s_a

        if use_policy:
            scores = self.forward_policy(s_a)
        else:
            scores = self.forward(s_a, which="q_min")

        raw_probs = F.softmax(scores, dim=0)

        if self.constraint_regularization:
            # same comment as in select(): enable if constraint_id exists
            probs = raw_probs
        else:
            probs = raw_probs

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), s_a[idx].view(-1, d)
        else:
            idx = torch.multinomial(probs, 1)
            return idx.item(), s_a[idx].view(-1, d)


    # ---- constraint helpers  ----
    def constraint_regularization(self, raw_probs, s_a):
        
        # constraints (constraint hard bias) 
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
        
        return probs


    def _regulate_constraint_strict(self, constraint):
        mask = (constraint >= 1.0).float()
        regulated_probs = mask / (mask.sum() + 1e-12)
        return regulated_probs


    def _regulate_constraint_push(self, probs, constraint):
        push = 1.0 + (constraint >= 2 / 3).float()
        new_probs = probs + push
        regulated_probs = new_probs / (new_probs.sum() + 1e-12)
        return regulated_probs


    def select_from_logit(self, logits, batch=False, batch_max=None):
        """
        Selects actions based on logits and does not apply constraint regularization.
        logits: shape [n, 1] or [n]
        """
        n = logits.shape[0]

        if batch and (batch_max is not None) and n <= batch_max:
            idx = torch.arange(n, device=logits.device)
            return idx.tolist(), logits, probs[idx.tolist()]

        probs = F.softmax(logits.squeeze(), dim=0)

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), logits[idx].view(-1, 1), probs[idx.tolist()]
        else:
            idx = torch.multinomial(probs, 1)
            return idx.item(), logits[idx].view(-1, 1), probs[idx.tolist()]


    # ---- policy distribution helper (useful for CQL actor-critic) ----
    def log_probs_from_logits(self, logits_1d):

        """
        given per-candidate logits (shape [n]), return:
          log_probs [n], probs [n]
        """

        logp = F.log_softmax(logits_1d.view(-1), dim=0)

        return logp, logp.exp()


    # ---- checkpointing  ----
    def load_chpt(self, checkpoint_path, config=None):
        load_chpt_mixed(self, checkpoint_path, config=config, verbose=False)


    def load_chpt_no_mix(self, checkpoint_path, config=None):

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict["model_state_dict"])

        # flag not resync
        if config and getattr(config, "policy_sync", False):
            config.policy_sync = False

        # update explore value if in explore mode
        if self.explore_mode:
            temp = state_dict.get("future_explore_value", None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)


    def load_chpt_wo_q_head(self, checkpoint_path):

        """
        Load a checkpoint but discard the pretrained Q-heads (both Q1 and Q2).
        Hidden layers are reused, final Linear layers are freshly initialized.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model_state_dict"]

        # load everything first
        self.load_state_dict(model_state)

        # reinitialize both Q heads
        self.q_net1[-1].reset_parameters()
        self.q_net2[-1].reset_parameters()

        # restore explore value if needed
        if self.explore_mode:
            temp = checkpoint.get("future_explore_value", None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)


    def reset_q_head(self):

        """Reinitialize both Q-heads."""

        self.q_net1[-1].reset_parameters()
        self.q_net2[-1].reset_parameters()


    def freeze_q_backbone(self):

        """
        Freeze all layers except the Q-heads (for both Q1 and Q2).
        """

        for param in self.q_net1.parameters():
            param.requires_grad = False
        for param in self.q_net2.parameters():
            param.requires_grad = False

        for param in self.q_net1[-1].parameters():
            param.requires_grad = True
        for param in self.q_net2[-1].parameters():
            param.requires_grad = True


    def unfreeze_q_backbone(self):

        """Unfreeze all layers including both Q-heads."""

        for param in self.q_net1.parameters():
            param.requires_grad = True
        for param in self.q_net2.parameters():
            param.requires_grad = True


    def reset_policy_head(self):

        """Reinitialize policy head."""
        self.policy_net[-1].reset_parameters()


    def freeze_policy_backbone(self):

        """Freeze policy backbone and keep last layer trainable."""
        for param in self.policy_net.parameters():
            param.requires_grad = False
        for param in self.policy_net[-1].parameters():
            param.requires_grad = True


    def unfreeze_policy_backbone(self):

        """Unfreeze policy network."""
        for param in self.policy_net.parameters():
            param.requires_grad = True


    def save_chpt(
        self,
        checkpoint_path,
        loss=None,
        iteration=None,
        optimizer_state_dict=None,
        future_explore_value=None,
        extra=None,
    ):
        torch.save(
            {
                "algorithm": f"{self.agent_name}",
                "iteration": iteration,
                "loss": loss,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer_state_dict,
                "future_explore_value": future_explore_value,
                "extra": extra,
            },
            checkpoint_path,
        )


    # ---- target-network helper (in case not created by algorithm directly)  ----
    def make_target_copy(self):

        tgt = copy.deepcopy(self)
        tgt.eval()
        for p in tgt.parameters():
            p.requires_grad_(False)
        return tgt


