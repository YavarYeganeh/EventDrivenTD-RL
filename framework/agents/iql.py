import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.agent import load_chpt_mixed
from framework.encoder import constraint_id, d_s_a, d_s


class IQL_Agent(nn.Module):
    
    """
        selects actions based on softmax sampling of scores.
        without skipping action!
        with checkpoint loading if path is provided.
        constraint regularization is implemented as a hard inductive bias

        iQL:
          - twin Q networks (q_net1, q_net2)
          - one V network (v_net)
          - one policy network (policy_net) used for action selection / AWR extraction

        Here there is no target network.
        this uses a stochastic policy for action selection (when explore_mode=True).
    """

    def __init__(
        self,
        input_dim=d_s_a,
        state_dim=d_s,
        hidden_dim=512,
        dropout=0.1,
        precision=torch.float32,
        checkpoint_path=None,
        device='cpu',
        constraint_regularization=True,
        explore_mode=False,
        *args,
        **kwargs
    ):

        super().__init__()

        self.precision = precision
        self.device = device
        self.constraint_regularization = constraint_regularization

        # dims
        self.input_dim = input_dim
        self.state_dim = state_dim 

        # twin Q networks (clipped double Q)
        self.q_net1 = self._build_network(self.input_dim, hidden_dim, dropout, out_dim=1)
        self.q_net2 = self._build_network(self.input_dim, hidden_dim, dropout, out_dim=1)

        # V network
        self.v_net = self._build_network(self.state_dim, hidden_dim, dropout, out_dim=1)

        # policy network (produces logits/score per candidate action)
        self.policy_net = self._build_network(self.input_dim, hidden_dim, dropout, out_dim=1)

        # explore flags
        self.explore_mode = explore_mode
        self.explore_value = 1.0  # temperature; 1.0 means no explore smoothing

        if checkpoint_path is not None:
            self.load_chpt(checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")

        self.eval()                      # eval mode by default
        self.to(device=self.device)      # move to device

        print("IQL Agent is initialized -- Twin Q + V + Policy, without skipping action!")
        print("This agent version assumes inputs scaled and pre-processed, therefore not clamping during forward pass!")
        print("Selection can be based on policy logits (default) or on min(Q1,Q2) if requested.")


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


    def forward(self, x, which='q1'):

        """
        Caution: This version assumes inputs scaled and pre-processed therefore not clamping here.

        which:
          - 'q1', 'q2' : individual Q
          - 'q_min'    : min(Q1,Q2)
          - 'v'        : V(s) (expects state input with state_dim)
          - 'pi'       : policy logits (expects state-action input with input_dim)
        """

        if which == 'q1':
            return self.q_net1(x).squeeze(-1)

        if which == 'q2':
            return self.q_net2(x).squeeze(-1)

        if which == 'q_min':
            q1 = self.forward(x, which='q1')
            q2 = self.forward(x, which='q2')
            return torch.min(q1, q2)

        if which == 'v':
            return self.v_net(x).squeeze(-1)

        if which == 'pi':
            return self.policy_net(x).squeeze(-1)

        raise ValueError(f"Unknown head '{which}'. Use one of: q1, q2, q_min, v, pi.")


    def forward_v(self, s):

        """ V(s): expects state tensor of shape (b, state_dim) """
        return self.forward(s, which='v')


    def forward_policy(self, s_a):

        """ policy logits/score for each candidate: expects tensor of shape (n, input_dim) """
        return self.forward(s_a, which='pi')


    def select(self, s_a, batch_list=None, batch=False, batch_max=None, use_policy=True):

        """
        Select from candidate set s_a: shape (n, d=input_dim).
        - use_policy=True  : select according to policy logits (recommended from the paper for IQL inference)
        - use_policy=False : select according to min(Q1,Q2) values (critic-based)
        """

        n, d = s_a.shape

        # compute logits/scores
        if use_policy:
            scores = self.forward_policy(s_a)         # shape: (n,)
        else:
            scores = self.forward(s_a, which='q_min') # shape: (n,)

        # temperature (smoothing)
        T = self.explore_value if self.explore_mode else 1.0
        T = max(float(T), 1e-6)

        logits = (scores - scores.max()) / T
        raw_probs = torch.softmax(logits, dim=0)

        # early return
        if batch and n <= batch_max:
            idx = torch.arange(n, device=s_a.device)
            return idx.tolist(), s_a, raw_probs[idx.tolist()]

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

        # selection
        if self.explore_mode:  # stochastic
            if batch:
                idx = torch.multinomial(probs, batch_max, replacement=False)
                return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
            else:
                idx = torch.multinomial(probs, 1)
                return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]

        else:  # greedy
            if batch:
                _, idx = torch.topk(probs, batch_max, largest=True, sorted=False)
                return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
            else:
                idx = torch.argmax(probs)
                return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]


    def select_stochastic(self, s_a, batch=False, batch_max=None, use_policy=True):

        n, d = s_a.shape

        if batch and n <= batch_max:
            idx = torch.arange(n, device=s_a.device)
            return idx.tolist(), s_a

        if use_policy:
            scores = self.forward_policy(s_a)
        else:
            scores = self.forward(s_a, which='q_min')

        raw_probs = F.softmax(scores, dim=0)

        # Apply a hard inductive bias to the probabilities for constraint (non-differentiable selection)
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

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), s_a[idx].view(-1, d)
        else:
            idx = torch.multinomial(probs, 1)

        return idx.item(), s_a[idx].view(-1, d)


    def _regulate_constraint_strict(self, constraint):

        mask = (constraint >= 1.0).float()
        regulated_probs = mask / (mask.sum())
        return regulated_probs


    def _regulate_constraint_push(self, probs, constraint):

        push = 1.0 + (constraint >= 2/3).float()
        new_probs = probs + push
        regulated_probs = new_probs / new_probs.sum()
        return regulated_probs


    def select_from_logit(self, logits, batch=False, batch_max=None):

        """
        Selects actions based on logits and does not apply constraint regularization.
        logits: shape [n, 1] or [n]
        """

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
        else:
            idx = torch.multinomial(probs, 1)

        return idx.item(), logits[idx].view(-1, 1)


    # ---- checkpointing / head management ----
    def load_chpt(self, checkpoint_path, config=None):
        load_chpt_mixed(self, checkpoint_path, config=config, verbose=False)


    def load_chpt_no_mix(self, checkpoint_path, config=None):

        # loading pretrained agent weights from checkpoint.
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict['model_state_dict'])

        # flag not resync
        if config and getattr(config, "policy_sync", False):
            config.policy_sync = False

        # setting/updating the explore value if in the explore mode
        if self.explore_mode:
            temp = state_dict.get('future_explore_value', None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)


    def load_chpt_wo_q_head(self, checkpoint_path):

        """
        Load a checkpoint but discard the pretrained Q-heads (both Q1 and Q2).
        Hidden layers are reused, final Linear layers are freshly initialized.
        """

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state = checkpoint['model_state_dict']

        # load everything first
        self.load_state_dict(model_state)

        # reinitialize both Q heads
        self.q_net1[-1].reset_parameters()
        self.q_net2[-1].reset_parameters()

        # restore explore value if needed
        if self.explore_mode:
            temp = checkpoint.get('future_explore_value', None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)


    def reset_q_head(self):

        """
        Reinitialize both Q-heads.
        Hidden layers are reused, final Linear layers are freshly initialized.
        """

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

        """
        Unfreeze all layers including both Q-heads.
        """

        for param in self.q_net1.parameters():
            param.requires_grad = True
        for param in self.q_net2.parameters():
            param.requires_grad = True


    def save_chpt(self, checkpoint_path, loss=None, iteration=None, optimizer_state_dict=None, future_explore_value=None):

        # saving agent weights to checkpoint.
        torch.save({
            'algorithm': 'iql',
            'iteration': iteration,
            'loss': loss,
            'model_state_dict': self.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            'future_explore_value': future_explore_value,  # set here for ease of train-sim communication
        }, checkpoint_path)

