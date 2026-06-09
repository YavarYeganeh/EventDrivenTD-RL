import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.encoder import constraint_id, d_s_a
from framework.agent import load_chpt_mixed


class DDQN_Agent(nn.Module):

    """
        Selects actions based on softmax sampling of scores.  
        Without skipping action!
        with checkpoint loading if path is provided.
        constraint is regularization is implemented as a hard inductive bias
        this uses a stochastic policy for action selection.
    """
    
    def __init__(self, input_dim=d_s_a, hidden_dim=1024, dropout=0.1, precision=torch.float32, checkpoint_path=None, device='cpu', constraint_regularization=True, explore_mode=False, *args, **kwargs):
   
        super().__init__()
        
        self.precision = precision
        self.device = device
        self.constraint_regularization = constraint_regularization
        
        # we need Q-network and target Q-network
        self.q_net = self._build_network(input_dim, hidden_dim, dropout)
        self.target_net = self._build_network(input_dim, hidden_dim, dropout)

        # initialize target network with same weights as we also need to update it once in a while
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.explore_mode = explore_mode # if true it explores stochastically with ration synced when loading checkpoints with self.explore_value as the ratio
        self.explore_value = 1.0  # default explore value which is no explore  

        if checkpoint_path is not None:
            self.load_chpt(checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")

        self.eval() # set to eval mode by default # only needs to be set to train mode during training and in that module only
        self.to(device=self.device) # move to device

        print("DDQN Agent is initialized -- Without skipping action and with Boltzmann-style if exploring!")
        print("This agent version assumes inputs scaled and pre-processed, therefore not clamping during forward pass!")


    def _build_network(self, input_dim, hidden_dim, dropout):
      
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4, dtype=self.precision),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1, dtype=self.precision)
        )


    def forward(self, x, target=False):

        """ Caution: This version assumes inputs scaled and pre-processed therefore not clamping here """
        # x = torch.clamp(x, 0., 1.).to(device=self.device, dtype=self.precision) ## for numerical stability

        net = self.target_net if target else self.q_net
        
        return net(x).squeeze(-1)  # shape: (n,)


    def select(self, s_a, batch_list=None, batch=False, batch_max=None, target=False):
    
        n, d = s_a.shape

        # scores
        scores = self.forward(s_a, target=target)  # shape: (n,)
        T = self.explore_value if self.explore_mode else 1.0 # 1.0 is no explore (smoothing)
        logits = (scores - scores.max()) / T
        raw_probs = torch.softmax(logits, dim=0)        

        # early return 
        if batch and n <= batch_max:
            idx = torch.arange(n, device=s_a.device)
            return idx.tolist(), s_a, raw_probs[idx.tolist()]

        # constraints
        # apply a hard inductive bias to the probabilities for constraint (this implementation is only selecting actions and non differentiable)
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
        if self.explore_mode: # stochastic (sampling)
            if batch:
                idx = torch.multinomial(probs, batch_max, replacement=False)
                return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
            else:
                idx = torch.multinomial(probs, 1)
                return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]
        else:

            if batch: # greedy based on Qs
                _, idx = torch.topk(probs, batch_max, largest=True, sorted=False)
                return idx.tolist(), s_a[idx].view(-1, d), probs[idx.tolist()]
            else:
                idx = torch.argmax(probs)
                return idx.item(), s_a[idx].view(-1, d), probs[idx.item()]


    def select_stochastic(self, s_a, batch=False, batch_max=None, target=False):

        n, d = s_a.shape

        if batch and n <= batch_max:
            idx = torch.arange(n)
            return idx.tolist(), s_a  #list(range(n)), s_a

        scores = self.forward(s_a, target=target)  # shape: (n,)
        raw_probs = F.softmax(scores, dim=0)

        # Apply a hard inductive bias to the probabilities for constraint (this implementation is only selecting actions and non differentiable)
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
        regulated_probs = mask / (mask.sum())  # re-normalize in case multiple fully constraintoded actions
        # print(f"Regulated probabilities (strict): {regulated_probs}")
        return regulated_probs

    def _regulate_constraint_push(self, probs, constraint):

        # push (boost) if constraint >= 2/3
        push = 1.0 + (constraint >= 2/3).float()
        new_probs = probs + push # ensure at least half chance for partially constraintoded actions (0.5 probability for single partially constraintoded action)
        regulated_probs = new_probs / new_probs.sum()  # re-normalize in case multiple entries are 1
        # print(f"Regulated probabilities (push): {regulated_probs}")
        return regulated_probs


    def select_from_logit(self, logits, batch=False, batch_max=None):

        """  
        This function selects actions based on logits and does not apply constraint regularization.        
        """

        # logits are [n, 1]
        n = logits.shape[0]

        if batch and n <= batch_max:
            idx = torch.arange(n)
            return idx.tolist(), logits  #list(range(n)), s_a

        probs = F.softmax(logits.squeeze(), dim=0)

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), logits[idx].view(-1, 1)
        else:
            idx = torch.multinomial(probs, 1)

        return idx.item(), logits[idx].view(-1, 1)


    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


    # ---- checkpointing / head management ----
    def load_chpt(self, checkpoint_path, config=None):
        load_chpt_mixed(self, checkpoint_path, config=config, verbose=False)


    def load_chpt_not_mix(self, checkpoint_path, config=None):

        # loading pretrained agent weights from checkpoint.
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict['model_state_dict'])
        self.update_target_network()
        # print(f"Checkpoint loaded from {checkpoint_path}")
        
        # flag not resync
        if config and config.policy_sync:
            config.policy_sync = False
        
        # setting/updating the explore value if in the explore mode
        if self.explore_mode:
            temp = state_dict.get('future_explore_value', None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)  # prevent divide-by-zero


    def load_chpt_wo_q_head(self, checkpoint_path):
        
        """
        Load a checkpoint but discard the pretrained Q-head.
        Hidden layers are reused, final Linear is freshly initialized.
        Caution: Target network is synced with resetted parameters.
        """

        # loading pretrained agent weights from checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state = checkpoint['model_state_dict']

        # loading pretrained agent weights from checkpoint # everything (including the old Q-head)
        self.load_state_dict(model_state)

        # reinitialize the Q-head so it's fresh & not with the scale of the previous reward
        q_head = self.q_net[-1] # q_net is an nn.Sequential, so [-1] is the last Linear (hidden_dim//4 -> 1)
        q_head.reset_parameters()

        # sync target_net from q_net so target_net also has a fresh head
        self.update_target_network()

        # restore explore value if needed
        if self.explore_mode:
            temp = checkpoint.get('future_explore_value', None)
            self.explore_value = 1.0 if temp is None else float(temp)
            self.explore_value = max(self.explore_value, 1e-6)  # prevent divide-by-zero

    
    def reset_q_head(self):
        
        """
        Reinitialize the Q-head.
        Hidden layers are reused, final Linear is freshly initialized.
        Target network remain intact.
        """

        # reinitialize the Q-head so it's fresh & not with the scale of the previous reward
        q_head = self.q_net[-1] # q_net is an nn.Sequential, so [-1] is the last Linear (hidden_dim//4 -> 1)
        q_head.reset_parameters()


    def freeze_q_backbone(self):
        
        """
        Freeze all layers except the Q-head.
        Except the last layer (i.e., q head).
        Target network remain intact.
        """

        for param in self.q_net.parameters():
            param.requires_grad = False

        q_head = self.q_net[-1] # q_net is an nn.Sequential, so [-1] is the last Linear (hidden_dim//4 -> 1)
        for param in q_head.parameters():
            param.requires_grad = True


    def unfreeze_q_backbone(self):
        
        """
        Unfreeze all layers including the Q-head.
        Target network remain intact.
        """

        for param in self.q_net.parameters():
            param.requires_grad = True


    def save_chpt(self, checkpoint_path, loss=None, iteration=None, optimizer_state_dict=None, future_explore_value=None):
        # saving agent weights to checkpoint.
        torch.save({
            "algorithm": "dql",
            'iteration': iteration,
            'loss': loss,
            'model_state_dict': self.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            'future_explore_value': future_explore_value, # set here for ease of train-sim communication
        }, checkpoint_path)
        # print(f"Checkpoint saved at {checkpoint_path}")


class DDQN_Agent_stochastic_policy_explore(nn.Module):

    """
        Selects actions based on softmax sampling of scores.  
        Without skipping action!
        with checkpoint loading if path is provided.
        constraint is regularization is implemented as a hard inductive bias
        this uses a stochastic policy for action selection.
    """
    
    def __init__(self, input_dim=d_s_a, hidden_dim=512, dropout=0.1, precision=torch.float32, checkpoint_path=None, device='cpu', constraint_regularization=True, *args, **kwargs):
   
        super().__init__()
        
        self.precision = precision
        self.device = device
        self.constraint_regularization = constraint_regularization
        
        # we need Q-network and target Q-network
        self.q_net = self._build_network(input_dim, hidden_dim, dropout)
        self.target_net = self._build_network(input_dim, hidden_dim, dropout)

        # initialize target network with same weights as we also need to update it once in a while
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target network in eval mode

        if checkpoint_path is not None:
            self.load_chpt(checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")

        self.eval() # set to eval mode by default # only needs to be set to train mode during training and in that module only
        self.to(device=self.device) # move to device

        print("DDQN Agent is initialized -- Without skipping action!")
        print("This agent version assumes inputs scaled and pre-processed, therefore not clamping during forward pass!")

    def _build_network(self, input_dim, hidden_dim, dropout):
      
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2, dtype=self.precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4, dtype=self.precision),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1, dtype=self.precision)
        )

    def forward(self, x, target=False):

        """ Caution: This version assumes inputs scaled and pre-processed therefore not clamping here """
        # x = torch.clamp(x, 0., 1.).to(device=self.device, dtype=self.precision) ## for numerical stability

        net = self.target_net if target else self.q_net
        
        return net(x).squeeze(-1)  # shape: (n,)

    def select(self, s_a, batch_list=None, batch=False, batch_max=None, target=False):
    
        n, d = s_a.shape

        if batch and n <= batch_max:
            idx = torch.arange(n)
            return idx.tolist(), s_a  #list(range(n)), s_a

        scores = self.forward(s_a, target=target)  # shape: (n,)
        raw_probs = F.softmax(scores, dim=0)

        # Apply a hard inductive bias to the probabilities for constraint (this implementation is only selecting actions and non differentiable)
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
        regulated_probs = mask / (mask.sum())  # re-normalize in case multiple fully constraintoded actions
        # print(f"Regulated probabilities (strict): {regulated_probs}")
        return regulated_probs

    def _regulate_constraint_push(self, probs, constraint):

        # push (boost) if constraint >= 2/3
        push = 1.0 + (constraint >= 2/3).float()
        new_probs = probs + push # ensure at least half chance for partially constraintoded actions (0.5 probability for single partially constraintoded action)
        regulated_probs = new_probs / new_probs.sum()  # re-normalize in case multiple entries are 1
        # print(f"Regulated probabilities (push): {regulated_probs}")
        return regulated_probs


    def select_from_logit(self, logits, batch=False, batch_max=None):

        """  
        This function selects actions based on logits and does not apply constraint regularization.        
        """

        # logits are [n, 1]
        n = logits.shape[0]

        if batch and n <= batch_max:
            idx = torch.arange(n)
            return idx.tolist(), logits  #list(range(n)), s_a

        probs = F.softmax(logits.squeeze(), dim=0)

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), logits[idx].view(-1, 1)
        else:
            idx = torch.multinomial(probs, 1)

        return idx.item(), logits[idx].view(-1, 1)

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


    def load_chpt(self, checkpoint_path, config=None):

        # loading pretrained agent weights from checkpoint.
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict['model_state_dict'])
        self.update_target_network()
        # print(f"Checkpoint loaded from {checkpoint_path}")
        
        # flag not resync
        if config and config.policy_sync:
            config.policy_sync = False


    def save_chpt(self, checkpoint_path, loss=None, iteration=None, optimizer_state_dict=None):
        # saving agent weights to checkpoint.
        torch.save({
            "algorithm": "dql",
            'iteration': iteration,
            'loss': loss,
            'model_state_dict': self.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
        }, checkpoint_path)
        # print(f"Checkpoint saved at {checkpoint_path}")


class Base_Agent(nn.Module):

    """
        Selects actions based on softmax sampling of scores.  
        Without skipping action!
    """

    def __init__(self, input_dim=d_s_a, hidden_dim=128, dropout=0.2, precision='float32', *args, **kwargs):
        
        super().__init__()
        
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=precision),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, dtype=precision),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, dtype=precision)
        )

        print("Base Agent is initialized -- Without skipping action!")

    def forward(self, x):
        return self.q_net(x).squeeze(-1)  # shape: (n,)

    def select(self, s_a, batch_list=None, batch=False, batch_max=None):

        n, d = s_a.shape

        if batch and n <= batch_max:
            return list(range(n)), s_a

        scores = self.forward(s_a)  # shape: (n,)
        probs = F.softmax(scores, dim=0)

        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be specified when batch=True.")
            idx = torch.multinomial(probs, batch_max, replacement=False)
            return idx.tolist(), s_a[idx].view(-1, d)
        else:
            idx = torch.multinomial(probs, 1)

        return idx.item(), s_a[idx].view(-1, d)

