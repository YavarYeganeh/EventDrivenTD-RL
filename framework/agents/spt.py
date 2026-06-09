import torch
import torch.nn as nn
from framework.encoder import constraint_id, pt_id


class SPT_Agent(nn.Module):

    def __init__(self, constraint_regularization=True, *args, **kwargs):
    
        super().__init__()
        self.constraint_regularization = constraint_regularization

        print(f"SPT (can be stochastic in practice) Agent is Selected -- With implemented constraint constraint = {self.constraint_regularization}!")


    def spt(self, x):

        spt = x[:, pt_id]
        spt = torch.softmax(-spt, dim=0)

        return spt # shape: (n,)


    def select(self, s_a, batch_list=None, batch=False, batch_max=None, target=False):
    
        n, d = s_a.shape

        # scores
        raw_probs = self.spt(s_a)  # shape: (n,)  

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

        if batch: # greedy based on Qs
            _, idx = torch.topk(probs, batch_max, largest=True, sorted=False)
            return idx.tolist(), s_a[idx].view(-1, d), raw_probs[idx.tolist()]
        else:
            idx = torch.argmax(probs)
            return idx.item(), s_a[idx].view(-1, d), raw_probs[idx.item()]


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
        
