import torch
from framework.encoder import constraint_id
import random


class Random_Agent(object):

    def __init__(self, constraint_regularization=True, *args, **kwargs):

        super().__init__()
        self.constraint_regularization = constraint_regularization

        print(f"Random Agent is Selected -- With implemented constraint constraint = {self.constraint_regularization}!")

    def select(self, s_a, batch_list=None, batch=False, batch_max=None):
    
        n, d = s_a.shape

        raw_probs = torch.ones(n, device=s_a.device) / n  # uniform distribution

        if batch and n <= batch_max:
            idx = torch.arange(n)
            return idx.tolist(), s_a, raw_probs[idx.tolist()]  #list(range(n)), s_a

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
            return idx.tolist(), s_a[idx].view(-1, d), raw_probs[idx.tolist()]
        else:
            idx = torch.multinomial(probs, 1)

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


class Random_Agent_Vanilla(object):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Random Agent is Selected -- Without constraint constraint!")

    def select(self, s_a=None, batch_list=None, batch=False, batch_max=None):

        # Determine n
        if batch_list is not None:
            n = len(batch_list)
        elif s_a is not None:
            n = s_a.shape[0]
        else:
            raise ValueError("Either batch_list or s_a must be provided.")

        if n == 0:
            return [] if batch else None

        # ---------- single ----------
        if not batch:
            idx = random.randrange(n)
            return idx

        # ---------- batch ----------
        if batch_max is not None and batch_max < n:
            return random.sample(range(n), batch_max)

        return list(range(n))

