import torch
from framework.encoder import constraint_id
import heapq
import random


class FIFO_Agent(object):
    
    def __init__(self, constraint_regularization=True, *args, **kwargs):

        super().__init__()
        self.constraint_regularization = constraint_regularization

        print(f"FIFO (stochastic in practice) Agent is Selected -- With implemented constraint constraint = {self.constraint_regularization}!")


    def select(self, s_a, batch_list, batch=False, batch_max=None):

        """
        FIFO-based selection using each batch_list's `get_last_move_out_time()`.

        Rules
        -----
        let n = batch_max if batch=True, otherwise n = 1.

        1) If there are >= n None times → sample n indices among the None entries.
        2) If there are <  n None times → take all None indices, then fill the
           rest from the smallest non-None times.
        3) If there are no None values → choose n smallest times.

        constraint constraint (same logic as Random_Agent)
        -------------------------------------------------
        max_constraint = max(constraint)

        - if max_constraint >= 1.0  (constraint_strict): only choose among constraint >= 1.0
        - elif max_constraint > 2/3 (constraint_push)  : prioritize constraint >= 2/3, then fill rest FIFO
        - else: standard FIFO

        Returns
        -------
        If batch=False:
            idx : int
        If batch=True:
            idxs : list[int]
        """
        _, d = s_a.shape
        n_items = len(batch_list)
        if n_items == 0:
            return [] if batch else None

        # ---------- number to return ----------
        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be provided when batch=True.")
            if batch_max <= 0:
                return []
            n = min(batch_max, n_items)
        else:
            n = 1

        # ---------- gather FIFO times ----------
        # each item must implement: get_last_move_out_time()
        times = [item.get_last_move_out_time() for item in batch_list]

        # ---------- constraint constraint ----------
        if self.constraint_regularization:

            constraint = s_a[:, constraint_id]

            max_constraint = torch.max(constraint)
            constraint_strict = (max_constraint >= 1.0)
            constraint_push = (max_constraint > 2/3)

            if constraint_strict:
                chosen = self._regulate_constraint_strict(times, constraint, n)

            elif constraint_push:
                chosen = self._regulate_constraint_push(times, constraint, n)

            else:
                chosen = self._select_fifo(times, n)

        else:
            chosen = self._select_fifo(times, n)

        # just in case data is used for training 
        probs = torch.ones(len(chosen), device=s_a.device) / len(chosen)

        # ---------- return ----------
        if batch:
            return chosen, s_a[chosen].view(-1, d), probs

        # single selection
        idx = chosen[0]
        return idx, s_a[idx].view(-1, d), probs


    def _select_fifo(self, times, n):

        # Split into None and non-None
        none_idxs = [i for i, t in enumerate(times) if t is None]
        non_none = [(t, i) for i, t in enumerate(times) if t is not None]

        # ---------- case 1: enough None entries ----------
        if len(none_idxs) >= n:
            chosen = random.sample(none_idxs, n)

        else:
            # Take all None indices
            chosen = list(none_idxs)
            need = n - len(chosen)

            # Fill with smallest FIFO-time values
            if need > 0 and non_none:
                smallest = heapq.nsmallest(need, non_none, key=lambda t: (t[0], t[1]))
                chosen.extend(i for _, i in smallest)

        return chosen


    def _regulate_constraint_strict(self, times, constraint, n):

        # strict: allow only fully constraintoded (constraint >= 1.0), then apply FIFO on that subset
        constraint_idxs = [i for i, c in enumerate(constraint) if c >= 1.0]

        # If for some reason empty, fall back to standard FIFO
        if len(constraint_idxs) == 0:
            return self._select_fifo(times, n)

        # FIFO on subset (preserve same FIFO rules)
        sub_times = [times[i] for i in constraint_idxs]
        sub_chosen = self._select_fifo(sub_times, min(n, len(constraint_idxs)))

        # Map back to original indices
        chosen = [constraint_idxs[i] for i in sub_chosen]
        return chosen


    def _regulate_constraint_push(self, times, constraint, n):

        # push: prioritize partially constraintoded (constraint >= 2/3), then fill remaining FIFO from others
        constraint_idxs = [i for i, c in enumerate(constraint) if c >= 2/3]
        other_idxs = [i for i, c in enumerate(constraint) if c < 2/3]

        chosen = []

        # First: FIFO within the pushed subset
        if len(constraint_idxs) > 0:
            sub_times = [times[i] for i in constraint_idxs]
            sub_chosen = self._select_fifo(sub_times, min(n, len(constraint_idxs)))
            chosen.extend([constraint_idxs[i] for i in sub_chosen])

        # Second: fill remaining with FIFO among the rest
        need = n - len(chosen)
        if need > 0 and len(other_idxs) > 0:
            sub_times = [times[i] for i in other_idxs]
            sub_chosen = self._select_fifo(sub_times, min(need, len(other_idxs)))
            chosen.extend([other_idxs[i] for i in sub_chosen])

        return chosen


class FIFO_Agent_Vanilla(object):
    
    def __init__(self, *args, **kwargs):

        super().__init__()
        print("FIFO Agent is Selected -- Without constraint constraint!")

    def select(self, s_a, batch_list, batch=False, batch_max=None):
        """
        FIFO-based selection using each item's `get_last_move_out_time()`.

        Rules
        -----
        Let n = batch_max if batch=True, otherwise n = 1.

        1) If there are >= n None times → sample n indices among the None entries.
        2) If there are <  n None times → take all None indices, then fill the
           rest from the smallest non-None times.
        3) If there are no None values → choose n smallest times.

        Returns
        -------
        If batch=False:
            idx : int
        If batch=True:
            idxs : list[int]
        """

        n_items = len(batch_list)
        if n_items == 0:
            return [] if batch else None

        # ---------- number to return ----------
        if batch:
            if batch_max is None:
                raise ValueError("batch_max must be provided when batch=True.")
            if batch_max <= 0:
                return []
            n = min(batch_max, n_items)
        else:
            n = 1

        # ---------- gather FIFO times ----------
        # each item must implement: get_last_move_out_time()
        times = [item.get_last_move_out_time() for item in batch_list]

        # Split into None and non-None
        none_idxs = [i for i, t in enumerate(times) if t is None]
        non_none = [(t, i) for i, t in enumerate(times) if t is not None]

        # ---------- case 1: enough None entries ----------
        if len(none_idxs) >= n:
            chosen = random.sample(none_idxs, n)

        else:
            # Take all None indices
            chosen = list(none_idxs)
            need = n - len(chosen)

            # Fill with smallest FIFO-time values
            if need > 0 and non_none:
                smallest = heapq.nsmallest(need, non_none, key=lambda t: (t[0], t[1]))
                chosen.extend(i for _, i in smallest)

        # ---------- return ----------
        if batch:
            return chosen

        # single selection
        return chosen[0]

