

import random
import numpy as np
import torch

from framework.encoder import d_s_a, extract_s_1_started



""" Offline """


# for offline agents
# using memory-mapped files to load data (to compensate for the large size of the data that can not be loaded into memory) with only using meta_data
def sample_exp_offline(meta_data, aggregate, len_exps, num_sampled_seeds, gamma, resample=20, avg_reward=False, add_s_a_0_all=False):

    """ 
    only considers event-level elements for reward calculation, system-level elements are not included in the reward calculation in the offline phase
    """

    s0_sizes, s1_sizes = [], []
    s_a_0_b, s_a_1_b, group_sizes, r_b, r_elements_b = [], [], [], [], []
    gammas = []

    if add_s_a_0_all:
        s_a_0_all_b = []  # 
        s0_all_sizes = []   

    # small speedups: avoid rebuilding keys each loop + localize random fns
    seeds = list(len_exps.keys())
    k = min(num_sampled_seeds, len(seeds))
    rand_sample = random.sample
    rand_range = random.randrange

    for _ in range(resample):

        for seed in rand_sample(seeds, k):

            n = len_exps[seed]
            exp_id = rand_range(n)
            md_seed = meta_data[seed]
            exp = md_seed[exp_id]

            # Load the state-action pairs from files
            s0 = np.fromfile(exp["s_a_0"], dtype=exp["dtype"]).reshape(-1, d_s_a)
            s1 = np.fromfile(exp["s_a_1"], dtype=exp["dtype"]).reshape(-1, d_s_a)
            s0 = torch.from_numpy(s0)
            s1 = torch.from_numpy(s1)

            s0_sizes.append(s0.shape[0])  # number of selected lots in the experience
            s1_sizes.append(s1.shape[0])  # number of proposed lots in the experience

            s_a_0_b.append(s0.view(-1, d_s_a))
            s_a_1_b.append(s1.view(-1, d_s_a))

            if add_s_a_0_all: # this includes all at the time action selection
                s0_all = np.fromfile(exp["s_a_0_all"], dtype=exp["dtype"]).reshape(-1, d_s_a)
                s0_all = torch.from_numpy(s0_all)
                s_a_0_all_b.append(s0_all)
                s0_all_sizes.append(s0_all.shape[0])

            # mean r_event over (parent + nested) without torch.stack (to be faster)
            nested = exp.get("nested", [])
            sum_r = None
            cnt = 0

            # parent
            r_event, r_elements = aggregate(exp, for_offline=True)
            sum_r = r_event.view(1, 1)
            cnt = 1
            r_elements_b.append(r_elements)

            # nested
            if avg_reward: # i.e., 1-step discounted if applicable

                for gid in nested:
                    exp_g = md_seed[gid]
                    r_event, r_elements = aggregate(exp_g, for_offline=True)
                    sum_r = sum_r + r_event.view(1, 1)
                    cnt += 1
                    r_elements_b.append(r_elements)

                r_b.append(sum_r / cnt)
                gammas.append(gamma)          # 1-step

            else: # n-step discounted

                for gid in nested:
                    exp_g = md_seed[gid]
                    r_event, r_elements = aggregate(exp_g, for_offline=True)
                    sum_r = sum_r + ((gamma)**cnt) * r_event.view(1, 1)
                    cnt += 1
                    r_elements_b.append(r_elements)

                r_b.append(sum_r)
                gammas.append(gamma ** cnt)   # n-step bootstrap discount

            # only one tuple (s_t, avg. reward or discounted, s_t_n)
            group_sizes.append(1)

    # cat
    s_a_0_b = torch.cat(s_a_0_b, dim=0) 
    s_a_1_b = torch.cat(s_a_1_b, dim=0)
    r_b = torch.cat(r_b, dim=0) 
    r_elements_b = torch.cat(r_elements_b, dim=0) 
    gammas = torch.tensor(gammas).reshape_as(r_b) 

    if add_s_a_0_all:
        s_a_0_all_b = torch.cat(s_a_0_all_b, dim=0)
        return s_a_0_all_b, s_a_0_b, s_a_1_b, r_b, r_elements_b, group_sizes, s0_all_sizes, s0_sizes, s1_sizes, gammas
    else:
        return s_a_0_b, s_a_1_b, r_b, r_elements_b, group_sizes, s0_sizes, s1_sizes, gammas


def sample_exp_offline_events(meta_data, aggregate, len_exps, num_sampled_seeds, gamma, resample=None, avg_reward=None, add_s_a_0_all=False):
    
    """needs mods agg(exp, for_offline=True) """
    
    group_sizes = []
    s0_sizes = []
    s1_sizes = []

    s_a_0_b, s_a_1_b, r_b, r_elements_b = [], [], [], []
    gammas = []

    if add_s_a_0_all:
        s_a_0_all_b = []
        s0_all_sizes = []

    sampled_seeds = random.sample(list(len_exps.keys()), num_sampled_seeds)

    for seed in sampled_seeds:
        
        exp_id = random.randint(0, len_exps[seed]-1)

        parent_out = meta_data[seed][exp_id]["out_time"] 

        group_ids = [exp_id] + meta_data[seed][exp_id].get("nested", [])

        n_started = 0

        for gid in group_ids:

            exp = meta_data[seed][gid]
            
            if exp["out_time"] > parent_out:
                n_started += 1
                continue # skipping as the event is not finished and therefore not nested
            
            # Load the state-action pairs from memory-mapped files
            s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, d_s_a)  # reshape to (batch_size, d_s_a)
            s1 = np.fromfile(exp["s_a_1"], dtype=exp['dtype']).reshape(-1, d_s_a)
            s0 = torch.from_numpy(s0)
            s1 = torch.from_numpy(s1)
            
            r_event, r_elements = aggregate(exp, for_offline=True)

            s0_size = s0.shape[0]  # number of selected lots in the experience
            s1_size = s1.shape[0]  # number of proposed lots in the experience

            if add_s_a_0_all:

                s0_all = np.fromfile(exp["s_a_0_all"], dtype=exp['dtype']).reshape(-1, d_s_a)
                s0_all = torch.from_numpy(s0_all)
                s_a_0_all_b.append(s0_all)

                s0_all_size = s0_all.shape[0]
                s0_all_sizes.append(s0_all_size)

            s_a_0_b.append(s0.view(-1,d_s_a))
            s_a_1_b.append(s1.view(-1,d_s_a))
            r_b.append((r_event).view(1,1))
            r_elements_b.append(r_elements)
            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)

        group_sizes.append(len(group_ids) - n_started) # handles even if nesting scheme is based on actions actions

    s_a_0_b = torch.cat(s_a_0_b, dim=0)
    s_a_1_b = torch.cat(s_a_1_b, dim=0)
    r_b = torch.cat(r_b, dim=0)
    r_elements_b = torch.cat(r_elements_b, dim=0)
    gammas = torch.ones_like(r_b) * gamma

    if add_s_a_0_all:
        s_a_0_all_b = torch.cat(s_a_0_all_b, dim=0)
        return s_a_0_all_b, s_a_0_b, s_a_1_b, r_b, r_elements_b, group_sizes, s0_all_sizes, s0_sizes, s1_sizes, gammas
    else:
        return s_a_0_b, s_a_1_b, r_b, r_elements_b, group_sizes, s0_sizes, s1_sizes, gammas    



""" Online """

def segment_experiences(
    rb,
    segment_reward,
    s_a_dim=d_s_a,
    scenario_list=None,
    num_sampled_scenarios=None,
    length=480,
    gamma=0.99,
    latest=True,
    with_candidate_info=False,
):

    """ 
    Segment of experiences
    with latest option (if False would sample to replay)
    .
    """
    
    group_sizes = []
    s0_sizes = []
    s0_all_sizes = []
    s1_sizes = []

    if not scenario_list:  # random sample scenarios
        scenario_list = random.sample(list(rb.keys()), num_sampled_scenarios)

    s_a_0_all, s_a_1_all, r_e_all, r_g_all, r_elements_all = [], [], [], [], []

    # optional candidate-info outputs
    log_prob_all = [] if with_candidate_info else None
    s_a_0_cands_all = [] if with_candidate_info else None

    for seed in scenario_list:  # seed is scenario id

        # segment sample
        seg_start_time, seg_start_pos, seg_end_time, seg_end_pos, seg_r, seg_r_elements = sample_segment(
            rb[seed], segment_reward, length, latest=latest
        )

        # extract s_1_started from parent
        s_a_1_parent_path = rb[seed][seg_end_pos]["s_a_1"]
        s_a_1_parent_dtype = rb[seed][seg_end_pos]["dtype"]

        s_a_1_parent = np.fromfile(s_a_1_parent_path, dtype=s_a_1_parent_dtype).reshape(-1, s_a_dim)
        s_a_1_parent = torch.from_numpy(s_a_1_parent)
        s_1_started = extract_s_1_started(s_a_1_parent)

        # experience IDs
        nested_ids = segment_nested(rb[seed], seg_start_time, seg_end_time)
        started_ids = segment_started(rb[seed], seg_start_time, seg_end_time)

        exp_ids = nested_ids + started_ids
        group_sizes.append(len(exp_ids))

        # -----------------------------------------------------
        # 1. Process nested experiences
        # -----------------------------------------------------
        for gid in nested_ids:
        
            exp = rb[seed][gid]

            s0_path = exp["s_a_0"]
            s1_path = exp["s_a_1"]
            exp_dtype = exp["dtype"]

            s0 = np.fromfile(s0_path, dtype=exp_dtype).reshape(-1, s_a_dim)
            s1 = np.fromfile(s1_path, dtype=exp_dtype).reshape(-1, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = torch.from_numpy(s1)

            r_event = exp["r_event"]
            r_elements_event = exp["r_elements"][:3]
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]
            s1_size = s1.shape[0]

            s_a_0_all.append(s0.view(-1, s_a_dim))
            s_a_1_all.append(s1.view(-1, s_a_dim))
            r_e_all.append(torch.tensor(r_event, dtype=torch.float32).view(1, 1))
            r_g_all.append(seg_r.view(1, 1))
            r_elements_all.append(torch.cat([r_elements_event, seg_r_elements], dim=1))

            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)

            if with_candidate_info:

                s0_all_path = exp["s_a_0_all"]

                s0_all = np.fromfile(s0_all_path, dtype=exp_dtype).reshape(-1, s_a_dim)
                s0_all = torch.from_numpy(s0_all)

                log_prob = torch.tensor(exp["log_prob"], dtype=torch.float32).view(1, 1)

                s_a_0_cands_all.append(s0_all.view(-1, s_a_dim))
                log_prob_all.append(log_prob)

                s0_all_sizes.append(s0_all.shape[0])

        # -----------------------------------------------------
        # 2. Process started-but-unfinished experiences
        # -----------------------------------------------------
        for gid in started_ids:

            exp = rb[seed][gid]

            s0_path = exp["s_a_0"]
            exp_dtype = exp["dtype"]

            s0 = np.fromfile(s0_path, dtype=exp_dtype).reshape(-1, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = s_1_started  # override with parent s_1_started

            r_event = exp["r_event"]
            r_elements_event = exp["r_elements"][:3]
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]
            s1_size = s1.shape[0]

            s_a_0_all.append(s0.view(-1, s_a_dim))
            s_a_1_all.append(s1.view(-1, s_a_dim))
            r_e_all.append(torch.tensor(r_event, dtype=torch.float32).view(1, 1))
            r_g_all.append(seg_r.view(1, 1))
            r_elements_all.append(torch.cat([r_elements_event, seg_r_elements], dim=1))

            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)

            if with_candidate_info:

                s0_all_path = exp["s_a_0_all"]

                s0_all = np.fromfile(s0_all_path, dtype=exp_dtype).reshape(-1, s_a_dim)
                s0_all = torch.from_numpy(s0_all)

                log_prob = torch.tensor(exp["log_prob"], dtype=torch.float32).view(1, 1)

                s_a_0_cands_all.append(s0_all.view(-1, s_a_dim))
                log_prob_all.append(log_prob)

                s0_all_sizes.append(s0_all.shape[0])

    # ---------------------------------------------------------
    # Final concatenation
    # ---------------------------------------------------------
    s_a_0_all = torch.cat(s_a_0_all, dim=0)
    s_a_1_all = torch.cat(s_a_1_all, dim=0)
    r_e_all = torch.cat(r_e_all, dim=0)
    r_g_all = torch.cat(r_g_all, dim=0)
    r_elements_all = torch.cat(r_elements_all, dim=0)
    weights_all = torch.full_like(r_g_all, gamma)

    if with_candidate_info:
        s_a_0_cands_all = torch.cat(s_a_0_cands_all, dim=0)
        log_prob_all = torch.cat(log_prob_all, dim=0)
    else:
        s_a_0_cands_all = None
        log_prob_all = None
        s0_all_sizes = None

    return (
        s_a_0_all,
        log_prob_all,
        s_a_0_cands_all,
        s_a_1_all,
        r_e_all,
        r_g_all,
        r_elements_all,
        group_sizes,
        s0_sizes,
        s0_all_sizes,
        s1_sizes,
        weights_all,
    )


def segment_experiences_wo_s_a_all_pob(rb, segment_reward, s_a_dim=d_s_a, scenario_list=None, num_sampled_scenarios=None, length=480, gamma=0.99, latest=True):

    """ 
    Segment of experiences
    with latest option (if False would sample to replay)
    .
    """
    
    group_sizes = []
    s0_sizes = []
    s1_sizes = []

    if not scenario_list:  # random sample scenarios
        scenario_list = random.sample(list(rb.keys()), num_sampled_scenarios)

    s_a_0_all, s_a_1_all, r_e_all, r_g_all, r_elements_all = [], [], [], [], []

    for seed in scenario_list:  # seed is scenario id

        # segment sample
        seg_start_time, seg_start_pos, seg_end_time, seg_end_pos, seg_r, seg_r_elements = sample_segment(rb[seed], segment_reward, length, latest=latest)

        # extract s_1_started from parent
        s_a_1_parent_path = rb[seed][seg_end_pos]["s_a_1"]
        s_a_1_parent_dtype = rb[seed][seg_end_pos]['dtype']
        s_a_1_parent = np.fromfile(s_a_1_parent_path, dtype=s_a_1_parent_dtype).reshape(-1, s_a_dim)
        s_a_1_parent = torch.from_numpy(s_a_1_parent)
        s_1_started = extract_s_1_started(s_a_1_parent)

        # experience IDs
        nested_ids =  segment_nested(rb[seed], seg_start_time, seg_end_time)
        started_ids = segment_started(rb[seed], seg_start_time, seg_end_time) 

        exp_ids = nested_ids + started_ids
        group_sizes.append(len(exp_ids))

        # -----------------------------------------------------
        # 1. Process nested experiences
        # -----------------------------------------------------
        for gid in nested_ids:
        
            exp = rb[seed][gid]
            s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)
            s1 = np.fromfile(exp["s_a_1"], dtype=exp['dtype']).reshape(-1, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = torch.from_numpy(s1)

            r_event = exp["r_event"]
            r_elements_event = exp["r_elements"][:3]
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]
            s1_size = s1.shape[0]

            s_a_0_all.append(s0.view(-1, s_a_dim))
            s_a_1_all.append(s1.view(-1, s_a_dim))
            r_e_all.append(torch.tensor(r_event, dtype=torch.float32).view(1,1))
            r_g_all.append(seg_r.view(1,1))
            r_elements_all.append(torch.cat([r_elements_event, seg_r_elements], dim=1))

            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)

        # -----------------------------------------------------
        # 2. Process started-but-unfinished experiences
        # -----------------------------------------------------
        for gid in started_ids:

            exp = rb[seed][gid]
            s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = s_1_started  # override with parent s_1_started

            r_event = exp["r_event"]
            r_elements_event = exp["r_elements"][:3]
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]
            s1_size = s1.shape[0]

            s_a_0_all.append(s0.view(-1, s_a_dim))
            s_a_1_all.append(s1.view(-1, s_a_dim))
            r_e_all.append(torch.tensor(r_event, dtype=torch.float32).view(1,1))
            r_g_all.append(seg_r.view(1,1))
            r_elements_all.append(torch.cat([r_elements_event, seg_r_elements], dim=1))

            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)


    # ---------------------------------------------------------
    # Final concatenation
    # ---------------------------------------------------------
    s_a_0_all = torch.cat(s_a_0_all, dim=0)
    s_a_1_all = torch.cat(s_a_1_all, dim=0)
    r_e_all = torch.cat(r_e_all, dim=0)
    r_g_all = torch.cat(r_g_all, dim=0)
    r_elements_all = torch.cat(r_elements_all, dim=0)
    weights_all = torch.full_like(r_g_all, gamma)  

    return (
        s_a_0_all, 
        s_a_1_all, 
        r_e_all,
        r_g_all,
        r_elements_all, 
        group_sizes, 
        s0_sizes, 
        s1_sizes,
        weights_all   
    )


def segment_experiences_w_replay(rb, segment_reward, s_a_dim=d_s_a, scenario_list=None, num_sampled_scenarios=None, length=480, gamma=0.99, replay=1):

    """ 
    Segment of experiences
    .
    """
    
    group_sizes = []
    s0_sizes = []
    s1_sizes = []

    if not scenario_list:  # random sample scenarios
        scenario_list = random.sample(list(rb.keys()), num_sampled_scenarios)

    s_a_0_all, s_a_1_all, r_e_all, r_g_all, r_elements_all = [], [], [], [], []

    for seed in scenario_list:  # seed is scenario id

        for i in range(replay):
            
            if i==0:
                latest = True
            else:
                latest = False
            
            # segment sample
            seg_start_time, seg_start_pos, seg_end_time, seg_end_pos, seg_r, seg_r_elements = sample_segment(rb[seed], segment_reward, length, latest=latest)

            # extract s_1_started from parent
            s_a_1_parent_path = rb[seed][seg_end_pos]["s_a_1"]
            s_a_1_parent_dtype = rb[seed][seg_end_pos]['dtype']
            s_a_1_parent = np.fromfile(s_a_1_parent_path, dtype=s_a_1_parent_dtype).reshape(-1, s_a_dim)
            s_a_1_parent = torch.from_numpy(s_a_1_parent)
            s_1_started = extract_s_1_started(s_a_1_parent)

            # experience IDs
            nested_ids =  segment_nested(rb[seed], seg_start_time, seg_end_time)
            started_ids = segment_started(rb[seed], seg_start_time, seg_end_time) 

            exp_ids = nested_ids + started_ids
            group_sizes.append(len(exp_ids))

            # -----------------------------------------------------
            # 1. Process nested experiences
            # -----------------------------------------------------
            for gid in nested_ids:
            
                exp = rb[seed][gid]
                s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)
                s1 = np.fromfile(exp["s_a_1"], dtype=exp['dtype']).reshape(-1, s_a_dim)
                s0 = torch.from_numpy(s0)
                s1 = torch.from_numpy(s1)

                r_event = exp["r_event"]
                r_elements_event = exp["r_elements"][:3]
                r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

                s0_size = s0.shape[0]
                s1_size = s1.shape[0]

                s_a_0_all.append(s0.view(-1, s_a_dim))
                s_a_1_all.append(s1.view(-1, s_a_dim))
                r_e_all.append(torch.tensor(r_event, dtype=torch.float32).view(1,1))
                r_g_all.append(seg_r.view(1,1))
                r_elements_all.append(torch.cat([r_elements_event, seg_r_elements], dim=1))

                s0_sizes.append(s0_size)
                s1_sizes.append(s1_size)

            # -----------------------------------------------------
            # 2. Process started-but-unfinished experiences
            # -----------------------------------------------------
            for gid in started_ids:

                exp = rb[seed][gid]
                s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)
                s0 = torch.from_numpy(s0)
                s1 = s_1_started  # override with parent s_1_started

                r_event = exp["r_event"]
                r_elements_event = exp["r_elements"][:3]
                r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

                s0_size = s0.shape[0]
                s1_size = s1.shape[0]

                s_a_0_all.append(s0.view(-1, s_a_dim))
                s_a_1_all.append(s1.view(-1, s_a_dim))
                r_e_all.append(torch.tensor(r_event, dtype=torch.float32).view(1,1))
                r_g_all.append(seg_r.view(1,1))
                r_elements_all.append(torch.cat([r_elements_event, seg_r_elements], dim=1))

                s0_sizes.append(s0_size)
                s1_sizes.append(s1_size)


    # ---------------------------------------------------------
    # Final concatenation
    # ---------------------------------------------------------
    s_a_0_all = torch.cat(s_a_0_all, dim=0)
    s_a_1_all = torch.cat(s_a_1_all, dim=0)
    r_e_all = torch.cat(r_e_all, dim=0)
    r_g_all = torch.cat(r_g_all, dim=0)
    r_elements_all = torch.cat(r_elements_all, dim=0)
    weights_all = torch.full_like(r_g_all, gamma)  

    return (
        s_a_0_all, 
        s_a_1_all, 
        r_e_all,
        r_g_all,
        r_elements_all, 
        group_sizes, 
        s0_sizes, 
        s1_sizes,
        weights_all   
    )


def sample_segment(rb, segment_reward, length=480, latest=True, max_trials=10000): 
    
    """
    
    Final latest or sample one segment of valid based on with at least provided length in rb[seed] moving backward

    returns: seg_start, seg_end, seg_r
    
    Find the last valid position in rb[seed] moving backward (incl. negative indexing).
    
    Conditions:
    - Must be active
    - Must be completed (to have all information registered.) 
    """

    active_mask = rb["active"]
    completed_mask = rb["Completed"]  # since completed mask register all information

    indices = np.where(active_mask & completed_mask)[0]

    if latest:

        # latest time among active completed experiences
        seg_end_ind = indices[np.argmax(rb["out_time"][indices])]
        seg_end_time = rb["out_time"][seg_end_ind]

        # experiences that start at least `length` after seg_end
        meets_length = (seg_end_time - rb["queue_time"]) >= length
        length_mask = np.where(meets_length & active_mask & completed_mask)[0]

    else:

        for i in range(max_trials):
           
            seg_end_ind = np.random.choice(indices)
            seg_end_time = rb["out_time"][seg_end_ind]
        
            # experiences that start at least `length` after seg_end
            meets_length = (seg_end_time - rb["queue_time"]) >= length
            length_mask = np.where(meets_length & active_mask & completed_mask)[0]

            if len(length_mask) > 0:
                break
        
        if i== max_trials-1: raise RuntimeError("Could not find valid segment after sampling end candidates.")

    # latest queue_time among those candidates # queue is correlated to actions
    seg_start_time = rb["queue_time"][length_mask].max()

    # index of that start
    seg_start_ind = length_mask[
        np.argmax(rb["queue_time"][length_mask])
    ]

    seg_r, seg_r_elements = segment_reward(rb, seg_start_ind, seg_end_ind)

    return seg_start_time, seg_start_ind, seg_end_time, seg_end_ind, seg_r, seg_r_elements


def latest_segment(rb, segment_reward, length=480): 
    
    """
    
    Final latest segment of valid based on with at least provided length in rb[seed] moving backward

    returns: seg_start, seg_end, seg_r
    
    Find the last valid position in rb[seed] moving backward (incl. negative indexing).
    
    Conditions:
    - Must be active
    - Must be completed (to have all information registered.) 
    """

    active_mask = rb["active"]
    completed_mask = rb["Completed"]  # since completed mask register all information

    indices = np.where(active_mask & completed_mask)[0]

    # latest time among active completed experiences
    seg_end_ind = indices[np.argmax(rb["out_time"][indices])]
    seg_end_time = rb["out_time"][seg_end_ind]

    # experiences that start at least `length` after seg_end
    meets_length = (seg_end_time - rb["queue_time"]) >= length

    length_mask = np.where(meets_length & active_mask & completed_mask)[0]

    # latest queue_time among those candidates # queue is correlated to actions
    seg_start_time = rb["queue_time"][length_mask].max()

    # index of that start
    seg_start_ind = length_mask[
        np.argmax(rb["queue_time"][length_mask])
    ]

    seg_r, seg_r_elements = segment_reward(rb, seg_start_ind, seg_end_ind)

    return seg_start_time, seg_start_ind, seg_end_time, seg_end_ind, seg_r, seg_r_elements


def segment_nested(rb, seg_start_time, seg_end_time):

    q_ref = seg_start_time
    o_ref = seg_end_time

    active_mask = rb["active"]
    completed_mask = rb["Completed"]
    inside_mask = (
        (rb["queue_time"] >= q_ref) & (rb["queue_time"] <= o_ref) &
        (rb["out_time"]   >= q_ref) & (rb["out_time"]   <= o_ref)
    )

    indices = np.where(active_mask & completed_mask & inside_mask)[0]
    indices = indices.tolist()

    return indices


def segment_started(rb, seg_start_time, seg_end_time):

    q_ref = seg_start_time
    o_ref = seg_end_time

    active_mask = rb["active"]

    # (A) Active & not completed: started inside [q_ref, o_ref)
    not_completed_mask = ~rb["Completed"]
    started_mask = (rb["queue_time"] >= q_ref) & (rb["queue_time"] < o_ref)

    indices_not_completed = np.where(
        active_mask & not_completed_mask & started_mask
    )[0]

    # (B) Active & completed: finished strictly after the reference out_time
    completed_mask = rb["Completed"]
    finished_after = rb["out_time"] > o_ref

    indices_completed_after = np.where(
        active_mask & completed_mask & started_mask & finished_after
    )[0]

    # Combine
    indices = np.concatenate((indices_not_completed, indices_completed_after))
    indices = indices.tolist()

    return indices


def latest_experiences(rb, cmd, s_a_dim=d_s_a, scenario_list=None, num_sampled_scenarios=None, threshold=None):

    """ This version also include weighting of the experiences within each scenario group based on their durations."""
    
    group_sizes = []
    s0_sizes = []
    s1_sizes = []

    if not scenario_list:  # random sample scenarios
        scenario_list = random.sample(list(rb.keys()), num_sampled_scenarios)

    s_a_0_all, s_a_1_all, r_all, r_elements_all = [], [], [], []
    weights_all = []  

    for seed in scenario_list:  # seed is scenario id

        # parent position
        position = find_last_valid(seed, cmd, rb, threshold)

        # reward of the system
        r_system = rb[seed][position]["r_system"]
        r_elements_system = rb[seed][position]["r_elements"][-3:]
        r_elements_system = torch.as_tensor(r_elements_system, dtype=torch.float32).view(1, -1)

        # extract s_1_started from parent
        s_a_1_parent_path = rb[seed][position]["s_a_1"]
        s_a_1_parent_dtype = rb[seed][position]['dtype']
        s_a_1_parent = np.fromfile(s_a_1_parent_path, dtype=s_a_1_parent_dtype).reshape(-1, s_a_dim)
        s_a_1_parent = torch.from_numpy(s_a_1_parent)
        s_1_started = extract_s_1_started(s_a_1_parent)

        # out_time from the parent experience # also used for started-but-unfinished experiences
        parent_out_time = rb[seed][position]["out_time"]

        # experience IDs
        nested_ids = list(set([position] + nested_experiences(position, rb[seed])))
        started_ids = started_experiences(position, rb[seed])

        exp_ids = nested_ids + started_ids
        group_sizes.append(len(exp_ids))

        # -----------------------------------------------------
        # 1. Compute raw weights for this seed
        # -----------------------------------------------------
        raw_weights = []
        epsilon = 1e-6  # small constant to avoid zero weights
        for gid in nested_ids:
            exp = rb[seed][gid]
            duration = exp["out_time"] - exp["queue_time"]   # weight = duration
            w = max(duration, epsilon) # ensure non-zero weight and nans!
            raw_weights.append(w)
        
        for gid in started_ids:
            exp = rb[seed][gid]
            duration = parent_out_time - exp["queue_time"]   # weight = duration # use parent out_time as it's not finished
            w = max(duration, epsilon) # ensure non-zero weight and nans!
            raw_weights.append(w)              

        raw_weights = torch.tensor(raw_weights, dtype=torch.float32)
        # weights_seed = torch.softmax(raw_weights, dim=0)  # softmax normalize
        weights_seed = raw_weights / raw_weights.sum()  # sum normalize instead of softmax which can overemphasize the largest weight/duration
        widx = 0  # index pointer

        # -----------------------------------------------------
        # 2. Process nested experiences
        # -----------------------------------------------------
        for gid in nested_ids:
        
            exp = rb[seed][gid]
            s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)
            s1 = np.fromfile(exp["s_a_1"], dtype=exp['dtype']).reshape(-1, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = torch.from_numpy(s1)

            r_event = exp["r_event"]
            r_elements_event = exp["r_elements"][:3]
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]
            s1_size = s1.shape[0]

            s_a_0_all.append(s0.view(-1, s_a_dim))
            s_a_1_all.append(s1.view(-1, s_a_dim))
            r_all.append(torch.tensor(r_event + r_system, dtype=torch.float32).view(1,1))
            r_elements_all.append(torch.cat([r_elements_event, r_elements_system], dim=1))

            # append weight
            weights_all.append(weights_seed[widx].view(1,1))
            widx += 1

            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)

        # -----------------------------------------------------
        # 3. Process started-but-unfinished experiences
        # -----------------------------------------------------
        for gid in started_ids:

            exp = rb[seed][gid]
            s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = s_1_started  # override with parent s_1_started

            r_event = exp["r_event"]
            r_elements_event = exp["r_elements"][:3]
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]
            s1_size = s1.shape[0]

            s_a_0_all.append(s0.view(-1, s_a_dim))
            s_a_1_all.append(s1.view(-1, s_a_dim))
            r_all.append(torch.tensor(r_event + r_system, dtype=torch.float32).view(1,1))
            r_elements_all.append(torch.cat([r_elements_event, r_elements_system], dim=1))

            # append weight
            weights_all.append(weights_seed[widx].view(1,1))
            widx += 1

            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)


    # ---------------------------------------------------------
    # Final concatenation
    # ---------------------------------------------------------
    s_a_0_all = torch.cat(s_a_0_all, dim=0)
    s_a_1_all = torch.cat(s_a_1_all, dim=0)
    r_all = torch.cat(r_all, dim=0)
    r_elements_all = torch.cat(r_elements_all, dim=0)
    weights_all = torch.cat(weights_all, dim=0)  

    return (
        s_a_0_all, 
        s_a_1_all, 
        r_all, 
        r_elements_all, 
        group_sizes, 
        s0_sizes, 
        s1_sizes,
        weights_all   
    )


def nested_experiences(position: int, rb: np.ndarray):
    
    """
    Return a list of integer indices of active experiences in the replay buffer whose
    (queue_time, out_time) lie inside the interval of the given position.

    This also considers completed status of the experiences.

    Parameters
    ----------
    position : int
        Index of the reference experience.
    rb : np.ndarray
        Replay buffer with dtype=experience_dtype.

    Returns
    -------
    list[int]
        Indices of matching experiences. // also return the parent experience
    """
    if not rb[position]["active"] or not rb[position]["Completed"]:
        raise ValueError(f"Reference/Parent position {position} is not active or completed")

    q_ref = rb[position]["queue_time"]
    o_ref = rb[position]["out_time"]

    active_mask = rb["active"]
    completed_mask = rb["Completed"]
    inside_mask = (
        (rb["queue_time"] >= q_ref) & (rb["queue_time"] <= o_ref) &
        (rb["out_time"]   >= q_ref) & (rb["out_time"]   <= o_ref)
    )

    indices = np.where(active_mask & completed_mask & inside_mask)[0]
    indices = indices.tolist()

    return indices


def started_experiences(position: int, rb: np.ndarray):
    
    """
    Return a list of integer indices of active experiences in the replay buffer that:

        Select experiences that STARTED within the parent window
        → and are either
        (1) unfinished, or
        (2) finished outside the parent window.

    Parameters
    ----------
    position : int
        Index of the reference experience.
    rb : np.ndarray
        Replay buffer with dtype=experience_dtype.

    Returns
    -------
    list[int]
        Indices of matching experiences.
    """
    # Validate reference experience
    if not rb[position]["active"] or not rb[position]["Completed"]:
        raise ValueError(f"Reference/Parent position {position} is not active or completed.")

    q_ref = rb[position]["queue_time"]
    o_ref = rb[position]["out_time"]

    active_mask = rb["active"]

    # (A) Active & not completed: started inside [q_ref, o_ref)
    not_completed_mask = ~rb["Completed"]
    started_mask = (rb["queue_time"] >= q_ref) & (rb["queue_time"] < o_ref)

    indices_not_completed = np.where(
        active_mask & not_completed_mask & started_mask
    )[0]

    # (B) Active & completed: finished strictly after the reference out_time
    completed_mask = rb["Completed"]
    finished_after = rb["out_time"] > o_ref

    indices_completed_after = np.where(
        active_mask & completed_mask & started_mask & finished_after
    )[0]

    # Combine
    indices = np.concatenate((indices_not_completed, indices_completed_after))
    indices = indices.tolist()

    return indices


def find_last_valid(seed, cmd, rb, threshold=None, max_steps=10000): 
    
    """
    Find the last valid position in rb[seed] moving backward (incl. negative indexing).
    
    Conditions:
    - Must be active
    - Must be completed
    - If threshold is provided: enforce it strictly until max_steps exceeded,
      then relax the threshold condition.
    """

    initial_position = position = int(cmd[seed]["position"] - 1)
    steps = 0
    strict = threshold is not None

    while True:   # infinite loop allowed — safety break inside

        entry = rb[seed][position]
        active = entry["active"]
        completed = entry["Completed"]

        # threshold rule (only active in strict mode)
        if strict:
            meets_threshold = (entry["out_time"] - entry["queue_time"]) >= threshold
        else:
            meets_threshold = True

        # if all conditions satisfied → done
        if active and completed and meets_threshold:
            return position

        # otherwise move backward (negative indexing ok)
        position -= 1
        steps += 1

        # relax threshold after too many steps
        if strict and steps >= max_steps:
            position = initial_position  # reset position
            strict = False


def latest_experiences_w_find_last_valid(rb, cmd, s_a_dim=d_s_a, scenario_list=None, num_sampled_scenarios=None, threshold=None):

    # this version also handles started but not finished nested experiences by using s_1_started extracted from the parent experience
    
    group_sizes = []
    s0_sizes = []
    s1_sizes = []

    if not scenario_list: # if the list of scenarios not provided randomly select num_sampled_scenarios
        scenario_list = random.sample(list(rb.keys()), num_sampled_scenarios)

    s_a_0_all, s_a_1_all, r_all, r_elements_all = [], [], [], []

    for seed in scenario_list: # seed here is the scenario id

        # parent position
        position = find_last_valid(seed, cmd, rb, threshold)

        # reward of the system
        r_system = rb[seed][position]["r_system"]
        r_elements_system = rb[seed][position]["r_elements"][-3:] # the last three elements are system that we also use for the child events
        # ensure torch 2D tensor for concat along dim=1
        r_elements_system = torch.as_tensor(r_elements_system, dtype=torch.float32).view(1, -1)
        
        # Extracting s_1_started from parent experience
        s_a_1_parent_path = rb[seed][position]["s_a_1"]
        s_a_1_parent_dtype = rb[seed][position]['dtype']
        s_a_1_parent = np.fromfile(s_a_1_parent_path, dtype=s_a_1_parent_dtype).reshape(-1, s_a_dim)
        s_a_1_parent = torch.from_numpy(s_a_1_parent)
        s_1_started = extract_s_1_started(s_a_1_parent)

        # extract nested experiences including the parent 
        nested_ids = list(set([position] + nested_experiences(position, rb[seed]))) # or nested_experiences(position, rb[seed])  # nested func also return the parent experience 

        # extract started but not finished experiences within the parent experience time window
        started_ids = started_experiences(position, rb[seed])

        group_sizes += [len(nested_ids) + len(started_ids)]

        # process nested experiences first
        for gid in nested_ids:
        
            exp = rb[seed][gid]
            # Load the state-action pairs from memory-mapped files
            s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)  # reshape to (batch_size, s_a_dim)
            s1 = np.fromfile(exp["s_a_1"], dtype=exp['dtype']).reshape(-1, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = torch.from_numpy(s1)
            
            r_event = exp["r_event"]

            r_elements_event = exp["r_elements"][:3] # the first three elements are related to the event 
            # ensure torch 2D tensor for concat along dim=1
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]  # number of selected lots in the experience
            s1_size = s1.shape[0]  # number of proposed lots in the experience
            s_a_0_all.append(s0.view(-1,s_a_dim))
            s_a_1_all.append(s1.view(-1,s_a_dim))
            r_all.append(torch.tensor(r_event + r_system, dtype=torch.float32).view(1,1)) # r_system is related to the parent experience and r_event is related to the nested experience, loss calculation should be separate for each seed #     loss = ((q_value-target).mean()-r_event_system)**2  # shape: (-1,s_a_dim or 1)
            r_elements_all.append(torch.cat([r_elements_event, r_elements_system], dim=1))
            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)  

        # process started experiences next
        for gid in started_ids:
    
            exp = rb[seed][gid]
            # Load the state-action pairs from memory-mapped files
            s0 = np.fromfile(exp["s_a_0"], dtype=exp['dtype']).reshape(-1, s_a_dim)  # reshape to (batch_size, s_a_dim)
            s0 = torch.from_numpy(s0)
            s1 = s_1_started # use s_1_started, extracted from the parent experience 
            
            r_event = exp["r_event"]

            r_elements_event = exp["r_elements"][:3] # the first three elements are related to the event 
            # ensure torch 2D tensor for concat along dim=1
            r_elements_event = torch.as_tensor(r_elements_event, dtype=torch.float32).view(1, -1)

            s0_size = s0.shape[0]  # number of selected lots in the experience
            s1_size = s1.shape[0]  # number of proposed lots in the experience
            s_a_0_all.append(s0.view(-1,s_a_dim))
            s_a_1_all.append(s1.view(-1,s_a_dim))
            r_all.append(torch.tensor(r_event + r_system, dtype=torch.float32).view(1,1)) # r_system is related to the parent experience and r_event is related to the nested experience, loss calculation should be separate for each seed #     loss = ((q_value-target).mean()-r_event_system)**2  # shape: (-1,s_a_dim or 1)
            r_elements_all.append(torch.cat([r_elements_event, r_elements_system], dim=1))
            s0_sizes.append(s0_size)
            s1_sizes.append(s1_size)

    s_a_0_all = torch.cat(s_a_0_all, dim=0)
    s_a_1_all = torch.cat(s_a_1_all, dim=0)
    r_all = torch.cat(r_all, dim=0)
    r_elements_all = torch.cat(r_elements_all, dim=0)

    return s_a_0_all, s_a_1_all, r_all, r_elements_all, group_sizes, s0_sizes, s1_sizes

