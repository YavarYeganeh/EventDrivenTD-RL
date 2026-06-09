

""" 

Example replay buffer implementation for the interaction between the simulation worker and the trainer through shared memory and control channels. The code is extracted from a larger codebase and not optimized for performance or robustness, but it demonstrates the workflow protocol and can be adapted for other environments.

"""

import numpy as np

import os




# Define a dtype for cmd and replay buffer position
cmd_dtype = np.dtype([
    ("cmd", np.int32),
    ("position", np.int32),  # position in the replay buffer
])


# Define dtype for replay buffer experience metadata
experience_dtype_without_rewards = np.dtype([
    ("active", np.bool_), # boolean field to indicate if the slot is filled with valid data
    ("seed", np.int32),      # scenario id  
    ("call_time", np.float64),
    ("entity_name", "S64"),   # fixed-length string (64 bytes should be enough, adjust if needed)
    ("s_a_0", "S256"), # for path of the memory-mapped state-action 0
    ("s_a_1", "S256"), # for path of the memory-mapped state-action 1
    ("dtype", "S16"),  # data type of the state-action arrays (e.g., 'float32')
    ("idle_time", np.float64),
    ("in_time", np.float64),
    ("out_time", np.float64),
    ("queue_time", np.float64),
    ("process_time", np.float64),
    ("out_sat", np.float64),
    ("queue_sat", np.float64),
    ("out_wl", np.float64),
    ("queue_wl", np.float64),
    ("out_moves", np.int32),
    ("out_end", np.int32),
    ("queue_moves", np.int32),
    ("queue_end", np.int32),
])


# Define dtype for replay buffer experience metadata
experience_dtype_0 = np.dtype([
    ("active", np.bool_), # boolean field to indicate if the slot is filled with valid data
    ("call_time", np.float64),
    ("entity_name", "S64"),   # fixed-length string (64 bytes should be enough, adjust if needed)
    ("s_a_0", "S256"), # for path of the memory-mapped state-action 0
    ("s_a_1", "S256"), # for path of the memory-mapped state-action 1
    ("dtype", "S16"),  # data type of the state-action arrays (e.g., 'float32')
    ("idle_time", np.float64),
    ("in_time", np.float64),
    ("out_time", np.float64),
    ("queue_time", np.float64),
    ("process_time", np.float64),
    ("out_sat", np.float64),
    ("queue_sat", np.float64),
    ("out_wl", np.float64),
    ("queue_wl", np.float64),
    ("out_moves", np.int32),
    ("out_end", np.int32),
    ("queue_moves", np.int32),
    ("queue_end", np.int32),
    ("r_event", np.float32),  # reward for the experience related to the event
    ("r_system", np.float32), # global reward for the experience related to the system
    ("r_elements", np.float32, (6,)), # different reward elements
])



def register_experience_0(experience, rb, cmd, rb_cap, mm_dir, sce_id):

    position = cmd[0]['position']

    if rb[position]["active"]:  # needs the previous experience to removed from disk (memory-mapped files)
        
        # remove memory-mapped files
        s_a_0_path = rb[position]["s_a_0"].decode('utf-8')
        s_a_1_path = rb[position]["s_a_1"].decode('utf-8')
        
        try:
            os.remove(s_a_0_path)
            os.remove(s_a_1_path)
        
        except FileNotFoundError:
            pass  # file already removed

   
    # Assign all fields from experience to the buffer slot
   
    rb[position]["call_time"] = experience["call_time"]
    rb[position]["entity_name"] = experience["entity_name"]
    rb[position]["idle_time"] = experience["idle_time"]
    rb[position]["in_time"] = experience["in_time"]
    rb[position]["out_time"] = experience["out_time"]
    rb[position]["queue_time"] = experience["queue_time"]
    rb[position]["process_time"] = experience["process_time"]
    rb[position]["out_sat"] = experience["out_sat"]
    rb[position]["queue_sat"] = experience["queue_sat"]
    rb[position]["out_wl"] = experience["out_wl"]
    rb[position]["queue_wl"] = experience["queue_wl"]
    rb[position]["out_moves"] = experience["out_moves"]
    rb[position]["out_end"] = experience["out_end"]
    rb[position]["queue_moves"] = experience["queue_moves"]
    rb[position]["queue_end"] = experience["queue_end"]
    rb[position]["r_event"] = experience["r_event"]
    rb[position]["r_system"] = experience["r_system"]
    rb[position]["r_elements"] = experience["r_elements"]

    
    # memory-mapped tensors paths

    s_a_0_path = os.path.join(mm_dir, f'{position}_{sce_id}_0.bin')
    s_a_1_path = os.path.join(mm_dir, f'{position}_{sce_id}_1.bin')
    
    s0 = experience["s_a_0"].numpy() # to store dtype as well
    dtype = s0.dtype
    s0.tofile(s_a_0_path)  # save the state-action pair 0
    
    s1 = experience["s_a_1"].numpy()
    s1.tofile(s_a_1_path) # save the state-action pair 1

    rb[position]["s_a_0"] = s_a_0_path
    rb[position]["s_a_1"] = s_a_1_path
    rb[position]["dtype"] = dtype # help to load the memory-mapped files later from np to torch
    

    # activate the slot in the replay buffer

    rb[position]["active"] = True
    
    # next position in the replay buffer
    
    if position < rb_cap - 1:

        cmd[0]['position'] = position + 1 

    else:
    
        cmd[0]['position'] = 0



# Define dtype for replay buffer experience metadata
experience_dtype = np.dtype([
    ("active", np.bool_), # boolean field to indicate if the slot is filled with valid data
    ("Completed", np.bool_), # whether the experience is completed if True (or just a started experience if False)
    ("call_time", np.float64),
    ("entity_name", "S64"),   # fixed-length string (64 bytes should be enough, adjust if needed)
    ("s_a_0", "S256"), # for path of the memory-mapped state-action 0
    ("log_prob", np.float32), # log-probability of the selected action(s) at t=0 (sum of log probs over selected items)
    ("s_a_0_all", "S256"), # for path of the memory-mapped state-action candidates at 0
    ("s_a_1", "S256"), # for path of the memory-mapped state-action 1
    ("dtype", "S16"),  # data type of the state-action arrays (e.g., 'float32')
    ("idle_time", np.float64),
    ("in_time", np.float64),
    ("out_time", np.float64),
    ("queue_time", np.float64),
    ("process_time", np.float64),
    ("out_sat", np.float64),
    ("queue_sat", np.float64),
    ("out_wl", np.float64),
    ("queue_wl", np.float64),
    ("out_moves", np.int32),
    ("out_end", np.int32),
    ("queue_moves", np.int32),
    ("queue_end", np.int32),
    ("r_event", np.float32),  # reward for the experience related to the event
    ("r_system", np.float32), # global reward for the experience related to the system
    ("r_elements", np.float32, (6,)), # different reward elements
])




def register_started_experience(started_experience, rb, cmd, rb_cap, mm_dir, sce_id):
    

    position = cmd[0]['position']

    if rb[position]["active"]:  # needs the previous experience to removed from disk (memory-mapped files)
        
        # remove memory-mapped files
        s_a_0_path = rb[position]["s_a_0"].decode('utf-8')
        s_a_0_all_path = rb[position]["s_a_0_all"].decode('utf-8')
        s_a_1_path = rb[position]["s_a_1"].decode('utf-8')
        
        try:
            os.remove(s_a_0_path)
            os.remove(s_a_0_all_path)
            os.remove(s_a_1_path)
        
        except FileNotFoundError:
            pass  # file already removed


    #  Assign possible fields from partial/started experience to the buffer slot

    rb[position]["call_time"]   = started_experience["call_time"]
    rb[position]["entity_name"] = started_experience["entity_name"]
    rb[position]["queue_time"]  = started_experience["queue_time"]
    rb[position]["r_event"]     = started_experience["r_event"]
    rb[position]["r_elements"] = started_experience["r_elements"]


    # memory-mapped tensors paths # only s_a_0 is stored at this point

    s_a_0_path = os.path.join(mm_dir, f'{position}_{sce_id}_0.bin')
    s_a_0_all_path = os.path.join(mm_dir, f'{position}_{sce_id}_0_all.bin')

    s0 = started_experience["s_a_0"].numpy()
    s0_all = started_experience["s_a_0_all"].numpy()
    dtype = s0.dtype

    s0.tofile(s_a_0_path)
    s0_all.tofile(s_a_0_all_path)    

    rb[position]["s_a_0"]  = s_a_0_path
    rb[position]["log_prob"] = started_experience["log_prob"]
    rb[position]["s_a_0_all"]  = s_a_0_all_path
    
    rb[position]["dtype"]  = dtype

    # activate the slot in the replay buffer
    
    rb[position]["active"] = True
    rb[position]["Completed"] = False

    # next position in the replay buffer
    
    if position < rb_cap - 1:

        cmd[0]['position'] = position + 1 

    else:
    
        cmd[0]['position'] = 0


    return position



def register_completed_experience(experience, rb, position, mm_dir, sce_id):


    """ This version also handles if the experience is None, i.e., skipped due to invalid times """
    

    if not rb[position]["active"] or rb[position]["Completed"]:
        
        raise ValueError("The experience at the given position is either inactive or already completed")


    # checking if the experience is valid -> registering in the buffer 
    if experience is not None:


        # Assign all fields from experience to the buffer slot
    
        rb[position]["call_time"] = experience["call_time"]
        rb[position]["entity_name"] = experience["entity_name"]
        rb[position]["idle_time"] = experience["idle_time"]
        rb[position]["in_time"] = experience["in_time"]
        rb[position]["out_time"] = experience["out_time"]
        rb[position]["queue_time"] = experience["queue_time"]
        rb[position]["process_time"] = experience["process_time"]
        rb[position]["out_sat"] = experience["out_sat"]
        rb[position]["queue_sat"] = experience["queue_sat"]
        rb[position]["out_wl"] = experience["out_wl"]
        rb[position]["queue_wl"] = experience["queue_wl"]
        rb[position]["out_moves"] = experience["out_moves"]
        rb[position]["out_end"] = experience["out_end"]
        rb[position]["queue_moves"] = experience["queue_moves"]
        rb[position]["queue_end"] = experience["queue_end"]
        rb[position]["r_event"] = experience["r_event"]
        rb[position]["r_system"] = experience["r_system"]
        rb[position]["r_elements"] = experience["r_elements"]

        
        # memory-mapped tensors paths # only s_a_1 is stored at this point and s_a_0 is already stored when started

        # s_a_0_path = os.path.join(mm_dir, f'{position}_{sce_id}_0.bin') # already stored when started
        # s_a_0_all_path and log_prob done as well.
        s_a_1_path = os.path.join(mm_dir, f'{position}_{sce_id}_1.bin')
        
        # s0 = experience["s_a_0"].numpy() # to store dtype as well
        # dtype = s0.dtype
        # s0.tofile(s_a_0_path)  # save the state-action pair 0
        
        s1 = experience["s_a_1"].numpy()
        s1.tofile(s_a_1_path) # save the state-action pair 1

        # rb[position]["s_a_0"] = s_a_0_path
        rb[position]["s_a_1"] = s_a_1_path
        # rb[position]["dtype"] = dtype # help to load the memory-mapped files later from np to torch
        
        
        # mark as completed experience
        rb[position]["active"] = True
        rb[position]["Completed"] = True

    
    # invalid -> skipping
    else:
        
        # first remove memory-mapped files
        s_a_0_path = rb[position]["s_a_0"].decode('utf-8')
        s_a_0_all_path = rb[position]["s_a_0_all"].decode('utf-8')
        
        try:
            os.remove(s_a_0_path)
            os.remove(s_a_0_all_path)
        
        except FileNotFoundError:
            pass  # file already removed 

        # making the buffer slot inactive
        rb[position]["active"] = False
        rb[position]["Completed"] = False





