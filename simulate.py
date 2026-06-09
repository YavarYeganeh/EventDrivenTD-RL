from sim.worker import main

if __name__ == "__main__":
    main()





""" 
##############
# Example in interaction of (Simpy) simulation worker with the trainer through shared memory and control channels. The code is intentionally simple and not optimized for performance or robustness, but it demonstrates the basic protocol and can be adapted for more complex scenarios.
##############



import sim.config as config


###### Simulator initialization 


### This method allows you to start the simulation time for either testing or training.


# testing/evaluation

if not config.training_mode:
    
    run_env(config.rand_init_time)  # run until init time with random policy

    config.init_agent = True # flag to initialize the agent in ml_handler

    if config.load_agent: 
        config.policy_sync = True  # to flag for agent load
    
    run_env(config.simulation_time)


# training

else:
    
    sim_continue = True

    while sim_continue:


        #####  Commands
                
        #     {
        #         0: Trainer → Sim = Initialize and run until INIT_TIME with the policy
        #         1: Trainer → Run for INTERVAL_TIME with the policy to obtain at least one new experience, then trainer updates the policy and require policy synchronization.
        #         2: Sim → Trainer = Task completed, waiting for the next command
        #         -1: Trainer → Sim = Finish the simulation and close
        #     }

        #

        
        if config.cmd[0]["cmd"] == 0: 

            run_env(config.rand_init_time)  # run until init time with random policy

            config.init_agent = True # flag to initialize the agent in ml_handler

            if config.load_agent: 
                config.policy_sync = True  # to flag for agent load

            run_env(config.replay_prefill_time + config.rand_init_time)  # run to fill the replay buffer before training starts

            config.cmd[0]["cmd"] = 2  # set the command to waiting for the next command
            print(f"Simulator: Scenario {config.scenario_id}: Signals initialization completed. Waiting for the next command.")


        elif config.cmd[0]["cmd"] == 1:
    
            start_position = config.cmd[0]['position']
            has_any_completed = None
            safety_counter = 0

            while config.cmd[0]['position'] == start_position and not has_any_completed:
                
                current_time = ENV.now # get the current simulation time
                target_time = current_time + config.sim_interval_time # define the target time to run
                run_env(target_time)

                safety_counter += 1
                if safety_counter > 1000:  # avoid infinite loop
                    print("Warning: no new experience after 1000 intervals")
                    break
                
                # check if there are at least one completed experience
                active_mask = config.rb["active"]
                completed_mask = config.rb["Completed"]
                has_any_completed = (active_mask & completed_mask).any()

            config.policy_sync = True  # to push for agent resync after its update

            config.cmd[0]["cmd"] = 2  # set the command to waiting for the next command


        elif config.cmd[0]["cmd"] == -1:

            sim_continue = False

            print(f"Trainer signals finishing the simulation: Scenario {config.scenario_id}")

            break

        
        time.sleep(config.cmd_wait)  # wait for some seconds before checking the command again

    
    # at the end of training close the shared memory
    config.existing_cmd_shm.close()
    config.existing_cmd_shm.unlink()

    config.existing_rb_shm.close()
    config.existing_rb_shm.unlink()

"""
