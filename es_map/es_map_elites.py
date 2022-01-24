

from es_map import map_elite_utils
from es_map import es_update
from es_map import distributed_evaluate
from es_map.interaction import interaction
from es_map import behavior_map
from es_map import novelty_archive
from es_map import nd_sort

import random
import numpy as np
import copy
import pickle
import json

import wandb

###################
## ES MAP ELITES ##
###################

def select_matching_metrics_for_update_modes_or_select_randomly(metrics,update_mode):
    if update_mode == "eval_fitness":
        if "fitness" in metrics:
            return "fitness"
        else:
            return np.random.choice(metrics) # no matching metric, select randomly
        
    elif update_mode == "evolvability":
        if "evolvability" in metrics:
            return "evolvability"
        else:
            return np.random.choice(metrics) # no matching metric, select randomly
        
    elif update_mode == "innovation":
        if "innovation" in metrics:
            return "innovation"
        else:
            return np.random.choice(metrics) # no matching metric, select randomly
        
    elif update_mode == "quality_evolvability":
        matching_metrics = []
        if "fitness" in metrics:
            matching_metrics.append("fitness")
        if "evolvability" in metrics:
            matching_metrics.append("evolvability")

        if len(matching_metrics) == 0:
            return np.random.choice(metrics) # no matching metric, select randomly
        else:
            return np.random.choice(matching_metrics)
    
    else:
        raise "Error unknown update mode"
        
    
    
    
    
    
    

def run_es_map_elites(client,config,wandb_logging=True):
    print("staring run_es_map_elites")
    
    if type(config) != type(dict()):
        config = config.as_dict()
        
    # TODO HACK this should be set elswhere properly
    from es_map import submission_common
    config["map_elites_grid_description"] = submission_common.get_bc_descriptor_for_env(config["env_id"])
    
    DEBUG = False
    if DEBUG is True:
        config["ES_NUM_GENERATIONS"] = 15
        config["ES_popsize"] = 12
        config["ES_NUM_INITIAL_RANDOM_INDIVIDUALS_TO_POPULATE_MAP"] = 1
        config["ES_CENTRAL_NUM_EVALUATIONS"] = 3
        config["ES_STEPS_UNTIL_NEW_PARENT_SELECTION"] = 3
    
    if wandb_logging is True:   
        run_name = wandb.run.dir.split("/")[-2]
    else:
        from datetime import datetime
        run_name = "local_dubug_run_" + datetime.now().strftime("_%m_%d___%H:%M")
    run_checkpoint_path = "/scratch/ak1774/runs/large_files/" + run_name
    
    generation_number = 0
    evaluations_so_far = 0
    best_fitness_so_far = 0
    best_model_so_far = None
    best_evolvability_so_far = 0
    obs_stats_history = []
    
    get_next_individual_id = map_elite_utils.create_id_generator()
    
    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        b_map = behavior_map.Grid_behaviour_map(config)
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        b_map = behavior_map.Grid_behaviour_map(config)
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        b_map = behavior_map.Grid_behaviour_multi_map(config)
    else:
        raise "Unknown BMAP_type"
        
    
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))
    obs_shape = map_elite_utils.get_obs_shape(config)
    observation_stats = {                  # this is a single obs stats to keep track of during the whole experiment. 
        "sum" : np.zeros(obs_shape),       # This is always expanded, and always used to calculate the current mean and std
        "sumsq" : np.zeros(obs_shape),
        "count" : 0,
    }
    
    while True:
        if generation_number >= config["ES_NUM_GENERATIONS"]:
            print("Done, reached iteration: ",config["ES_NUM_GENERATIONS"])
            break
        
        ##############################################
        ## Populate the map with random individuals ##
        ##############################################
        if generation_number == 0:
            current_obs_mean,current_obs_std = map_elite_utils.calculate_obs_stats(observation_stats)   
            for _ in range(config["ES_NUM_INITIAL_RANDOM_INDIVIDUALS_TO_POPULATE_MAP"]):
                print("CREATING RANDOM INDIVIDUAL")
                new_individual_params = map_elite_utils.get_random_individual(config)

                # for the single map we are only interested in the fitness
                # but we maybe should calculate evolvability and innovation so we can compare them to the other algos

                eval_results = distributed_evaluate.evaluate_individual_repeated(theta=new_individual_params,
                                                                obs_mean=current_obs_mean,obs_std=current_obs_std,use_action_noise=False,record_obs=False,  # not using action noise for evaluation runs, also not recording obs stats
                                                                config=config,repeat_n=config["ES_CENTRAL_NUM_EVALUATIONS"])

                # TODO, maybe also evaluate children to get evolvability, especially if evolvability is out metric...
                     
                new_individual = {
                    "params" : new_individual_params,
                    "ID" : get_next_individual_id(),
                    "parent_ID" : None,
                    "generation_created" : generation_number,
                    
                    "child_eval" : None, # TODO
                    
                    "eval_fitness" : eval_results["fitness"],
                    "eval_bc" : eval_results["bc"],
                    
                    "evolvability" : None,
                    "innovation" : None,
                    "entropy" : None,

                    "innovation_over_time" : None,  # innovation decreases over time, as we add new individuals to the archive
                    #{  
                    #    generation_number : innovation_at_generation,
                    #    generation_number : innovation_at_generation,
                    #},
                }
                if new_individual["eval_fitness"] > best_fitness_so_far:
                    best_fitness_so_far = new_individual["eval_fitness"]
                    
                map_elite_utils.try_insert_individual_in_map(new_individual,b_map,b_archive,config)
        
        ######################################
        ## PARENT AND UPDATE MODE SELECTION ##
        ######################################
        
        # TODO we select an update mode and do that update for a coulple of steps
        # But we can also do the other kind of updates using the same evaluations, and calculate fitness.
        # It might be worth doing to do them, try to insert them into the map, 
        # even through we will not evaluate evolvability and innovation for them
        
        if config["BMAP_type_and_metrics"]["type"] == "single_map":
            
            # Choose an update mode
            es_update_mode = np.random.choice(config["ES_UPDATES_MODES_TO_USE"])
            
            non_empty_cells = b_map.get_non_empty_cells()
            if config["ES_PARENT_SELECTION_MODE"] == "uniform":
                parent_cell = np.random.choice(non_empty_cells)  # NOTE, here goes cell selection method
            elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
                # rank proportional means we rank based on the score
                # What this score should be? Fitness?
                # Do we have versions where we decide insertion based on evolvability or novelty when we have single map?
                 
                sorted_cells = sorted(non_empty_cells,key=lambda x : x["elite"]["eval_fitness"])                
                selected_index = map_elite_utils.rank_based_selection(num_parent_candidates=len(sorted_cells),
                                                        num_children=None, # only want a single parent
                                                        agressiveness=config["ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
                parent_cell = sorted_cells[selected_index]
            current_individual = parent_cell["elite"]
            
        elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
            # Choose an update mode
            es_update_mode = np.random.choice(config["ES_UPDATES_MODES_TO_USE"])
            # For nd sorted map, we have a bunch of cells, which each of them have a list of elites.
            # Do we uniformly select between all elites, or we select a cell, and then select an elite.
            # And even then, do we select elite with probability based on the kind of update we are about to do?
            # Probably we should. TODO
            # For now just uniformly select a cell, then uniformly select an elite.
            non_empty_cells = b_map.get_non_empty_cells()
            selected_cell = np.random.choice(non_empty_cells)
            current_individual = np.random.choice(selected_cell["elites"])
            
        elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
            # Choose an update mode
            es_update_mode = np.random.choice(config["ES_UPDATES_MODES_TO_USE"])
            
            # Now here we should defineltly select an individual from the type of map which is related to the update we selected
            # We have 3 metric: "eval_fitness","evolvability","innovation"
            # And we have a bunch of update modes: "fitness",evolvability,innovation,quality_evolvability
            
            #selected_metric = select_matching_metrics_for_update_modes_or_select_randomly(config["BMAP_type_and_metrics"]["metrics"],
            #                                                                              es_update_mode)
            # try to get a cell with selected metric
            # What to do here. If we dont have a specific update mode, we never select the metric, which is a bit pointless.
            # Maybe we should just select randomly across.
            # Or maybe we should differentiate between them. Like innovation is always useful i guess.
            
            # I rethought this, actually everything is always useful for everything.
            # - If we want high innovation and evolvability fitness probably helps
            # - Evo and innov also helps fitness.
            # - Does evo help innov and vica versa? Probably yes.
            # So just randomly select both the update mode and the cell.
            
            selected_metric = np.random.choice(config["BMAP_type_and_metrics"]["metrics"]) 
            non_empty_cells = b_map.get_non_empty_cells(selected_metric)
            if len(non_empty_cells) == 0:
                # this can only happen in the beginning, because we dont evaluate evolvability and innovation
                # for the initial seed individuals.
                # for this case select fitness, we always evaluate fitness. (and i guess fitness should always be a metric for multi map)
                selected_metric = "eval_fitness"
                non_empty_cells = b_map.get_non_empty_cells(selected_metric)
            selected_cell = np.random.choice(non_empty_cells)
            current_individual = selected_cell["elite"]
        
        
        
        print("new parent selected with update mode: ",es_update_mode)
                        
        current_optim_state = None # Whenever a new parent is selected, we reset optimizer state
                                   # This is the same as original es me. Note that in theory we could remember optimizer state 
                                   # for every elite in the map, with every update mode, but that would be expensive
        for update_step_i in range(config["ES_STEPS_UNTIL_NEW_PARENT_SELECTION"]):
            print("new update started")
            current_obs_mean,current_obs_std = map_elite_utils.calculate_obs_stats(observation_stats)  
            
            #######################
            ## EVALUATE CHILDREN ##
            #######################
            # does this cell already have cached evaluations?, then we can reuse it
            if current_individual["child_eval"] is None:  # maybe this should never happen, we will see
                current_individual["child_eval"] = distributed_evaluate.es_evaluate_children(client,current_individual["params"],obs_mean=current_obs_mean,
                                                                                                                                    obs_std=current_obs_std,config=config)
                # record the observation stats
                observation_stats["sum"] += current_individual["child_eval"]["child_obs_sum"]
                observation_stats["sumsq"] += current_individual["child_eval"]["child_obs_sq"]
                observation_stats["count"] += current_individual["child_eval"]["child_obs_count"]
                current_obs_mean,current_obs_std = map_elite_utils.calculate_obs_stats(observation_stats)
                
                
            ##################
            ## DO ES UPDATE ##
            ##################
            updated_theta,current_optim_state = es_update.es_update(current_individual["params"],current_individual["child_eval"],
                                                config,es_update_type=es_update_mode,novelty_archive=b_archive,optimizer_state=current_optim_state)
            
            if current_individual["evolvability"] is None:
                current_evolvability = map_elite_utils.calculate_behavioural_variance(current_individual["child_eval"],config)
                current_innovation = map_elite_utils.calculate_innovativeness(current_individual["child_eval"],b_archive,config)
                current_entropy = map_elite_utils.calculate_behavioural_distribution_entropy(current_individual["child_eval"],config)
                current_individual["evolvability"] = current_evolvability
                current_individual["innovation"] = current_innovation
                current_individual["entropy"] = current_entropy
                
            
            
            ############################
            ## EVALUATE UPDATED THETA ##
            ############################
            updated_eval_results = distributed_evaluate.evaluate_individual_repeated(theta=updated_theta,
                                                            obs_mean=current_obs_mean,obs_std=current_obs_std,use_action_noise=False,record_obs=False,
                                                            config=config,repeat_n=config["ES_CENTRAL_NUM_EVALUATIONS"])
            
            ##################################
            ## EVALUATE CHILDREN OF UPDATED ##
            ##################################
            updated_child_eval = distributed_evaluate.es_evaluate_children(client,updated_theta,obs_mean=current_obs_mean,obs_std=current_obs_std,config=config)
            updated_evolvability = map_elite_utils.calculate_behavioural_variance(updated_child_eval,config)
            updated_innovation = map_elite_utils.calculate_innovativeness(updated_child_eval,b_archive,config)
            updated_entropy = map_elite_utils.calculate_behavioural_distribution_entropy(updated_child_eval,config)
            
            # record the observation stats
            observation_stats["sum"] += updated_child_eval["child_obs_sum"]
            observation_stats["sumsq"] += updated_child_eval["child_obs_sq"]
            observation_stats["count"] += updated_child_eval["child_obs_count"]
            
            new_individual = {
                "params" : updated_theta,  # 1d torch tensor containing the parameters 
                "ID" : get_next_individual_id(),
                "parent_ID" : current_individual["ID"],
                "generation_created" : generation_number,

                "child_eval" : updated_child_eval,

                "eval_fitness" : updated_eval_results["fitness"],
                "eval_bc" : updated_eval_results["bc"],

                "evolvability" : updated_evolvability,
                "innovation" : updated_innovation,
                "entropy" : updated_entropy,

                "innovation_over_time" : {   # innovation decreases over time, as we add new individuals to the archive
                    generation_number : updated_innovation,
                },
            }
            
            if new_individual["evolvability"] > best_evolvability_so_far:
                best_evolvability_so_far = new_individual["evolvability"]
            if new_individual["eval_fitness"] > best_fitness_so_far:
                best_fitness_so_far = new_individual["eval_fitness"]
            
            #########################
            ## INSERT INTO ARCHIVE ##
            #########################
            
            # Weather or not we insert it, we add it to the behavior archive
            b_archive.add_to_archive(new_individual["eval_bc"])
            
            map_elite_utils.try_insert_individual_in_map(new_individual,b_map,b_archive,config)
                
            ################################
            ## PREPARE FOR NEXT ITERATION ##
            ################################
            current_individual = new_individual
            
            ########################
            ## EVERY STEP LOGGING ##
            ########################

            if config["BMAP_type_and_metrics"]["type"] == "multi_map":
                non_empty_cells = b_map.get_non_empty_cells("eval_fitness")  # for logging use the fitness map
                b_map_size = b_map.data[0].size  # only want to know the size of a single map, not the whole multi map
            else:
                non_empty_cells = b_map.get_non_empty_cells()
                b_map_size = b_map.data.size
            print(generation_number,len(non_empty_cells),best_fitness_so_far)

            qd_score = map_elite_utils.calculate_qd_score(b_map,config) 

            # Do the step logging 
            step_logs = {
                "generation_number" : generation_number,
                "evaluations_so_far" : generation_number*config["ES_popsize"],
                "nonempty_cells" : len(non_empty_cells),
                "nonempty_ratio" : float(len(non_empty_cells)) / b_map_size,
                "qd_score" : qd_score,
                "best_fitness_so_far" : best_fitness_so_far,
                "best_evolvability_so_far" : best_evolvability_so_far,
                "current_children_fitness_mean" : np.mean(new_individual["child_eval"]["fitnesses"]),
                "current_children_fitness_std" : np.std(new_individual["child_eval"]["fitnesses"]),
                "current_eval_fitness" : new_individual["eval_fitness"],
                "current_evolvability" : new_individual["evolvability"],
                "current_innovation" : new_individual["innovation"],
                "current_entropy" : new_individual["entropy"],
            }
            if wandb_logging is True:
                wandb.log(step_logs)
            
            ####################
            ## N STEP LOGGING ##
            ####################

            # Do the n-step logging
            if generation_number % config["PLOT_FREQUENCY"] == 10:
                # do plot map
                fig_f,ax = map_elite_utils.plot_b_map(b_map,metric="eval_fitness",config=config)
                n_step_log = {
                    "generation_number" : generation_number,
                    "evaluations_so_far" : generation_number*config["ES_popsize"],
                    "b_map_plot" : fig_f,
                }
                if "evolvability" in config["BMAP_type_and_metrics"]["metrics"]: 
                    fig_evo,ax = map_elite_utils.plot_b_map(b_map,metric="evolvability",config=config)
                    n_step_log["b_map_evolvability_plot"] = fig_evo
                
                if wandb_logging is True:
                    wandb.log(n_step_log)

                # also print some stuff to console
                print(generation_number)
                for k,v in step_logs.items():
                    print(k,v)
            
            ###################
            ## CHECKPOINTING ##
            ###################

            # Do checkpointing
            if generation_number % config["CHECKPOINT_FREQUENCY"] == 10:
                np.save(run_checkpoint_path+"/b_map.npy",b_map.data,allow_pickle=True)
                b_archive.save_to_file(run_checkpoint_path+"/b_archive.npy")
                obs_stats_history.append(copy.deepcopy(observation_stats))
                with open(run_checkpoint_path+'/obs_stats.pickle', 'wb') as handle:
                    pickle.dump(obs_stats_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        
            generation_number += 1
            
                
    ################
    ## FINAL SAVE ##
    ################
    print("Doing final save!")
    np.save(run_checkpoint_path+"/b_map.npy",b_map.data,allow_pickle=True)
    b_archive.save_to_file(run_checkpoint_path+"/b_archive.npy")
    print("Final save done: ", run_checkpoint_path)
                
                