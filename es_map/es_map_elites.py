

from es_map import map_elite_utils
from es_map import es_update
from es_map import distributed_evaluate
from es_map.interaction import interaction
from es_map import behavior_map
from es_map import novelty_archive

import random
import numpy as np

import wandb

###################
## ES MAP ELITES ##
###################

def run_es_map_elites_single_map(client,config,wandb_logging=True):
    pass

    print("staring run_es_map_elites_single_map")
    
    if type(config) != type(dict()):
        config = config.as_dict()
    
    DEBUG = True
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
    
    get_next_individual_id = map_elite_utils.create_id_generator()
    
    b_map = behavior_map.Grid_behaviour_map(config)
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))
    
    while True:
        if generation_number >= config["ES_NUM_GENERATIONS"]:
            print("Done, reached iteration: ",config["ES_NUM_GENERATIONS"])
            break
            
        non_empty_cells = b_map.get_non_empty_cells()
        
        
        ##############################################
        ## Populate the map with random individuals ##
        ##############################################
        if len(non_empty_cells) == 0: # no parent available
            for _ in range(config["ES_NUM_INITIAL_RANDOM_INDIVIDUALS_TO_POPULATE_MAP"]):
                print("CREATING RANDOM INDIVIDUAL")
                new_individual_params = map_elite_utils.get_random_individual(config)

                # for the single map we are only interested in the fitness
                # but we maybe should calculate evolvability and innovation so we can compare them to the other algos

                eval_results = distributed_evaluate.evaluate_individual_repeated(theta=new_individual_params,
                                                                obs_mean=None,obs_std=None,eval=True,
                                                                config=config,repeat_n=config["ES_CENTRAL_NUM_EVALUATIONS"])

                # TODO, maybe also evaluate children to get evolvability
                
                # insert the new individual in the map
                map_coords = b_map.get_cell_coords(eval_results["bc"])

                needs_adding = False
                if b_map.data[map_coords] is None:
                    needs_adding = True
                elif b_map.data[map_coords]["elite"]["eval_fitness"] < eval_results["fitness"]:
                    needs_adding = True
                
                if needs_adding is True:      
                    new_individual = {
                        "params" : new_individual_params,
                        "ID" : get_next_individual_id(),
                        "parent_ID" : None,
                        "generation_created" : generation_number,
                        
                        "child_eval" : None, # TODO
                        
                        "eval_fitness" : eval_results["fitness"],
                        "eval_bc" : eval_results["bc"],
                        
                        "obs_stats" : {  
                            "obs_sum" : None,  # TODO handle obs stats
                            "obs_sq" : None,
                            "obs_count" : None,
                        },
                        
                        "evolvability" : None,
                        "innovation" : None,

                        "innovation_over_time" : None,  # innovation decreases over time, as we add new individuals to the archive
                        #{  
                        #    generation_number : innovation_at_generation,
                        #    generation_number : innovation_at_generation,
                        #},
                    }
                    
                    new_cell = {
                        "elite" : new_individual
                    }
                    b_map.data[map_coords] = new_cell
                    
                    if new_individual["eval_fitness"] > best_fitness_so_far:
                        best_fitness_so_far = new_individual["eval_fitness"]
                
        
        ######################
        ## PARENT SELECTION ##
        ######################
        else:
            if config["ES_PARENT_SELECTION_MODE"] == "uniform":
                parent_cell = np.random.choice(non_empty_cells)  # NOTE, here goes cell selection method
            elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
                sorted_cells = sorted(non_empty_cells,key=lambda x : x["elite"]["eval_fitness"])                
                selected_index = map_elite_utils.rank_based_selection(num_parent_candidates=len(sorted_cells),
                                                        num_children=None, # only want a single parent
                                                        agressiveness=config["ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
                parent_cell = sorted_cells[selected_index]
                
            
            current_individual = parent_cell["elite"]
            print("new parent selected")
                
            # decide on update mode (explore,exploit,build_evolvability)
            es_update_mode = np.random.choice(config["ES_UPDATES_MODES_TO_USE"])
                
            for update_step_i in range(config["ES_STEPS_UNTIL_NEW_PARENT_SELECTION"]):
                print("new update started")
                
                #######################
                ## EVALUATE CHILDREN ##
                #######################
                # does this cell already have cached evaluations?, then we can reuse it
                if current_individual["child_eval"] is None:  # maybe this should never happen, we will see
                    #current_individual -> obs_mean,obs_std # TODO
                    obs_mean,obs_std = None,None
                    current_individual["child_eval"] = distributed_evaluate.es_evaluate_children(client,current_individual["params"],obs_mean,obs_std,config)
                    
                    
                ##################
                ## DO ES UPDATE ##
                ##################
                updated_theta = es_update.es_update(current_individual["params"],current_individual["child_eval"],
                                                    config,es_update_type=es_update_mode,novelty_archive=b_archive)
                
                if current_individual["evolvability"] is None:
                    current_evolvability = es_update.calculate_behavioural_variance(current_individual["child_eval"],config)
                    current_innovation = es_update.calculate_innovativeness(current_individual["child_eval"],b_archive,config)
                    current_individual["evolvability"] = current_evolvability
                    current_individual["innovation"] = current_innovation
                    
                
                
                ############################
                ## EVALUATE UPDATED THETA ##
                ############################
                
                updated_eval_results = distributed_evaluate.evaluate_individual_repeated(theta=updated_theta,
                                                                obs_mean=None,obs_std=None,eval=True,
                                                                config=config,repeat_n=config["ES_CENTRAL_NUM_EVALUATIONS"])
                
                ##################################
                ## EVALUATE CHILDREN OF UPDATED ##
                ##################################
                
                updated_child_eval = distributed_evaluate.es_evaluate_children(client,updated_theta,obs_mean=None,obs_std=None,config=config)
                updated_evolvability = es_update.calculate_behavioural_variance(updated_child_eval,config)
                updated_innovation = es_update.calculate_innovativeness(updated_child_eval,b_archive,config)
                
                new_individual = {
                    "params" : updated_theta,  # 1d torch tensor containing the parameters 
                    "ID" : get_next_individual_id(),
                    "parent_ID" : current_individual["ID"],
                    "generation_created" : generation_number,

                    "child_eval" : updated_child_eval,

                    "eval_fitness" : updated_eval_results["fitness"],
                    "eval_bc" : updated_eval_results["bc"],

                    "obs_stats" : {   # TODO
                        "obs_sum" : None,
                        "obs_sq" : None,
                        "obs_count" : None,
                    },

                    "evolvability" : updated_evolvability,
                    "innovation" : updated_innovation,

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
                
                map_coords = b_map.get_cell_coords(new_individual["eval_bc"])
                if b_map.data[map_coords] is None:
                    new_cell = { "elite" : new_individual }
                    b_map.data[map_coords] = new_cell
                elif b_map.data[map_coords]["elite"]["eval_fitness"] < new_individual["eval_fitness"]:
                    b_map.data[map_coords]["elite"] = new_individual 
                    
                ################################
                ## PREPARE FOR NEXT ITERATION ##
                ################################
                current_individual = new_individual
                
                
                ########################
                ## EVERY STEP LOGGING ##
                ########################

                non_empty_cells = b_map.get_non_empty_cells()
                print(generation_number,len(non_empty_cells),best_fitness_so_far)

                # Do the step logging 
                step_logs = {
                    "generation_number" : generation_number,
                    "nonempty_cells" : len(non_empty_cells),
                    "nonempty_ratio" : float(len(non_empty_cells)) / b_map.data.size,
                    "best_fitness_so_far" : best_fitness_so_far,
                    "best_evolvability_so_far" : best_evolvability_so_far,
                    "current_children_fitness_mean" : np.mean(new_individual["child_eval"]["fitnesses"]),
                    "current_children_fitness_std" : np.std(new_individual["child_eval"]["fitnesses"]),
                    "current_eval_fitness" : new_individual["eval_fitness"],
                    "current_evolvability" : new_individual["evolvability"],
                    "current_innovation" : new_individual["innovation"],
                }
                if wandb_logging is True:
                    wandb.log(step_logs)
                
                ####################
                ## N STEP LOGGING ##
                ####################

                # Do the n-step logging
                if generation_number % config["PLOT_FREQUENCY"] == 10:
                    # do plot map
                    # save map without heavy stuff???
                    fig_f,ax = map_elite_utils.plot_4d_map(b_map,metric="eval_fitness")
                    fig_evo,ax = map_elite_utils.plot_4d_map(b_map,metric="evolvability")
                    n_step_log = {
                        "b_map_plot" : fig_f,
                        "b_map_evolvability_plot" : fig_evo, 
                    }
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
                    
            
                generation_number += 1
                
                
    ################
    ## FINAL SAVE ##
    ################
    print("Doing final save!")
    np.save(run_checkpoint_path+"/b_map.npy",b_map.data,allow_pickle=True)
    b_archive.save_to_file(run_checkpoint_path+"/b_archive.npy")
    print("Final save done: ", run_checkpoint_path)
                
                