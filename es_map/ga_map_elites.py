
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
## GA MAP ELITES ##
###################


def run_ga_map_elites(client,config,wandb_logging=True):
    
        
    print("staring run_ga_map_elites")
    
    config = config.as_dict()
    
    DEBUG = False
    if DEBUG is True:
        if type(config) != type(dict()):
            config = config.as_dict()
            
        config["GA_CHILDREN_PER_GENERATION"] = 20
        config["GA_NUM_EVALUATIONS"] = 4
    
    b_map = behavior_map.Grid_behaviour_map(config)
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))
    
    if wandb_logging is True:   
        run_name = wandb.run.dir.split("/")[-2]
    else:
        from datetime import datetime
        run_name = "local_dubug_run_" + datetime.now().strftime("_%m_%d___%H:%M")
    run_checkpoint_path = "/scratch/ak1774/runs/large_files/" + run_name

    evaluations_per_generation = config["GA_CHILDREN_PER_GENERATION"] * config["GA_NUM_EVALUATIONS"]
    # es evaluations
    # evaluations_per_generation = config["ES_popsize"] + config["ES_CENTRAL_NUM_EVALUATIONS"] 
        
    generation_number = 0
    evaluations_so_far = 0
    best_fitness_so_far = 0
    best_model_so_far = None
        
    print("staring main loop")

    while True:
        if generation_number >= config["GA_MAP_ELITES_NUM_GENERATIONS"]:
            print("Done, reached iteration: ",config["GA_MAP_ELITES_NUM_GENERATIONS"])
            break
        
        ##########################################
        ## SELECT PARENTS AND EVALUATE_CHILDREN ##
        ##########################################
        non_empty_cells = b_map.get_non_empty_cells(config)
        
        # single perent mode, this was used in original es map elites implementation (also normal map elites)
        if config["GA_MULTI_PARENT_MODE"] is False:
            parent_params,parent_obs_mean,parent_obs_std = map_elite_utils.ga_select_parent_single_mode(non_empty_cells,config)
            child_results = distributed_evaluate.ga_evaluate_children_single_parent(client,theta=parent_params,
                                                                                    obs_mean=parent_obs_mean,
                                                                                    obs_std=parent_obs_std,
                                                                                    config=config)
        else:
            parent_datas = map_elite_utils.ga_select_parents_multi_parent_mode(non_empty_cells,config)
            child_results = distributed_evaluate.ga_evaluate_children_multi_parent(client,parent_datas,config)
        
        #####################################################
        ## DECIDE IF CHILDREN NEEDS TO BE ADDED TO ARCHIVE ## 
        #####################################################
        for child_res in child_results:
            
            mean_fitness = child_res["mean_fitness"]
            mean_bc = child_res["mean_bc"]
            
            # try to add them to the archive
            if mean_fitness > best_fitness_so_far:
                best_fitness_so_far = mean_fitness
                best_model_so_far = child_res["child"]
                
            coords = b_map.get_cell_coords(mean_bc,config)
            cell = b_map.data[coords]        
                    
            need_adding = False
            if cell is None:
                need_adding = True
            elif mean_fitness > cell["eval_fitness"]:
                need_adding = True
                
                
            if need_adding is True:
                updated_cell = {
                    "params" : child_res["child"],
                    "generation_created" : generation_number,

                    "eval_fitness" : mean_fitness,
                    "eval_bc" : mean_bc,

                    "obs_stats" : {   # used for observation normalization. These stats are used for the children as well ??
                        "obs_sum" : child_res["child_obs_sum"],
                        "obs_sq" : child_res["child_obs_sq"],
                        "obs_count" : child_res["child_obs_count"],
                    },
                }
            b_map.data[coords] = updated_cell
        
        ########################
        ## EVERY STEP LOGGING ##
        ########################
        
        print(generation_number,len(non_empty_cells),best_fitness_so_far)
        
        # Calculate things for logging
        generation_fitneses = [child_res["mean_fitness"] for child_res in child_results]
        generation_bc = np.stack([child_res["mean_bc"] for child_res in child_results])
        
        # Do the step logging 
        step_logs = {
            "generation_number" : generation_number,
            "nonempty_cells" : len(non_empty_cells),
            "nonempty_ratio" : float(len(non_empty_cells)) / b_map.data.size,
            "best_fitness_so_far" : best_fitness_so_far,
            "generation_fitness_mean" : np.mean(generation_fitneses),
            "generation_fitness_std" : np.std(generation_fitneses),
            "generation_bc_mean" : np.mean(generation_bc,axis=0),
            "generation_bc_std" : np.std(generation_bc,axis=0),
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
            fig,ax = map_elite_utils.plot_4d_map(b_map)
            n_step_log = {
                "b_map_plot" : fig,
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
            np.save(run_checkpoint_path+"/best_model.npy",best_model_so_far)
            np.save(run_checkpoint_path+"/b_map.npy",b_map.data,allow_pickle=True)
            b_archive.save_to_file(run_checkpoint_path+"/b_archive.npy")
            pass
                
        generation_number += 1
        

    ################
    ## FINAL SAVE ##
    ################
    print("Doing final save!")
    np.save(run_checkpoint_path+"/best_model.npy",best_model_so_far)
    np.save(run_checkpoint_path+"/b_map.npy",b_map.data,allow_pickle=True)
    b_archive.save_to_file(run_checkpoint_path+"/b_archive.npy")
    print("Final save done: ", run_checkpoint_path)











def select_parent_elite(parent_cell,config):
    if parent_cell is None:
        return get_random_individual(config)
    else:
        
        if config["MAP_TYPE"] == "SINGLE":
            return parent_cell["elite"]
        
        elif config["MAP_TYPE"] == "MULTIPLE_INDEPENDENT":
            # Select one from the type of elites, and return the elite of that type
            raise "TODO, not implemented"
        
        elif config["MAP_TYPE"] == "MULTIPLE_ND_SORT":
            # then select a parent randomly from the maintained elites in the pareto front
            raise "TODO, not implemented"
        
        else:
            raise "Unknown MAP_TYPE"








