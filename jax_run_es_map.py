import os
import numpy as np

import wandb
from es_map import submission_common
from es_map import jax_es_map_train

def run_es_experiment():
    
    config_defaults = {
        "env_name" : "ant", #  ant, walker, hopper, halfcheetah, humanoid, ant_omni, humanoid_omni
        "episode_max_length" : 1000,
        "env_deterministic" : True, 

        "ES_NUM_GENERATIONS" : 500,  # was 1000
        "ES_popsize" : 1000,
        "ES_sigma" : 0.02,
        "ES_OPTIMIZER_TYPE" : "ADAM",
        "ES_lr" : 0.01,
        "ES_L2_COEFF" : 0.005,  
        
        "ES_CENTRAL_NUM_EVALUATIONS" : 50,# How many times central individual evaluated
        "EVALUATION_BATCH_SIZE" : 50, 
        "ES_STEPS_UNTIL_NEW_PARENT_SELECTION" : 5,
        
        
        "ES_UPDATES_MODES_TO_USE" : ["fitness"], # "fitness","evo_var","evo_ent","innovation",...
        
        "NOVELTY_CALCULATION_NUM_NEIGHBORS" : 10,
        "ENTROPY_CALCULATION_KERNEL_BANDWIDTH" : 0.25, # TODO maybe this should be different for final pos and foot contacts...
        
        "CHECKPOINT_FREQUENCY" : 100,
        
        
        
        "BMAP_type_and_metrics" : {
            "type" : "single_map",
            "metrics" : ["eval_fitness"],
        }
    
    config_defaults = {
        
        "ES_NUM_INITIAL_RANDOM_INDIVIDUALS_TO_POPULATE_MAP" : 20,
        
        "ES_NUM_GENERATIONS" : 1000,  # was 1000
        "ES_popsize" : 2000,
        "ES_sigma" : 0.02,
        "ES_EVALUATION_BATCH_SIZE" : 5, # only computational efficiency, no effect on results
        "ES_lr" : 0.01,
        "ES_L2_COEFF" : 0.005,  
        "ES_OPTIMIZER_TYPE" : "ADAM",   # NOTE, for es_map we currently dont save the optimizer state, so maybe use SGD only
        
        "ES_CENTRAL_NUM_EVALUATIONS" : 30,
        "ES_STEPS_UNTIL_NEW_PARENT_SELECTION" : 5,
        
        "map_elites_grid_description" : {
            "bc_limits" : [[0,1],[0,1],[0,1],[0,1]],
            "grid_dims" : [6,6,6,6],
        },
        
        # BMAP settings
        "BMAP_type_and_metrics" : {
            "type" : "multi_map",    # type can be: "single_map","multi_map","nd_sorted_map" 
            "metrics" : ["eval_fitness","evolvability"],    # metric can be: ["f"],["f",e], ["f,e,i"], etc... 
                         # TODO eval_fitness, excpected_fitness, see if there is a differec
        },
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],#"evolvability","innovation"], # list of updates to use # TODO add QE
        "ES_PARENT_SELECTION_MODE" : "rank_proportional",  # "uniform", "rank_proportional"
        "ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS" : 1.0,  # 0.0 uniform, 1.0 normal , higher more agressive
        
        "ES_ND_SORT_MAX_FRONT_SIZE_TO_KEEP" : 6,
        "NOVELTY_CALCULATION_NUM_NEIGHBORS" : 10,
        "ENTROPY_CALCULATION_KERNEL_BANDWIDTH" : 0.25,
        
        # Using plain ES to fill MAP
        "FILL_MAP_WITH_OFFSPRING_NUM_CHILDREN" : 2000,  # Note that this is multiplied with ES_CENTRAL_NUM_EVALUATIONS
        "FILL_MAP_WITH_OFFSPRING_MEASURE_EVERY_N_GENERATIONS" : 100,  
        
        "CHECKPOINT_FREQUENCY" : 100,
        "PLOT_FREQUENCY" : 100,
    }
    
    n_workers = 50
    
    ######## DEBUG ##################
    #config_defaults["ALGORITHM_TYPE"] = "PLAIN_ES"
    #config_defaults["BMAP_type_and_metrics"]["metrics"] = ["eval_fitness","evolvability","innovation"]
    #config_defaults["ES_UPDATES_MODES_TO_USE"] = ["fitness","evolvability","innovation"]
    #config_defaults["env_id"] ="QDAntBulletEnv-v0"
    #n_workers = 8
    
    print("Initializing wandb")
    config = submission_common.setup_wandb(config_defaults,project_name="ES_SINGLE")
    print("Initializing wandb DONE!!!")
    
    
    print("Setting up client")
    client = submission_common.set_up_dask_client(n_workers=n_workers)
    print("Client set up DONE!!!")
    
    print("Starting algorithm... ",config["ALGORITHM_TYPE"])
    if config["ALGORITHM_TYPE"] == "PLAIN_ES":
        from es_map import plain_es
        plain_es.run_plain_es(client,config)
    elif config["ALGORITHM_TYPE"] == "MAP_ES":
        from es_map import es_map_elites
        es_map_elites.run_es_map_elites(client,config)
    elif config["ALGORITHM_TYPE"] == "MAP_GA":
        from es_map import ga_map_elites
        ga_map_elites.run_ga_map_elites(client,config)
    else:
        raise "ERROR unknown ALGORITHM_TYPE!!!"
    
    print("ALGORITHM Finished properly")
    
    
if __name__ == '__main__':  
    run_es_experiment()