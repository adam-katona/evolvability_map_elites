

# this file is called by wandb agent when doing a sweep
# or alternatively, it can be called by itself to run with the default config

import os
import numpy as np

import wandb
from es_map import submission_common
from es_map import jax_simple_es_train



def run_es_experiment():
    
    config_defaults = {
        "env_name" : "ant", #  ant, walker, hopper, halfcheetah, humanoid, ant_omni, humanoid_omni
        "env_mode" : "NORMAL_CONTACT",
        "episode_max_length" : 250,
        "env_deterministic" : True, 
        
        
        "ALGORITHM_TYPE" : "PLAIN_ES",   # "PLAIN_ES", "MAP_ES", "MAP_GA"

        "ES_NUM_GENERATIONS" : 500,  # was 1000
        "ES_popsize" : 1000,
        "ES_sigma" : 0.02,
        "ES_OPTIMIZER_TYPE" : "ADAM",
        "ES_lr" : 0.01,
        "ES_L2_COEFF" : 0.005,  
        
        "ES_CENTRAL_NUM_EVALUATIONS" : 50, # How many times central individual evaluated (pointless for deterministic...)
        
        "ES_UPDATES_MODES_TO_USE" : ["fitness"], # "fitness","evo_var","evo_ent","innovation",...
        
        "NOVELTY_CALCULATION_NUM_NEIGHBORS" : 10,
        "ENTROPY_CALCULATION_KERNEL_BANDWIDTH" : 0.25, # maybe this should be different for final pos and foot contacts...
        
        "CHECKPOINT_FREQUENCY" : 100,
        
        "PLAIN_ES_TEST_ELITE_MAPPING_PERFORMANCE" : True,
        "PLAIN_ES_TEST_ELITE_MAPPING_FREQUENCY" : 1, # try filling the map every n steps
        
        "BMAP_type_and_metrics" : {
            "type" : "single_map",
            "metrics" : ["eval_fitness"],
        }
        
        
        # TODO
        #"map_elites_grid_description" : {
        #    "bc_limits" : [[0,1],[0,1],[0,1],[0,1]],
        #    "grid_dims" : [6,6,6,6],
        #},
        
        # BMAP settings
        #"BMAP_type_and_metrics" : {
        #    "type" : "multi_map",    # type can be: "single_map","multi_map","nd_sorted_map" 
        #    "metrics" : ["eval_fitness","evolvability"],    # metric can be: ["f"],["f",e], ["f,e,i"], etc... 
        #},
        #"ES_UPDATES_MODES_TO_USE" : ["fitness"],#"evolvability","innovation"], # list of updates to use # TODO add QE
        #"ES_PARENT_SELECTION_MODE" : "rank_proportional",  # "uniform", "rank_proportional"
        #"ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS" : 1.0,  # 0.0 uniform, 1.0 normal , higher more agressive
        
        #"ES_ND_SORT_MAX_FRONT_SIZE_TO_KEEP" : 6,
        #"NOVELTY_CALCULATION_NUM_NEIGHBORS" : 10,
        
        # Using plain ES to fill MAP
        #"FILL_MAP_WITH_OFFSPRING_NUM_CHILDREN" : 2000,  # Note that this is multiplied with ES_CENTRAL_NUM_EVALUATIONS
        #"FILL_MAP_WITH_OFFSPRING_MEASURE_EVERY_N_GENERATIONS" : 100,  
        
        #"CHECKPOINT_FREQUENCY" : 100,
        #"PLOT_FREQUENCY" : 100,
    }

    print("Initializing wandb")
    config = submission_common.setup_wandb_jax(config_defaults,project_name="JAX_ES_SINGLE")
    print("Initializing wandb DONE!!!")
    
    jax_simple_es_train.train(config)
    
    


if __name__ == '__main__':  
    run_es_experiment()






