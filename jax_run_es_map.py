import os
import numpy as np

import wandb
from es_map import submission_common
from es_map import jax_es_map_train

def run_es_experiment():
    
    config_defaults = {
        "env_name" : "ant", #  ant, walker, hopper, halfcheetah, humanoid, ant_omni, humanoid_omni
        "env_mode" : "NORMAL_CONTACT",
        "episode_max_length" : 250,
        "env_deterministic" : True, 

        "ES_NUM_GENERATIONS" : 5000,  # was 1000
        "ES_popsize" : 10000,
        "ES_sigma" : 0.02,
        "ES_OPTIMIZER_TYPE" : "ADAM",
        "ES_lr" : 0.01,
        "ES_L2_COEFF" : 0.005,  
        
        "ES_CENTRAL_NUM_EVALUATIONS" : 100, # How many times central individual evaluated
        "ES_STEPS_UNTIL_NEW_PARENT_SELECTION" : 5,
        
        # Will randomly select an update mode each time a new parent is slelected
        "ES_UPDATES_MODES_TO_USE" : ["fitness"], # "fitness","evo_var","evo_ent","innovation",...
        
        "NOVELTY_CALCULATION_NUM_NEIGHBORS" : 10,
        "ENTROPY_CALCULATION_KERNEL_BANDWIDTH" : 0.25, # TODO maybe this should be different for final pos and foot contacts...
        "ES_ND_SORT_MAX_FRONT_SIZE_TO_KEEP" : 8,
        
        "CHECKPOINT_FREQUENCY" : 100,
        
        "BMAP_type_and_metrics" : {
            "type" : "single_map",
            "metrics" : ["eval_fitness"],
        },
        
        "ES_PARENT_SELECTION_MODE" : "rank_proportional",  # "uniform", "rank_proportional"
        "ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS" : 1.0,  # 0.0 uniform, 1.0 normal , higher more agressive
    }
    
    print("Initializing wandb")
    config = submission_common.setup_wandb_jax(config_defaults,project_name="JAX_ES_MAP")
    print("Initializing wandb DONE!!!")
    
    jax_es_map_train.train(config,wandb_logging=True)
    
    print("ALGORITHM Finished properly")
    
    
if __name__ == '__main__':  
    run_es_experiment()