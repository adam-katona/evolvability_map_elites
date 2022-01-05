import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np

import wandb
from es_map import submission_common

def run_es_experiment():
    
    config_defaults = {
        "env_id" : "DamageAnt-v2",
        "policy_args" : {
            "init" : "normc",
            "layers" :[256, 256],
        "activation" : 'tanh',
        "action_noise" : 0.01,
        },
        "env_args" : {
            "use_norm_obs" : True,
        },
        
        "ES_NUM_INITIAL_RANDOM_INDIVIDUALS_TO_POPULATE_MAP" : 20,
        
        "ES_NUM_GENERATIONS" : 1000,
        "ES_popsize" : 100,
        "ES_sigma" : 0.02,
        "ES_EVALUATION_BATCH_SIZE" : 5,
        "ES_lr" : 0.01,
        
        "ES_CENTRAL_NUM_EVALUATIONS" : 30,
        "ES_STEPS_UNTIL_NEW_PARENT_SELECTION" : 5,
        
        "GA_MAP_ELITES_NUM_GENERATIONS" : 1000,
        
        "GA_CHILDREN_PER_GENERATION" : 200,
        "GA_NUM_EVALUATIONS" : 10,
        
        "GA_MULTI_PARENT_MODE" : True,
        "GA_PARENT_SELECTION_MODE" : "rank_proportional",  # "uniform", "rank_proportional"
        "GA_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS" : 1.0,  # 0.0 uniform, 1.0 normal , higher more agressive
        "GA_MUTATION_POWER" : 0.02,
        
        "map_elites_grid_description" : {
            "bc_limits" : [[0,1],[0,1],[0,1],[0,1]],
            "grid_dims" : [6,6,6,6],
        },
        
        # BMAP settings
        "BMAP_type_and_metrics" : ["single_map",    # type can be: "single_map","multi_map","nd_sorted_map" 
                                ["fitness"]],    # metric can be: ["f"],["f",e], ["f,e,i"], etc... 
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],#"evolvability","innovation"], # list of updates to use
        "ES_PARENT_SELECTION_MODE" : "rank_proportional",  # "uniform", "rank_proportional"
        "ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS" : 1.0,  # 0.0 uniform, 1.0 normal , higher more agressive
        
        "NOVELTY_CALCULATION_NUM_NEIGHBORS" : 10,
        
        
        "CHECKPOINT_FREQUENCY" : 100,
        "PLOT_FREQUENCY" : 100,
    }
    
    print("Initializing wandb")
    config = submission_common.setup_wandb(config_defaults,project_name="ES_SINGLE")
    print("Initializing wandb DONE!!!")
    
    
    print("Setting up client")
    client = submission_common.set_up_dask_client()
    print("Client set up DONE!!!")
    
    print("Starting algorithm...")
    from es_map import es_map_elites
    es_map_elites.run_es_map_elites_single_map(client,config)
    print("ALGORITHM Finished properly")

    
    
if __name__ == '__main__':    
        
    sweep_configuration = {
        "name": "es_single_map_sweep",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "ES_UPDATES_MODES_TO_USE" : {
                "values" :  [ 
                    ["fitness"],
                    ["evolvability"],
                    ["innovation"],
                    ["fitness","evolvability"],
                    ["fitness","innovation"],
                    ["fitness","evolvability","innovation"],
                ],
            },
            "env_args" : {
                "values" : [
                    {"use_norm_obs" : True,},
                    {"use_norm_obs" : False,},    
                ],
            },    
            #"env_args.use_norm_obs" : {
            #    "values" : [ True,False],
            #},  
        }
    }

    print("Starting SWEEP!!!")
    sweep_id = wandb.sweep(sweep_configuration)
    print("Sweep started!   Starting agent now...")
    wandb.agent(sweep_id, function=run_es_experiment)
    print("Agent Done!!!")