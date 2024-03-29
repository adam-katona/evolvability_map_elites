
import copy
#from dask.distributed import Client
#import dask
import os

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np

import wandb
from es_map import submission_common

def run_ga_experiment():
    
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
        
        "ES_popsize" : 100,
        "ES_sigma" : 0.02,
        "ES_EVALUATION_BATCH_SIZE" : 5,
        "ES_lr" : 0.01,
        
        "ES_CENTRAL_NUM_EVALUATIONS" : 30,
        
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
        
        "CHECKPOINT_FREQUENCY" : 100,
        "PLOT_FREQUENCY" : 100,
    }
    
    print("Initializing wandb")
    config = submission_common.setup_wandb(config_defaults,project_name="GA")
    print("Initializing wandb DONE!!!")
    
    
    print("Setting up client")
    client = submission_common.set_up_dask_client()
    print("Client set up DONE!!!")
    
    print("Starting algorithm...")
    from es_map import ga_map_elites
    ga_map_elites.run_ga_map_elites(client,config)
    print("ALGORITHM Finished properly")




if __name__ == '__main__':
    
    sweep_configuration = {
        "name": "ga_sweep",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "GA_MULTI_PARENT_MODE": {
                "values": [True,False],
            },
            "GA_PARENT_SELECTION_MODE" : {
                "values": ["uniform","rank_proportional"],
            }
        }
    }
    
    print("Starting SWEEP!!!")
    sweep_id = wandb.sweep(sweep_configuration)
    print("Sweep started!   Starting agent now...")
    wandb.agent(sweep_id, function=run_ga_experiment)
    print("Agent Done!!!")
    
        
        
    #import sys
    #if len(sys.argv) > 1:
    #    import wandb
    #    print("starting agent")
    #    wandb.agent(sys.argv[1], function=run_single_experiment)
        
    #import os
    #os.chdir("/scratch/ak1774/runs")
    
    #TEST_RUN = False
    
    #if TEST_RUN is False:
    #    import wandb
    
    
    
    #    sweep_id = wandb.sweep(sweep_configuration)
        
        # run the sweep
    #    wandb.agent(sweep_id, function=run_single_experiment)
    
#    else:

#        print("setting up client")
#        client = set_up_dask_client(5)
#        print("client set up finished")
        
#        from es_map import map_elites
        
#        map_elites.run_ga_map_elites(client,ga_default_config,wandb_logging=False)
    
    
    # local run
    
    
    