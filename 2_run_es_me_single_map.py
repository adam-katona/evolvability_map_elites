import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np

import wandb
from es_map import submission_common

    
    
if __name__ == '__main__':    
        
    sweep_configuration = {
        "name" : "es_single_map_sweep",
        "program" : "/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/run_es_map.py",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "ES_UPDATES_MODES_TO_USE" : {
                "values" :  [ 
                    ["fitness"],
            #        ["evolvability"],
            #        ["innovation"],
            #        ["fitness","evolvability"],
            #        ["fitness","innovation"],
            #        #["evolvability","innovation"],
            #        ["fitness","evolvability","innovation"],
                ],
            },
            "env_args" : {
                "values" : [
                    {"use_norm_obs" : True,},
                    #{"use_norm_obs" : False,},    
                ],
            },    

            "env_id" : {
                "values" : [
                    #"DamageAnt-v2",
                    "QDAntBulletEnv-v0",
                    "QDWalker2DBulletEnv-v0",
                    "QDHalfCheetahBulletEnv-v0",
                    "QDHopperBulletEnv-v0",
                ]
            }
        },
    }

    #run_es_experiment()
    #import sys
    #sys.quit()

    print("Starting SWEEP!!!")
    sweep_id = wandb.sweep(sweep_configuration)
    print("Sweep started!   Starting agent now...")
    #wandb.agent(sweep_id, function=run_es_experiment)
    #print("Agent Done!!!")