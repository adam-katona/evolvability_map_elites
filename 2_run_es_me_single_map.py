import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np

import wandb
from es_map import submission_common

    
    
if __name__ == '__main__':    
        
    single_map_sweep_config = {
        "name" : "es_single_map_sweep",
        "program" : "/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/run_es_map.py",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "ALGORITHM_TYPE" : {
                "values" : ["MAP_ES"]
            },
            "ES_OPTIMIZER_TYPE" : {
                "values" : ["ADAM"]  
            },    
            
            "BMAP_type_and_metrics" : {
                "values" : [
                    {
                       "type" : "single_map",
                        "metrics" : ["eval_fitness"],
                    },
                    #{
                    #   "type" : "multi_map",
                    #    "metrics" : [],
                    #},
                    #{
                    #   "type" : "nd_sorted_map",
                    #    "metrics" : ["eval_fitness","innovation"],
                    #},
                    #{
                    #   "type" : "nd_sorted_map",
                    #    "metrics" : ["eval_fitness","evolvability"],
                    #},
                    #{
                    #   "type" : "nd_sorted_map",
                    #    "metrics" : ["eval_fitness","evolvability","innovation"],
                    #},
                ]
            },
            
            "ES_UPDATES_MODES_TO_USE" : {
                "values" :  [ 
                    ["fitness"],
            #        ["evolvability"],
            #        ["innovation"],
                    ["fitness","evolvability"],
                    ["fitness","innovation"],
            #        #["evolvability","innovation"],
                    ["fitness","evolvability","innovation"],
            #        ["fitness","evolvability","innovation","quality_evolvability","quality_innovation","quality_evolvability_innovation"], 
            
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
    
    
    
    
    combined_sweep_config = {
       "name" : "es_single_map_sweep",
        "program" : "/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/run_es_map.py",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "ALGORITHM_TYPE" : {
                "values" : ["MAP_ES"]
            },
            "ES_OPTIMIZER_TYPE" : {
                "values" : ["ADAM"]  
            },                      
            
            "BMAP_type_and_metrics" : {
                "values" : [
                    #{
                    #   "type" : "single_map",
                    #    "metrics" : ["eval_fitness"],
                    #},
                    #{
                    #   "type" : "multi_map",
                    #    "metrics" : [],
                    #},
                    #{
                    #   "type" : "nd_sorted_map",
                    #    "metrics" : ["eval_fitness","innovation"],
                    #},
                    #{
                    #   "type" : "nd_sorted_map",
                    #    "metrics" : ["eval_fitness","evolvability"],
                    #},
                    {
                       "type" : "nd_sorted_map",
                        "metrics" : ["eval_fitness","evolvability","innovation"],
                    },
                    {
                       "type" : "multi_map",
                        "metrics" : ["eval_fitness","evolvability","innovation"],
                    },
                ]
            },
            
        #"quality_evolvability" or 
        #"quality_innovation" or
        #"quality_evolvability_innovation"
            
            
            "ES_UPDATES_MODES_TO_USE" : {
                "values" :  [ 
            #        ["fitness"],
            #        ["evolvability"],
            #        ["innovation"],
            #        ["fitness","evolvability"],
            #        ["fitness","innovation"],
            #        #["evolvability","innovation"],
                     ["fitness","evolvability","innovation"],
                     #["quality_evolvability","quality_innovation","quality_evolvability_innovation"], 
                     ["fitness","evolvability","innovation","quality_evolvability","quality_innovation","quality_evolvability_innovation"], 
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
    
    
    
    multi_map_sweep_config = {
       "name" : "es_single_map_sweep",
        "program" : "/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/run_es_map.py",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "ALGORITHM_TYPE" : {
                "values" : ["MAP_ES"]
            },
            "ES_OPTIMIZER_TYPE" : {
                "values" : ["ADAM"]  
            },    
            
            "BMAP_type_and_metrics" : {
                "values" : [
                    #{
                    #   "type" : "single_map",
                    #    "metrics" : ["eval_fitness"],
                    #},
                    #{
                    #   "type" : "multi_map",
                    #    "metrics" : ["eval_fitness","innovation"],
                    #},
                    #{
                    #   "type" : "multi_map",
                    #    "metrics" : ["eval_fitness","evolvability"],
                    #},
                    {
                       "type" : "multi_map",
                        "metrics" : ["eval_fitness","evolvability","innovation"],
                    },
                ]
            },
          
            "ES_UPDATES_MODES_TO_USE" : {
                "values" :  [ 
            #        ["fitness"],
            #        ["evolvability"],
            #        ["innovation"],
            #        ["fitness","evolvability"],
            #        ["fitness","innovation"],
            #        #["evolvability","innovation"],
                    ["fitness","evolvability","innovation"],
            #        ["quality_evolvability"],
                    ["fitness","evolvability","innovation","quality_evolvability","quality_innovation","quality_evolvability_innovation"], 
            
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
    sweep_id = wandb.sweep(single_map_sweep_config)
    print("Sweep started!   Starting agent now...")
    #wandb.agent(sweep_id, function=run_es_experiment)
    #print("Agent Done!!!")