import numpy as np
import copy

import wandb
from es_map import submission_common

from es_map import custom_configs
# Let us have a separate sweep for 
# - single map
# - multi map
# - nd_sorted map

# for each we want to vary env_name and env_mode

# For single map, we want to vary
# - update modes
# - metric  (insertion decision)
# - ES_PARENT_SELECTION_MODE

# For multi map
# - update modes
# - metrics
# - ES_PARENT_SELECTION_MODE

# For metrics in we probably want to test several conbinations
# - original stuff, select for eval fitness, with various update modes 
# - select for evolvability only

# Hipothesis 1:
# Selecting for evolvability instead of fitness will result in both higher evolvability and fitness in some envs
# This is tested with the basic single map stuff

# Hipothesis 2:
# Selecting for both is even better (either with mulit map or nd sorted map)

# Hipothesis 3:
# Which one is better multi map or nd sorted map


# I probably want to do update modes and map metric pairs.
# Or do I?






# The problem with sweeps is it is doing grid search
# So far i have been abusing the sweep to run the configs i want, but now let us switch to a config list
# I still want to use wandb so i can utilise the agents to run experiments in multiple machines.

# The way this will work is that i have a global list of configs
# The sweep varies the config id
# When the run is starting, i set the config from the global list







if __name__ == '__main__':    
            
    basic_sweep_conf = {
            "name" : "jax_map_es",
            "program" : "/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/jax_run_es_map.py",
            "metric": {"name": "max_dist", "goal": "maximize"},
            "method": "grid",
            "parameters": {
                "RUN_ID" : {
                    "values" : [0,1]
                },
                "env_name" : {
                    "values" : [
                        "ant",
                        "humanoid", 
                        #"walker", 
                        #"hopper", 
                        #"halfcheetah",
                    ],
                },
                "env_mode" : {
                    "values" : [
                        "NORMAL_CONTACT",#      same fitness as the original env, bd is foot contacts
                        #"NORMAL_FINAL_POS",#    same fitness as the original env, bd is final pos
                        #"DISTANCE_CONTACT",#    fitness is distance, bd is foot contact
                        #"DISTANCE_FINAL_POS",#  fitness is distance, bd is final pos
                        "CONTROL_FINAL_POS",#   fitness is control cost only, bd is final pos  (for control we dont use foot contacts)
                        #"DIRECTIONAL_CONTACT",#
                    ],
                },
                "ES_PARENT_SELECTION_MODE" : {
                    "values" : [
                        "rank_proportional",
                        #"uniform"
                    ],
                },
            },
        }
    
    custom_config_list_sweep = {
        "config_index" : {
            "values" : list(range(len(custom_configs.config_list)))
        },
    }
    custom_long_config_list_sweep = {
        "config_index" : {
            "values" : [1,3,7,8,10,11]
        },
        "config_list_name" : {
            "values" : ["default_list"]
        },
        "ES_NUM_GENERATIONS" : {
             "values" : [40000]
        }
    }
    custom_combined_config_list_sweep = {
        "config_index" : {
            "values" : list(range(len(custom_configs.combined_config_list)))
        },
        "config_list_name" : {
            "values" : ["combined_update_list"]
        },
    }
    
    single_map_sweep_parameters = {    
        "ES_UPDATES_MODES_TO_USE" : {
            "values" : [
                ["fitness"],
                #["evo_var"],
                ["evo_ent"],
                ["innovation"],
                #["quality_evo_var"],
                ["quality_evo_ent"],
                ["quality_innovation"],
                #["quality_evo_var_innovation"],
                ["quality_evo_ent_innovation"],
            ],
        },
        "BMAP_type_and_metrics" : {
            "values" : [
                {"type" : "single_map", "metrics" : ["eval_fitness"]},
                {"type" : "single_map", "metrics" : ["excpected_fitness"]},
                {"type" : "single_map", "metrics" : ["innovation"]},
                #{"type" : "single_map", "metrics" : ["evo_var"]},
                {"type" : "single_map", "metrics" : ["evo_ent"]},
                
            ]
        }
    }
    
    
    multi_map_sweep_parameters = {    
        "ES_UPDATES_MODES_TO_USE" : {
            "values" : [
                ["fitness"],
                ["evo_ent"],
                ["innovation"],
                ["fitness","evo_ent","innovation"],
                ["quality_evo_ent_innovation"],
            ],
        },
        "BMAP_type_and_metrics" : {
            "values" : [
                {"type" : "multi_map", "metrics" : ["excpected_fitness","evo_ent"]},
                {"type" : "multi_map", "metrics" : ["excpected_fitness","innovation"]},
                {"type" : "multi_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
            ]
        }
    }
    
    nd_sorted_map_sweep_parameters = {    
        "ES_UPDATES_MODES_TO_USE" : {
            "values" : [
                ["fitness"],
                ["evo_ent"],
                ["innovation"],
                ["fitness","evo_ent","innovation"],
                ["quality_evo_ent_innovation"],
            ],
        },
        "BMAP_type_and_metrics" : {
            "values" : [
                {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent"]},
                {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","innovation"]},
                {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
            ]
        }
    }
    
    
    single_map_sweep_conf = copy.deepcopy(basic_sweep_conf)
    single_map_sweep_conf["parameters"].update(single_map_sweep_parameters)
    
    multi_map_sweep_conf = copy.deepcopy(basic_sweep_conf)
    multi_map_sweep_conf["parameters"].update(multi_map_sweep_parameters)
    
    nd_sorted_map_sweep_conf = copy.deepcopy(basic_sweep_conf)
    nd_sorted_map_sweep_conf["parameters"].update(nd_sorted_map_sweep_parameters)
    
    custom_sweep_conf = copy.deepcopy(basic_sweep_conf)
    custom_sweep_conf["parameters"].update(custom_config_list_sweep)
    
    custom_long_sweep_conf = copy.deepcopy(basic_sweep_conf)
    custom_sweep_conf["parameters"].update(custom_long_config_list_sweep)
    
    custom_combined_sweep_conf = copy.deepcopy(basic_sweep_conf)
    custom_sweep_conf["parameters"].update(custom_combined_config_list_sweep)
    
    print("Starting SWEEP!!!")
    sweep_id = wandb.sweep(custom_long_sweep_conf)
    print("Sweep started!   Starting agent now...")
    