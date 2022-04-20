import numpy as np

import wandb
from es_map import submission_common


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



# Hipothesis 1:
# Selecting for evolvability instead of fitness will result in both higher evolvability and fitness in some envs
# This is tested with the basic single map stuff

# Hipothesis 2:
# Selecting for both is even better (either with mulit map or nd sorted map)

# Hipothesis 3:
# Which one is better multi map or nd sorted map





if __name__ == '__main__':    
        
    sweep_conf = {
            "name" : "jax_map_es",
            "program" : "/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/jax_run_es_map.py",
            "metric": {"name": "eval_fitness", "goal": "maximize"},
            "method": "grid",
            "parameters": {
                "RUN_ID" : {
                    "values" : [0,1]
                },
                "env_name" : {
                    "values" : [
                        "ant","humanoid", "walker", "hopper", "halfcheetah",
                    ],
                },
                "env_mode" : {
                    "values" : [
                        "NORMAL_CONTACT",#      same fitness as the original env, bd is foot contacts
                        "NORMAL_FINAL_POS",#    same fitness as the original env, bd is final pos
                        "DISTANCE_CONTACT",#    fitness is distance, bd is foot contact
                        "DISTANCE_FINAL_POS",#  fitness is distance, bd is final pos
                        "CONTROL_FINAL_POS",#   fitness is control cost only, bd is final pos  (for control we dont use foot contacts)
                        "DIRECTIONAL_CONTACT",#
                    ],
                },
                
                # We want to vary a few things
                # - 
                
                "ES_UPDATES_MODES_TO_USE" : {
                    "values" : [
                        ["fitness"],
                        ["evo_var"],
                        ["evo_ent"],
                        ["innovation"],
                        ["quality_evo_var"],
                        ["quality_evo_ent"],
                        ["quality_innovation"],
                        ["quality_evo_var_innovation"],
                        ["quality_evo_ent_innovation"],
                    ],
                },
            },
        }
    
    print("Starting SWEEP!!!")
    sweep_id = wandb.sweep(sweep_conf)
    print("Sweep started!   Starting agent now...")
    