import numpy as np

import wandb
from es_map import submission_common


if __name__ == '__main__':    
        
    sweep_conf = {
            "name" : "jax_simple_es",
            "program" : "/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/jax_run_es.py",
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
    