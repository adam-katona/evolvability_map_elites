import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np

import wandb
from es_map import submission_common

def run_es_experiment():
    
    config_defaults = {
        "env_id" : "QDAntBulletEnv-v0", # "DamageAnt-v2",
        "policy_args" : {
            "init" : "normc",
            "layers" :[128, 128],#[256, 256],
        "activation" : 'tanh',
        "action_noise" : 0.01,
        },
        "env_args" : {
            "use_norm_obs" : True,
        },
        
        "ALGORITHM_TYPE" : "MAP_ES",   # "PLAIN_ES", "MAP_ES", "MAP_GA"
        
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
        "BMAP_type_and_metrics" : {
            "type" : "multi_map",    # type can be: "single_map","multi_map","nd_sorted_map" 
            "metrics" : ["eval_fitness","evolvability"],    # metric can be: ["f"],["f",e], ["f,e,i"], etc... 
        },
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],#"evolvability","innovation"], # list of updates to use # TODO add QE
        "ES_PARENT_SELECTION_MODE" : "rank_proportional",  # "uniform", "rank_proportional"
        "ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS" : 1.0,  # 0.0 uniform, 1.0 normal , higher more agressive
        
        "ES_ND_SORT_MAX_FRONT_SIZE_TO_KEEP" : 6,
        "NOVELTY_CALCULATION_NUM_NEIGHBORS" : 10,
        
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