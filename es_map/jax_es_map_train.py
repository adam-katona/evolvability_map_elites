import random
import numpy as np
import torch
import brax
from brax.envs import wrappers
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

import random
import numpy as np
import copy
import pickle
import json

import wandb

from es_map import jax_evaluate
from es_map import jax_es_update

from es_map import submission_common
from es_map import behavior_map
from es_map import map_elite_utils
from es_map import behavior_map
from es_map import nd_sort


def train(config,wandb_logging=True):

    print("staring run_es_map_elites")
    
    if type(config) != type(dict()):
        config = config.as_dict()
    
    from es_map import jax_evaluate
    config_defaults["map_elites_grid_description"] = jax_evaluate.brax_get_bc_descriptor_for_env(config["env_name"])

    if wandb_logging is True:   
        run_name = wandb.run.dir.split("/")[-2]
    else:
        from datetime import datetime
        run_name = "local_dubug_run_" + datetime.now().strftime("_%m_%d___%H:%M")
    run_checkpoint_path = "/scratch/ak1774/runs/large_files_jax/" + run_name

    # setup random seed
    seed = random.randint(0, 10000000000)
    print("starting run with seed: ",seed)
    config["seed"] = seed
    key = jax.random.PRNGKey(seed)
    key, key_init_model = jax.random.split(key, 2)
    
    # Dump config
    with open(run_checkpoint_path+'/config.json', 'w') as file:
        json.dump(config, file,indent=4)
        
    # setup env and model
    env = jax_evaluate.create_env(config["env_name"],config["env_mode"],config["ES_popsize"],
                                  config["ES_CENTRAL_NUM_EVALUATIONS"],config["episode_max_length"])
    model = jax_evaluate.create_MLP_model(env.observation_space.shape[1],env.action_space.shape[1])

    # setup batched functions
    batch_model_apply_fn = jax.jit(jax.vmap(model.apply))                                   # This does not work (hangs) if not jitted  WTF???
    batch_vec_to_params = jax.vmap(jax_evaluate.vec_to_params_tree,in_axes=[0, None,None])  # this does not work (hangs) when jitted    WTF???
    calculate_novelty_fn = jax_evaluate.get_calculate_novelty_fn(k=config["NOVELTY_CALCULATION_NUM_NEIGHBORS"])
    batch_calculate_novelty_fn = jax.jit(jax.vmap(calculate_novelty_fn,in_axes=[0, None]))

    # get initial parameters
    initial_model_params = model.init(key_init_model)
    vec,shapes,indicies = jax_evaluate.params_tree_to_vec(initial_model_params)
    
    # I use torch, because i want to use torch optimizer, and since it is so little part of the copmutation (the grad update compared to the evaluations and grad caluclations),
    # i dont care about it being slow, or needing copy to cpu
    first_parent_params = np.array(vec)
    first_parent = {
        "params" : first_parent_params,
        "child_eval" : None,
    }
    
    b_archive = [] # we append bcs, and do: b_archive_array = jnp.stack(b_archive)   
    b_map = behavior_map.create_b_map_grid(config)

    observation_stats = {                  # this is a single obs stats to keep track of during the whole experiment. 
        "sum" : np.zeros(env.observation_space.shape[1]),       # This is always expanded, and always used to calculate the current mean and std
        "sumsq" : np.zeros(env.observation_space.shape[1]),
        "count" : 0,
    }

    generation_number = 0
    evaluations_so_far = 0
    best_fitness_so_far = 0
    best_model_so_far = None
    best_evolvability_so_far = 0
    obs_stats_history = []

    while True:
        if generation_number >= config["ES_NUM_GENERATIONS"]:
            print("Done, reached iteration: ",config["ES_NUM_GENERATIONS"])
            break

        # NOTE we skip initial random population, we use a single individual as the first parent
        
        ######################################
        ## PARENT AND UPDATE MODE SELECTION ##
        ######################################
        # Choose an update mode
        es_update_mode = np.random.choice(config["ES_UPDATES_MODES_TO_USE"])
        if generation_number == 0:
            current_individual = first_parent
        else:
            if config["BMAP_type_and_metrics"]["type"] == "single_map":
                # choose parent
                non_empty_cells = b_map.get_non_empty_cells()
                
                if config["ES_PARENT_SELECTION_MODE"] == "uniform":
                    parent_cell = np.random.choice(non_empty_cells) 
                elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
                    # rank proportional means we rank based on the score
                    # What this score should be? Fitness?
                    # Do we have versions where we decide insertion based on evolvability or novelty when we have single map?
                    
                    sorted_cells = sorted(non_empty_cells,key=lambda x : x["elite"]["eval_fitness"])                
                    selected_index = map_elite_utils.rank_based_selection(num_parent_candidates=len(sorted_cells),
                                                            num_children=None, # only want a single parent
                                                            agressiveness=config["ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
                    parent_cell = sorted_cells[selected_index]
                current_individual = parent_cell["elite"]
            
            elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
                
            
            elif config["BMAP_type_and_metrics"]["type"] == "multi_map":



            # NOTE, rewrite loop with new rollout function












