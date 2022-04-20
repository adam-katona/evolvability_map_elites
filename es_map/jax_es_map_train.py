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
from es_map.my_brax_envs import brax_envs

from es_map import submission_common
from es_map import behavior_map
from es_map import map_elite_utils
from es_map import behavior_map
from es_map import nd_sort


# One of the contribution of evolvability map elites
# is to actually select for evolvability or innovation and even excpected fitness
# Because previous work always selected for fitness. I understand why, 
# because they actually want elite performers, not elite evolvers
# But since we select for elite evolvers, we lose the ability to find the elite performers/ For this reason we introduce 2 maps
# One is for selecting parents, the other is for finding the elite performers.
# elite_performance_map: is used for remembering the elite performers
# elite_evolver_map (b_map): used for selecting perents, contains the elite evolvers.

# INSERTION DECISION
# Elite performance map works like previous map elite maps, always make insertion decision based on eval_fitness
# For Elite evolver map, there are all kinds of options for the inserition decision, but it is based on 3 values
# - excepcted fitness of offspring
# - excpected novelty of offspring
# - entropy of offspring

# Also elite_evolver_map have 3 kinds
# - single map -> use single metric
# - multi map -> use multiple metric,
#                each metric have a separate single map
#                on insertion, we try inserting in each map
#                on selection, we can select from each map
# - nd_sorted map -> multiple metric, each cell in map contains a nondominated front of elites
#                    on insertion, we sort elites and the candidate, and keep the first n
#                    on selection, we select from the front.









# One detail is that every time we compare innovcation, we need to recalculate it for the current archive.
def update_innovation_of_cells(cells,b_archive_array,batch_calculate_novelty_fn):
    for cell in cells:
        child_bds = jnp.array(cell["child_eval"]["bcs"])
        child_novelties = batch_calculate_novelty_fn(child_bds,b_archive_array)
        innovation = jnp.mean(child_novelties).item() # innovation is excpected novelty
        cell["innovation"] = innovation


def select_parent_from_single_map(b_map,config,b_archive_array,batch_calculate_novelty_fn):
    non_empty_cells = b_map.get_non_empty_cells()
    if config["ES_PARENT_SELECTION_MODE"] == "uniform":
        parent_cell = np.random.choice(non_empty_cells)  # NOTE, here goes cell selection method
    elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
        # use the metric of the map, to do ranked selection
        metric = config["BMAP_type_and_metrics"]["metrics"][0] # single_map have 1 metric
        if metric == "innovation":
            update_innovation_of_cells(non_empty_cells,b_archive_array,batch_calculate_novelty_fn)
        sorted_cells = sorted(non_empty_cells,key=lambda x : x["elite"][metric])                
        selected_index = map_elite_utils.rank_based_selection(num_parent_candidates=len(sorted_cells),
                                                num_children=None, # only want a single parent
                                                agressiveness=config["ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
        parent_cell = sorted_cells[selected_index]
    return parent_cell["elite"]

def select_parent_from_multi_map(b_map,config,b_archive_array,batch_calculate_novelty_fn):
    # Here we could use selected_update_mode, and try to select a metric which plays well with selected_update_mode
    # For example we we are updating for evolvability, maybe select high evolvability parents...
    # But for 2 reasons we dont do this
    # Reason 1: we want to benefit from synergies, maybe high fitness will lead to high evolvability, etc...
    # Reason 2: If we dont have an update mode for it, we will never select it, so it is pointless to have that map...
    
    selected_metric = np.random.choice(config["BMAP_type_and_metrics"]["metrics"])
    non_empty_cells = b_map.get_non_empty_cells(selected_metric)
    
    if config["ES_PARENT_SELECTION_MODE"] == "uniform":
        parent_cell = np.random.choice(non_empty_cells)  # NOTE, here goes cell selection method
        
    elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
        # TODO if selected metric is innovation, we actually need to recalcualte it
        if selected_metric == "innovation":
            update_innovation_of_cells(non_empty_cells,b_archive_array,batch_calculate_novelty_fn)
        sorted_cells = sorted(non_empty_cells,key=lambda x : x["elite"][selected_metric])                
        selected_index = map_elite_utils.rank_based_selection(num_parent_candidates=len(sorted_cells),
                                                num_children=None, # only want a single parent
                                                agressiveness=config["ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
        parent_cell = sorted_cells[selected_index]
    return parent_cell["elite"]

    

def select_parent_from_nd_sorted_map(b_map,config,b_archive_array,batch_calculate_novelty_fn):
    
    non_empty_cells = b_map.get_non_empty_cells()
    if config["ES_PARENT_SELECTION_MODE"] == "uniform":
        selected_cell = np.random.choice(non_empty_cells)
        selected_individual = np.random.choice(selected_cell["elites"])
    elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
        if "innovation" in config["BMAP_type_and_metrics"]["metrics"]:
            update_innovation_of_cells(non_empty_cells,b_archive_array,batch_calculate_novelty_fn)
        # We could do a nondominated sort on the combined elites of every cell, and use the nd rank
        # Let us do that
        all_elite_objectives = []
        all_elite_indicies = []
        for cell_i,cell in enumerate(non_empty_cells):
            for elite_i,elite in enumerate(cell["elites"]):
                all_elite_indicies.append((cell_i,elite_i))
                objectives = []
                for metric in config["BMAP_type_and_metrics"]["metrics"]:
                    objectives.append(elite[metric])
                all_elite_objectives.append(np.array(objectives))
        
        all_elite_objectives = np.stack(all_elite_objectives)
        multi_objective_fitnesses = jnp.array(all_elite_objectives)
        domination_matrix = nd_sort.jax_calculate_domination_matrix(multi_objective_fitnesses)
        
        # copy back to cpu for rest of the computation
        domination_matrix = np.array(domination_matrix) 
        multi_objective_fitnesses = np.array(multi_objective_fitnesses)
         
        fronts = nd_sort.numba_calculate_pareto_fronts(domination_matrix)
        nondomination_rank_dict = nd_sort.fronts_to_nondomination_rank(fronts)
        crowding = nd_sort.calculate_crowding_metrics(multi_objective_fitnesses,fronts)
        sorted_indicies = nd_sort.nondominated_sort(nondomination_rank_dict,crowding)
        # nondominated sort sorts, so best is first. This is opposite to what we did in the other cases, reverse the indicies.
        sorted_indicies = sorted_indicies[::-1]
        
        selected_index = map_elite_utils.rank_based_selection(num_parent_candidates=len(sorted_indicies),
                                                num_children=None, # only want a single parent
                                                agressiveness=config["ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
        
        selected_indicies = all_elite_indicies[sorted_indicies[selected_index]]
        selected_cell = non_empty_cells[selected_indicies[0]]
        selected_individual = selected_cell["elites"][selected_indicies[1]]
        
    return selected_individual
    
    
    

def select_parent_from_map(b_map,config,b_archive_array,batch_calculate_novelty_fn):
    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        return select_parent_from_single_map(b_map,config,b_archive_array,batch_calculate_novelty_fn)
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        return select_parent_from_multi_map(b_map,config,b_archive_array,batch_calculate_novelty_fn)
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        return select_parent_from_nd_sorted_map(b_map,config,b_archive_array,batch_calculate_novelty_fn)
    else:
        raise "unknown BMAP_type"


    


def train(config,wandb_logging=True):
    
    if type(config) != type(dict()):
        config = config.as_dict()
    
    from es_map import jax_evaluate
    bd_descriptor = brax_envs.env_to_bd_descriptor(config["env_name"],config["env_mode"])
    config["map_elites_grid_description"] = bd_descriptor

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



        ##############################################
        ## Populate the map with random individuals ##
        ##############################################
        
        # Let us skip this step, and start with a single random individual
        
        
        
        ######################################
        ## PARENT AND UPDATE MODE SELECTION ##
        ######################################
        
        # Choose an update mode
        es_update_mode = np.random.choice(config["ES_UPDATES_MODES_TO_USE"])
        
        # Choose a parent
        if generation_number == 0:
            current_individual = first_parent
        else:
            if config["BMAP_type_and_metrics"]["type"] == "single_map":
                pass
            elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
                # how do we 
            elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
                pass
        
        
        ##################
        ## UPDATE STEPS ##
        ##################
        for update_step_i in range(config["ES_STEPS_UNTIL_NEW_PARENT_SELECTION"]):
        
            #######################
            ## EVALUATE CHILDREN ##
            #######################
            
            
            ##################
            ## DO ES UPDATE ##
            ##################
            
            
            ############################
            ## EVALUATE UPDATED THETA ##
            ############################
            
            ##################################
            ## EVALUATE CHILDREN OF UPDATED ##
            ##################################
            
            
            #########################
            ## INSERT INTO ARCHIVE ##
            #########################
            
            
            ################################
            ## PREPARE FOR NEXT ITERATION ##
            ################################
            
            
            ########################
            ## EVERY STEP LOGGING ##
            ########################
            
            ####################
            ## N STEP LOGGING ##
            ####################
            
            
            ###################
            ## CHECKPOINTING ##
            ###################
            # Dont really do checkpointing any more, final sive is enough
            
            generation_number += 1
            
            
        ################
        ## FINAL SAVE ##
        ################















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












