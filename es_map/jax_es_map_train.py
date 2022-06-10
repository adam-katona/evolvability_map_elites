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

from es_map import custom_configs


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
#                    on selection, we do one big nd sort of the whole map






def calculate_lineage_stats(lineage_data):
    child_dict = {}
    for data in lineage_data:
        parent_ID = data["parent_ID"]
        if parent_ID in child_dict:
            child_dict[parent_ID].append(data)
        else:
            child_dict[parent_ID] = [data]  
    
    number_of_child = [len(children) for parent_id,children in child_dict.items()]
    lineage_lengths = [data["lineage_length_before_individual"] for data in lineage_data]
    
    return {
        "mean_num_child" : np.mean(number_of_child),
        "max_num_child" : np.max(number_of_child),
        "mean_lineage_length" : np.mean(lineage_lengths),
        "max_lineage_length" : np.max(lineage_lengths),
    }
    
def calculate_evolvability_diversity(b_map_evolver,evolver_config,evolvability_type):
    
    # we need to get the best evolvability in each non empty cell, and sum them
    # If it is a single map, we just grab the evolvability
    # If it is nd sorted map, we get the highest evolvability from the nd front
    # If it is multimap, we get the highest evolvability out of the each metric (we get this, even if evolvavility is not a metric used)
    
    if evolver_config["BMAP_type_and_metrics"]["type"] == "single_map":
        non_empty_cells = b_map_evolver.get_non_empty_cells()
        return np.sum([cell["elite"][evolvability_type] for cell in non_empty_cells])
        
    elif evolver_config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        non_empty_cells = b_map_evolver.get_non_empty_cells()
        evolvabilities = []
        for cell in non_empty_cells:
            cell_evolvabilities = [elite[evolvability_type] for elite in cell["elites"]]
            evolvabilities.append(np.max(cell_evolvabilities))
        return np.sum(evolvabilities)
        
    elif evolver_config["BMAP_type_and_metrics"]["type"] == "multi_map":
        return b_map_evolver.get_ed_score(evolvability_type)
        
    else:
        raise "Unknown bmap type in evolvability diversity calculation"
    


# One detail is that every time we compare innovation, we need to recalculate it for the current archive.
def update_innovation_of_map(non_empty_cells,config,b_archive_array,batch_calculate_novelty_fn):

    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        individuals = [cell["elite"] for cell in non_empty_cells]
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        individuals = []
        for cell in non_empty_cells:
            individuals.extend(cell["elites"])
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        individuals = [cell["elite"] for cell in non_empty_cells]
        
    for individual in individuals:
        child_bds = jnp.array(individual["child_eval"]["rollout_results"]["bds"][:config["ES_popsize"]])
        child_novelties = batch_calculate_novelty_fn(child_bds,b_archive_array)
        innovation = jnp.mean(child_novelties).item() # innovation is excpected novelty
        individual["innovation"] = innovation


def select_parent_from_single_map(b_map,config,b_archive_array,batch_calculate_novelty_fn):
    non_empty_cells = b_map.get_non_empty_cells()
    if config["ES_PARENT_SELECTION_MODE"] == "uniform":
        parent_cell = np.random.choice(non_empty_cells)  
    elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
        # use the metric of the map, to do ranked selection
        metric = config["BMAP_type_and_metrics"]["metrics"][0] # single_map have 1 metric
        if metric == "innovation":
            update_innovation_of_map(non_empty_cells,config,b_archive_array,batch_calculate_novelty_fn)
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
    
    # We select a metric randomly, it is complately independent from the es_update_mode we use.
    selected_metric = np.random.choice(config["BMAP_type_and_metrics"]["metrics"])
    non_empty_cells = b_map.get_non_empty_cells(selected_metric)
    
    if config["ES_PARENT_SELECTION_MODE"] == "uniform":
        parent_cell = np.random.choice(non_empty_cells) 
        
    elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
        # TODO if selected metric is innovation, we actually need to recalcualte it
        if selected_metric == "innovation":
            update_innovation_of_map(non_empty_cells,config,b_archive_array,batch_calculate_novelty_fn)
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
        selected_individual = np.random.choice(selected_cell["elites"]) # select randomly from the front
    elif config["ES_PARENT_SELECTION_MODE"] == "rank_proportional":
        if "innovation" in config["BMAP_type_and_metrics"]["metrics"]:
            update_innovation_of_map(non_empty_cells,config,b_archive_array,batch_calculate_novelty_fn)
        # We could do a nondominated sort on the combined elites of every cell, and use the nd rank
        # Let us do that
        # The complexity of sorting is n^2, so we might have problem for large maps.
        # Especially, because each cell can have multiple elites.
        # If we are below few 10K we are ok, our biggest map is 1296, so we are very OK.
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
        # nondominated sort sorts, so best is first. 
        # rank_based_selection assumes that last is best, so let us reverse the order
        sorted_indicies = sorted_indicies[::-1]
        
        selected_index = map_elite_utils.rank_based_selection(num_parent_candidates=len(sorted_indicies),
                                                num_children=None, # only want a single parent
                                                agressiveness=config["ES_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
        
        selected_indicies = all_elite_indicies[sorted_indicies[selected_index]]
        cell_i,elite_i = selected_indicies
        selected_individual = non_empty_cells[cell_i]["elites"][elite_i]
        
    return selected_individual
    
    
    

def select_parent_from_map(b_map,config,b_archive_array,batch_calculate_novelty_fn):
    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        return select_parent_from_single_map(b_map,config,b_archive_array,batch_calculate_novelty_fn)
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        return select_parent_from_multi_map(b_map,config,b_archive_array,batch_calculate_novelty_fn)
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        return select_parent_from_nd_sorted_map(b_map,config,b_archive_array,batch_calculate_novelty_fn)
    else:
        raise "unknown BMAP_type"


    


def train(config,wandb_logging=True):
    
    if type(config) != type(dict()):
        config = config.as_dict()
    
    if "config_index" in config:
        config = custom_configs.get_config_from_index(config,config["config_index"])
    
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
        
    population_size = config["ES_popsize"]
    get_next_individual_id = map_elite_utils.create_id_generator()
        
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
        "ID" : get_next_individual_id(),
        "parent_ID" : None,
        "generation_created" : 0,
        "lineage_length_before_individual" : 0,
        "params" : first_parent_params,
        "child_eval" : None,
    }
    
    b_archive = [] # we append bcs, and do: b_archive_array = jnp.stack(b_archive)   
    b_archive_array = None
    evolver_config = copy.deepcopy(config) 
    evolver_bd_descriptor = brax_envs.env_to_evolver_bd_descriptor(evolver_config["env_name"],evolver_config["env_mode"])
    evolver_config["map_elites_grid_description"] = evolver_bd_descriptor
    b_map_evolver = behavior_map.create_b_map_grid(evolver_config)
    
    # create a b_map for performance only. We will try to insert every individual into this map as well.
    # for this create a config which which we use when calling functions which try to insert individuals into it.
    perf_config = copy.deepcopy(config) 
    perf_config["BMAP_type_and_metrics"]["type"] = "single_map"
    perf_config["BMAP_type_and_metrics"]["metrics"] = ["eval_fitness"]
    b_map_performance = behavior_map.create_b_map_grid(perf_config)
    

    observation_stats = {                  # this is a single obs stats to keep track of during the whole experiment. 
        "sum" : np.zeros(env.observation_space.shape[1]),       # This is always expanded, and always used to calculate the current mean and std
        "sumsq" : np.zeros(env.observation_space.shape[1]),
        "count" : 0,
    }

    generation_number = 0
    
    best_ever_eval_normal_fitness = -99999999999.0
    best_ever_eval_control_cost = -99999999999.0
    best_ever_eval_distance_walked = -99999999999.0
    
    best_ever_child_mean_normal_fitness = -99999999999.0
    best_ever_child_mean_control_cost = -99999999999.0
    best_ever_child_mean_distance_walked = -99999999999.0
    
    best_ever_child_normal_fitness = -99999999999.0
    best_ever_child_control_cost = -99999999999.0
    best_ever_child_distance_walked = -99999999999.0
    
    best_ever_evo_var = 0.0
    best_ever_evo_ent = 0.0
    
    
    
    # phylogeny_data format: (parent_id,child_id,generation_number)
    # phylogeny_data = []  # Not doing it, maybe in the future
    
    all_step_logs = []
    
    # lineage data is stored as a list:
    # {
    #  "ID" :  
    #  "parent_ID"
    #  "generation_created"  
    #  "lineage_length_before_individual" 
    # }
    # I store the lineage data because i want to answer questions like:
    # How long is the longest lineage (average)
    # How many times a parent is selected (what is the largest / average forking)
    # Maybe i can even plot the whole thing (would be fun)
    # What else? Any other data needs saving?
    # I save the lineage data separatly because i want to remember the whole thing, not just the elites in the map
    lineage_data = []
    

    while True:
        if generation_number >= config["ES_NUM_GENERATIONS"]:
            print("Done, reached iteration: ",config["ES_NUM_GENERATIONS"])
            break


        ##############################################
        ## Populate the map with random individuals ##
        ##############################################
        
        # Only on first iteration
        # Let us skip this step, and start with a single random individual
        
        
        ######################################
        ## PARENT AND UPDATE MODE SELECTION ##
        ######################################
        
        # Choose an update mode
        es_update_mode = np.random.choice(config["ES_UPDATES_MODES_TO_USE"])
        
        # Choose a parent
        if generation_number == 0:
            current_idividual = first_parent
        else:
            current_idividual = select_parent_from_map(b_map_evolver,evolver_config,b_archive_array,batch_calculate_novelty_fn)
        
        
        ##################
        ## UPDATE STEPS ##
        ##################
        optimizer_state = None # Whenever a new parent is selected, we reset optimizer state
        for update_step_i in range(config["ES_STEPS_UNTIL_NEW_PARENT_SELECTION"]):
        
            #######################
            ## EVALUATE CHILDREN ##
            #######################
            
            # Only needed if we dont have cached evaluations already (first parent)
            if current_idividual["child_eval"] is None:
                key, key_create_pop = jax.random.split(key, 2)
                all_params,perturbations = jax_es_update.jax_es_create_population(
                                                                        current_idividual["params"],key_create_pop,
                                                                        popsize=config["ES_popsize"],
                                                                        eval_batch_size=config["ES_CENTRAL_NUM_EVALUATIONS"],
                                                                        sigma=config["ES_sigma"])
                all_model_params = batch_vec_to_params(all_params,shapes,indicies)
                
                rollout_results,new_obs_stats = jax_evaluate.rollout_episodes(env,all_model_params,observation_stats,config,
                                                                                    batch_model_apply_fn)
                observation_stats = new_obs_stats 
                mean_eval_bd = jnp.mean(rollout_results["bds"][population_size:],axis=0) # for the first parent, add it to the archive
                b_archive.append(mean_eval_bd)
                
                current_idividual["child_eval"] = {
                    "rollout_results" : rollout_results,
                    "key_create_pop" : key_create_pop,
                }
            else:
                # recreate perturbations from random seed
                perturbations = jax_es_update.jax_create_perturbations(key=current_idividual["child_eval"]["key_create_pop"],
                                                                       popsize=config["ES_popsize"],
                                                                       params_shape=current_idividual["params"].shape)
            
            ##################
            ## DO ES UPDATE ##
            ##################       
            child_bds = rollout_results["bds"][:population_size]
            child_fitness = rollout_results["fitnesses"][:population_size]
            
            b_archive_array = jnp.stack(b_archive)
            child_novelties = batch_calculate_novelty_fn(child_bds,b_archive_array)

        
            grad = jax_es_update.jax_calculate_gradient(perturbations=perturbations,
                                                child_fitness=child_fitness,
                                                bds=child_bds,config=config,
                                                mode=es_update_mode,novelties=child_novelties)
            grad = torch.from_numpy(np.array(grad))
            updated_params,optimizer_state = jax_es_update.do_gradient_step(current_idividual["params"],grad,optimizer_state,config)
            
            ##################################
            ## EVALUATE CHILDREN OF UPDATED ##
            ##################################
            key, key_create_pop = jax.random.split(key, 2)
            all_params,perturbations = jax_es_update.jax_es_create_population(
                                                                    updated_params,key_create_pop,
                                                                    popsize=config["ES_popsize"],
                                                                    eval_batch_size=config["ES_CENTRAL_NUM_EVALUATIONS"],
                                                                    sigma=config["ES_sigma"])
            all_model_params = batch_vec_to_params(all_params,shapes,indicies)
            
            rollout_results,new_obs_stats = jax_evaluate.rollout_episodes(env,all_model_params,observation_stats,config,
                                                                                batch_model_apply_fn)
            observation_stats = new_obs_stats 
            
            fitness = rollout_results["fitnesses"]
            bds = rollout_results["bds"]
            
            child_bds = bds[:population_size]
            eval_bds = bds[population_size:]
            child_fitness = fitness[:population_size]
            eval_fitness = fitness[population_size:]
            child_novelties = batch_calculate_novelty_fn(rollout_results["bds"],b_archive_array)
            
            mean_eval_bd = jnp.mean(eval_bds,axis=0)
            mean_eval_fitness = jnp.mean(eval_fitness).item()
            
            excpected_fitness = jnp.mean(child_fitness).item()
            innovation = jnp.mean(child_novelties).item() # innovation is excpected novelty
            evo_var = jax_evaluate.calculate_evo_var(bds)
            evo_ent = jax_evaluate.calculate_evo_ent(bds,config)
        
            updated_individual = {
                
                "ID" : get_next_individual_id(),
                "parent_ID" : current_idividual["ID"],
                "generation_created" : generation_number,
                "lineage_length_before_individual" : current_idividual["lineage_length_before_individual"]+1,
                
                "params" : updated_params,
                "child_eval" : {
                    "rollout_results" : rollout_results,
                    "key_create_pop" : key_create_pop,
                },
                "eval_fitness" : mean_eval_fitness,
                "eval_bc" : mean_eval_bd,
                    
                "excpected_fitness" : excpected_fitness,
                "innovation" : innovation,
                "evo_var" : evo_var,
                "evo_ent" : evo_ent,
            }
            
            lineage_data.append({
                "ID" : updated_individual["ID"],
                "parent_ID" : updated_individual["parent_ID"],
                "generation_created" : updated_individual["generation_created"],
                "lineage_length_before_individual" : updated_individual["lineage_length_before_individual"],
            })
            
            ##############################
            ## TRY INSERT INTO ARCHIVES ##
            ##############################
            
            # Try to insert into both b_map_evolver and b_map_performance
            map_elite_utils.try_insert_individual_in_map(updated_individual,b_map_evolver,b_archive=b_archive_array,
                                                         config=evolver_config,batch_calculate_novelty_fn=batch_calculate_novelty_fn)
            map_elite_utils.try_insert_individual_in_map(updated_individual,b_map_performance,b_archive=None,
                                                         config=perf_config,batch_calculate_novelty_fn=batch_calculate_novelty_fn)
            
            # Only add the archive after i calculated all the novelties and stuff for this generation
            b_archive.append(mean_eval_bd)
            
            ################################
            ## PREPARE FOR NEXT ITERATION ##
            ################################
            current_idividual = updated_individual
            
            ########################
            ## EVERY STEP LOGGING ##
            ########################
            
            # stats from current individual
            normal_fitness = rollout_results["normal_fitness"]
            distance_walked = rollout_results["distance_walked"]
            control_cost = rollout_results["control_cost_fitness"]
            
            mean_dist = jnp.mean(distance_walked[:population_size])
            max_dist = jnp.max(distance_walked[:population_size])
            eval_mean_dist = jnp.mean(distance_walked[population_size:])
            
            mean_normal_fitness = jnp.mean(normal_fitness[:population_size])
            max_normal_fitness = jnp.max(normal_fitness[:population_size])
            eval_normal_fitness = jnp.mean(normal_fitness[population_size:])
            
            mean_control_cost = jnp.mean(control_cost[:population_size])
            max_control_cost = jnp.max(control_cost[:population_size])
            eval_control_cost = jnp.mean(control_cost[population_size:])
            
            current_idividual["eval_mean_dist"] = eval_mean_dist
            current_idividual["eval_normal_fitness"] = eval_normal_fitness
            current_idividual["eval_control_cost"] = eval_control_cost
            
            
            if evo_var > best_ever_evo_var:
                best_ever_evo_var = evo_var
            if evo_ent > best_ever_evo_ent:
                best_ever_evo_var = evo_ent
            
            if eval_normal_fitness > best_ever_eval_normal_fitness:
                best_ever_eval_normal_fitness = eval_normal_fitness
            if eval_control_cost > best_ever_eval_control_cost:
                best_ever_eval_control_cost = eval_control_cost
            if eval_mean_dist > best_ever_eval_distance_walked:
                best_ever_eval_distance_walked = eval_mean_dist
                
            if  mean_normal_fitness > best_ever_child_mean_normal_fitness:
                best_ever_child_mean_normal_fitness = mean_normal_fitness
            if  mean_control_cost > best_ever_child_mean_control_cost:
                best_ever_child_mean_control_cost = mean_control_cost
            if  mean_dist > best_ever_child_mean_distance_walked:
                best_ever_child_mean_distance_walked = mean_dist
                
            if  max_normal_fitness > best_ever_child_normal_fitness:
                best_ever_child_normal_fitness = max_normal_fitness
            if  max_control_cost > best_ever_child_control_cost:
                best_ever_child_control_cost = max_control_cost
            if  max_dist > best_ever_child_distance_walked:
                best_ever_child_distance_walked = max_dist
                
            print(generation_number,eval_normal_fitness,eval_mean_dist,max_dist)
            
            # get stats from performance map
            perf_non_empty_cells = b_map_performance.get_non_empty_cells()  # for logging use the fitness map
            perf_b_map_size = b_map_performance.data.size
            perf_nonempty_ratio = len(perf_non_empty_cells) / float(perf_b_map_size)
            perf_qd_score = np.sum([cell["elite"]["eval_fitness"] for cell in perf_non_empty_cells])
            perf_qd_normal_fitness = np.sum([cell["elite"]["eval_normal_fitness"] for cell in perf_non_empty_cells])
            perf_qd_control_cost = np.sum([cell["elite"]["eval_control_cost"] for cell in perf_non_empty_cells])
            perf_qd_mean_dist = np.sum([cell["elite"]["eval_mean_dist"] for cell in perf_non_empty_cells])
            
            # ed stands for evolvability diversity 
            evolver_ed_var_score = calculate_evolvability_diversity(b_map_evolver,evolver_config,"evo_var")
            evolver_ed_ent_score = calculate_evolvability_diversity(b_map_evolver,evolver_config,"evo_ent")
            
            # Lineage data calculation
            lineage_stats = calculate_lineage_stats(lineage_data)
            
            # get stats from evolvbility map
            # I dont know if i am particularly interseted in this stuff...
            # We can do the final analysis after the run anyway, just not the learning curves.
            
            # IMPORTANT Metrics
            # Best ever fitness DONE
            # Best ever eval fitness DONE
            # Best ever evolvability DONE
            # QD DONE
            # ED DONE
            
            
            step_logs = {
                "generation_number" : generation_number,
                #"evaluations_so_far" : generation_number*config["ES_popsize"],

                # Log the 3 kind of fitness
                "mean_normal_fitness" : mean_normal_fitness,
                "eval_normal_fitness" : eval_normal_fitness,
                
                "mean_control_cost" : mean_control_cost,
                "max_control_cost" : max_control_cost,
                "eval_control_cost" : eval_control_cost,
                
                "mean_dist" : mean_dist,
                "max_dist" : max_dist,
                "eval_mean_dist" : eval_mean_dist,

                # we dont log the bd, because it is multi dimensinal.
                # but we log entropy, variance and innovation    
                "excpected_fitness" : excpected_fitness,        
                "innovation" : innovation,
                "evo_var" : evo_var,
                "evo_ent" : evo_ent,
            
                # best ever stats
                "best_ever_eval_normal_fitness" : best_ever_eval_normal_fitness,
                "best_ever_eval_control_cost" : best_ever_eval_control_cost,
                "best_ever_eval_distance_walked" : best_ever_eval_distance_walked,
                
                "best_ever_child_mean_normal_fitness" : best_ever_child_mean_normal_fitness,
                "best_ever_child_mean_control_cost" : best_ever_child_mean_control_cost,
                "best_ever_child_mean_distance_walked" : best_ever_child_mean_distance_walked,
                
                "best_ever_child_normal_fitness" : best_ever_child_normal_fitness,
                "best_ever_child_control_cost" : best_ever_child_control_cost,
                "best_ever_child_distance_walked" : best_ever_child_distance_walked,
                
                "best_ever_evo_var" : best_ever_evo_var,
                "best_ever_evo_ent" : best_ever_evo_ent,
            
                "perf_nonempty_ratio" : perf_nonempty_ratio,
                "perf_qd_score" : perf_qd_score,
                "perf_qd_normal_fitness" : perf_qd_normal_fitness,
                "perf_qd_control_cost" : perf_qd_control_cost,
                "perf_qd_mean_dist" : perf_qd_mean_dist,
                
                # evolver map stats
                "evolver_ed_var_score" : evolver_ed_var_score,
                "evolver_ed_ent_score" : evolver_ed_ent_score,
                
                # lineage stats
                "lineage_mean_num_child" : lineage_stats["mean_num_child"],
                "lineage_max_num_child" : lineage_stats["max_num_child"],
                "mean_lineage_length" : lineage_stats["mean_lineage_length"],
                "max_lineage_length" : lineage_stats["max_lineage_length"],        
            }
            if wandb_logging is True:   
                wandb.log(step_logs)   
            
            all_step_logs.append(step_logs)
            
            ####################
            ## N STEP LOGGING ##
            ####################

            # No N step logging, but we can print some stuff
            #if generation_number % config["PLOT_FREQUENCY"] == 10:
            #    print(generation_number)
            #    for k,v in step_logs.items():
            #        print(k,v)
            
            ###################
            ## CHECKPOINTING ##
            ###################
            # Dont really do checkpointing any more, final sive is enough
            
            generation_number += 1
            
            
    ################
    ## FINAL SAVE ##
    ################
    np.save(run_checkpoint_path+"/b_map_evolver.npy",b_map_evolver.data,allow_pickle=True)
    np.save(run_checkpoint_path+"/b_map_performance.npy",b_map_performance.data,allow_pickle=True)
    b_archive_array = np.stack(b_archive)  
    np.save(run_checkpoint_path+"/b_archive.npy",b_archive_array)
    with open(run_checkpoint_path+'/all_step_logs.pickle', 'wb') as handle:
        pickle.dump(all_step_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_checkpoint_path+'/lineage_data.pickle', 'wb') as handle:
        pickle.dump(lineage_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    final_rollout_arrays = {name : np.array(arr) for name,arr in rollout_results.items()}
    np.savez(run_checkpoint_path+"/final_rollout_arrays.npz",final_rollout_arrays)




















