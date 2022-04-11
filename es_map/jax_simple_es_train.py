

# A simple ES loop
# A training loop for plain es with jax
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

#from es_map import map_elite_utils
from es_map import jax_evaluate
from es_map import jax_es_update

import wandb

from es_map import submission_common
from es_map import behavior_map
from es_map import map_elite_utils

# stratefy for map fillup
# We have 2 cases
# Deterministic envs
# For deterministic we can use the child evaluations to fill the map.
# Do we want to do it every generation or like every 50 generation...
# Also i am interested in the cummulative map, and the map of the current individual, both of this should be measured


# Stochastic env
# For stochastic env, we need a lot of evaluations to determine bd
# For this we can sample offspring (evaluating each like 20 times)
# For this we cannot reuse the child evaluations, since those are only evaluated once
# We dont care about this case for now, NOT IMPLEMENTED


# In the case of deterministic, we can simply use the evaluations to fill up the map


# note that we olny do this with fitness maps, no need to recalculate innovation or stuff like that
# note that instead of sampling children, we just use the children used for calculating the grad
# we can do this because evaluating once is fine if env is deterministic
def test_elite_mapping_performance(b_map_cumm,child_fitness,child_bds,config):
    
    b_map = behavior_map.create_b_map_grid(config)
    
    fitnesses = np.array(child_fitness)
    bds = np.array(child_bds)
    
    #b_map_fillup_data = []
    #b_map_cumm_fillup_data = []
    #b_map_current_count = 0
    #b_map_cumm_current_count = np.sum(b_map_cumm.data != None)
    
    for i in range(child_fitness.shape[0]):
        
        individual = {
                "eval_fitness" : fitnesses[i],
                "eval_bc" : bds[i],
        }
        
        b_map_added = map_elite_utils.try_insert_individual_in_map(individual,b_map,b_archive=None,config=config)
        b_map_cumm_added = map_elite_utils.try_insert_individual_in_map(individual,b_map_cumm,b_archive=None,config=config)
        #b_map_current_count += int(b_map_added) 
        #b_map_cumm_current_count += int(b_map_cumm_added) 
        #b_map_fillup_data.append(b_map_current_count)
        #b_map_cumm_fillup_data.append(b_map_cumm_current_count)
        
    # Now calculate the QD scores and nonempty ratios for both the cummulative and fresh map
    non_empty_cells = b_map.get_non_empty_cells()
    qd_score = np.sum([cell["elite"]["eval_fitness"] for cell in non_empty_cells])
    cumm_non_empty_cells = b_map_cumm.get_non_empty_cells()
    cumm_qd_score = np.sum([cell["elite"]["eval_fitness"] for cell in cumm_non_empty_cells])
    
    # also do some plots
    #fig_f,ax = map_elite_utils.plot_b_map(b_map,metric="eval_fitness",config=config)
    #fig_f_cumm,ax = map_elite_utils.plot_b_map(b_map_cumm,metric="eval_fitness",config=config)             
    
    # also plot how fast they accumulate
    #b_map_fillup_data = np.array(b_map_fillup_data) / float(b_map.data.size)
    #b_map_cumm_fillup_data = np.array(b_map_cumm_fillup_data) / float(b_map.data.size)
    #fig_accumulation,ax = plt.subplots()
    #ax.plot(b_map_fillup_data)
    #ax.plot(b_map_cumm_fillup_data)
    
    results = {
        #"nonempty_cells" : len(non_empty_cells),
        "nonempty_ratio" : float(len(non_empty_cells)) / b_map.data.size,
        "qd_score" : qd_score,
        #"cumm_nonempty_cells" : len(cumm_non_empty_cells),
        "cumm_nonempty_ratio" : float(len(cumm_non_empty_cells)) / b_map_cumm.data.size,
        "cumm_qd_score" : cumm_qd_score,
        #"b_map_plot" : fig_f,
        #"b_map_cumm_plot" : fig_f_cumm,
        #"b_map_accumulation_plot" : fig_accumulation,
    }
    return results
    
    
    
    
    
    

def train(config,wandb_logging=True):

    if type(config) != type(dict()):
        config = config.as_dict()

    bd_descriptor = jax_evaluate.brax_get_bc_descriptor_for_env(config["env_name"])
    config["map_elites_grid_description"] = bd_descriptor

    env_name = config["env_name"]
    env_mode = config["env_mode"]
    population_size = config["ES_popsize"]

    
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
    env = jax_evaluate.create_env(env_name,env_mode,population_size,config["ES_CENTRAL_NUM_EVALUATIONS"],config["episode_max_length"])
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
    current_params = np.array(vec)

    b_archive = [] # we append bcs, and do: b_archive_array = jnp.stack(b_archive)   
    if config["PLAIN_ES_TEST_ELITE_MAPPING_PERFORMANCE"] is True:
        b_map_cumm = behavior_map.create_b_map_grid(config)
     
    all_step_logs = []
    optimizer_state = None
    observation_stats = {                  # this is a single obs stats to keep track of during the whole experiment. 
        "sum" : np.zeros(env.observation_space.shape[1]),       # This is always expanded, and always used to calculate the current mean and std
        "sumsq" : np.zeros(env.observation_space.shape[1]),
        "count" : 0,
    }
    
    generation_number = 0
    while True:
        if generation_number >= config["ES_NUM_GENERATIONS"]:
            print("Done, reached iteration: ",config["ES_NUM_GENERATIONS"])
            break

        print("starting iteration: ",generation_number)

        # Create pop
        key, key_create_pop = jax.random.split(key, 2)
        all_params,perturbations = jax_es_update.jax_es_create_population(current_params,key_create_pop,
                                                                          popsize=population_size,
                                                                          eval_batch_size=config["ES_CENTRAL_NUM_EVALUATIONS"],sigma=config["ES_sigma"])
        all_model_params = batch_vec_to_params(all_params,shapes,indicies)
        
        rollout_results,new_obs_stats = jax_evaluate.rollout_episodes(env,all_model_params,observation_stats,config,
                                                                            batch_model_apply_fn)
        observation_stats = new_obs_stats 
        
        # used for es updates
        fitness = rollout_results["fitnesses"]
        bds = rollout_results["bds"]
        
        child_bds = bds[:population_size]
        eval_bds = bds[population_size:]
        child_fitness = fitness[:population_size]
        eval_fitness = fitness[population_size:]
        

        # Calculate novelties, innovation end evolvabilities
        mean_eval_bd = jnp.mean(eval_bds,axis=0)
        b_archive.append(mean_eval_bd)
        
        b_archive_array = jnp.stack(b_archive)
        child_novelties = batch_calculate_novelty_fn(child_bds,b_archive_array)
        innovation = jnp.mean(child_novelties).item() # innovation is excpected novelty
        
        evo_var = jax_evaluate.calculate_evo_var(bds)
        evo_ent = jax_evaluate.calculate_evo_ent(bds,config)
        
        # Calculate mapping performance
        if config["PLAIN_ES_TEST_ELITE_MAPPING_PERFORMANCE"] is True:
            if generation_number % config["PLAIN_ES_TEST_ELITE_MAPPING_FREQUENCY"] == 0:
                map_perf = test_elite_mapping_performance(b_map_cumm,child_fitness,child_bds,config)
                wandb.log(map_perf,commit=False)   
                
        
        # Calculate grad and do grad update
        es_update_mode = config["ES_UPDATES_MODES_TO_USE"][0]
        grad = jax_es_update.jax_calculate_gradient(perturbations=perturbations,child_fitness=child_fitness,
                                            bds=child_bds,config=config,mode=es_update_mode,novelties=child_novelties)
        grad = torch.from_numpy(np.array(grad))
        updated_params,optimizer_state = jax_es_update.do_gradient_step(current_params,grad,optimizer_state,config)
        
        ########################
        ## LOGGING / PLOTTING ##
        ########################

        normal_fitness = rollout_results["normal_fitness"]
        distance_walked = rollout_results["distance_walked"]
        control_cost = rollout_results["control_cost_fitness"]
        
        mean_dist = jnp.mean(distance_walked[:population_size])
        max_dist = jnp.max(distance_walked[:population_size])
        eval_mean_dist = jnp.mean(distance_walked[population_size:])
        
        mean_normal_fitness = jnp.mean(normal_fitness[:population_size])
        eval_normal_fitness = jnp.mean(normal_fitness[population_size:])
        
        mean_control_cost = jnp.mean(control_cost[:population_size])
        max_control_cost = jnp.max(control_cost[:population_size])
        eval_control_cost = jnp.mean(control_cost[population_size:])
        
        
        print(generation_number,eval_normal_fitness,eval_mean_dist,max_dist)
        
        # Do the step logging 
        step_logs = {
            "generation_number" : generation_number,
            "evaluations_so_far" : generation_number*config["ES_popsize"],

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
            "innovation" : innovation,
            "evo_var" : evo_var,
            "evo_ent" : evo_ent,
        }
        
        if wandb_logging is True:   
            wandb.log(step_logs)    
            
        # Do checkpointing
        if generation_number % config["CHECKPOINT_FREQUENCY"] == 10:
            if len(b_archive) > 0:
                b_archive_array = np.stack(b_archive)  
                np.save(run_checkpoint_path+"/b_archive.npy",b_archive_array)
            with open(run_checkpoint_path+'/obs_stats.pickle', 'wb') as handle:
                pickle.dump(observation_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(run_checkpoint_path+"/theta.npy",current_params)
            
        current_params = updated_params
        generation_number += 1
    
    # final save
    final_rollout_arrays = {name : np.array(arr) for name,arr in rollout_results}
    np.savez(run_checkpoint_path+"/final_rollout_arrays.npz",final_rollout_arrays) # to load do: dict(np.load(run_checkpoint_path+"/final_rollout_arrays.npz"))
    
    if len(b_archive) > 0:
        b_archive_array = np.stack(b_archive)  
        np.save(run_checkpoint_path+"/b_archive.npy",b_archive_array)
    with open(run_checkpoint_path+'/obs_stats.pickle', 'wb') as handle:
        pickle.dump(observation_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(run_checkpoint_path+"/theta.npy",current_params)
    if config["PLAIN_ES_TEST_ELITE_MAPPING_PERFORMANCE"] is True:
        np.save(run_checkpoint_path+"/b_map_cumm.npy",b_map_cumm.data,allow_pickle=True)
    
    
            
        
        
        
        
        
