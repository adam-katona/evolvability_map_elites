

import os
os.environ["MKL_NUM_THREADS"] = "1"  # set this beofre importing numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from dask.distributed import Client
import dask

import wandb

from es_map import map_elites
from es_map import es_update
from es_map import novelty_archive
from es_map import behavior_map

from es_map import random_table

            
def calculate_obs_stats(sum,sumsq,count):
    mean = sum / count
    std = np.sqrt(np.maximum(sumsq / count - np.square(mean), 1e-2))
    return mean,std


default_config = {
    "evaluate_batch_size" : 10,        # how many evaluations to calculate per remote call (purely comutational parameter, no effect on result)
    "child_evaluate_batch_size" : 20,  # how many evaluations to calculate per remote call (purely comutational parameter, no effect on result)
    
    "GA_MAP_ELITES_NUM_GENERATIONS" : 100000,
    "ES_MAP_ELITES_NUM_GENERATIONS" : 1000,
    
    "GA_MUTATION_POWER" : 0.02,
}        

def get_random_individual(config):
    from es_map.interaction import interaction
    env = interaction.build_interaction_module(config["env_id"],config)
    theta = env.initial_theta()
    return theta


def set_up_weights_and_biases_run(config,run_name):
    wandb.config = config
    wandb.init(project="evolvability_map_elites", entity="adam_katona")  # TODO reinit=True, if want to run more experiments in single run
    wandb.run.name = run_name
    wandb.run.config.update(wandb.config)
    wandb.run.tags = wandb.run.tags + (config["env_id"],)
    
    # create a folder for checkpoints and large files
    run_name = wandb.run.dir.split("/")[-2]
    run_checkpoint_path = "/scratch/ak1774/runs/large_files/" + run_name
    os.makedirs(run_checkpoint_path,exist_ok=True)
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f)
        


def set_up_dask_client():
    client = Client(n_workers=60, threads_per_worker=1)
    def set_up_worker():
        import os
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1" 
        os.environ["OMP_NUM_THREADS"] = "1" 
    client.run(set_up_worker)
    return client

def save_model_params(params):
    # save the params into wandb.run.dir folder. 
    # Once wandb.finish() is called it will sync it to the cloud
    pass


# simple map elites means we only have one map, and only do exploit updates
def run_simple_ga_map_elites(config,register_weights_and_biases=False):

    # prepare dask client
    client = set_up_dask_client()
    
    b_map = behavior_map.Grid_behaviour_map(config)
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))

    generation_number = 0
    while True:
        if generation_number >= config["GA_MAP_ELITES_NUM_GENERATIONS"]:
            break
        
        # get parent
        non_empty_cells = b_map.get_non_empty_cells(config)
        if len(non_empty_cells) == 0:
            parent_cell = None
            parent_params = get_random_individual(config)
            parent_obs_mean = None
            parent_obs_std = None
        else:
            parent_cell = np.random.choice(non_empty_cells)  # NOTE, here goes cell selection method
            parent_params = parent_cell["params"]
            parent_obs_mean,parent_obs_std = calculate_obs_stats( parent_params["obs_stats"]["obs_sum"],
                                                                  parent_params["obs_stats"]["obs_sq"],
                                                                  parent_params["obs_stats"]["obs_count"])

        # create child
        child = parent_params + np.random.randn(*parent_params.shape) * config["GA_MUTATION_POWER"]
        
        # evaluate child
        child_f = client.scatter(child,broadcast=True)
        
        res = [client.submit(es_update.evaluate_individual,theta=child_f,obs_mean=parent_obs_mean,obs_std=parent_obs_std,eval=True,config=config,
                             pure=False) for _ in range(10)]
        res = client.gather(res)

        


def run_simple_es_map_elites(config,register_weights_and_biases=False):
    
    b_map = behavior_map.Grid_behaviour_map(config)
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))

    generation_number = 0
    while True:
        if generation_number >= config["ES_MAP_ELITES_NUM_GENERATIONS"]:
            break



def run_map_elites_experiment(config,register_weights_and_biases=False):
  
    b_map = behavior_map.Grid_behaviour_map(config)
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))

    ## for a few steps, evaluate random individuals, to have a few non empty cells in the map
    #for seed_i in range(config["seeding_steps"]):
    #    random_params = get_random_individual(config)
    #    
    #    fitnesses,bcs = evaluate_individual_multiple_times(random_params,n=50)
    #    evaluate_fitness = np.mean(fitnesses)
    #    evaluate_bcs = np.mean(bcs,axis=0)
        
        

    while True:
        
        # TODO select mode (eg explore, exploit, innovate...)
        mode = "exploit"
        
        parent_cell = map_elites.select_parent_cell(grid,config,mode=mode)  # will return none if no cell available yet
        
        if parent_cell is None:
            parent = None
            parent_params = get_random_individual(config)
        else:
            # TODO mode dependent 
            parent = parent_cell["elite"]
            parent_params = parent["params"]
            
            
        # ensure we have valid observation statistics
        if parent is None
        
            
    
        # choose ES update mode
        es_update_mode = "fitness" # can be "fitness", "variability", "innovation"
        
        # update / mutate individual
        
        do_es_update(update_mode,return_parent_evolvability,return_parent_innovation)
        
        
        updated_individual_1,updated_individual_0_child_evaluations = es_update(parent,es_update_mode,config,bc_archive)
        evo_0 = calculate_evolvability(updated_individual_0_child_evaluations)
        innov_0 = calcualte_innovation(updated_individual_0_child_evaluations,bc_archive)
        
        updated_individual_2,updated_individual_1_child_evaluations = es_update(updated_individual_1,es_update_mode,config,bc_archive) 
        evo_1 = calculate_evolvability(updated_individual_1_child_evaluations)
        innov_1 = calcualte_innovation(updated_individual_1_child_evaluations,bc_archive)
        # TODO cache updated_individual_2, if we ever want to update updated_individual_1 in the future
        # Also we can use evo_0,innov_0, to update the stats of parent (especially innovation, since the meaning changes over time)
        
        # evaluate
        fitness_1,bc_1 = evaluate(updated_individual_1)
        # Also we can evaluate updated_individual_2 and put in in the fitness archive if it turns out to be good
        
        child = {
            "params" : updated_individual_1,
            "offspring_evaluations" : updated_individual_1_child_evaluations,
            "fitness" : fitness_1,
            "bc" : bc_1,
            "evolvability" : evo_1,
            "innovation" : innov_1
        }
        
        grid = update_grid(grid,child,config)
        
        # update grid
        grid = update_grid(grid,fitness_1,bc_1,updated_individual_1,evo_1,innov_1)
        bc_archive = update_archive(bc_archive,bc_1,updated_individual_1)
        
        



# General API challanges
# - es update can return multiple individuals -> needs to return a list
# - to evaluate all of them, we need to do many secondary evaluations...




    


