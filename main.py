

import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from es_map import map_elites
from es_map import es_update
from es_map import novelty_archive
from es_map import behavior_map

from es_map import random_table


default_config = {
    "evaluate_batch_size" : 10,        # how many evaluations to calculate per remote call (purely comutational parameter, no effect on result)
    "child_evaluate_batch_size" : 20,  # how many evaluations to calculate per remote call (purely comutational parameter, no effect on result)
    
    
}        

def get_random_individual(config):
    return np.randn(10)   

def evaluate(params,config):
    # calculate fitness and bc of params
    return np.random.randn(1)[0],np.random.randn(2)

def evaluate_children(individual,config):
    return {
        "seed" : 42,  # used to reconstruct the perturbations, which can be used to do an ES step
        "fitnesses" : np.random.randn(config["ES_pop_size"]),
        "bcs" : np.random.randn([config["ES_pop_size"],2]),
    }

def evaluate_individual_multiple_times(params,n):
    results = [evaluate(params) for _ in range(n)]
    fitnesses = np.array([res[0] for res in results])
    bcs = np.array([res[1] for res in results])
    return fitnesses,bcs

def run_map_elites_experiment(config,register_weights_and_biases=False):
  
    b_map = behavior_map.Grid_behaviour_map(config)
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))

    # for a few steps, evaluate random individuals, to have a few non empty cells in the map
    for seed_i in range(config["seeding_steps"]):
        random_params = get_random_individual(config)
        
        fitnesses,bcs = evaluate_individual_multiple_times(random_params,n=50)
        evaluate_fitness = np.mean(fitnesses)
        evaluate_bcs = np.mean(bcs,axis=0)
        
        

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




    


