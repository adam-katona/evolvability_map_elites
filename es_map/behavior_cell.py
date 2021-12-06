



# A cell is a dict with all kind of data about elites and evaluations of elites
# Elites are also a dict
example_evaluated_individual = {
    "params" : None,  # 1d torch tensor containing the parameters 

    "child_eval" : {
        "seed" : 42,  # used to reconstruct the perturbations, which can be used to do an ES step
        "fitnesses" : None,
        "bcs" : None,
    },
    
    "eval_fitness" : 2.2,
    "eval_bc" : [1,2],
    
    "obs_stats" : {   # used for observation normalization. These stats are used for the children as well ??
        "obs_sum" : None,
        "obs_sq" : None,
        "obs_count" : None,
    },
    
    "evolvability" : 2.1,
    "innovation" : 1.5,
    
    
}

example_cell = {
    "cell_coords" : (2,7),   # coords in the behavior map
    
    # In case of single fitness cell
    "elite" : example_evaluated_individual,
    
    # In case of multiple independent
    "fitness_elite" : example_evaluated_individual,
    "evolvability_elite" : example_evaluated_individual,
    "innovation_elite" : example_evaluated_individual,

    # in case of multiple, nd sorted
    "elites" : [example_evaluated_individual,example_evaluated_individual] # list of non dominating elites



}

# TODO function for multi independent elite to get list of non empty elites
def get_non_empty_elites():
    pass

