
import numpy as np
import torch
from es_map import nd_sort

# update type
# - excpected fitness
# - evolvability
# - innovativeness
# + all combinations of the 3 with nondominated sorting (4 other)


# Selection type
# eval fitness
# eval evolvability
# eval innovativeness




example_evaluated_individual = {
    
    "params" : None,  # 1d torch tensor containing the parameters 
    "ID" : None,
    "parent_ID" : None,
    "generation_created" : 0,
    


    "child_eval" : {
        "noise_descriptors" : None,
        "fitnesses" : None,
        "bcs" : None,
        
        "child_obs_sum" : None,
        "child_obs_sq" : None,
        "child_obs_count" : None,
        
        "batch_time" : None,
    },
    
    "eval_fitness" : 2.2,
    "eval_bc" : [1,2],
    
    #"obs_stats" : {   # used for observation normalization. These stats are used for the children as well ??
    #    "obs_sum" : None,
    #    "obs_sq" : None,
    #    "obs_count" : None,
    #},
    
    "evolvability" : 2.1,
    "innovation" : 1.5,
    
    "innovation_over_time" : {   # innovation decreases over time, as we add new individuals to the archive
        4 : 1.2,
        6 : 1.1,
    }
}


def es_update(theta,child_evaluations,config,es_update_type="fitness",novelty_archive=None,optimizer_state=None):
    # novelty_archive is only needed if we want to calculate innovation (for now we always do)
    # do a whole es update
    
    # calculate indicies based on the type of ES update
    pop_size = len(child_evaluations["fitnesses"])  # can be slightly different from config["ES_popsize"]
    
    if es_update_type == "fitness":
        sorted_indicies = np.argsort(child_evaluations["fitnesses"]) 
        pass
    elif es_update_type == "evolvability":
        # for evolvability we want to maximize expected variance.
        # In order to do this, we go towards the individuals whose contribution to the variance is largest.
        # also this is the simple case of 1 d bc
        # if bc is a vector, what we are maximizing is the sum of the variance in each dimension.
        # so in order to maximize this, we need to go towards the ponts which contribution to this is largest.
        bcs = child_evaluations["bcs"]
        bc_mean = np.mean(bcs,axis=0)
        
        contributions_to_variance = np.sum(((bcs - bc_mean) ** 2),axis=1)
        evolvability_of_parent = np.mean(contributions_to_variance)
        
        # and then we rank
        sorted_indicies = np.argsort(contributions_to_variance) 
         
        pass
    elif es_update_type == "innovation":
        # for innovation we want to maximize the excpected novelty.
        # for this we want to go towards points which are novel.
        novelties = novelty_archive.calculate_novelty(child_evaluations["bcs"],k_neerest=config["NOVELTY_CALCULATION_NUM_NEIGHBORS"])
        sorted_indicies = np.argsort(novelties) 
        
        # calculate the innovation, which we define as the excpected novelty of offspring
        innovation_of_parent = np.mean(novelties)
        
        pass
    
    elif es_update_type == "quality_evolvability":
        # we do a nondominating sort on fitness and evolvability, and use the ranks to do the update
        
        fitnesses = np.array(child_evaluations["fitnesses"])
        bcs = child_evaluations["bcs"]
        bc_mean = np.mean(bcs,axis=0)
        contributions_to_variance = np.sum(((bcs - bc_mean) ** 2),axis=1)

        multi_objective_fitnesses = np.concatenate([fitnesses.reshape(-1,1),contributions_to_variance.reshape(-1,1)],axis=1)
        # multi_objective_fitnesses shape is (pop_size,2)

        fronts = nd_sort.calculate_pareto_fronts(multi_objective_fitnesses)
        nondomination_rank_dict = nd_sort.fronts_to_nondomination_rank(fronts)
        crowding = nd_sort.calculate_crowding_metrics(multi_objective_fitnesses,fronts)
        sorted_indicies = nd_sort.nondominated_sort(nondomination_rank_dict,crowding)
        # nondominated sort sorts, so best is first. This is opposite to what we did in the other cases, reverse the indicies.
        sorted_indicies = sorted_indicies[::-1]
        
        
    else:
        raise "unknown es_update_type"
    
        
    # calculate ranks
    all_ranks = np.linspace(-0.5,0.5,len(sorted_indicies)) 
    perturbation_ranks = np.zeros(len(sorted_indicies))
    perturbation_ranks[sorted_indicies] = all_ranks
    perturbation_ranks = torch.from_numpy(perturbation_ranks).float()
    
    # reconstruct perturbations
    from es_map import random_table
    noise_table = random_table.noise_table
    perturbation_array = [torch.from_numpy(noise_table[rand_i:rand_i+theta.size])*direction 
                          for rand_i,direction in child_evaluations["noise_descriptors"]]
    perturbation_array = torch.stack(perturbation_array)
    
    grad = torch.matmul(perturbation_ranks,perturbation_array)  # ES update, calculate the weighted sum of the perturbations
    grad = grad / pop_size / config["ES_sigma"]
    
    # TODO, what to do with optimizer, should we save optimizer state?, For now not.
    theta = torch.nn.Parameter(torch.from_numpy(theta)) 
    if config["ES_OPTIMIZER_TYPE"] == "ADAM":
        optimizer = torch.optim.Adam([theta],lr=config["ES_lr"],weight_decay=config["ES_L2_COEFF"])
    elif config["ES_OPTIMIZER_TYPE"] == "SGD":
        optimizer = torch.optim.SGD([theta],lr=config["ES_lr"],weight_decay=config["ES_L2_COEFF"])
        
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    theta.grad = -grad # we are maximizing, but torch optimizer steps in the opposite direction of the gradient, multiply by -1 so we can maximize.
    # also add term for l2 regularization
    theta.grad = theta.grad 
    optimizer.step()
    
    updated_theta = theta.detach().numpy()
    new_optimizer_state = optimizer.state_dict()
    
    return updated_theta,new_optimizer_state


