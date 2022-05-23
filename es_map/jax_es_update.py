
import jax.numpy as jnp
import numpy as np
import jax
import torch

from es_map import nd_sort

def jax_create_perturbations(key,popsize,params_shape):
    half_popsize = int(popsize/2)
    if 2*half_popsize != popsize:
        raise "error, popsize must be divisible by 2"
    perturbations = jax.random.normal(key,[half_popsize, params_shape[0]])
    perturbations = jnp.concatenate([perturbations, -1 * perturbations])
    return perturbations

def jax_es_create_population(parent_params,key,popsize,eval_batch_size,sigma):
    
    half_popsize = int(popsize/2)
    if 2*half_popsize != popsize:
        raise "error, popsize must be divisible by 2"
    
    # copy params to gpu, if not there already
    parent_params = jnp.array(parent_params)
    
    perturbations = jax.random.normal(key,[half_popsize, parent_params.shape[0]])
    
    # mirrored sampling
    perturbations = jnp.concatenate([perturbations, -1 * perturbations])
    child_params = parent_params + sigma * perturbations
    
    parent_eval_params = jnp.repeat(parent_params.reshape(1,-1),eval_batch_size,axis=0)
    all_params = jnp.concatenate([child_params,parent_eval_params],axis=0)

    return all_params,perturbations



def calculate_entropy_contributions(bds,config):
    # i dont want to implement this in jax, so copy the bds to cpu
    bds_cpu = np.array(bds)
    # contribution to entropy
    import scipy.spatial.distance
    k_sigma = config["ENTROPY_CALCULATION_KERNEL_BANDWIDTH"]  # kernel standard deviation
    pairwise_sq_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(bds_cpu, "sqeuclidean"))
    k = np.exp(-pairwise_sq_dists / k_sigma ** 2)

    p = k.mean(axis=1)
    entropy_term = -np.log(p)
    other_term = - (k / k.mean(axis=0)).mean(axis=1)
    
    perturbation_weights = entropy_term + other_term
    return perturbation_weights

def jax_calculate_gradient(perturbations,child_fitness,bds,mode,config,novelties=None):
    
    # should we do this on cpu or gpu?  Remember that this is only once per es step, not need to be very efficient...
    # still stuff like novelty calculation is 100 ms on cpu and 3ms on gpu... Entropy calculation likely even more.
    # The weighted sum calculation should definetly happen on the gpu, because we dont want to copy the whole perturbation array to the cpu...
    # But calculating the ranks can happen on cpu, if we dont have an easy gpu implementation. 
    # Copying the fitnesses and bds is no big deal in the ballpark of 1ms
    # The domination martix calculatinon is much faster on gpu (20ms instead of 7 sec) 

    # if using ranked_weighting, we sort the metric, and asign weights between -0.5 and 0.5
    # else, we use the metric directly (perturbation_weights)
    using_ranked_weighting = True  # Some modes work with ranking, others not
    pop_size = child_fitness.size
    
    if mode == "fitness":
        child_fitness = jnp.array(child_fitness) # ensure it is on gpu
        sorted_indicies = jnp.argsort(child_fitness) 
        
    elif mode == "evo_var":
        bd_mean = jnp.mean(bds,axis=0)
        contributions_to_variance = jnp.sum(((bds - bd_mean) ** 2),axis=1)
        sorted_indicies = jnp.argsort(contributions_to_variance) 
        
    elif mode == "evo_ent":
        perturbation_weights = calculate_entropy_contributions(bds,config)
        using_ranked_weighting = False
        
    elif mode == "innovation":
        sorted_indicies = jnp.argsort(novelties) 

    elif (mode == "quality_evo_var" or 
          mode == "quality_evo_ent" or 
          mode == "quality_innovation" or
          mode == "quality_evo_var_innovation" or
          mode == "quality_evo_ent_innovation"):
        
        # first objective is always fitness
        objetives = [child_fitness.reshape(-1,1)]
        
        if "evo_var" in mode:
            bd_mean = jnp.mean(bds,axis=0)
            contributions_to_variance = jnp.sum(((bds - bd_mean) ** 2),axis=1)
            objetives.append(contributions_to_variance.reshape(-1,1))
            
        if "evo_ent" in mode:
            entropy_contribution = calculate_entropy_contributions(bds,config)
            objetives.append(entropy_contribution.reshape(-1,1))
            
        if "innovation" in mode:
            objetives.append(novelties.reshape(-1,1))
        
        multi_objective_fitnesses = jnp.concatenate(objetives,axis=1)
        # multi_objective_fitnesses shape is (pop_size,2) or (pop_size,3) 

        # old slow code
        #fronts = nd_sort.calculate_pareto_fronts(multi_objective_fitnesses)
        
        # with new code we do domination matrix calculation on gpu and front calculation with numba
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
        
        
    else:
        raise "unknonw grad calculation mode"
    
    
    
    
    if using_ranked_weighting is True:
        # calculate ranks
        sorted_indicies = np.array(sorted_indicies) # ensure sorted_indicies is on cpu
        all_ranks = np.linspace(-0.5,0.5,len(sorted_indicies)) 
        perturbation_ranks = np.zeros(len(sorted_indicies))
        perturbation_ranks[sorted_indicies] = all_ranks
    else:
        perturbation_ranks = perturbation_weights
    
    
    perturbation_ranks = jnp.array(perturbation_ranks)
    grad = jnp.matmul(perturbation_ranks,perturbations)  # ES update, calculate the weighted sum of the perturbations
    grad = grad / pop_size / config["ES_sigma"]
    
    return grad
    

def do_gradient_step(theta,grad,optimizer_state,config):
    # do the gradient update, we can do this on cpu with torch
    # normally we would create an optimizer in the beginning for plain es, but we do the same as in map elites,
    # where we need to create new optimizers when we select new parents
    theta = torch.nn.Parameter(torch.from_numpy(theta)) 
    if config["ES_OPTIMIZER_TYPE"] == "ADAM":
        optimizer = torch.optim.Adam([theta],lr=config["ES_lr"],weight_decay=config["ES_L2_COEFF"])
    elif config["ES_OPTIMIZER_TYPE"] == "SGD":
        optimizer = torch.optim.SGD([theta],lr=config["ES_lr"],weight_decay=config["ES_L2_COEFF"])
    
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    
    theta.grad = -grad # we are maximizing, but torch optimizer steps in the opposite direction of the gradient, multiply by -1 so we can maximize.
    optimizer.step()

    new_theta = theta.detach().numpy()
    optimizer_state = optimizer.state_dict()
    
    
    return new_theta,optimizer_state
