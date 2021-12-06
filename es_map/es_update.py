
import numpy as np
import torch



def es_update(theta,child_evaluations,config,es_update_type="fitness",bc_archive=None,):
    # bc_archive is only needed if we want to calculate innovation (for now we always do)
    # do a whole es update
    
    # calculate indicies based on the type of ES update
    
    if es_update_type == "fitness":
        sorted_indicies = np.argsort(child_evaluations["fitnesses"]) 
        pass
    elif es_update_type == "evolvability":
        # for evolvability we want to maximize expected variance.
        # In order to do this, we go towards the individuals whose contribution to the variance is largest.
        # also this is the simple case of 1 d bc
        # if bc is a vector, what we are maximizing is the sum of the variance in each dimension.
        # so in order to maximize this, we need to go towards the ponts which contribution to this is largest.
        
        contribution_to_variance = sum((bc - bc_mean) ** 2)
        evolvability_of_parent = mean(sum((bc - bc_mean) ** 2))
        
        # and then we rank
        
        # as oposed to rank each dimension, then 
         
        pass
    elif es_update_type == "innovation":
        # for innovation we want to maximize the excpected novelty.
        # for this we want to go towards points which are novel.
        
        
        pass
        
    else:
        raise "unknown es_update_type"
    
    sorted_indicies = np.argsort(child_evaluations["fitnesses"]) 
        
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
    grad = grad / config["ES_popsize"] / config["ES_sigma"]
    
    # TODO, what to do with optimizer, should we save optimizer state?, For now not.
    theta = torch.nn.Parameter(torch.from_numpy(theta)) 
    optimizer = torch.optim.Adam([theta],lr=config["ES_lr"])
    theta.grad = -grad # we are maximizing, but torch optimizer steps in the opposite direction of the gradient, multiply by -1 so we can maximize.
    optimizer.step()
    
    updated_theta = theta.detach().numpy()
    
    return updated_theta



def calculate_var_evolvability(evaluations):
    # evolvability is the excpected variance of the behaviour (summed over components)
    mean_bc = torch.mean(evaluations["bc"],dim=0)
    evolvability = torch.sum((evaluations["bc"] - mean_bc) ** 2) / evaluations["bc"].shape[0]
    return evolvability


def calcualte_innovation(evaluations,bc_archive):
    # innovation is the excpected novelty
    pass



def evaluate_individual_local(theta,obs_mean,obs_std,eval,config):
    
    # create the env
    from es_map.interaction import interaction
    env = interaction.build_interaction_module(config["env_id"],config)

    total_return, length, bc, final_xpos, obs_sum, obs_sq, obs_count = env.rollout(
        theta,random_state=np.random.RandomState(),eval=eval, obs_mean=obs_mean, obs_std=obs_std, render=False)
    
    return {
        "fitness" : total_return,
        "length" : length,
        "bc" : bc,
        "final_xpos" : final_xpos,
        "obs_sum" : obs_sum,
        "obs_sq" : obs_sq,
        "obs_count" : obs_count,
    }
        

def evaluate_children_remote(noise_descriptors,central_theta,obs_mean,obs_std,config):
    
    import time
    
    now = time.time()
    
    # create the env
    from es_map.interaction import interaction
    env = interaction.build_interaction_module(config["env_id"],config)
    
    # calculate the parameters from central_theta and the noise_descriptors
    from es_map import random_table
    params = []
    for noise_i,direction in noise_descriptors:
        noise = random_table.noise_table[noise_i:noise_i+central_theta.size]
        mutation = noise * direction * config["ES_sigma"]
        params.append(central_theta+mutation)
    
    returns, lengths, bcs, final_xpos, obs_sum, obs_sq, obs_count = env.rollout_batch(
                                        params,batch_size=len(params),random_state=np.random.RandomState(),eval=False,
                                        obs_mean=obs_mean, obs_std=obs_std, render=False)
    
    elapsed = time.time() - now
    
    return {
        "fitnesses" : returns,
        "lengths" : lengths,
        "bcs" : bcs,
        "final_xpos" : final_xpos,
        "obs_sum" : obs_sum,
        "obs_sq" : obs_sq,
        "obs_count" : obs_count,
        "elapsed_time" : elapsed,
    }
    

def es_evaluate_children(client,central_theta,obs_mean,obs_std,config):
    from es_map import random_table
    
    HALF_POP_SIZE = config["ES_popsize"] // 2

    # get some random indicies to the noise table
    random_table_indicies = random_table.get_random_indicies(param_dim=central_theta.size,num_indicies=HALF_POP_SIZE)
    pop_list = [(rand_i,1) for rand_i in random_table_indicies] # (rand_index,direction)
    pop_list.extend([(rand_i,-1) for rand_i in random_table_indicies]) # mirrored sampling
    pop_batches = [pop_list[x:x+config["ES_EVALUATION_BATCH_SIZE"]] for x in range(0, len(pop_list), config["ES_EVALUATION_BATCH_SIZE"])]

    # distribute this to cluster
    # firs, we need to send some info to everyone
    central_theta_f = client.scatter(central_theta,broadcast=True)
    
    # then we need to distribute the batches between workers
    res = client.map(evaluate_children_remote,pop_batches,central_theta=central_theta_f,obs_mean=obs_mean,obs_std=obs_std,config=config)
    res = client.gather(res)
    
    
    fitnesses = np.concatenate([r["fitnesses"] for r in res])
    bcs = np.concatenate([r["bcs"] for r in res])
    
    # observation statistics tracking (NOTE the obs statistics are tracked for 1% of the evaluations)
    obs_sum = np.sum(np.stack([r["obs_sum"] for r in res]),axis=0)
    obs_sq = np.sum(np.stack([r["obs_sq"] for r in res]),axis=0)
    obs_count = np.sum([r["obs_count"] for r in res])
        
    batch_time = np.mean([r["elapsed_time"] for r in res])
    print("mean batch time is: ",batch_time,[r["elapsed_time"] for r in res])
    
    return {
        "noise_descriptors" : pop_list,
        "fitnesses" : fitnesses,
        "bcs" : bcs,
        
        "child_obs_sum" : obs_sum,
        "child_obs_sq" : obs_sq,
        "child_obs_count" : obs_count,
    }
    