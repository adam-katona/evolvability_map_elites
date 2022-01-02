import numpy as np
import torch




# evaluate a single theta
def evaluate_individual(theta,obs_mean,obs_std,eval,config):
    
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
        

# evaluate a single theta repeatedly
def evaluate_individual_repeated(theta,obs_mean,obs_std,eval,config,repeat_n=1):
    import time
    now = time.time()
    
    from es_map.interaction import interaction
    env = interaction.build_interaction_module(config["env_id"],config)
    
    fitnesses = []
    lengths = []
    bcs = []
    all_obs_sum = []
    all_obs_sq = []
    all_obs_count = []
    
    for _ in range(repeat_n):
        total_return, length, bc, final_xpos, obs_sum, obs_sq, obs_count = env.rollout(
            theta,random_state=np.random.RandomState(),eval=eval, obs_mean=obs_mean, obs_std=obs_std, render=False)
        fitnesses.apped(total_return)
        lengths.apped(length)
        bcs.apped(bc)
        all_obs_sum.apped(obs_sum)
        all_obs_sq.apped(obs_sq)
        all_obs_count.apped(obs_count)
    
    elapsed = time.time() - now

    return {
        "fitnesses" : fitnesses,
        "length" : lengths,
        "bc" : bcs,

        "obs_sum" : np.sum(obs_sum,axis=0),
        "obs_sq" :  np.sum(obs_sq,axis=0),
        "obs_count" :  np.sum(obs_count),
    }

     

# evaluate children defined by a center theta and nose descriptors
def evaluate_children_remote(noise_descriptors,central_theta,obs_mean,obs_std,config,repeat_n=1):
    
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
        for _ in range(repeat_n):
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
    
# recieve parent list, and create and evaluate a child for each of them. Does not use random table indicies.
def ga_evaluate_children_multi_parent(client,parent_datas,config):
    # here there is no point using a random table, since every individual is different.
    # Just pass the whole arrays around for now, could later reimplement with random seeds and deterministic mutations.
        
    all_children_results = []
    all_children = []
    final_child_results = []
    for theta,obs_mean,obs_std in parent_datas:
        
        child = theta + np.random.randn(theta.size) * config["GA_MUTATION_POWER"] 
        child_results = [client.submit(evaluate_individual,theta=child,obs_mean=obs_mean,obs_std=obs_std,eval=False,config=config,pure=False) 
                                        for _ in range(config["GA_NUM_EVALUATIONS"])]
        all_children_results.append(child_results)
        all_children.append(child)
    
    for results,child in zip(all_children_results,all_children):
        results = client.gather(results)
    
        child_result = {
            "child" : child,
            "mean_fitness" : np.mean([res["fitness"] for res in results]),
            "mean_bc" : np.mean([res["bc"] for res in results],axis=0),
            
            "child_obs_sum" : np.sum([res["obs_sum"] for res in results],axis=0),
            "child_obs_sq" : np.sum([res["obs_sq"] for res in results],axis=0),
            "child_obs_count" : np.sum([res["obs_count"] for res in results]),
        }
        final_child_results.append(child_result)
    return final_child_results
        
        
# create and evaluate children of a single parent, using random table indicies
def ga_evaluate_children_single_parent(client,theta,obs_mean,obs_std,config):
    from es_map import random_table
    
    num_children = config["GA_CHILDREN_PER_GENERATION"]
    num_evaluation_per_children = config["GA_NUM_EVALUATIONS"]
    
    random_table_indicies = random_table.get_random_indicies(param_dim=theta.size,num_indicies=num_children)
    child_list = [[(rand_i,1)] for rand_i in random_table_indicies] # (rand_index,direction) not used for ga, but keep it so i can have 1 evaluate child function

    theta_f = client.scatter(theta,broadcast=True)
    
    res = client.map(evaluate_children_remote,child_list,central_theta=theta_f,obs_mean=obs_mean,obs_std=obs_std,config=config,
                     repeat_n=num_evaluation_per_children)
    res = client.gather(res)
    
    child_results = []
    for i,r in enumerate(res):
        noise_i = child_list[i][0][0]
        child_params = theta + random_table.noise_table[noise_i:noise_i+theta.size] * config["ES_sigma"]
        
        child_result = {
            "child" : child_params,
            "mean_fitness" : np.mean(r["fitnesses"]),
            "mean_bc" : np.mean(r["bcs"],axis=0),
            
            "child_obs_sum" : r["obs_sum"],
            "child_obs_sq" : r["obs_sq"],
            "child_obs_count" : r["obs_count"],
        }
        child_results.append(child_result)
    
    return child_results
    
    
# Create and evaluate childre of a central theta using random table indicies
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
    
    # join the list of batch results into single arrays 
    fitnesses = np.concatenate([r["fitnesses"] for r in res])
    bcs = np.concatenate([r["bcs"] for r in res],axis=0)
    
    # observation statistics tracking (NOTE the obs statistics are tracked for 1% of the evaluations)
    obs_sum = np.sum(np.stack([r["obs_sum"] for r in res]),axis=0)
    obs_sq = np.sum(np.stack([r["obs_sq"] for r in res]),axis=0)
    obs_count = np.sum([r["obs_count"] for r in res])
        
    batch_time = np.mean([r["elapsed_time"] for r in res])
    #print("mean batch time is: ",batch_time,[r["elapsed_time"] for r in res])
    
    return {
        "noise_descriptors" : pop_list,
        "fitnesses" : fitnesses,
        "bcs" : bcs,
        
        "child_obs_sum" : obs_sum,
        "child_obs_sq" : obs_sq,
        "child_obs_count" : obs_count,
        
        "batch_time" : batch_time,
    }
    
    
    





