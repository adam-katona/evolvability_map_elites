


import numpy as np
import brax
from brax.envs import wrappers
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
import jax

from brax.training import distribution
from brax.training import networks


from es_map.my_brax_envs import brax_envs
from es_map import map_elite_utils

def create_env(env_name,env_mode,population_size,evaluation_batch_size,episode_max_length):
     
    # envs to consider:
    # 3d envs: ant, humanoid
    # 2d envs: walker, halfcheetah, hopper
    
    # There are several variants of the envs
    # We can have fitness: movement forward (+ control cost), movement distance, controll cost
    # We can have bd: foot contacts, final pos
    
    # The combinations that we want to do are:
    # - movement forward, foot contacts   
    # - controll cost, final pos
    
    env = brax_envs.create_base_env(env_name,env_mode)
   
    env = wrappers.EpisodeWrapper(env, episode_length=episode_max_length, action_repeat=1)
    env = wrappers.VectorWrapper(env, batch_size=population_size+evaluation_batch_size) # for each iteration, we evaluate the children and the parent performance in parallel
    env = wrappers.AutoResetWrapper(env) # not sure if this is nessasary, we only look at the first episode
    env = wrappers.VectorGymWrapper(env) 
    return env


def create_MLP_model(observation_size, action_size):
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    return networks.make_model(
      [64, 64, parametric_action_distribution.param_size],
      observation_size,
    )

# The stochastic version would look something like this  
#def get_stochastic_action(parametric_action_distribution,model_out,key):
#    actions = parametric_action_distribution.sample(model_out, key)
#    return action
# Where parametric_action_distribution is eg: parametric_action_distribution = brax.training.distribution.NormalTanhDistribution(event_size=action_size)

    
def get_deterministic_actions(model_out):
    loc, scale = jnp.split(model_out, 2, axis=-1) # splits it into 2 even subarrays along last axis
    act = jnp.tanh(loc)
    return act

##################
## MODEL USAGE: ##
##################
# model = create_MLP_model(observation_size, action_size)
# model_params = model.init(key)
# model_out = model.apply(model_params, observation)
# action = get_deterministic_actions(model_out)
#
# for batched version:
# batch_model_apply = jp.vmap(model.apply)
# model_out = batch_model_apply(batched_model_params,batched_observation)
# action = get_deterministic_actions(model_out)




# turn a parameter dict into a flat 1d parameter vector, + stuff needed to recunstruct 1d vector it into original dict format
def params_tree_to_vec(params):
    numelements = jax.tree_map(lambda x:x.size ,params)
    numel_leafs,treedef = jax.tree_flatten(numelements)
    ending_indicies = np.cumsum(numel_leafs)
    ending_indicies_tree = jax.tree_unflatten(treedef,ending_indicies)
    
    shapes_tree = jax.tree_map(lambda x:x.shape ,params)
    
    flat_shaped_tree = jax.tree_map(lambda x:x.reshape(-1) ,params)
    flat_shaped_leafs,treedef = jax.tree_flatten(flat_shaped_tree)
    
    vec = jnp.concatenate(flat_shaped_leafs)
    
    return vec,shapes_tree,ending_indicies_tree
    
def vec_to_params_tree(vec,shapes,indicies):
    return jax.tree_multimap(lambda i,shape:vec[i-np.prod(shape):i].reshape(shape),indicies,shapes)

##########################
## MODEL TO FLAT USAGE: ##
##########################
# vec,shapes,indicies = params_tree_to_vec(model_params)   # now we can do anything with vec, like get mutated copies, calculate weighted sums for ES gradient...
# reconstructed_model_params = vec_to_params_tree(vec,shapes,indicies)
# FOR BATCHED CASE:
# batch_vec_to_params = jax.vmap(vec_to_params_tree,in_axes=[0, None,None])
# param_vecs = jnp.random.normal(10000,6000)  # population of 10000 parameter vectors
# batch_model_params = batch_vec_to_params(param_vecs,shapes,indicies)



def get_calculate_novelty_fn(k):
    def calculate_novelty(bd,archive):
        # For no
        distances = jnp.sqrt(jnp.sum((archive - bd)**2,axis=1))
        actual_k = min(k,archive.shape[0])
        nearest_neighbors,nearest_indicies = jax.lax.top_k(-distances, actual_k) # take negative to calculate neerest instead of furthest
        novelty = -jnp.mean(nearest_neighbors)   # take negative again, to get the mean distance
        return novelty
    
    return calculate_novelty

################################
## NOVELTY CALCULATION USAGE: ##
################################
# calculate_novelty_fn = get_calculate_novelty_fn(k=10)
# batch_calculate_novelty_fn = jax.jit(jax.vmap(calculate_novelty_fn,in_axes=[0, None]))
# novelties = batch_calculate_novelty_fn(bds,archive)

def calculate_evo_var(bds):
    bd_mean = jnp.mean(bds,axis=0)
    contributions_to_variance = jnp.sum(((bds - bd_mean) ** 2),axis=1)
    evolvability_of_parent = jnp.mean(contributions_to_variance).item()
    return evolvability_of_parent

def calculate_evo_ent(bds,config):
    bds_cpu = np.array(bds)
    import scipy.spatial.distance
    k_sigma = config["ENTROPY_CALCULATION_KERNEL_BANDWIDTH"]  # kernel standard deviation
    pairwise_sq_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(bds_cpu, "sqeuclidean"))
    k = np.exp(-pairwise_sq_dists / k_sigma ** 2)
    p = k.mean(axis=1)
    entropy = -np.log(p).mean()
    return entropy
    




def rollout_flexible_episodes(env,params,obs_stats,config,
                              batched_model_apply_fn):
    # env - gym wrapped batched brax env
    # params - batched parameter tree
    # obs_stats - obervation statistics for mean and var calculation for normalization
    #
    # Return
 
    if config["env_deterministic"] is True:
        # if deterministic, we set the same key for every reset
        # the key is only used to add some random perturbation to the starting position, so it is only relevant to do before we call reset
        env.seed(666)
    obs = env.reset()

    IS_DIRECTIONAL_MODE = ("DIRECTIONAL" in config["env_mode"])

    control_cost_fitness = jnp.zeros(env._state.info["fitness_control_cost_and_survive"])
    forward_fitness = jnp.zeros(env._state.info["fitness_forward"])
    distance_walked = jnp.zeros(env._state.info["fitness_distance_walked"])
    if IS_DIRECTIONAL_MODE is True:
        directional_fitness = jnp.zeros(env._state.info["fitness_directional_distance"])

    final_pos = jnp.zeros_like(env._state.info["bd_final_pos"])
    foot_contacts = jnp.zeros_like(env._state.info["bd_foot_contacts"])

    # we have 2 kinds of values, the cummulatives, which needs to be summed until the end of the first episode,
    #                            and the ones which have to be read only once at the end of the episode
    # Cummulative values:
    #  - control_cost_fitness
    #  - forward_fitness
    # Only read at the end values
    #  - distance_walked
    #  - directional_fitness
    #  - final_pos
    #  - foot_contacts
    
    # Also, the position is zerod the last step of the episode, 
    # so the values connected to poition must be read out before the last step
    # These are 
    #  - distance_walked
    #  - directional_fitness
    #  - final_pos

    active_episode = jnp.ones_like(control_cost_fitness)

    # prepare obs stats
    mean,var = map_elite_utils.calculate_obs_stats(obs_stats)
    mean = jnp.array(mean) # copy to gpu
    var = jnp.array(var)
    
    # also prepare variable to accumulate values to calcuate obs stats, we will add new obs and return these
    obs_sums = jnp.array(obs_stats["sum"])
    obs_squared_sums = jnp.array(obs_stats["sumsq"])
    obs_count = jnp.array([obs_stats["count"]])
    
    max_steps = config["episode_max_length"]
    
    for step_i in range(max_steps):
    
        normalized_obs = (obs-mean) / var
        
        model_out = batched_model_apply_fn(params,normalized_obs)
        action = get_deterministic_actions(model_out)
        
        before_step_pos = env._state.info["bd_final_pos"]
        before_step_distance_walked = env._state.info["fitness_distance_walked"]
        if IS_DIRECTIONAL_MODE is True:
            before_step_directional_fitness = env._state.info["fitness_directional_distance"]
        
        
        obs,reward,done,info = env.step(action)
        
        if step_i == (max_steps-1): # if we reached the max time limit, set done to 1 
            done = jnp.ones_like(done) 
        last_step_of_first_episode = active_episode * done # will only ever be 1 when we are at last step of first episode
        active_episode = active_episode * (1 - done) # once the first episode is done, active_episode will become and stay 0
    
        control_cost_fitness += env._state.info["fitness_control_cost_and_survive"] * active_episode
        forward_fitness += env._state.info["fitness_forward"] * active_episode
    
        current_foot_contacts = env._state.info["bd_foot_contacts"]
        # bd is sometimes nan and inf (we multiply by 0 in those cases, but still infects with nan...)
        current_foot_contacts = jnp.nan_to_num(info["bd"],nan=0.0, posinf=0.0, neginf=0.0)
        
        final_pos = final_pos + last_step_of_first_episode.reshape(-1,1) * before_step_pos
        foot_contacts = foot_contacts + last_step_of_first_episode.reshape(-1,1) * current_foot_contacts
        distance_walked = distance_walked + last_step_of_first_episode * before_step_distance_walked
        if IS_DIRECTIONAL_MODE is True:
            directional_fitness = directional_fitness + last_step_of_first_episode * before_step_directional_fitness

        # record observation stats, only count active episodes (zero out others)
        active_obs = active_episode.reshape(-1,1) * obs
        obs_sums = obs_sums + jnp.sum(active_obs,axis=0)
        obs_squared_sums = obs_squared_sums + jnp.sum(active_obs*active_obs,axis=0)
        obs_count = obs_count + jnp.sum(active_episode)

        # no more active episodes, we can return
        if jnp.sum(active_episode) == 0:
            break

    # turn back obs stats into normal cpu format
    new_obs_stats = {
        "sum" : np.array(obs_sums),
        "sumsq" : np.array(obs_squared_sums),
        "count" : obs_count[0],
    }
    
    results = {
        "control_cost_fitness" : control_cost_fitness,
        "forward_fitness" : forward_fitness,
        "distance_walked" : distance_walked,
        "normal_fitness" : control_cost_fitness + forward_fitness,
        
        "final_pos" : final_pos,
        "foot_contacts" : foot_contacts,
    }
    if IS_DIRECTIONAL_MODE is True:
        results["directional_fitness"] = directional_fitness
    
    fitness_mode,bd_mode = brax_envs.env_mode_interpret(config["env_mode"])
    if fitness_mode == "NORMAL":
        results["fitnesses"] = results["normal_fitness"]
    elif fitness_mode == "DISTANCE":
        results["fitnesses"] = results["distance_walked"]
    elif fitness_mode == "CONTROL":
        results["fitnesses"] = results["control_cost_fitness"]
    elif fitness_mode == "DIRECTIONAL":
        results["fitnesses"] = results["directional_fitness"]
    
    if bd_mode == "CONTACT":
        results["bds"] = results["foot_contacts"]
    elif bd_mode == "FINAL_POS":
        results["bds"] = results["final_pos"] 

    return results,new_obs_stats




# def rollout_episodes(env,params,obs_stats,config,
#                      batched_model_apply_fn):
#     # env - gym wrapped batched brax env
#     # params - batched parameter tree
#     # obs_stats - obervation statistics for mean and var calculation for normalization
#     #
#     # Return
#     # cumulative_reward - episode rewards
#     # bds - episode behavior descriptors
#     # new_obs_stats - updated obs_stats
    
    
    
    
#     if config["env_deterministic"] is True:
#         # if deterministic, we set the same key for every reset
#         # the key is only used to add some random perturbation to the starting position, so it is only relevant to do before we call reset
#         env.seed(666)
#     obs = env.reset()
    
    
#     cumulative_reward = jnp.zeros(obs.shape[0])
#     active_episode = jnp.ones_like(cumulative_reward)
#     bd_dim = env._state.info["bd"].shape[1]
#     bds = jnp.zeros([obs.shape[0],bd_dim])

#     final_pos = jnp.zeros([obs.shape[0],2]) 

#     # prepare obs stats
#     mean,var = map_elite_utils.calculate_obs_stats(obs_stats)
#     mean = jnp.array(mean) # copy to gpu
#     var = jnp.array(var)
    
#     # also prepare variable to accumulate values to calcuate obs stats, we will add new obs and return these
#     obs_sums = jnp.array(obs_stats["sum"])
#     obs_squared_sums = jnp.array(obs_stats["sumsq"])
#     obs_count = jnp.array([obs_stats["count"]])
    
    
#     max_steps = config["episode_max_length"]
    
#     for step_i in range(max_steps):
    
#         normalized_obs = (obs-mean) / var
        
#         model_out = batched_model_apply_fn(params,normalized_obs)
#         action = get_deterministic_actions(model_out)
    
#         # even when bd is not final pos, it is good to have the final position for plotting purpuses
#         # we want to take the xy pos of the body (coord system is z up, right handed)
#         # the pos is in the shape of [batch_dim,num_bodies,3]
#         # we want to take body 0, because that is the torso i think for all robots
#         # we have to take the pos before step, because once the episode is done, pos will be zerod out
#         before_step_pos = env._state.qp.pos[:,0,0:2]
    
#         obs,reward,done,info = env.step(action)
        
#         if step_i == (max_steps-1): # if we reached the max time limit, set done to 1 
#             done = jnp.ones_like(done) 
#         last_step_of_first_episode = active_episode * done # will only ever be 1 when we are at last step of first episode
#         active_episode = active_episode * (1 - done) # once the first episode is done, active_episode will become and stay 0
        
#         cumulative_reward += reward * active_episode
    
        

#         # bd is sometimes nan and inf (we multiply by 0 in those cases, but still infects with nan...)
#         info_bd = jnp.nan_to_num(info["bd"],nan=0.0, posinf=0.0, neginf=0.0)
#         bds = bds + last_step_of_first_episode.reshape(-1,1) * info_bd
        
#         final_pos = final_pos + last_step_of_first_episode.reshape(-1,1) * before_step_pos

#         # record observation stats, only count active episodes (zero out others)
#         active_obs = active_episode.reshape(-1,1) * obs
#         obs_sums = obs_sums + jnp.sum(active_obs,axis=0)
#         obs_squared_sums = obs_squared_sums + jnp.sum(active_obs*active_obs,axis=0)
#         obs_count = obs_count + jnp.sum(active_episode)
    
#         # no more active episodes, we can return
#         if jnp.sum(active_episode) == 0:
#             break
    
#     # turn back obs stats into normal cpu format
#     new_obs_stats = {
#         "sum" : np.array(obs_sums),
#         "sumsq" : np.array(obs_squared_sums),
#         "count" : obs_count[0],
#     }
    
#     return cumulative_reward,bds,new_obs_stats,final_pos

