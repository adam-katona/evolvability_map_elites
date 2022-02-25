


import numpy as np
import brax
from brax.envs import wrappers
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
import jax

from es_map.qdax_envs.unidirectional_envs import ant, walker, hopper, halfcheetah, humanoid

from es_map import map_elite_utils


def create_ant(env_name,population_size,evaluation_batch_size):
    if env_name is "ant":
        env = ant.QDUniAnt()
    else:
        raise "unknown env name"
    
    env = wrappers.EpisodeWrapper(env, episode_length=1000, action_repeat=1)
    env = wrappers.VectorWrapper(env, batch_size=population_size+evaluation_batch_size) # for each iteration, we evaluate the children and the parent performance in parallel
    env = wrappers.AutoResetWrapper(env) # not sure if this is nessasary, we only look at the first episode
    env = wrappers.VectorGymWrapper(env) 
    return env


def create_MLP_model(observation_size, action_size):
    parametric_action_distribution = brax.training.distribution.NormalTanhDistribution(event_size=action_size)
    return brax.training.networks.make_model(
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



def rollout_episodes(env,params,obs_stats,config,
                     batched_model_apply_fn):
    # env - gym wrapped batched brax env
    # params - batched parameter tree
    # obs_stats - obervation statistics for mean and var calculation for normalization
    #
    # Return
    # cumulative_reward - episode rewards
    # bds - episode behavior descriptors
    # new_obs_stats - updated obs_stats
    
    obs = env.reset()
    
    cumulative_reward = jnp.zeros(obs.shape[0])
    active_episode = jnp.ones_like(cumulative_reward)
    bd_dim = env._state.info["bd"].shape[1]
    bds = jnp.zeros([obs.shape[0],bd_dim])

    # prepare obs stats
    mean,var = map_elite_utils.calculate_obs_stats(obs_stats)
    mean = jnp.array(mean) # copy to gpu
    var = jnp.array(var)
    
    # also prepare variable to accumulate values to calcuate obs stats, we will add new obs and return these
    obs_sums = jnp.array(obs_stats["sum"])
    obs_squared_sums = jnp.array(obs_stats["sumsq"])
    obs_count = jnp.array([obs_stats["count"]])
    
    
    # max_steps = config[""] # TODO
    max_steps = 1000
    
    for step_i in range(max_steps):
    
        normalized_obs = (obs-mean) / var
        
        model_out = batched_model_apply_fn(params,normalized_obs)
        action = get_deterministic_actions(model_out)
    
        obs,reward,done,info = env.step(action)
        
        last_step_of_first_episode = active_episode * done # will only ever be 1 when we are at last step of first episode
        active_episode = active_episode * (1 - done) # once the first episode is done, active_episode will become and stay 0
        
        cumulative_reward += reward * active_episode
    
        # bd is sometimes nan and inf (we multiply by 0 in those cases, but still infects with nan...)
        info_bd = jnp.nan_to_num(info["bd"],nan=0.0, posinf=0.0, neginf=0.0)
        bds = bds + last_step_of_first_episode.reshape(-1,1) * info_bd

        # record observation stats, only count active episodes (zero out others)
        active_obs = active_episode.reshape(-1,1) * obs
        obs_sums = obs_sums + jnp.sum(active_obs,axis=0)
        obs_squared_sums = obs_squared_sums + jnp.sum(active_obs*active_obs,axis=0)
        obs_count = obs_count + jnp.sum(active_episode)
    
    # turn back obs stats into normal cpu format
    new_obs_stats = {
        "sum" : np.array(obs_sums),
        "sumsq" : np.array(obs_squared_sums),
        "count" : obs_count[0],
    }
    return cumulative_reward,bds,new_obs_stats