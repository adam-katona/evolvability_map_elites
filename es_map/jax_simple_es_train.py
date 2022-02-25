

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

from es_map import map_elite_utils
from es_map import jax_evaluate


def train():
    

    env_name = "ant"
    population_size = 10000
    evaluation_batch_size = 50
    
    # setup random seed
    seed = random.randint(0, 10000000000)
    print("starting run with seed: ",seed)
    key = jax.random.PRNGKey(seed)
    key, key_init_model = jax.random.split(key, 2)
    
    # setup env and model
    env = jax_evaluate.create_ant(env_name,population_size,evaluation_batch_size):
    model = jax_evaluate.create_MLP_model(env.observation_space.shape[1],env.action_space.shape[1])

    # setup batched functions
    batch_model_apply = jp.vmap(model.apply)
    batch_vec_to_params = jax.vmap(vec_to_params_tree,in_axes=[0, None,None])

    # get initial parameters
    initial_model_params = model.init(key_init_model)
    vec,shapes,indicies = jax_evaluate.params_tree_to_vec(model_params)
    
    # I use torch, because i want to use torch optimizer, and since it is so little part of the copmutation (the grad update compared to the evaluations and grad caluclations),
    # i dont care about it being slow
    current_params = np.array(vec)


    optimizer_state = None
