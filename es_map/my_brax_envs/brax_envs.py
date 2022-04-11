


import brax
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
import jax
import numpy as np

from brax.envs import ant, walker2d, hopper, halfcheetah, humanoid

# Here I define env classes, which inherit from the original brax envs,
# The extra thing they do is add the following fields to state.info:
#  - control_cost_and_survive_reward  # penalty and reward for control and being alive
#  - forward_reward                   # reward for going forward  
#  - distance_walked_from_origin      # can be used as alternative fitness
#  - distance walked in
#  -
#  - 


# directional version of envs.
# on reset, select random direction.
# add this direction to observaton

# We handle the 2d and 3d envs separatly
# Only 3d envs have directional stuff

###################
## FLEXIBLE ENVS ##
###################

def init_extra_info_fields(state,num_legs,num_dimensions,foot_indicies):
    # num_dimensions how the robot can move from a top down perspective, in 2d or on 1d. 
    info = state.info
    info["fitness_control_cost_and_survive"] = 0
    info["fitness_forward"] = 0
    info["fitness_distance_walked"] = 0
    info["fitness_directional_distance"] = 0
    info["bd_foot_contacts"] = jnp.zeros(num_legs)
    info["bd_final_pos"] = jnp.zeros(num_dimensions)
    
    info['contact_cum'] = jnp.zeros((num_legs))
    return info

def calculate_extra_info_fields(state,num_legs,num_dimensions,foot_indicies):
    info = state.info
    
    # calculate foot contacts
    contact = jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1)), axis=1) != 0
    contact = contact[foot_indicies]
    info["contact_cum"] = info["contact_cum"] + contact
    info["bd_foot_contacts"] = info["contact_cum"] / info["steps"]
    
    # calculate final pos (always overwrite with current pos)
    current_pos = state.qp.pos[0,0:num_dimensions]
    info["bd_final_pos"] = current_pos
    info["fitness_distance_walked"] = jnp.sqrt(jnp.sum(current_pos ** 2),axis=1)
    return info
    
    


class FelxibleHumanoid(humanoid.Humanoid):
    FOOT_INDEX = [4,6]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=2,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info = calculate_extra_info_fields(state,num_legs=2,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_quadctrl"]  
                 - state.metrics["reward_impact"]  
                 + state.metrics["reward_alive"]) 
        info["fitness_forward"] = state.metrics["reward_linvel"]
        
        return state.replace(info=info)
    
class FelxibleAnt(ant.Ant):
    FOOT_INDEX = [2,4,6,8]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=4,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info = calculate_extra_info_fields(state,num_legs=4,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_ctrl_cost"]  
                 - state.metrics["reward_contact_cost"]  
                 + state.metrics["reward_survive"]) 
        info["fitness_forward"] = state.metrics["reward_forward"]

        return state.replace(info=info)



class FelxibleWalker(walker2d.Walker2d):
    FOOT_INDEX = [3,6]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info = calculate_extra_info_fields(state,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_ctrl"]  
                 + state.metrics["reward_healthy"]) 
        info["fitness_forward"] = state.metrics["reward_forward"]

        return state.replace(info=info)

class FelxibleHalfCheetah(halfcheetah.Halfcheetah):
    FOOT_INDEX = [3,6]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info = calculate_extra_info_fields(state,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = - state.metrics["reward_ctrl_cost"] 
        info["fitness_forward"] = state.metrics["reward_forward"]
        return state.replace(info=info)


class FelxibleHopper(hopper.Hopper):
    FOOT_INDEX = [3]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=1,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info = calculate_extra_info_fields(state,num_legs=1,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_ctrl"]  
                 + state.metrics["reward_healthy"]) 
        info["fitness_forward"] = state.metrics["reward_forward"]

        return state.replace(info=info)




######################
## DIRECTIONAL ENVS ##
######################


def add_direction_to_obs(obs,dir):
    new_obs = jnp.concatenate([obs,dir])
    return new_obs

def init_extra_info_for_directional(state,rng):
    info = state.info
    angle = jax.random.uniform(rng, shape=[1], minval=-np.pi, maxval=np.pi)
    info["direction"] = jnp.array([jnp.cos(angle),jnp.sin(angle)])
    obs = add_direction_to_obs(state.obs,dir)
    return info,obs

def calculate_extra_info_for_directional(state):
    info = state.info
    info["fitness_directional_distance"] = jnp.dot(info["bd_final_pos"],info["direction"])
    obs = add_direction_to_obs(state.obs,dir)
    return info,obs

class DirectionalHumanoid(FelxibleHumanoid):
        
    @property
    def observation_size(self) -> int:
        return super().observation_size + 2 # add the direction as input
    
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info,obs = init_extra_info_for_directional(state,rng)
        return state.replace(obs=obs,info=info)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info,obs = calculate_extra_info_for_directional(state)
        return state.replace(obs=obs,info=info)
    
class DirectionalAnt(FelxibleHumanoid):
        
    @property
    def observation_size(self) -> int:
        return super().observation_size + 2 # add the direction as input
    
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info,obs = init_extra_info_for_directional(state,rng)
        return state.replace(obs=obs,info=info)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info,obs = calculate_extra_info_for_directional(state)
        return state.replace(obs=obs,info=info)



def env_mode_interpret(env_mode):
    if env_mode == "NORMAL_CONTACT":
        fitness_mode = "NORMAL"
        bd_mode = "CONTACT"
    elif env_mode == "NORMAL_FINAL_POS":
        fitness_mode = "NORMAL"
        bd_mode = "FINAL_POS"
    elif env_mode == "DISTANCE_CONTACT":
        fitness_mode = "DISTANCE"
        bd_mode = "CONTACT"
    elif env_mode == "DISTANCE_FINAL_POS":
        fitness_mode = "DISTANCE"
        bd_mode = "FINAL_POS"
    elif env_mode == "CONTROL_FINAL_POS":
        fitness_mode = "CONTROL"
        bd_mode = "FINAL_POS"
    elif env_mode == "DIRECTIONAL_CONTACT":
        fitness_mode = "DIRECTIONAL"
        bd_mode = "CONTACT"
    else:
        raise "Unknown env_mode"
    
    return fitness_mode,bd_mode

def env_to_bd_descriptor(env_name,env_mode):
    fitness_mode,bd_mode = env_mode_interpret(env_mode)
    if env_name == "ant":
        if bd_mode == "CONTACT":
            return {
                "bc_limits" : [[0,1],[0,1],[0,1],[0,1]],
                "grid_dims" : [6,6,6,6],
            }
        elif bd_mode == "FINAL_POS":
            return {
                "bc_limits" : [[-50, 50], [-50, 50]],
                "grid_dims" : [32,32],
            }
            
    elif env_name == "humanoid":
        if bd_mode == "CONTACT":
            return {
                "bc_limits" : [[0,1],[0,1]],
                "grid_dims" : [32,32],
            }
        elif bd_mode == "FINAL_POS":
            return {
                "bc_limits" : [[-50, 50], [-50, 50]],
                "grid_dims" : [32,32],
            }
    
    
    elif env_name == "walker" or env_name == "halfcheetah":
        if bd_mode == "CONTACT":
            return {
                "bc_limits" : [[0,1],[0,1]],
                "grid_dims" : [32,32],
            }
        elif bd_mode == "FINAL_POS":
            return {
                "bc_limits" : [[-50, 50]],
                "grid_dims" : [100],
            }
    elif env_name == "hopper":
        if bd_mode == "CONTACT":
            return {
                "bc_limits" : [[0,1]],
                "grid_dims" : [100],
            }
        elif bd_mode == "FINAL_POS":
            return {
                "bc_limits" : [[-50, 50]],
                "grid_dims" : [100],
            }
    else:
        raise "unknown env name"
    

def create_base_env(env_name,env_mode):
    
        # env_mode can be:
        # "NORMAL_CONTACT"      same fitness as the original env, bd is foot contacts
        # "NORMAL_FINAL_POS"    same fitness as the original env, bd is final pos
        # "DISTANCE_CONTACT"    fitness is distance, bd is foot contact
        # "DISTANCE_FINAL_POS"  fitness is distance, bd is final pos
        # "CONTROL_FINAL_POS"   fitness is control cost only, bd is final pos  (for control we dont use foot contacts)
        # "DIRECTIONAL_CONTACT" # fitness is directional, bd is foot contacts (for directional we dont use final pos)
   
    if "DIRECTIONAL" in env_mode:
        if env_name == "ant":
            env = DirectionalAnt()
        elif env_name == "humanoid":
            env = DirectionalHumanoid()
        else:
            raise "only ant and humanoid have directional modes"
    else:
        if env_name == "ant":
            env = FelxibleAnt()
        elif env_name == "walker":
            env = FelxibleWalker()
        elif env_name == "hopper":
            env = FelxibleHopper()
        elif env_name == "halfcheetah":
            env = FelxibleHalfCheetah()
        elif env_name == "humanoid":
            env = FelxibleHumanoid()
        else:
            raise "unknown env name"
        
    return  env















