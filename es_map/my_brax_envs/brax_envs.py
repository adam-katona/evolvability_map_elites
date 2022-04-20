


import brax
from brax import jumpy as jp
from brax.envs import env
import jax.numpy as jnp
import jax
import numpy as np

from brax.envs import ant, walker2d, hopper, halfcheetah, humanoid

# Here I define env classes, which inherit from the original brax envs,
# The extra thing they do is add a bunch of fields to state.info:

# There was an issue, that step does not expose the info variable provided by the physics engine, which have the contact info
# Original plan was to call the parent step() function, and add some extra stuff
# But because parent does not expose foot contacts, had to rewrite that function.
# Now we copy pasted the step function for each class, and added some line to return contact info
# I know it is horrible...



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
    #info["bd_foot_contacts"] = jnp.zeros(num_legs)
    info["bd_final_pos"] = jnp.zeros(num_dimensions)
    
    info['contact_cum'] = jnp.zeros((num_legs))
    return info

def calculate_extra_info_fields(state,physics_info,num_legs,num_dimensions,foot_indicies):
    info = state.info
    
    # calculate foot contacts
    contact = jp.sum(jp.square(jp.clip(physics_info.contact.vel, -1, 1)), axis=1) != 0
    contact = jnp.take(contact,jnp.array(foot_indicies))
    #contact = contact[foot_indicies]
    info["contact_cum"] = info["contact_cum"] + contact
    #info["bd_foot_contacts"] = info["contact_cum"] / info["steps"]
    
    # calculate final pos (always overwrite with current pos)
    current_pos = state.qp.pos[0,0:num_dimensions]
    info["bd_final_pos"] = current_pos
    info["fitness_distance_walked"] = jnp.sqrt(jnp.sum(current_pos ** 2))
    return info
    
    


class FelxibleHumanoid(humanoid.Humanoid):
    FOOT_INDEX = [4,6]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=2,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        #state = super().step(state,action)
        state,physics_info = self._step(state,action)
        info = calculate_extra_info_fields(state,physics_info,num_legs=2,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_quadctrl"]  
                 - state.metrics["reward_impact"]  
                 + state.metrics["reward_alive"]) 
        info["fitness_forward"] = state.metrics["reward_linvel"]
        
        return state.replace(info=info)
    
    def _step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info, action)

        pos_before = state.qp.pos[:-1]  # ignore floor at last index
        pos_after = qp.pos[:-1]  # ignore floor at last index
        com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
        com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
        lin_vel_cost = 1.25 * (com_after[0] - com_before[0]) / self.sys.config.dt
        quad_ctrl_cost = .01 * jp.sum(jp.square(action))
        # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
        quad_impact_cost = jp.float32(0)
        alive_bonus = jp.float32(5)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        done = jp.where(qp.pos[0, 2] < 0.8, jp.float32(1), jp.float32(0))
        done = jp.where(qp.pos[0, 2] > 2.1, jp.float32(1), done)
        state.metrics.update(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=quad_impact_cost)

        return state.replace(qp=qp, obs=obs, reward=reward, done=done),info
        
class FelxibleAnt(ant.Ant):
    FOOT_INDEX = [2,4,6,8]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=4,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        #state = super().step(state,action)
        state,physics_info = self._step(state,action)
        #print(type(physics_info))
        #print(physics_info)
        info = calculate_extra_info_fields(state,physics_info,num_legs=4,num_dimensions=2,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_ctrl_cost"]  
                 - state.metrics["reward_contact_cost"]  
                 + state.metrics["reward_survive"]) 
        info["fitness_forward"] = state.metrics["reward_forward"]

        return state.replace(info=info)


    def _step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        x_before = state.qp.pos[0, 0]
        x_after = qp.pos[0, 0]
        forward_reward = (x_after - x_before) / self.sys.config.dt
        ctrl_cost = .5 * jp.sum(jp.square(action))
        contact_cost = (0.5 * 1e-3 *
                        jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
        survive_reward = jp.float32(1)
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        done = jp.where(qp.pos[0, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
        done = jp.where(qp.pos[0, 2] > 1.0, x=jp.float32(1), y=done)
        state.metrics.update(
            reward_ctrl_cost=ctrl_cost,
            reward_contact_cost=contact_cost,
            reward_forward=forward_reward,
            reward_survive=survive_reward)

        return state.replace(qp=qp, obs=obs, reward=reward, done=done),info


class FelxibleWalker(walker2d.Walker2d):
    FOOT_INDEX = [3,6]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        #state = super().step(state,action)
        state,physics_info = self._step(state,action)
        info = calculate_extra_info_fields(state,physics_info,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_ctrl"]  
                 + state.metrics["reward_healthy"]) 
        info["fitness_forward"] = state.metrics["reward_forward"]

        return state.replace(info=info)
    
    def _step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        # Reverse torque improves performance over a range of hparams.
        qp, info = self.sys.step(state.qp, -action)
        obs = self._get_obs(qp)

        # Ignore the floor at last index.
        pos_before = state.qp.pos[:-1]
        pos_after = qp.pos[:-1]
        com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
        com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
        x_velocity = (com_after[0] - com_before[0]) / self.sys.config.dt
        forward_reward = self._forward_reward_weight * x_velocity

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        rewards = forward_reward + healthy_reward

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        costs = ctrl_cost

        reward = rewards - costs

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_healthy=healthy_reward)
        return state.replace(qp=qp, obs=obs, reward=reward, done=done),info

class FelxibleHalfCheetah(halfcheetah.Halfcheetah):
    FOOT_INDEX = [3,6]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        #state = super().step(state,action)
        state,physics_info = self._step(state,action)
        info = calculate_extra_info_fields(state,physics_info,num_legs=2,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = - state.metrics["reward_ctrl_cost"] 
        info["fitness_forward"] = state.metrics["reward_forward"]
        return state.replace(info=info)
    
    def _step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        x_before = state.qp.pos[0, 0]
        x_after = qp.pos[0, 0]
        forward_reward = (x_after - x_before) / self.sys.config.dt
        ctrl_cost = -.1 * jp.sum(jp.square(action))
        reward = forward_reward + ctrl_cost
        state.metrics.update(
            reward_ctrl_cost=ctrl_cost, reward_forward=forward_reward)

        return state.replace(qp=qp, obs=obs, reward=reward),info


class FelxibleHopper(hopper.Hopper):
    FOOT_INDEX = [3]
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info = init_extra_info_fields(state,num_legs=1,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        return state.replace(info=info)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        #state = super().step(state,action)
        state,physics_info = self._step(state,action)
        info = calculate_extra_info_fields(state,physics_info,num_legs=1,num_dimensions=1,foot_indicies=self.FOOT_INDEX)
        info["fitness_control_cost_and_survive"] = (
                 - state.metrics["reward_ctrl"]  
                 + state.metrics["reward_healthy"]) 
        info["fitness_forward"] = state.metrics["reward_forward"]

        return state.replace(info=info)

    def _step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        # Reverse torque improves performance over a range of hparams.
        qp, info = self.sys.step(state.qp, -action)
        obs = self._get_obs(qp)

        # Ignore the floor at last index.
        pos_before = state.qp.pos[:-1]
        pos_after = qp.pos[:-1]
        com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
        com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
        x_velocity = (com_after[0] - com_before[0]) / self.sys.config.dt
        forward_reward = self._forward_reward_weight * x_velocity

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        rewards = forward_reward + healthy_reward

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        costs = ctrl_cost

        reward = rewards - costs

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_healthy=healthy_reward)
        return state.replace(qp=qp, obs=obs, reward=reward, done=done),info


######################
## DIRECTIONAL ENVS ##
######################


def add_direction_to_obs(obs,direction):
    new_obs = jnp.concatenate([obs,direction])
    return new_obs

def init_extra_info_for_directional(state,rng):
    info = state.info
    angle = jax.random.uniform(rng, shape=[1], minval=-np.pi, maxval=np.pi)
    info["direction"] = jnp.array([jnp.cos(angle),jnp.sin(angle)]).reshape(-1)
    obs = add_direction_to_obs(state.obs,info["direction"])
    return info,obs

def calculate_extra_info_for_directional(state):
    info = state.info
    info["fitness_directional_distance"] = jnp.dot(info["bd_final_pos"],info["direction"])
    obs = add_direction_to_obs(state.obs,info["direction"])
    return info,obs

class DirectionalHumanoid(FelxibleHumanoid):
        
    #@property
    #def observation_size(self) -> int:
    #    return super().observation_size + 2 # add the direction as input
    
    def reset(self, rng: jp.ndarray) -> env.State:
        state = super().reset(rng)
        info,obs = init_extra_info_for_directional(state,rng)
        return state.replace(obs=obs,info=info)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super().step(state,action)
        info,obs = calculate_extra_info_for_directional(state)
        return state.replace(obs=obs,info=info)
    
class DirectionalAnt(FelxibleAnt):
        
    #@property
    #def observation_size(self) -> int:
    #    return super().observation_size + 2 # add the direction as input
    
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















