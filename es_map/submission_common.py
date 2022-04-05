

import os
import wandb

from es_map import jax_evaluate


def set_up_dask_client(n_workers=50):
    #import shutil
    #shutil.rmtree("/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/dask-worker-space")
    
    import resource
    soft_limit,hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print("Open file descriptor limit: ",soft_limit," ",hard_limit)
    if soft_limit < hard_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE,(hard_limit,hard_limit)) 
        print("Raised the limit to: ",resource.getrlimit(resource.RLIMIT_NOFILE))
    
    from dask.distributed import Client
    import dask
    #dask.config.set("temporary-directory", "/scratch/ak1774/runs/dask_tmp")
    
    #client = Client(n_workers=n_workers, threads_per_worker=1,local_directory="/scratch/ak1774/runs/dask_tmp")
    client = Client(n_workers=n_workers, threads_per_worker=1,local_directory="/tmp")
    def set_up_worker():
        import os
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1" 
        os.environ["OMP_NUM_THREADS"] = "1" 
        import numpy as np
        
    print("running worker setup")
    client.run(set_up_worker)
    print("worker setup done")
    return client

def get_bc_descriptor_for_env(env_id):
    if 'HumanoidDeceptive' in env_id:
        return {
            "bc_limits" : [[-50, 50], [-50, 50]],
            "grid_dims" : [91,91],
        }
    elif 'AntMaze' in env_id:
        return {
            "bc_limits" : [[-40, 40], [-40, 40]],
            "grid_dims" : [10,10],
        }
    elif 'DamageAnt' in env_id:
        return {
            "bc_limits" : [[0,1],[0,1],[0,1],[0,1]],
            "grid_dims" : [10,10,10,10],
        }
    elif 'QDAntBulletEnv' in env_id:
        return {
            "bc_limits" : [[0,1],[0,1],[0,1],[0,1]],
            "grid_dims" : [6,6,6,6],
        }
    elif 'QDWalker2DBulletEnv' in env_id:
        return {
            "bc_limits" : [[0,1],[0,1]],
            "grid_dims" : [32,32],
        }
    elif 'QDHalfCheetahBulletEnv' in env_id:
        return {
            "bc_limits" : [[0,1],[0,1]],
            "grid_dims" : [32,32],
        }
    elif 'QDHopperBulletEnv' in env_id:
        return {
            "bc_limits" : [[0,1]],
            "grid_dims" : [1000],
        }
    else:
        raise "Error, dont know bc dims for env"
       

    
       

def setup_wandb(config_defaults,project_name):
    
    
    import types
    if isinstance(wandb.config, types.FunctionType):
        # wandb config is a function, which means wandb is not inited, we are not in a sweep
        config_defaults["map_elites_grid_description"] = get_bc_descriptor_for_env(config_defaults["env_id"])
    else:
        config_defaults["map_elites_grid_description"] = get_bc_descriptor_for_env(wandb.config["env_id"])
    
    print(config_defaults["map_elites_grid_description"])
    
    wandb.init(project=project_name, entity="adam_katona",config=config_defaults,dir="/scratch/ak1774/runs") # this is a bit vauge how this happens but default will not set values already set by a sweep
    config = wandb.config
    print(config)
    
    
    run_name = wandb.run.dir.split("/")[-2]
    run_checkpoint_path = "/scratch/ak1774/runs/large_files/" + run_name
    os.makedirs(run_checkpoint_path,exist_ok=True)
    print("created folder: ",run_checkpoint_path)
    
    return config



    
def setup_wandb_jax(config_defaults,project_name):
    
    import types
    if isinstance(wandb.config, types.FunctionType):
        # wandb config is a function, which means wandb is not inited, we are not in a sweep
        bd_descriptor = jax_evaluate.brax_get_bc_descriptor_for_env(config_defaults["env_name"])
        config_defaults["map_elites_grid_description"] = bd_descriptor
    else:
        bd_descriptor = jax_evaluate.brax_get_bc_descriptor_for_env(wandb.config["env_name"])
        config_defaults["map_elites_grid_description"] = bd_descriptor

    
    
    
    wandb.init(project=project_name, entity="adam_katona",config=config_defaults,dir="/scratch/ak1774/runs") # this is a bit vauge how this happens but default will not set values already set by a sweep
    config = wandb.config
    print(config)
    
    
    run_name = wandb.run.dir.split("/")[-2]
    run_checkpoint_path = "/scratch/ak1774/runs/large_files_jax/" + run_name
    os.makedirs(run_checkpoint_path,exist_ok=True)
    print("created folder: ",run_checkpoint_path)

    return config


