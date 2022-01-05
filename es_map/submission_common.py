

import os
import wandb


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

def setup_wandb(config_defaults,project_name):
    wandb.init(project=project_name, entity="adam_katona",config=config_defaults,dir="/scratch/ak1774/runs") # this is a bit vauge how this happens but default will not set values already set by a sweep
    config = wandb.config
    print(config)
    
    run_name = wandb.run.dir.split("/")[-2]
    run_checkpoint_path = "/scratch/ak1774/runs/large_files/" + run_name
    os.makedirs(run_checkpoint_path,exist_ok=True)
    print("created folder: ",run_checkpoint_path)
    
    return config


    
    




