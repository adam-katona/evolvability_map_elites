
import copy
#from dask.distributed import Client
#import dask
import os

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

def set_up_dask_client(n_workers=50):
    import shutil
    #shutil.rmtree("/home/userfs/a/ak1774/workspace/evolvability_map_elites/evolvability_map_elites/dask-worker-space")
    
    from dask.distributed import Client
    import dask
    #dask.config.set("temporary-directory", "/scratch/ak1774/runs/dask_tmp")
    
    client = Client(n_workers=n_workers, threads_per_worker=1,local_dir="/scratch/ak1774/runs/dask_tmp")
    def set_up_worker():
        import os
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1" 
        os.environ["OMP_NUM_THREADS"] = "1" 
    print("running worker setup")
    client.run(set_up_worker)
    print("worker setup done")
    return client

def run_single_experiment():
    
    #import os
    #os.chdir("/scratch/ak1774/runs")
    
    wandb.init(project="ga_map_elites_debug", entity="adam_katona",config=ga_default_config,dir="/scratch/ak1774/runs") # this is a bit vauge how this happens but default will not set values already set by a sweep
    config = wandb.config
    print(config)
    
    run_name = wandb.run.dir.split("/")[-2]
    run_checkpoint_path = "/scratch/ak1774/runs/large_files/" + run_name
    os.makedirs(run_checkpoint_path,exist_ok=True)
    
    from es_map import map_elites
    
    print("setting up client")
    client = set_up_dask_client()
    print("client set up finished")
    map_elites.run_ga_map_elites(client,config)

if __name__ == '__main__':
    

        
    
    # start a weights and biases sweep to run all the GA experiments 


    # we have 4 settings
    # single or multi parent
    # ranked selection or uniform

    # Also we want to run a few environments (ant, humanoid)

    # these are the default values, the sweep will overwrite this
    ga_default_config = {
        "env_id" : "DamageAnt-v2",
        "policy_args" : {
            "init" : "normc",
            "layers" :[256, 256],
        "activation" : 'tanh',
        "action_noise" : 0.01,
        },
        "env_args" : {
            "use_norm_obs" : True,
        },
        
        "ES_popsize" : 100,
        "ES_sigma" : 0.02,
        "ES_EVALUATION_BATCH_SIZE" : 5,
        "ES_lr" : 0.01,
        
        "ES_CENTRAL_NUM_EVALUATIONS" : 30,
        
        "GA_MAP_ELITES_NUM_GENERATIONS" : 1000,
        
        "GA_CHILDREN_PER_GENERATION" : 200,
        "GA_NUM_EVALUATIONS" : 10,
        
        "GA_MULTI_PARENT_MODE" : True,
        "GA_PARENT_SELECTION_MODE" : "rank_proportional",  # "uniform", "rank_proportional"
        "GA_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS" : 1.0,  # 0.0 uniform, 1.0 normal , higher more agressive
        "GA_MUTATION_POWER" : 0.02,
        
        "map_elites_grid_description" : {
            "bc_limits" : [[0,1],[0,1],[0,1],[0,1]],
            "grid_dims" : [6,6,6,6],
        },
        
        "CHECKPOINT_FREQUENCY" : 100,
        "PLOT_FREQUENCY" : 100,
    }


    sweep_configuration = {
        "name": "ga_sweep",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            "GA_MULTI_PARENT_MODE": {
                "values": [True,False],
            },
            "GA_PARENT_SELECTION_MODE" : {
                "values": ["uniform","rank_proportional"],
            }
        }
    }
        
        
    import sys
    if len(sys.argv) > 1:
        import wandb
        print("starting agent")
        wandb.agent(sys.argv[1], function=run_single_experiment)
        
    #import os
    #os.chdir("/scratch/ak1774/runs")
    
    TEST_RUN = False
    
    if TEST_RUN is False:
        import wandb
    
    
    
        sweep_id = wandb.sweep(sweep_configuration)
        
        # run the sweep
        wandb.agent(sweep_id, function=run_single_experiment)
    
    else:

        print("setting up client")
        client = set_up_dask_client(5)
        print("client set up finished")
        
        from es_map import map_elites
        
        map_elites.run_ga_map_elites(client,ga_default_config,wandb_logging=False)
    
    
    # local run
    
    
    