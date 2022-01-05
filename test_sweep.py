

import wandb

import wandb

def my_train_func():

    wandb.init(config=default_config) # this is a bit vauge how this happens but default will not set values already set by a sweep
    config = wandb.config
    print(config)
    
    from dask.distributed import Client
    client = Client(n_workers=2, threads_per_worker=1)
    def f():
        print("SETUP")
    client.run(f)

    wandb.log({"a": config["a"], "accuracy": config["a"] + 1})

if __name__ == "__main__":

    sweep_configuration = {
        "name": "my-awesome-sweep",
        "metric": {"name": "best_fitness_so_far", "goal": "maximize"}, # only for nice plotting and summary
        "method": "grid",
        "parameters": {
            "a": {
                "values": [1, 2, 3, 4]
            }
        }
    }

    default_config = {
        "a" : 2,
        "optim" : "sgd",
    }



    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=my_train_func)