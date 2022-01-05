

import wandb

def my_train_func():

    default_config = {
        "a" : 2,
        "optim" : "sgd",
    }

    wandb.init(config=default_config) # this is a bit vauge how this happens but default will not set values already set by a sweep
    config = wandb.config
    print(config)
    
    from dask.distributed import Client
    client = Client(n_workers=2, threads_per_worker=1)
    def f():
        print("SETUP")
    client.run(f)

    def add_one(x):
        return x+1
    
    res = client.submit(add_one,config["a"])
    res = client.gather(res)

    wandb.log({"a": config["a"], "accuracy": res})
    
    wandb.finish()


def simplest_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    a = wandb.config.a
    wandb.log({"a": a, "accuracy": a + 1})



if __name__ == "__main__":


    sweep_configuration = {
            "name": "my-awesome-sweep",
            "metric": {"name": "accuracy", "goal": "maximize"}, # only for nice plotting and summary
            "method": "grid",
            "parameters": {
                "a": {
                    "values": [1, 2, 3, 4]
                }
            }
        }

    sweep_id = wandb.sweep(sweep_configuration)
    wandb.agent(sweep_id, function=my_train_func)

    #import sys
    #wandb.agent(sys.argv[1], function=simplest_train_func)