

import wandb

import wandb
sweep_configuration = {
    "name": "my-awesome-sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},
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

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init(config=default_config)
    a = wandb.config.a
    
    print(wandb.config)

    wandb.log({"a": a, "accuracy": a + 1})

sweep_id = wandb.sweep(sweep_configuration)

# run the sweep
wandb.agent(sweep_id, function=my_train_func)