
import copy

# start a weights and biases sweep to run all the GA experiments 


# we have 4 settings
# single or multi parent
# ranked selection or uniform

# Also we want to run a few environments (ant, humanoid)

ga_default_config = {
    
}


import wandb
sweep_configuration = {
    "name": "ga_sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {
        "a": {
            "values": [1, 2, 3, 4]
        }
    }
}

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    a = wandb.config.a

    wandb.log({"a": a, "accuracy": a + 1})

sweep_id = wandb.sweep(sweep_configuration)

# run the sweep
wandb.agent(sweep_id, function=my_train_func)



def run_single_experiment():
    wandb.init()
    a = wandb.config.a
    
    config = copy.deepcopy(ga_default_config)
    
    
    
    
    