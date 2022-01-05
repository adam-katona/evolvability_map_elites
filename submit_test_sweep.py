import wandb

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

