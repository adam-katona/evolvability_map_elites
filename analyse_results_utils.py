import os
import wandb
import copy 
import json

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np




# now some functions which can do this with multiple keys
def get_unique_combinations(runs,keys):
    all_tuples = []
    for run in runs:
        all_tuples.append(tuple(run["config"][key] for key in keys))
    return set(all_tuples)

def select_where_keys_are(runs,keys,key_values):
    selected_runs = []
    for run in runs:
        if tuple(run["config"][key] for key in keys) == key_values:
            selected_runs.append(run)
    return selected_runs
    
def group_by_keys(runs,keys):
    unique_vals = get_unique_combinations(runs,keys)
    return {val:select_where_keys_are(runs,keys,val) for val in unique_vals}

# group runs togeather which have same config but different run_id
def group_runs_with_same_conf(runs):
    grouped_runs = {}
    for run in runs:
        run_conf = copy.deepcopy(run["config"])
        run_conf["RUN_ID"] = 0
        run_conf = json.dumps(run_conf)
        if run_conf in grouped_runs:
            grouped_runs[run_conf].append(run)
        else:
            grouped_runs[run_conf] = [run]
    return grouped_runs

def group_runs_by_selection_type(runs):
    grouped_runs = {}
    for run in runs:
        run_conf = copy.deepcopy(run["config"])
        mode = run_conf["ES_PARENT_SELECTION_MODE"]
        run_conf["ES_PARENT_SELECTION_MODE"] = 0
        run_conf = json.dumps(run_conf)
        if run_conf in grouped_runs:
            grouped_runs[run_conf][mode] = run
        else:
            grouped_runs[run_conf] = {mode : run}
    return grouped_runs
    
    

def id_to_path(id):
    from glob import glob    
    path = glob("/scratch/ak1774/runs/large_files_jax/*"+id, recursive = False)[0]
    return path



