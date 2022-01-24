from es_map import map_elite_utils
from es_map import es_update
from es_map import distributed_evaluate
from es_map.interaction import interaction
from es_map import behavior_map
from es_map import novelty_archive
from es_map import nd_sort

import random
import numpy as np
import copy
import pickle
import json
import matplotlib.pyplot as plt

import wandb



def get_individual_elite_mapping_performance(client,theta,b_map_cumm,obs_mean,obs_std,config):
    # This function is used to test an alternative to map elites to fill the map
    # Instead of map elites algorithm, we evolve an evolvable individual, and sample its offspring to fill the map
    
    # We call this function periodically, to get 2 measuremnets
    # Raw map elites performance (how well it is mapping elites with only the current individual offsping)
    # Cummulative map elites performance (how well we can map elites by peridically trying (like every 50 generation))
    
    # create a fresh b_map
    b_map = behavior_map.Grid_behaviour_map(config)
    
    # let us sample the offspring of theta
    child_results = distributed_evaluate.ga_evaluate_children_single_parent(client,theta,obs_mean,obs_std,config,
                                                       num_children=config["FILL_MAP_WITH_OFFSPRING_NUM_CHILDREN"],
                                                       num_evaluation_per_children=config["ES_CENTRAL_NUM_EVALUATIONS"])
    
    # try to insert them to both the new and the cummulative maps
    b_map_fillup_data = []
    b_map_cumm_fillup_data = []
    b_map_current_count = 0
    b_map_cumm_current_count = np.sum(b_map_cumm.data != None)
    for child_res in child_results:
        new_individual = {  # create an individual with a bunch of None-s since we dont really want to remember all this
                "params" : None, # i dont really need to keep this in memeory, maybe later
                "ID" : None, # Dont care
                "parent_ID" : None,
                "generation_created" : None,
                "child_eval" : None, #
                "eval_fitness" : child_res["mean_fitness"],
                "eval_bc" : child_res["mean_bc"],
                "evolvability" : None,
                "innovation" : None,
                "innovation_over_time" : None, 
        }
        b_map_added = map_elite_utils.try_insert_individual_in_map(new_individual,b_map,b_archive=None,config=config)
        b_map_cumm_added = map_elite_utils.try_insert_individual_in_map(new_individual,b_map_cumm,b_archive=None,config=config)
        b_map_current_count += int(b_map_added) 
        b_map_cumm_current_count += int(b_map_cumm_added) 
        b_map_fillup_data.append(b_map_current_count)
        b_map_cumm_fillup_data.append(b_map_cumm_current_count)
        
    # Now calculate the QD scores and nonempty ratios for both the cummulative and fresh map
    non_empty_cells = b_map.get_non_empty_cells()
    qd_score = np.sum([cell["elite"]["eval_fitness"] for cell in non_empty_cells])
    cumm_non_empty_cells = b_map_cumm.get_non_empty_cells()
    cumm_qd_score = np.sum([cell["elite"]["eval_fitness"] for cell in cumm_non_empty_cells])
    
    # also do some plots
    fig_f,ax = map_elite_utils.plot_b_map(b_map,metric="eval_fitness",config=config)
    fig_f_cumm,ax = map_elite_utils.plot_b_map(b_map_cumm,metric="eval_fitness",config=config)             
    
    # also plot how fast they accumulate
    b_map_fillup_data = np.array(b_map_fillup_data) / float(b_map.data.size)
    b_map_cumm_fillup_data = np.array(b_map_cumm_fillup_data) / float(b_map.data.size)
    fig_accumulation,ax = plt.subplots()
    ax.plot(b_map_fillup_data)
    ax.plot(b_map_cumm_fillup_data)
    
    results = {
        "nonempty_cells" : len(non_empty_cells),
        "nonempty_ratio" : float(len(non_empty_cells)) / b_map.data.size,
        "qd_score" : qd_score,
        "cumm_nonempty_cells" : len(cumm_non_empty_cells),
        "cumm_nonempty_ratio" : float(len(cumm_non_empty_cells)) / b_map_cumm.data.size,
        "cumm_qd_score" : cumm_qd_score,
        "b_map_plot" : fig_f,
        "b_map_cumm_plot" : fig_f_cumm,
        "b_map_accumulation_plot" : fig_accumulation,
    }
    return results
    

    
    
    
  
def run_plain_es(client,config,wandb_logging=True):
    if type(config) != type(dict()):
        config = config.as_dict()
        
    # TODO HACK this should be set elswhere properly
    from es_map import submission_common
    config["map_elites_grid_description"] = submission_common.get_bc_descriptor_for_env(config["env_id"])
    
    DEBUG = False
    if DEBUG is True:
        config["ES_NUM_GENERATIONS"] = 15
        config["ES_popsize"] = 12
        config["ES_NUM_INITIAL_RANDOM_INDIVIDUALS_TO_POPULATE_MAP"] = 1
        config["ES_CENTRAL_NUM_EVALUATIONS"] = 3
        config["FILL_MAP_WITH_OFFSPRING_NUM_CHILDREN"] = 5
        
    if wandb_logging is True:   
        run_name = wandb.run.dir.split("/")[-2]
    else:
        from datetime import datetime
        run_name = "local_dubug_run_" + datetime.now().strftime("_%m_%d___%H:%M")
    run_checkpoint_path = "/scratch/ak1774/runs/large_files/" + run_name
    
    b_archive = novelty_archive.NoveltyArchive(bc_dim = len(config["map_elites_grid_description"]["grid_dims"]))
    obs_shape = map_elite_utils.get_obs_shape(config)
    observation_stats = {                  # this is a single obs stats to keep track of during the whole experiment. 
        "sum" : np.zeros(obs_shape),       # This is always expanded, and always used to calculate the current mean and std
        "sumsq" : np.zeros(obs_shape),
        "count" : 0,
    }
    current_obs_mean,current_obs_std = map_elite_utils.calculate_obs_stats(observation_stats)  
    
    theta = map_elite_utils.get_random_individual(config)
    
    generation_number = 0
    evaluations_so_far = 0
    best_fitness_so_far = 0
    best_model_so_far = None
    best_evolvability_so_far = 0
    obs_stats_history = []
    optim_state=None
    b_map_cumm = behavior_map.Grid_behaviour_map(config) # this is only used to measure and plot, no role in the algo
    
    while True:
        if generation_number >= config["ES_NUM_GENERATIONS"]:
            print("Done, reached iteration: ",config["ES_NUM_GENERATIONS"])
            break
        
        child_evals = distributed_evaluate.es_evaluate_children(client,theta,obs_mean=current_obs_mean,
                                                                  obs_std=current_obs_std,config=config)
        
        
        # record the observation stats
        observation_stats["sum"] += child_evals["child_obs_sum"]
        observation_stats["sumsq"] += child_evals["child_obs_sq"]
        observation_stats["count"] += child_evals["child_obs_count"]
        current_obs_mean,current_obs_std = map_elite_utils.calculate_obs_stats(observation_stats)
        
        updated_theta,optim_state = es_update.es_update(theta,child_evals,config,es_update_type=config["ES_UPDATES_MODES_TO_USE"][0],
                                            novelty_archive=b_archive,optimizer_state=optim_state)
                
        current_evolvability = map_elite_utils.calculate_behavioural_variance(child_evals,config)
        current_innovation = map_elite_utils.calculate_innovativeness(child_evals,b_archive,config)
        current_entropy = map_elite_utils.calculate_behavioural_distribution_entropy(child_evals,config)
        
        
        eval_results = distributed_evaluate.evaluate_individual_repeated(theta=theta,
                                                                obs_mean=current_obs_mean,obs_std=current_obs_std,use_action_noise=False,record_obs=False,
                                                                config=config,repeat_n=config["ES_CENTRAL_NUM_EVALUATIONS"])
        
        b_archive.add_to_archive(eval_results["bc"])
        
        if current_evolvability > best_evolvability_so_far:
                    best_evolvability_so_far = current_evolvability
        if eval_results["fitness"] > best_fitness_so_far:
            best_fitness_so_far = eval_results["fitness"]
            
        if (generation_number % config["FILL_MAP_WITH_OFFSPRING_MEASURE_EVERY_N_GENERATIONS"] == 0 or
           generation_number == (config["ES_NUM_GENERATIONS"]-1)):  # or last generation
            map_perf = get_individual_elite_mapping_performance(client,theta=theta,b_map_cumm=b_map_cumm,
                                                                obs_mean=current_obs_mean,obs_std=current_obs_std,config=config)
            map_perf["generation_number"] = generation_number
            num_eval_per_measure = config["FILL_MAP_WITH_OFFSPRING_NUM_CHILDREN"] * config["ES_CENTRAL_NUM_EVALUATIONS"]
            map_perf["evaluations_so_far"] = generation_number*config["ES_popsize"] + num_eval_per_measure
            
            non_plot_stats = {key : val for key,val in map_perf.items() if "plot" not in key}
            print(json.dumps(non_plot_stats,indent=4))
            
            if wandb_logging is True:
                wandb.log(map_perf)
        
            
        # Do the step logging 
        step_logs = {
            "generation_number" : generation_number,
            "evaluations_so_far" : generation_number*config["ES_popsize"],
            "best_fitness_so_far" : best_fitness_so_far,
            "best_evolvability_so_far" : best_evolvability_so_far,
            "current_children_fitness_mean" : np.mean(child_evals["fitnesses"]),
            "current_children_fitness_std" : np.std(child_evals["fitnesses"]),
            "current_eval_fitness" : eval_results["fitness"],
            "current_evolvability" : current_evolvability,
            "current_innovation" : current_innovation,
            "current_entropy" : current_entropy,
        }
        if wandb_logging is True:
            wandb.log(step_logs)
        
        print(json.dumps(step_logs,indent=4))
           
        # Do checkpointing
        if generation_number % config["CHECKPOINT_FREQUENCY"] == 10:
            b_archive.save_to_file(run_checkpoint_path+"/b_archive.npy")
            obs_stats_history.append(copy.deepcopy(observation_stats))
            with open(run_checkpoint_path+'/obs_stats.pickle', 'wb') as handle:
                pickle.dump(obs_stats_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(run_checkpoint_path+"/theta.npy",theta)
            
        generation_number +=1
        theta = updated_theta
        
    b_archive.save_to_file(run_checkpoint_path+"/b_archive.npy")
    obs_stats_history.append(copy.deepcopy(observation_stats))
    with open(run_checkpoint_path+'/obs_stats.pickle', 'wb') as handle:
        pickle.dump(obs_stats_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(run_checkpoint_path+"/theta.npy",theta)
        