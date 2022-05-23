import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.spatial.distance
import jax.numpy as jnp

from es_map import nd_sort
        
def calculate_behavioural_variance(child_evaluations,config):
    bcs = child_evaluations["bcs"]
    bc_mean = np.mean(bcs,axis=0)
    contributions_to_variance = np.sum(((bcs - bc_mean) ** 2),axis=1)
    evolvability_of_parent = np.mean(contributions_to_variance) # NOTE this is not really a variance, because BC is a vector
    return evolvability_of_parent                               # it is the summed variance of each component of BC
    
    
def calculate_behavioural_distribution_entropy(child_evaluations,config):
    def kernel_func(b):
        k_sigma = 1.0  # kernel standard deviation
        pairwise_sq_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(b.reshape(-1, 1), "sqeuclidean"))
        # pylint: disable=invalid-unary-operand-type
        k = np.exp(-pairwise_sq_dists / k_sigma ** 2)
        return k
    
    bcs = child_evaluations["bcs"]
    entropy = -np.log(kernel_func(bcs).mean(axis=1)).mean()
    return entropy

def calculate_innovativeness(child_evaluations,novelty_archive,config): # innovation is excpected novelty
    novelties = novelty_archive.calculate_novelty(child_evaluations["bcs"],k_neerest=config["NOVELTY_CALCULATION_NUM_NEIGHBORS"])
    return np.mean(novelties) # expected novelty of children
        
def create_id_generator():
    counter = itertools.count()
    def get_next_id():
        return next(counter)
    return get_next_id

def get_random_individual(config):
    from es_map.interaction import interaction
    env = interaction.build_interaction_module(config["env_id"],config)
    theta = env.initial_theta()
    return theta

def get_obs_shape(config):
    from es_map.interaction import interaction
    env = interaction.build_interaction_module(config["env_id"],config)
    return env.env.observation_space.shape[0]

def calculate_obs_stats(obs_stats):
    sum = obs_stats["sum"]
    sumsq = obs_stats["sumsq"]
    count = obs_stats["count"]
    
    if count == 0:
        return np.zeros_like(sum),np.ones_like(sum)
    mean = sum / count
    std = np.sqrt(np.maximum(sumsq / count - np.square(mean), 1e-2))
    return mean,std

def rank_based_selection(num_parent_candidates,num_children,agressiveness=1.0):
    # agressiveness decide how agressively we want to select higher ranking individuals over lower ranking ones.
    # 1 is normal rank based probability, 0 is uniform,
    # The higher it is the more agressive the selection is.
    
    p = np.array(list(range(num_parent_candidates))) + 1.0
    p = p ** agressiveness
    p = p / np.sum(p)
    
    selected_indicies = np.random.choice(num_parent_candidates,size=num_children, replace=True, p=p)
    return selected_indicies


def plot_b_map(b_map,metric="eval_fitness",config=None,plt_inline_mode=False):
    
    data = []
    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        for val in b_map.data.reshape(-1):
            if val is not None:
                if val["elite"][metric] is not None:
                    data.append(val["elite"][metric])
                else:
                    data.append(None)
            else:
                data.append(None)
        data = np.array(data).reshape(*b_map.data.shape)
        
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        metric_i = b_map.get_metric_index(metric)
        for val in b_map.data[metric_i].reshape(-1):
            if val is not None:
                if val["elite"][metric] is not None:
                    data.append(val["elite"][metric])
                else:
                    data.append(None)
            else:
                data.append(None)
        data = np.array(data).reshape(*b_map.data[metric_i].shape)
        
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        for val in b_map.data.reshape(-1):
            if val is not None:
                metric_values = [elite[metric] for elite in val["elites"] if elite[metric] is not None]
                if len(metric_values) > 0:
                    data.append(max(metric_values))
                else:
                    data.append(None)
            else:
                data.append(None)
        data = np.array(data).reshape(*b_map.data.shape)
    
    if len(data.shape) == 4:
        return plot_4d_map(data,metric)
    elif len(data.shape) == 2:
        if plt_inline_mode is True:
            plt.imshow(data.astype(float))
            plt.colorbar()
        else:
            fig,ax = plt.subplots()
            ax.imshow(data.astype(float))
            return fig,ax
    elif len(data.shape) == 1:
        # 1d grid, make it 2d
        size = data.shape[0]
        # here you would find the closest divisor to the square root.
        # Or even it does not have to be a divisor, just fill with None-s in the end
        img_size = int(np.ceil(np.sqrt(size)))
        img = np.array([None]*(img_size*img_size))
        img[:size] = data
        img = img.reshape([img_size,img_size])
        if plt_inline_mode is True:
            plt.imshow(img.astype(float))
            plt.colorbar()
        else:
            fig,ax = plt.subplots()
            ax.imshow(img.astype(float))
            return fig,ax
        
    else:
        raise "Error, dont know how to plot b_map with shape: " + str(data.shape)
    
    

def plot_4d_map(data_4d,plt_inline_mode=False):
    
    dim = data_4d.shape
    data_2d = np.zeros([dim[0]*dim[2],dim[1]*dim[3]])
    for coord1 in range(dim[0]):
        for coord2 in range(dim[1]):
            data_2d[coord1*dim[2]:(coord1+1)*dim[2],coord2*dim[3]:(coord2+1)*dim[3]] = data_4d[coord1,coord2]

    if plt_inline_mode is True:
        import matplotlib.pyplot as plt
        plt.imshow(data_2d.astype(float))
        plt.colorbar()
    else:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.imshow(data_2d.astype(float))
        #ax.colorbar()
    
        return fig,ax



def get_nd_sorted_first_front(all_individuals,metrics):
    all_individual_metrics = []
    for individual in all_individuals:
        individual_metrics = []
        for metric in metrics:
            val = individual[metric]
            if val is None:
                if metric == "evolvability" or metric == "innovation":
                    val = 0.0 # this is all right, in some cases we dont evaluate children, but evaluate fitness
                              # both are positive numbers, set them to 0 which is the worst possible value for them 
                else:
                    raise "Error: fitness is None, this is not excpected"
            individual_metrics.append(val)
        all_individual_metrics.append(individual_metrics)
    all_individual_metrics = np.stack(all_individual_metrics)
    
    first_front_sorted_indicies = nd_sort.nd_sort_get_first_front(all_individual_metrics)
    return first_front_sorted_indicies

def calculate_qd_score(b_map,config):
    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        non_empty_cells = b_map.get_non_empty_cells()
        qd_score = np.sum([cell["elite"]["eval_fitness"] for cell in non_empty_cells])
        
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        def get_fitness_if_non_empty(x):
            if x is not None:
                return x["elite"]["eval_fitness"]
            else:
                return 0.0
        vfunc = np.vectorize(get_fitness_if_non_empty)
        fitnesses = vfunc(b_map.data)
        best_fitnesses = np.max(fitnesses,axis=0)
        qd_score = np.sum(best_fitnesses)
        
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        non_empty_cells = b_map.get_non_empty_cells()
        qd_score = np.sum([np.max([e["eval_fitness"] for e in cell["elites"]]) for cell in non_empty_cells])
        
    return qd_score

def ga_select_parent_single_mode(non_empty_cells,config):
    # selects a single parent who is going to seed the next population
    # Used in the ES map elite paper as baseline, parralelizes to cluster well, 
    # but I think it is not ideal because the there are so few parent selected (1 per generation) 
    # returns single tuple 
    
    if len(non_empty_cells) == 0:
            parent_cell = None
            parent_params = get_random_individual(config)
            
    else:
        if config["GA_PARENT_SELECTION_MODE"] == "uniform":
            parent_cell = np.random.choice(non_empty_cells)  # NOTE, here goes cell selection method
        elif config["GA_PARENT_SELECTION_MODE"] == "rank_proportional":
            sorted_cells = sorted(non_empty_cells,key=lambda x : x["eval_fitness"])                
            selected_index = rank_based_selection(num_parent_candidates=len(sorted_cells),
                                                     num_children=None, # only want a single parent
                                                     agressiveness=config["GA_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
            parent_cell = sorted_cells[selected_index]

        else:
            raise "NOT IMPLEMENTED"

        parent_params = parent_cell["params"]
        
    return parent_params
    
    
def ga_select_parents_multi_parent_mode(non_empty_cells,config):
    # select a separate parent for every new child in the population
    # need to send separate parameter for each worker (not parraliliz very well), 
    # but I feel like this will be faster to explore and solve problems (we will see)
    parent_param_list = []
        
    if len(non_empty_cells) == 0: # no parent available, create population from random individuals
        parent_param_list = [get_random_individual(config) for parent_i in range(config["GA_CHILDREN_PER_GENERATION"])]

    else:  # we have parents available
        if config["GA_PARENT_SELECTION_MODE"] == "uniform":
            parent_cell_list = np.random.choice(non_empty_cells,size=config["GA_CHILDREN_PER_GENERATION"])
        elif config["GA_PARENT_SELECTION_MODE"] == "rank_proportional":
            sorted_cells = sorted(non_empty_cells,key=lambda x : x["eval_fitness"])
            selected_indicies = rank_based_selection(num_parent_candidates=len(sorted_cells),
                                                     num_children=config["GA_CHILDREN_PER_GENERATION"],
                                                     agressiveness=config["GA_RANK_PROPORTIONAL_SELECTION_AGRESSIVENESS"])
            parent_cell_list = [sorted_cells[parent_i] for parent_i in selected_indicies]
        else:
            raise "NOT IMPLEMENTED"

        parent_param_list = [cell["params"] for cell in parent_cell_list]

    return parent_param_list
    
    
def recalculate_innovation(individual,b_archive,batch_calculate_novelty_fn,config):
    novelties = batch_calculate_novelty_fn(individual["child_eval"]["rollout_results"]["bds"][:config["ES_popsize"]],b_archive)
    innovation = jnp.mean(novelties).item()
    individual["innovation"] = innovation    

def try_insert_into_coords(new_individual,map_coords,metric,b_map,b_archive,config,batch_calculate_novelty_fn):
    needs_adding = False
    if b_map.data[map_coords] is None:
        needs_adding = True
    else:
        if metric == "innovation": # recalculate innovation
            recalculate_innovation(b_map.data[map_coords]["elite"],b_archive,batch_calculate_novelty_fn,config)
            
        if b_map.data[map_coords]["elite"][metric] < new_individual[metric]:
            needs_adding = True
        
    if needs_adding is True:
        new_cell = {"elite" : new_individual}
        b_map.data[map_coords] = new_cell
        return True
    return False
    

def try_insert_individual_in_map(new_individual,b_map,b_archive,config,batch_calculate_novelty_fn=None):
    # if multi map, try inserting it to each map
    # if nd sorted, but we only have fitness, assume evolvability and innovation are 0, and try to insert it that way
    # b_archive is needed to recalculate novelty of the elites, to see if they decreased since we last calculated it
    metrics = config["BMAP_type_and_metrics"]["metrics"]
    
    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        metric = metrics[0] # single map only have one metric
        if len(metrics) > 1:
            raise "Error, single map should only have one metric!!"
        
        map_coords = b_map.get_cell_coords(new_individual["eval_bc"])
        return try_insert_into_coords(new_individual,map_coords,metric,b_map,b_archive,config,batch_calculate_novelty_fn)
        
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        # for multi map, try inserting the individual in each map
        if len(metrics) < 2:
            raise "Error, why use multi map if you only have one metric!!"
        
        any_inserted = False
        for metric in metrics:
            map_coords = b_map.get_cell_coords(new_individual["eval_bc"],metric)
            inserted = try_insert_into_coords(new_individual,map_coords,metric,b_map,b_archive,config,batch_calculate_novelty_fn)
            any_inserted = any_inserted or inserted
        return any_inserted
        
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        # nd uses all the metrics at the same time, to sort individuals, and keep the 
        # least crowded n from the nondominated front
        
        if len(metrics) < 2:
            raise "Error, why do nd_sorted_map if you dont specify multiple objectives for it!!"
        
        map_coords = b_map.get_cell_coords(new_individual["eval_bc"])
        if b_map.data[map_coords] is None:
            # emtpy, just add it
            b_map.data[map_coords] = {"elites" : [new_individual]}
            return True
        else:
            current_elites = b_map.data[map_coords]["elites"]
            if "innovation" in metrics: # make sure innovation is up to date
                for elite in current_elites:
                    recalculate_innovation(elite,b_archive,batch_calculate_novelty_fn,config)
                        
            all_contestants = current_elites
            all_contestants.append(new_individual)
            new_id = len(all_contestants)

            first_front_sorted_indicies = get_nd_sorted_first_front(all_contestants,metrics)
            # take the first n elements (or as many as there is)
            sorted_contestants = [all_contestants[idx] for idx in first_front_sorted_indicies]
            new_elites = sorted_contestants[:config["ES_ND_SORT_MAX_FRONT_SIZE_TO_KEEP"]]
            b_map.data[map_coords]["elites"] = new_elites

            # test if we actually inserted the new individual
            if new_id in first_front_sorted_indicies[:config["ES_ND_SORT_MAX_FRONT_SIZE_TO_KEEP"]]:
                return True
            return False