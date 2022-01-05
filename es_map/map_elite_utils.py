import numpy as np
import itertools
        
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

def calculate_obs_stats(sum,sumsq,count):
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

def plot_4d_map(b_map,metric="eval_fitness",plt_inline_mode=False):
    data_4d = []
    for val in b_map.data.reshape(-1):
        if val is not None:
            if val["elite"][metric] is not None:
                data_4d.append(val["elite"][metric])
            else:
                data_4d.append(None)
        else:
            data_4d.append(None)
    data_4d = np.array(data_4d).reshape(*b_map.data.shape)
    
    dim = b_map.data.shape
    data_2d = np.zeros([dim[0]*dim[2],dim[1]*dim[3]])
    for coord1 in range(dim[0]):
        for coord2 in range(dim[1]):
            data_2d[coord1*dim[2]:(coord1+1)*dim[2],coord2*dim[3]:(coord2+1)*dim[3]] = data_4d[coord1,coord2]

    if plt_inline_mode is True:
        import matplotlib.pyplot as plt
        plt.imshow(data_2d)
        plt.colorbar()
    else:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.imshow(data_2d)
        #ax.colorbar()
    
        return fig,ax




def ga_select_parent_single_mode(non_empty_cells,config):
    # selects a single parent who is going to seed the next population
    # Used in the ES map elite paper as baseline, parralelizes to cluster well, 
    # but I think it is not ideal because the there are so few parent selected (1 per generation) 
    # returns single tuple 
    
    if len(non_empty_cells) == 0:
            parent_cell = None
            parent_params = get_random_individual(config)
            parent_obs_mean = None
            parent_obs_std = None
            
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
        parent_obs_mean,parent_obs_std = calculate_obs_stats( parent_cell["obs_stats"]["obs_sum"],
                                                              parent_cell["obs_stats"]["obs_sq"],
                                                              parent_cell["obs_stats"]["obs_count"])
        
    return parent_params,parent_obs_mean,parent_obs_std
    
    
def ga_select_parents_multi_parent_mode(non_empty_cells,config):
    # select a separate parent for every new child in the population
    # need to send separate parameter for each worker (not parraliliz very well), 
    # but I feel like this will be faster to explore and solve problems (we will see)
    parent_datas = []
        
    if len(non_empty_cells) == 0: # no parent available, create population from random individuals
        parent_list = [get_random_individual(config) for parent_i in range(config["GA_CHILDREN_PER_GENERATION"])]
        for parent_params in parent_list:
            parent_obs_mean = None
            parent_obs_std = None
            parent_datas.append((parent_params,parent_obs_mean,parent_obs_std))

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

        for parent_cell in parent_cell_list:
            parent_params = parent_cell["params"]
            parent_obs_mean,parent_obs_std = calculate_obs_stats( parent_cell["obs_stats"]["obs_sum"],
                                                                  parent_cell["obs_stats"]["obs_sq"],
                                                                  parent_cell["obs_stats"]["obs_count"])
            parent_datas.append((parent_params,parent_obs_mean,parent_obs_std))
    return parent_datas
    