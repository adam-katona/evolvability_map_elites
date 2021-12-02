

from es_map import behavior_map
from es_map import es_update

import random
import numpy as np

    


def select_parent_cell(grid,config,mode="Exploit"):
    # select a random non empty cell
    # if all cells are empty, return None
    non_empty_mask = grid != None
    if sum(non_empty_mask) == 0:
        return None
    else:
        # TODO use mode to filter cells further which have elites with that mode
        non_empty_coords = np.where(non_empty_mask)
        # TODO do different kinds of weighted selection
        choosen_cell = select_parent_uniform_probability(grid,non_empty_coords)
        return choosen_cell
    
def select_parent_uniform_probability(grid,non_empty_coords):
    choosen_coords = tuple(random.choice(non_empty_coords))     # convert the tuple, so grid[coords] gets you grid[coord[0],coord[1]] 
    return grid[choosen_coords]                                 # instead of [grid[coords[0]],grid[coords[1]]]
    
    
def get_grid_coordinates(bc,config):
    coords = []    
    for bc_dim,behaviour in enumerate(bc):
        
        limits = config["map_elites_grid_description"]["bc_limits"][bc_dim]
        num_grids = config["map_elites_grid_description"]["grid_dims"][bc_dim]
        
        if behaviour <= limits[0]:
            coord = 0
        elif behaviour >= limits[1]:
            coord = num_grids-1
        else:
            step = (limits[1]-limits[0]) / num_grids
            coord = int((behaviour - limits[0]) / step)
        coords.append(coord)
    return tuple(coords)
    
    



def select_parent_elite(parent_cell,config):
    if parent_cell is None:
        return get_random_individual(config)
    else:
        
        if config["MAP_TYPE"] == "SINGLE":
            return parent_cell["elite"]
        
        elif config["MAP_TYPE"] == "MULTIPLE_INDEPENDENT":
            # Select one from the type of elites, and return the elite of that type
            raise "TODO, not implemented"
        
        elif config["MAP_TYPE"] == "MULTIPLE_ND_SORT":
            # then select a parent randomly from the maintained elites in the pareto front
            raise "TODO, not implemented"
        
        else:
            raise "Unknown MAP_TYPE"








