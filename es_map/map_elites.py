

from es_map import behavior_map
from es_map import es_update

from es_map import map_elite_utils

import random
import numpy as np

    
###################
## GA MAP ELITES ##
###################


def run_ga_map_elites(client,config):
    pass



###################
## ES MAP ELITES ##
###################

def run_es_map_elites(client,config):
    pass









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








