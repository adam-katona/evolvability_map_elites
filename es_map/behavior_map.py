

import numpy as np

# The behaviour map can be 
# - grid
# - cvt  (see paper: Scaling Up MAP-Elites Using Centroidal Voronoi Tessellations)


# For now we only implement grid, but have a general API

#  The grid is described by config["map_elites_grid_description"]
# - bc_limits  eg: [[-1,1][-1,1]]
# - grid_dims   eg: [32,32]


# API for behaviour map
class Behavior_map_base:
    def __init__(self,config):
        self.data = None  # has the data in a numpy array
        
    def get_cell_coords(self,bc,config):
        # return cell coords belonging to bc
        # to get the cell do self.data[coords]
        pass
        
    def get_non_empty_cells(self,config):
        # returs list of non-empty cells
        pass

    




class Grid_behaviour_map:
    def __init__(self,config):
        self.data = np.empty(shape = config["map_elites_grid_description"]["grid_dims"], dtype=object)
        
    def get_cell_coords(self,bc,config):
        coords = []    
        for bc_dim,behaviour in enumerate(bc):  # NOTE, could do this vectorized...
            
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
        return tuple(coords) # tuple conversion needed, so can be used to index self.data
        
    def get_non_empty_cells(self,config):
        non_empty_mask = self.data != None
        if np.sum(non_empty_mask) == 0:
            return []
        else:
            return self.data[non_empty_mask]
    

# Centroidal Voronoi Tessellations, useful for high dimensional behavior spaces
class CVT_behaviour_map:
    pass
    




