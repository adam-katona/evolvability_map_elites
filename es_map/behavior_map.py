

import numpy as np

# The behaviour map can be 
# - grid
# - cvt  (see paper: Scaling Up MAP-Elites Using Centroidal Voronoi Tessellations)

#  The grid is described by config["map_elites_grid_description"]
# - bc_limits  eg: [[-1,1][-1,1]]
# - grid_dims   eg: [32,32]


def _get_cell_coords(bc,config):
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



# there are 3 types of b_map
# - single map
# - multi map
# - nd_sorted map

def create_b_map_grid(config):
    if config["BMAP_type_and_metrics"]["type"] == "single_map":
        b_map = Grid_behaviour_map(config)
    elif config["BMAP_type_and_metrics"]["type"] == "nd_sorted_map":
        b_map = Grid_behaviour_map(config)
    elif config["BMAP_type_and_metrics"]["type"] == "multi_map":
        b_map = Grid_behaviour_multi_map(config)
    else:
        raise "Unknown BMAP_type"
    return b_map


class Grid_behaviour_map:
    # This is a normal b_map with a single channel. 
    # This is used for single metric maps, and nd_sorted maps
    def __init__(self,config):
        self.config = config
        # sanity check
        b_map_type = config["BMAP_type_and_metrics"]["type"]
        if b_map_type == "multi_map":
            raise "ERROR, using the wrong kind of grid class"
        
        self.data = np.empty(shape = config["map_elites_grid_description"]["grid_dims"], dtype=object)
        
    def get_cell_coords(self,bc):
        return _get_cell_coords(bc,self.config)
        
    def get_non_empty_cells(self):
        non_empty_mask = self.data != None
        if np.sum(non_empty_mask) == 0:
            return []
        else:
            return self.data[non_empty_mask]
    
class Grid_behaviour_multi_map:
    def __init__(self,config):
        self.config = config
        self.b_map_type = config["BMAP_type_and_metrics"]["type"]
        self.b_map_metrics = config["BMAP_type_and_metrics"]["metrics"]
        self.num_metrics = len(self.b_map_metrics)
        
        if self.b_map_type != "multi_map":
            raise "ERROR, using the wrong kind of grid class"
        
        data_shape = [self.num_metrics,*config["map_elites_grid_description"]["grid_dims"]]
        self.data = np.empty(shape = data_shape, dtype=object)
        
    def get_cell_coords(self,bc,metric):
        coords = _get_cell_coords(bc,self.config)
        channel_i = self.get_metric_index(metric)
        return tuple([channel_i,*coords])
            
    
    def get_metric_index(self,metric):
        for channel_i,m in enumerate(self.b_map_metrics):
            if m == metric:
                return channel_i
        else:
            raise "bmap metric not in metric list"
    
    def get_non_empty_cells(self,metric):
        
        selected_channel_i = self.get_metric_index(metric)
        data_to_check_for_nonempty = self.data[selected_channel_i]
            
        non_empty_mask = data_to_check_for_nonempty != None
        if np.sum(non_empty_mask) == 0:
            return []
        else:
            return self.data[selected_channel_i][non_empty_mask]
           
    def get_ed_score(self,evolvability_type):
        
        def get_evolvability_or_zero(val):
            if val is None:
                return 0.0
            else:
                return val["elite"][evolvability_type]
            
        evo_or_zero = np.vectorize(get_evolvability_or_zero)(self.data)
        # maximize over channels
        evo_or_zero = np.max(evo_or_zero,axis=0)
        return np.sum(evo_or_zero)
        
        
        #best_evolvability_or_zero = np.zeros_like(self.data[0])
        #for channel_i,m in enumerate(self.b_map_metrics):
        #    channel_evolvability_or_zero = np.zeros_like(self.data[0])
        #    non_empty_mask = self.data[channel_i] != None
         #   
        #    evo_or_zero = np.array(map(f, x))
        #    
        #    
        #    channel_evolvability_or_zero[non_empty_mask] = self.data[channel_i][non_empty_mask]["elite"][evolvability_type]
        #    
        #    best_evolvability_or_zero = np.max(channel_evolvability_or_zero,best_evolvability_or_zero)
        
        #return np.sum(best_evolvability_or_zero)
        
        
        


# Centroidal Voronoi Tessellations, useful for high dimensional behavior spaces
class CVT_behaviour_map:
    pass
    




