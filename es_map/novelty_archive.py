

from sklearn.neighbors import NearestNeighbors
import numpy as np


class NoveltyArchive:
    def __init__(self,bc_dim):
        
        self.bc_dim = bc_dim
        self.all_bcs = []
        self.nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
        self.model_needs_fitting = True


    def add_to_archive(self,bc):
        self.all_bcs.append(bc)
        self.model_needs_fitting = True
        
    def ensure_model_fitted(self):
        if len(self.all_bcs) > 0:
            if self.model_needs_fitting is True:
                self.nn_model.fit(np.array(self.all_bcs).reshape(-1,self.bc_dim))
                self.model_needs_fitting = False
        
        
    def calculate_novelty(self,bcs,k_neerest=5):
        if len(self.all_bcs) > 0:
            self.ensure_model_fitted()
            distances, indicies = self.nn_model.kneighbors(np.array(bcs).reshape(-1,self.bc_dim), 
                                                           n_neighbors=min(k_neerest,len(self.all_bcs)))  # TODO ensure bcs is in the right format
            return np.mean(distances)
        else:
            # empty archive, cannot calculate novelty, return 0 (which corresponds to not novel at all)
            return 0
        
            
        
        
        

