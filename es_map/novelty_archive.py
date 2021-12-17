

from sklearn.neighbors import NearestNeighbors
import numpy as np


class NoveltyArchive:
    def __init__(self,bc_dim):
        
        self.bc_dim = bc_dim
        self.all_bcs = []
        self.nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
        self.model_needs_fitting = True


    def load_from_file(self,path):
        data = np.load(path)
        self.all_bcs = list(data)
        self.model_needs_fitting = True
        
        
    def save_to_file(self,path):
        if len(self.all_bcs) > 0:
            data = np.stack(self.all_bcs)
            np.save(path,data)
        
        

    def add_to_archive(self,bc):
        self.all_bcs.append(bc)
        self.model_needs_fitting = True
        
    def ensure_model_fitted(self):
        if len(self.all_bcs) > 0:
            if self.model_needs_fitting is True:
                self.nn_model.fit(np.stack(self.all_bcs))  
                self.model_needs_fitting = False
        
        
    def calculate_novelty(self,bcs,k_neerest=5):
        if len(bcs.shape) == 1: # make it a 2d array, even if we have a single bc
            bcs = bcs.reshape(1,-1)
        num_bcs = bcs.shape[0]
        
        if len(self.all_bcs) > 0:
            self.ensure_model_fitted()
            distances, indicies = self.nn_model.kneighbors(np.array(bcs).reshape(-1,self.bc_dim), 
                                                           n_neighbors=min(k_neerest,len(self.all_bcs)))  # TODO ensure bcs is in the right format
            return np.mean(distances,axis=1)
        else:
            # empty archive, cannot calculate novelty, return 0 (which corresponds to not novel at all)
            return np.zeros(num_bcs)
        
            
        
        
        

