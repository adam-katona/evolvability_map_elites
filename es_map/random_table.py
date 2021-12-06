import numpy as np
import torch

# This module is used like a singleton
# Only import it if you want to load the the random table into memory.
# On repeated import, it is not loaded again.
RAND_TABLE_SIZE = 100 * 1000 * 1000

np.random.seed(42)
noise_table = np.random.randn(RAND_TABLE_SIZE).astype(np.float32)

np.random.seed()

def get_random_indicies(param_dim,num_indicies):
    return np.random.randint(0,RAND_TABLE_SIZE-param_dim,num_indicies)