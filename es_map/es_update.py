




def es_update(params,children_evaluations,mode,config,bc_archive=None):
    # bc_archive is only needed if we want to calculate innovation (for now we always do)
    # do a whole es update
    
    evaluations = {
        "seed" : 42,
        "fitnesses" : torch.randn(10),
        "bc" : torch.randn(10,2),
    }
    
    new_params = params + torch.randn_like(params)
    
    return new_params,evaluations



def calculate_var_evolvability(evaluations):
    # evolvability is the excpected variance of the behaviour (summed over components)
    mean_bc = toch.mean(evaluations["bc"],dim=0)
    evolvability = torch.sum((evaluations["bc"] - mean_bc) ** 2) / evaluations["bc"].shape[0]
    return evolvability


def calcualte_innovation(evaluations,bc_archive):
    # innovation is the excpected novelty
    pass