
import numpy as np


def nd_sort_get_first_front(fitnesses)

    fronts = calculate_pareto_fronts(fitnesses)
    nondomination_rank_dict = fronts_to_nondomination_rank(fronts)
    
    crowding = calculate_crowding_metrics(fitnesses,fronts)
    
    # Sort the population, this returns a list where the best is first
    non_domiated_sorted_indicies = nondominated_sort(nondomination_rank_dict,crowding)
    return non_domiated_sorted_indicies[:len(fronts[0])] # only return the indicies of the first front


#################################
## NONDOMINATED SORT FUNCTIONS ##
#################################

def dominates(fitnesses_1,fitnesses_2):
    # fitnesses_1 is a array of objectives of solution 1 [objective1, objective2 ...]
    larger_or_equal = fitnesses_1 >= fitnesses_2
    larger = fitnesses_1 > fitnesses_2
    if np.all(larger_or_equal) and np.any(larger):
        return True
    return False


# returns a matrix with shape (pop_size,pop_size) to anwer the question does individial i dominates individual j -> domination_matrix[i,j]
# like a hundred times faster then calling dominates() in 2 for loops
def calculate_domination_matrix(fitnesses):    
    
    pop_size = fitnesses.shape[0]
    num_objectives = fitnesses.shape[1]
    
    # numpy meshgrid does not work if original array is 2d, so we have to build the mesh grid manually
    fitness_grid_x = np.zeros([pop_size,pop_size,num_objectives])
    fitness_grid_y = np.zeros([pop_size,pop_size,num_objectives])
    
    for i in range(pop_size):
        fitness_grid_x[i,:,:] = fitnesses[i]
        fitness_grid_y[:,i,:] = fitnesses[i]
    
    larger_or_equal = fitness_grid_x >= fitness_grid_y
    larger = fitness_grid_x > fitness_grid_y
    
    return np.logical_and(np.all(larger_or_equal,axis=2),np.any(larger,axis=2))

def calculate_pareto_fronts(fitnesses):
    
    # Calculate dominated set for each individual
    domination_sets = []
    domination_counts = []
    
    domination_matrix = calculate_domination_matrix(fitnesses)
    pop_size = fitnesses.shape[0]
    
    for i in range(pop_size):
        current_dimination_set = set()
        domination_counts.append(0)
        for j in range(pop_size):
            if domination_matrix[i,j]:
                current_dimination_set.add(j)
            elif domination_matrix[j,i]:
                domination_counts[-1] += 1
                
        domination_sets.append(current_dimination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts==0)[0]
        if len(current_front) == 0:
            #print("Done")
            break
        #print("Front: ",current_front)
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[individual] = -1 # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1
            
    return fronts


def calculate_crowding_metrics(fitnesses,fronts):
    
    num_objectives = fitnesses.shape[1]
    num_individuals = fitnesses.shape[0]
    
    # Normalise each objectives, so they are in the range [0,1]
    # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
    normalized_fitnesses = np.zeros_like(fitnesses)
    for objective_i in range(num_objectives):
        min_val = np.min(fitnesses[:,objective_i])
        max_val = np.max(fitnesses[:,objective_i])
        val_range = max_val - min_val
        normalized_fitnesses[:,objective_i] = (fitnesses[:,objective_i] - min_val) / val_range
    
    fitnesses = normalized_fitnesses
    crowding_metrics = np.zeros(num_individuals)

    for front in fronts:
        for objective_i in range(num_objectives):
            
            sorted_front = sorted(front,key = lambda x : fitnesses[x,objective_i])
            
            crowding_metrics[sorted_front[0]] = np.inf
            crowding_metrics[sorted_front[-1]] = np.inf
            if len(sorted_front) > 2:
                for i in range(1,len(sorted_front)-1):
                    crowding_metrics[sorted_front[i]] += fitnesses[sorted_front[i+1],objective_i] - fitnesses[sorted_front[i-1],objective_i]

    return  crowding_metrics


def fronts_to_nondomination_rank(fronts):
    nondomination_rank_dict = {}
    for i,front in enumerate(fronts):
        for x in front:   
            nondomination_rank_dict[x] = i
    return nondomination_rank_dict
        

def nondominated_sort(nondomination_rank_dict,crowding):
    
    num_individuals = len(crowding)
    indicies = list(range(num_individuals))

    def nondominated_compare(a,b):
        # returns 1 if a dominates b, or if they equal, but a is less crowded
        # return -1 if b dominates a, or if they equal, but b is less crowded
        # returns 0 if they are equal in every sense
        
        
        if nondomination_rank_dict[a] > nondomination_rank_dict[b]:  # domination rank, smaller better
            return -1
        elif nondomination_rank_dict[a] < nondomination_rank_dict[b]:
            return 1
        else:
            if crowding[a] < crowding[b]:   # crowding metrics, larger better
                return -1
            elif crowding[a] > crowding[b]:
                return 1
            else:
                return 0

    non_domiated_sorted_indicies = sorted(indicies,key = functools.cmp_to_key(nondominated_compare),reverse=True) # decreasing order, the best is the first
    return non_domiated_sorted_indicies



def get_normalized_nondominated_ranks(fitnesses):

    fronts = calculate_pareto_fronts(fitnesses)
    nondomination_rank_dict = fronts_to_nondomination_rank(fronts)
    
    crowding = calculate_crowding_metrics(fitnesses,fronts)
    
    # Sort the population
    non_domiated_sorted_indicies = nondominated_sort(nondomination_rank_dict,crowding)

    # non_domiated_sorted_indicies is the indices of individuals, sorted, so the first index in this list belongs to the best individual, the last index is the worst
    # we want to associate individuals with a number between 0.5 (best) and -0.5 (worst), based on their position in the sorted list
    # This code looks a bit strange, but it does exactly that.
    all_ranks = np.linspace(0.5,-0.5,len(non_domiated_sorted_indicies))
    pop_rank = np.zeros(len(non_domiated_sorted_indicies))
    pop_rank[non_domiated_sorted_indicies] = all_ranks


    return pop_rank,fronts


