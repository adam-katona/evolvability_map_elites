
import copy

config_list = {  
    # baselines
    "ME__explore" : {
        "ES_UPDATES_MODES_TO_USE" : ["innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["eval_fitness"]},
    },
    "ME__exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["eval_fitness"]},
    },
    "ME__explore-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["eval_fitness"]},
    },
    
    # elite evolver map (fitness only)
    "E-ME-f__exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    "E-ME-f__explore-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },

    # evolvability (with fitness only map)
     "E-ME-f__evolvability" : {
        "ES_UPDATES_MODES_TO_USE" : ["evo_ent"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    "E-ME-f__evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    "E-ME-f__explore-evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    
    # evolvability with evolvability map
    "E-ME-e__explore-evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["evo_ent"]},
    },
    "E-ME-i__explore-evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["innovation"]},
    },
    
    # multi maps
    "MM-ME-fei__explore-evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "multi_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "ND-ME-fei__explore-evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
}


# The configs here is kind of additive testing, adding one feature, and see if it helps.
# We can also do an ablation study, take the complate system, and starte removing comonents.

# Maybe i need to reevaluate all the modes and environments
 




config_index_to_names = {i:name for i,name in enumerate(config_list)}

def get_config_from_index(default_config,index):
    config_name = config_index_to_names[index]
    custom_config = config_list[config_name]

    config = copy.deepcopy(default_config)
    config.update(custom_config)
    config["config_name"] = config_name
    
    return config

