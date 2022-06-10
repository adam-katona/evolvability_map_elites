
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

# TODO, for future runs
# Add combined updates
# We have 8 configs where there are multiple update modes
# We could also do a test only with one type of map
# Eg for nondominated we do the combined, alternating or even combined + alternating...
# Let us do that.

combined_config_list = {
    "E-ME-f__explore-evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    "E-ME-f__explore-evolvability-exploit_C" : {
        "ES_UPDATES_MODES_TO_USE" : ["quality_evo_ent_innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    "E-ME-f__explore-evolvability-exploit_C+" : {
        "ES_UPDATES_MODES_TO_USE" : ["quality_evo_ent_innovation","fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    
    "ND-ME-fei__explore-evolvability-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "ND-ME-fei__explore-evolvability-exploit_C" : {
        "ES_UPDATES_MODES_TO_USE" : ["quality_evo_ent_innovation"],
        "BMAP_type_and_metrics" : {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "ND-ME-fei__explore-evolvability-exploit_C+" : {
        "ES_UPDATES_MODES_TO_USE" : ["quality_evo_ent_innovation","fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
}


ablation_config_list = {
    "FULL_MM" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "multi_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "FULL_ND" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "NO_EVOLVABILITY_UPDATE_MM" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","innovation"],
        "BMAP_type_and_metrics" : {"type" : "multi_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "NO_EVOLVABILITY_UPDATE_NO_INNOV_ND" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],
        "BMAP_type_and_metrics" : {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
     "NO_EVOLVABILITY_UPDATE_NO_INNOV_MM" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],
        "BMAP_type_and_metrics" : {"type" : "multi_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "NO_EVOLVABILITY_UPDATE_ND" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","innovation"],
        "BMAP_type_and_metrics" : {"type" : "nd_sorted_map", "metrics" : ["excpected_fitness","evo_ent","innovation"]},
    },
    "NO_EVOLVABILITY_SELECTION" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["excpected_fitness"]},
    },
    "NO_EVOLVABILITY_SELECTION_EVAL_FITNESS" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","evo_ent","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["eval_fitness"]},
    },
    "NO_BOTH_ME__exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["eval_fitness"]},
    },
    "NO_BOTH_ME__explore-exploit" : {
        "ES_UPDATES_MODES_TO_USE" : ["fitness","innovation"],
        "BMAP_type_and_metrics" : {"type" : "single_map", "metrics" : ["eval_fitness"]},
    },
}


# Metrics
# Best ever fitness
# Best ever eval fitness
# Best ever evolvability
# QD
# ED


# envs humanoid and ent dist_final_pos



# The configs here is kind of additive testing, adding one feature, and see if it helps.
# We can also do an ablation study, take the complate system, and starte removing comonents.

# Maybe i need to reevaluate all the modes and environments
 





def get_config_from_index(default_config,index):
    
    if "config_list_name" in default_config:
        if default_config["config_list_name"] == "default_list":
            selected_config_dict = config_list
        elif default_config["config_list_name"] == "combined_update_list":
            selected_config_dict = combined_config_list
        elif default_config["config_list_name"] == "ablation_list":
            selected_config_dict = ablation_config_list
        else:
            raise "Unknown config_list_name"
    else:
        selected_config_dict = config_list
        
        
    config_i_to_name = {i:name for i,name in enumerate(selected_config_dict)}
    config_name = config_i_to_name[index]
    custom_config = selected_config_dict[config_name]

    config = copy.deepcopy(default_config)
    config.update(custom_config)
    config["config_name"] = config_name
    
    return config

