import copy
import argparse

N_VISION_SCENES = 5000
N_TINY_VISION_SCENES = 1
N_TABLE1_SCENES = 100


EXPERIMENT_OPTIONS = {
    "vision": {  # Training data for the vision module.
        "plan": {
            "seed": 1,
            "task": "stack",
            "stage": "plan",
            "n_scenes": N_VISION_SCENES,
            "save_states": False,
        },
        "place": {
            "seed": 2,
            "task": "place",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
            "save_states": True,
        },
        "stack": {
            "seed": 3,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
            "save_states": True,
        },
    },
    "table1": {
        "stack": {
            "seed": 5,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
            "z_rot": True,
            "larger_table_range": True
        },
        "place": {
            "seed": 4,
            "task": "place",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
            "z_rot": True,
            "larger_table_range": True
        },
    },
    "table1_smaller_range_no_z_rot": {
        "stack": {
            "seed": 5,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
            "z_rot": False,
            "larger_table_range": False
        },
        "place": {
            "seed": 4,
            "task": "place",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
            "z_rot": False,
            "larger_table_range": False
        },
    },
}


# Create tiny experiments.
exp_names = list(EXPERIMENT_OPTIONS.keys())
for exp_name in exp_names:
    tiny_exp_name = f"{exp_name}_tiny"
    EXPERIMENT_OPTIONS[tiny_exp_name] = copy.deepcopy(EXPERIMENT_OPTIONS[exp_name])
    for set_name in EXPERIMENT_OPTIONS[tiny_exp_name].keys():
        EXPERIMENT_OPTIONS[tiny_exp_name][set_name]["n_scenes"] = N_TINY_VISION_SCENES

exp_names = list(EXPERIMENT_OPTIONS.keys())
for exp_name in exp_names:
    EXPERIMENT_OPTIONS[f"system_{exp_name}"] = copy.deepcopy(
        EXPERIMENT_OPTIONS[exp_name]
    )
