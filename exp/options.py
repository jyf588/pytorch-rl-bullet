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
    "t1": {
        "place": {
            "seed": 4,
            "task": "place",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
        "stack": {
            "seed": 5,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
    },
    "t1_zrot_lrange": {
        "stack": {
            "seed": 5,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
        "place": {
            "seed": 4,
            "task": "place",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
    },
    "t1_zrot_lrange_sphere": {
        "stack": {
            "seed": 5,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
        "place": {
            "seed": 4,
            "task": "place",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
    },
    "t1_sphere": {
        "stack": {
            "seed": 5,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
        "place": {
            "seed": 4,
            "task": "place",
            "stage": "place",
            "n_scenes": N_TABLE1_SCENES,
            "save_states": False,
        },
    },
    "demo": {
        "language": {
            "seed": 5,
            "n_scenes": 10,
            "save_states": False,
        }
    }
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
