import copy
import argparse

N_VISION_SCENES = 2500
N_TINY_VISION_SCENES = 1
N_TABLE1_SCENES = 100


EXPERIMENT_OPTIONS = {
    "seg": {  # Training data for the segmentation module.
        "plan": {
            "seed": 1,
            "task": "stack",
            "stage": "plan",
            "n_scenes": N_VISION_SCENES,
        },
        "place": {
            "seed": 2,
            "task": "place",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
        },
        "stack": {
            "seed": 3,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
        },
    },
    "vision": {  # Training data for the vision module.
        "plan": {
            "seed": 4,
            "task": "stack",
            "stage": "plan",
            "n_scenes": N_VISION_SCENES,
        },
        "place": {
            "seed": 5,
            "task": "place",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
        },
        "stack": {
            "seed": 6,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
        },
    },
    # "table1": {
    #     "place": {"seed": 7, "task": "place", "n_scenes": N_TABLE1_SCENES},
    #     "stack": {"seed": 8, "task": "stack", "n_scenes": N_TABLE1_SCENES},
    # },
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
