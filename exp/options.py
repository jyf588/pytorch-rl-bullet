import copy
import argparse

N_VISION_SCENES = 2500
N_TINY_VISION_SCENES = 1
N_TABLE1_SCENES = 100


EXPERIMENT_OPTIONS = {
    "seg": {  # Training data for the segmentation module.
        # "seg_plan": {"seed": 1, "task": "place", "n_scenes": N_VISION_SCENES},
        # "seg_place": {"seed": 2, "task": "place", "n_scenes": N_VISION_SCENES},
        "stack": {
            "seed": 3,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
        },
    },
    "vision": {  # Training data for the vision module.
        # "seg_plan": {"seed": 1, "task": "place", "n_scenes": N_VISION_SCENES},
        # "seg_place": {"seed": 2, "task": "place", "n_scenes": N_VISION_SCENES},
        "stack": {
            "seed": 3,
            "task": "stack",
            "stage": "place",
            "n_scenes": N_VISION_SCENES,
        },
    },
    "table1": {
        "place": {"seed": 7, "task": "place", "n_scenes": N_TABLE1_SCENES},
        "stack": {"seed": 8, "task": "stack", "n_scenes": N_TABLE1_SCENES},
    },
}


# Create tiny experiments.
for exp_name in ["seg", "vision"]:
    tiny_exp_name = f"{exp_name}_tiny"
    EXPERIMENT_OPTIONS[tiny_exp_name] = copy.deepcopy(EXPERIMENT_OPTIONS[exp_name])
    for set_name in EXPERIMENT_OPTIONS[tiny_exp_name].keys():
        EXPERIMENT_OPTIONS[tiny_exp_name][set_name]["n_scenes"] = N_TINY_VISION_SCENES
