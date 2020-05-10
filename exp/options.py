import copy
import argparse

N_VISION_SCENES = 2500
N_TINY_VISION_SCENES = 1
N_TABLE1_SCENES = 100


EXPERIMENT_OPTIONS = {
    "vision": {
        "seg_plan": {"seed": 1, "task": "place", "n_scenes": N_VISION_SCENES},
        "seg_place": {"seed": 2, "task": "place", "n_scenes": N_VISION_SCENES},
        "seg_stack": {"seed": 3, "task": "stack", "n_scenes": N_VISION_SCENES},
        "vision_plan": {"seed": 4, "task": "place", "n_scenes": N_VISION_SCENES},
        "vision_place": {"seed": 5, "task": "place", "n_scenes": N_VISION_SCENES},
        "vision_stack": {"seed": 6, "task": "stack", "n_scenes": N_VISION_SCENES},
    },
    "table1": {
        "place": {"seed": 7, "task": "place", "n_scenes": N_TABLE1_SCENES},
        "stack": {"seed": 8, "task": "stack", "n_scenes": N_TABLE1_SCENES},
    },
}

EXPERIMENT_OPTIONS["vision_tiny"] = copy.deepcopy(EXPERIMENT_OPTIONS["vision"])
for set_name in EXPERIMENT_OPTIONS["vision_tiny"].keys():
    EXPERIMENT_OPTIONS["vision_tiny"][set_name]["n_scenes"] = N_TINY_VISION_SCENES
