"""Options for scene generation."""
import math
import copy
import argparse

N_SCENES = 2000

MAX_OPENRAVE_OBJECTS = 6
MIN_TABLETOP_OBJECTS = 4
N_STACK_OBJECTS = 2
N_PLACE_OBJECTS = 1

BASE_OBJECT = argparse.Namespace(
    seed=None,
    n_objects=None,
    obj_dist_thresh=0.25,
    max_retries=50,
    shapes=None,
    colors=["red", "yellow", "green", "blue"],
    radius=(0.03, 0.05),
    height=(0.13, 0.18),
    x_pos=None,
    y_pos=None,
    z_pos=0.0,
    z_rot=(0.0, 2 * math.pi),
    mass=(1.0, 5.0),
    mu=(0.8, 1.2),
    position_mode="com",
    check_uniqueness=True,
)

# Options for manipulated objects.
MANIPULATED_OBJECT = copy.deepcopy(BASE_OBJECT)
MANIPULATED_OBJECT.n_objects = 1
MANIPULATED_OBJECT.shapes = ["box", "cylinder"]
MANIPULATED_OBJECT.x_pos = (-0.1, 0.25)
MANIPULATED_OBJECT.y_pos = (-0.1, 0.5)

# Options for surrounding objects.
SURROUND_OBJECT = copy.deepcopy(BASE_OBJECT)
SURROUND_OBJECT.shapes = ["box", "cylinder", "sphere"]
SURROUND_OBJECT.x_pos = (-0.1, 0.3)
SURROUND_OBJECT.y_pos = (-0.3, 0.7)

# Options for stacking.
STACK_TOP, STACK_BTM = [copy.deepcopy(MANIPULATED_OBJECT) for _ in range(2)]
STACK_SURROUND = copy.deepcopy(SURROUND_OBJECT)
STACK_BTM.radius = 0.045  # Slightly larger radius for the bottom.
STACK_SURROUND.n_objects = (
    MIN_TABLETOP_OBJECTS - N_STACK_OBJECTS,
    MAX_OPENRAVE_OBJECTS - 2,
)

# Options for placing.
PLACE_OBJECT = copy.deepcopy(MANIPULATED_OBJECT)
PLACE_SURROUND = copy.deepcopy(SURROUND_OBJECT)
PLACE_SURROUND.n_objects = (
    MIN_TABLETOP_OBJECTS - N_PLACE_OBJECTS,
    MAX_OPENRAVE_OBJECTS - 1,
)

TASK_LIST = ["place", "stack"]
TASK2OPTIONS = {
    "place": [PLACE_OBJECT, PLACE_SURROUND],
    "stack": [STACK_TOP, STACK_BTM, STACK_SURROUND],
}

EXPERIMENT2SEED = {
    "vision_planning": {"place": 1, "stack": 2,},
    "vision_placing": {"place": 3, "stack": 4,},
    "table1": {"place": 5, "stack": 6,},
}
