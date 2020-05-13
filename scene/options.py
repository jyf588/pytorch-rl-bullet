"""Options for scene generation."""
import math
import copy
import argparse

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
    # z_rot=None,
    mass=(1.0, 5.0),
    mu=(0.8, 1.2),
    position_mode="com",
    check_uniqueness=True,
)

# Options for manipulated objects.
MANIPULATED_OBJECTS = copy.deepcopy(BASE_OBJECT)
MANIPULATED_OBJECTS.x_pos = (-0.1, 0.25)
MANIPULATED_OBJECTS.y_pos = (-0.1, 0.5)

# Options for surrounding objects.
SURROUND_OBJECTS = copy.deepcopy(BASE_OBJECT)
SURROUND_OBJECTS.shapes = ["box", "cylinder", "sphere"]
SURROUND_OBJECTS.x_pos = (-0.1, 0.3)
SURROUND_OBJECTS.y_pos = (-0.3, 0.7)
# SURROUND_OBJECTS.x_pos = (-0.1, 0.25)
# SURROUND_OBJECTS.y_pos = (-0.1, 0.5)

# Options for stacking.
STACK_OBJECT = copy.deepcopy(MANIPULATED_OBJECTS)
STACK_OBJECT.n_objects = 1
STACK_OBJECT.shapes = ["box", "cylinder"]
STACK_TOP, STACK_BTM = [copy.deepcopy(STACK_OBJECT) for _ in range(2)]
STACK_BTM.radius = 0.045  # Slightly larger radius for the bottom.
STACK_SURROUND = copy.deepcopy(SURROUND_OBJECTS)
STACK_SURROUND.n_objects = (
    MIN_TABLETOP_OBJECTS - N_STACK_OBJECTS,
    MAX_OPENRAVE_OBJECTS - N_STACK_OBJECTS,
)

# Options for placing.
PLACE_OBJECTS = copy.deepcopy(MANIPULATED_OBJECTS)
PLACE_OBJECTS.n_objects = 2  # We include a second placing dst that will be deleted.
PLACE_OBJECTS.shapes = ["box", "cylinder"]
PLACE_SURROUND = copy.deepcopy(SURROUND_OBJECTS)
PLACE_SURROUND.n_objects = (
    MIN_TABLETOP_OBJECTS - N_PLACE_OBJECTS,
    MAX_OPENRAVE_OBJECTS - N_PLACE_OBJECTS,
)

# Map tasks to options.
TASK_LIST = ["place", "stack"]
TASK2OPTIONS = {
    "place": [PLACE_OBJECTS, PLACE_SURROUND],
    "stack": [STACK_TOP, STACK_BTM, STACK_SURROUND],
}
