"""Options for scene generation."""
import math
import copy
import argparse

from ns_vqa_dart.bullet import util


# TASK_LIST = ["place", "stack"]

N_PLACE_MANIP = 1
N_STACK_MANIP = 2


def create_options(json_path: str):
    j = util.load_json(path=json_path)
    task2opt = j["task_options"]
    s = j["generator_options"]
    min_objects = s["min_objects"]
    max_objects = s["max_objects"]

    base_opt = s["base"]
    manip = s["manip"]
    sur = s["surround"]
    stack_manip = s["stack_manip"]
    place_manip = s["place_manip"]
    stack_top = None if "stack_top" not in s else s["stack_top"]
    stack_btm = None if "stack_btm" not in s else s["stack_btm"]

    manip_opt = create_child_options(base_opt, manip)

    # Placing
    place_manip_opt = create_child_options(manip_opt, place_manip)

    # Stacking
    stack_manip_opt = create_child_options(manip_opt, stack_manip)
    stack_top_opt = create_child_options(stack_manip_opt, stack_top)
    stack_btm_opt = create_child_options(stack_manip_opt, stack_btm)

    # Surround
    sur_opt = create_child_options(base_opt, sur)
    place_sur_opt = create_child_options(sur_opt)
    stack_sur_opt = create_child_options(sur_opt)

    # We use constants instead of the `n` entry in the task dictionary. Main reason is
    # that for place, n=2, because we want to generate an imaginary target location, but
    # the imaginary object will be deleted later. So instead, we want the *actual*
    # number of place objects, which is 1.
    place_sur_opt["n"] = compute_sur_bounds(min_objects, max_objects, N_PLACE_MANIP)
    stack_sur_opt["n"] = compute_sur_bounds(min_objects, max_objects, N_STACK_MANIP)

    task2gen_opt = {
        "place": {"manip": [place_manip_opt], "surround": [place_sur_opt],},
        "stack": {
            "manip": [stack_top_opt, stack_btm_opt],
            "surround": [stack_sur_opt],
        },
    }
    return task2opt, task2gen_opt


def compute_sur_bounds(min_objects, max_objects, n_existing):
    min_sur_objects = min_objects - n_existing
    max_sur_objects = max_objects - n_existing
    return min_sur_objects, max_sur_objects


def create_child_options(base_opt, params=None):
    opt = copy.deepcopy(base_opt)
    if params is not None:
        for k, v in params.items():
            opt[k] = v
    return opt


# MAX_TABLETOP_OBJECTS = 6  # For system, 6 is the max number that OR can handle.
# MIN_TABLETOP_OBJECTS = 4
# N_STACK_OBJECTS = 2
# N_PLACE_OBJECTS = 1
# BASE_OBJECT = argparse.Namespace(
#     n_objects=None,
#     obj_dist_thresh=0.25,
#     max_retries=50,
#     shapes=None,
#     colors=["red", "yellow", "green", "blue"],
#     radius=(0.03, 0.05),
#     height=(0.13, 0.18),
#     x_pos=None,
#     y_pos=None,
#     z_pos=0.0,
#     z_rot=(0.0, 2 * math.pi),
#     mass=(1.0, 5.0),
#     mu=(0.8, 1.2),
#     position_mode="com",
# )

# Options for manipulated objects.
# MANIPULATED_OBJECTS = copy.deepcopy(BASE_OBJECT)
# MANIPULATED_OBJECTS.x_pos = (-0.1, 0.25)
# MANIPULATED_OBJECTS.y_pos = (-0.1, 0.5)

# Options for surrounding objects.
# SURROUND_OBJECTS = copy.deepcopy(BASE_OBJECT)
# SURROUND_OBJECTS.shapes = ["box", "cylinder", "sphere"]
# SURROUND_OBJECTS.x_pos = (-0.1, 0.3)
# SURROUND_OBJECTS.y_pos = (-0.3, 0.7)
# SURROUND_OBJECTS.x_pos = (-0.1, 0.25)
# SURROUND_OBJECTS.y_pos = (-0.1, 0.5)

# Options for stacking.
# STACK_OBJECT = copy.deepcopy(MANIPULATED_OBJECTS)
# STACK_OBJECT.n_objects = 1
# STACK_OBJECT.shapes = ["box", "cylinder"]
# STACK_TOP, STACK_BTM = [copy.deepcopy(STACK_OBJECT) for _ in range(2)]
# STACK_BTM.radius = 0.045  # Slightly larger radius for the bottom.
# STACK_SURROUND = copy.deepcopy(SURROUND_OBJECTS)
# STACK_SURROUND.n_objects = (
#     MIN_TABLETOP_OBJECTS - N_STACK_OBJECTS,
#     MAX_TABLETOP_OBJECTS - N_STACK_OBJECTS,
# )

# Options for placing.
# PLACE_OBJECTS = copy.deepcopy(MANIPULATED_OBJECTS)
# PLACE_OBJECTS.n_objects = 2  # We include a second placing dst that will be deleted.
# PLACE_OBJECTS.shapes = ["box", "cylinder"]
# PLACE_SURROUND = copy.deepcopy(SURROUND_OBJECTS)
# PLACE_SURROUND.n_objects = (
#     MIN_TABLETOP_OBJECTS - N_PLACE_OBJECTS,
#     MAX_TABLETOP_OBJECTS - N_PLACE_OBJECTS,
# )

# Map tasks to options.
# TASK_LIST = ["place", "stack"]
# TASK2OPTIONS = {
#     "place": [PLACE_OBJECTS, PLACE_SURROUND],
#     "stack": [STACK_TOP, STACK_BTM, STACK_SURROUND],
# }
