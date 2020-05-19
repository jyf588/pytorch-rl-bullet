import copy
import numpy as np
import os
import pprint
import time
from typing import *
import matplotlib.pyplot as plt


# homedir = os.path.expanduser("~")
# CONTAINER_DIR = os.path.join(homedir, "container_data")
STAGE2NAME = {"reach": "REACH", "transport": "MOVE", "retract": "RETRACT"}
MAX_OBJECTS = 6


def check_clean_container(container_dir: str):
    """Checks whether the openrave container is clean (e.g., no existing filepaths.)"""
    for stage, name in STAGE2NAME.items():
        sp, lp = construct_filepaths(container_dir=container_dir, name=name)
        assert not os.path.exists(sp)
        assert not os.path.exists(lp)


def construct_filepaths(container_dir: str, name: str):
    save_path = os.path.join(container_dir, f"PB_{name}.npz")
    load_path = os.path.join(container_dir, f"OR_{name}.npz")
    return save_path, load_path


def compute_trajectory(
    container_dir: str,
    odicts: Dict,
    target_idx: int,
    q_start: np.ndarray,
    q_end: np.ndarray,
    stage: str,
    src_base_z_post_placing: Optional[float] = None,
    default_base_z: Optional[float] = 0.0,
) -> np.ndarray:
    """Computes a trajectory using OpenRAVE.
    Args:
        odicts: A list of object dictionaries, with the format:
            [
                {
                    "position": <position>,
                    ...
                }
            ]
        target_idx: The object index to come first in the ordering of object 
            positions sent to OpenRAVE.
        q_start: The source / starting q of shape (7,).
        q_end: The destination q of shape (7,).
        stage: The stage we are computing the trajectory for.
        src_base_z_post_placing: The base z position of the object 
            corresponding to `target_idx` after placing.
        default_base_z: The default base z position of objects in `odicts`.
        
    Returns:
        traj: The trajectory computed by OpenRAVE of shape (T, 7).
            If we did not find the file from OpenRAVE, we return None.
            If we found the file but no solution was found, we return an empty
                array.
    """
    name = STAGE2NAME[stage]

    # Clip the number of objects at the maximum allowed by OpenRAVE.
    if len(odicts) > MAX_OBJECTS:
        odicts = copy.deepcopy(odicts[:MAX_OBJECTS])

    # Extract object positions from the state. Destination object needs to come
    # first.
    odicts = copy.deepcopy(odicts)  # We make a copy because we are modifying.

    ordered_idxs = [target_idx]
    for idx in range(len(odicts)):
        if idx != target_idx:
            ordered_idxs.append(idx)

    # Append object positions one by one.
    object_positions = []
    for idx in ordered_idxs:
        # Extract the position.
        position = odicts[idx]["position"]

        # OpenRAVE expects the z position to represent the bottom of the
        # objects.
        if position[2] > 0.15:      # CoM position > 0.15
            position[2] = 0.15      # on some other obj
        else:
            position[2] = default_base_z    # 0.0, on floor
        # if stage == "retract" and idx == 0:
        #     assert src_base_z_post_placing is not None
        #     position[2] = src_base_z_post_placing
        #     print("pos", position)
        # else:
        #     position[2] = default_base_z

        # Store the object position.
        object_positions.append(position)

    # print(stage)
    # print(object_positions)
    # Zero-pad each position with a fourth dimension because OpenRAVE expects
    # it.
    object_positions = np.array([p + [0.0] for p in object_positions])

    # Compute the trajectory using OpenRAVE.
    trajectory = get_traj_from_openrave_container(
        object_positions=object_positions,
        q_start=q_start,
        q_end=q_end,
        save_path=os.path.join(container_dir, f"PB_{name}.npz"),
        load_path=os.path.join(container_dir, f"OR_{name}.npz"),
    )
    return trajectory


def get_traj_from_openrave_container(
    object_positions: np.ndarray,
    q_start: Union[np.ndarray, None],
    q_end: Union[np.ndarray, None],
    save_path: str,
    load_path: str,
) -> np.ndarray:
    """Computes a trajectory using OpenRAVE.

    Args:
        object_positions: An array of object positions, of shape 
            (n_objects, 4). The fourth dimension is currently just a zero pad 
            because OpenRAVE expects it.
        q_start: The source / starting q of shape (7,).
        q_end: The destination q of shape (7,).
        save_path: The path to save input information for OpenRAVE to.
        load_path: The path to load OpenRAVE's output trajectory from.
    
    Returns:
        traj: The trajectory computed by OpenRAVE of shape (T, 7).
            If we did not find the file from OpenRAVE, we return None.
            If we found the file but no solution was found, we return an empty
                array.
    """
    # print("Printing inputs to computing trajectory")
    # print(f"object_positions: {object_positions}")
    # print(f"q_start: {q_start}")
    # print(f"q_end: {q_end}")

    if q_start is not None and q_end is not None:
        np.savez(save_path, object_positions, q_start, q_end)  # move
    elif q_end is not None:
        np.savez(save_path, object_positions, q_end)  # reach has q_start 0
    elif q_start is not None:
        np.savez(save_path, object_positions, q_start)  # retract always ends at zero
    else:
        assert False

    # Wait for command from OpenRave

    assert not os.path.exists(load_path)

    # Check for OpenRAVE's output file.
    start = time.time()
    while not os.path.exists(load_path):
        time.sleep(0.02)
        time_elapsed = time.time() - start

        # If longer than 5 seconds, return failure code.
        if time_elapsed > 5:
            print(f"Could not find OpenRAVE-generated file: {load_path}")
            return None
    if os.path.isfile(load_path):
        time.sleep(0.3)  # TODO: wait for file write
        loaded_data = np.load(load_path)
        traj_i = loaded_data["arr_0"]
        traj_s = loaded_data["arr_1"]
        # for k in range(7):
        #     plt.plot(range(300), traj_i[:,k])
        #     plt.plot(range(300), traj_s[:,k])
        #     plt.show()
        try:
            os.remove(load_path)
            # input("press enter")
        except OSError as e:  # name the Exception `e`
            print("Failed with:", e.strerror)  # look what it says
            # input("press enter")
    else:
        raise ValueError("%s isn't a file!" % load_path)
    # print("Trajectory obtained from OpenRave!")
    # input("press enter")
    return traj_s
