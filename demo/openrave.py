import copy
import numpy as np
import os
import time
from typing import *


homedir = os.path.expanduser("~")
CONTAINER_DIR = os.path.join(homedir, "container_data")
STAGE2NAME = {
    "reach": "REACH",
    "transport": "MOVE",
}


def compute_trajectory(
    state: Dict,
    dst_oid: int,
    q_start: np.ndarray,
    q_end: np.ndarray,
    stage: str,
) -> np.ndarray:
    """Computes a trajectory using OpenRAVE.
    Args:
        state: A state dictionary, with the format:
            {
                "objects": {
                    <oid>: {
                        "position": <position>
                    }
                }
            }
        q_start: The source / starting q of shape (7,).
        q_end: The destination q of shape (7,).
        stage: The stage we are computing the trajectory for.
        
    Returns:
        traj: The trajectory computed by OpenRAVE of shape (200, 7).
    """
    name = STAGE2NAME[stage]

    # Extract object positions from the state. Destination object needs to come
    # first.
    state = copy.deepcopy(state)
    object_positions = []
    oids = list(state["objects"].keys())
    ordered_oids = [dst_oid]
    for oid in oids:
        if oid != dst_oid:
            ordered_oids.append(oid)
    for oid in ordered_oids:
        position = state["objects"][oid]["position"]

        # Set object z to zero because that's what OR expects.
        position[2] = 0.0
        object_positions.append(position)
    # object_positions = [o["position"] for o in state["objects"].values()]

    # Pad a fourth dimension with zero because open rave expects it.
    object_positions = np.array([p + [0.0] for p in object_positions])

    # Compute the trajectory using OpenRAVE.
    trajectory = get_traj_from_openrave_container(
        object_positions=object_positions,
        q_start=q_start,
        q_end=q_end,
        save_path=os.path.join(CONTAINER_DIR, f"PB_{name}.npz"),
        load_path=os.path.join(CONTAINER_DIR, f"OR_{name}.npy"),
    )
    return trajectory


def get_traj_from_openrave_container(
    object_positions: np.ndarray,
    q_start: np.ndarray,
    q_end: np.ndarray,
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
        traj: The trajectory computed by OpenRAVE of shape (200, 7).
    """
    print("Printing inputs to computing trajectory")
    print(f"object_positions: {object_positions}")
    print(f"q_start: {q_start}")
    print(f"q_end: {q_end}")

    if q_start is not None:
        np.savez(save_path, object_positions, q_start, q_end)  # move
    else:
        np.savez(save_path, object_positions, q_end)  # reach has q_start 0

    # Wait for command from OpenRave

    assert not os.path.exists(load_path)
    while not os.path.exists(load_path):
        time.sleep(0.2)
    if os.path.isfile(load_path):
        traj = np.load(load_path)
        print("loaded")
        try:
            os.remove(load_path)
            print("deleted")
            # input("press enter")
        except OSError as e:  # name the Exception `e`
            print("Failed with:", e.strerror)  # look what it says
            # input("press enter")
    else:
        raise ValueError("%s isn't a file!" % load_path)
    print("Trajectory obtained from OpenRave!")
    # input("press enter")
    return traj
