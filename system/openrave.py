import copy
import numpy as np
import os
import pprint
import time
from typing import *
import matplotlib.pyplot as plt


homedir = os.path.expanduser("~")
CONTAINER_DIR = os.path.join(homedir, "container_data")
STAGE2NAME = {
    "reach": "REACH",
    "transport": "MOVE",
}


def compute_trajectory(
    odicts: Dict,
    target_idx: int,
    q_start: np.ndarray,
    q_end: np.ndarray,
    stage: str,
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
        
    Returns:
        traj: The trajectory computed by OpenRAVE of shape (200, 7). Returns
            None if OpenRAVE failed to give us a result.
    """
    name = STAGE2NAME[stage]

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

        # Set object z to zero because that's what OR expects.
        position[2] = 0.0

        # Store the object position.
        object_positions.append(position)

    # Zero-pad each position with a fourth dimension because OpenRAVE expects
    # it.
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
        traj: The trajectory computed by OpenRAVE of shape (200, 7). Returns
            None if OpenRAVE failed to give us a result.    # TODO
    """
    print("Printing inputs to computing trajectory")
    print(f"object_positions: {object_positions}")
    print(f"q_start: {q_start}")
    print(f"q_end: {q_end}")

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
            return None
    if os.path.isfile(load_path):
        time.sleep(0.3)  # TODO: wait for networking
        loaded_data = np.load(load_path)
        traj_i = loaded_data["arr_0"]
        traj_s = loaded_data["arr_1"]
        print("loaded")
        # for k in range(7):
        #     plt.plot(range(400), traj_i[:,k])
        #     plt.plot(range(400), traj_s[:,k])
        #     plt.show()
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
    return traj_s
