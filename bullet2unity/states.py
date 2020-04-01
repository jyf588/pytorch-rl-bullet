import json
import math
import numpy as np
import os
import pickle
import pprint
import pybullet as p
import random
from scipy.spatial.transform import Rotation as R
import time
from typing import *

import bullet2unity.const as const


def bullet2unity_state(bullet_state: Dict):
    """Converts a bullet state to a unity state.

    Args:
        bullet_state, with the format: {
            "objects": {
                "<oid>": {
                    "shape": shape,
                    "color": color,
                    "radius": radius,
                    "height": height,
                    "orientation": [x, y, z, w],
                    "position": [x, y, z]
                },
                ...
            },
            "robot": {
                "<joint_name>": <joint_angle>,
                ...
            }
        }. Note that if "robot" key is not present, the default robot pose will
        be used.
    
    Returns:
        unity_state: The Unity state, which is a list with the format: 
            [
                joint_angles[0],
                ...
                n_objects,
                objects[0].shape,
                objects[0].color,
                objects[0].size,
                objects[0].position,
                objects[0].rotation,
                ...
            ]
    """
    # Convert robot state from bullet to unity.
    try:
        bullet_robot_state = bullet_state["robot"]
    except KeyError:
        bullet_robot_state = const.DEFAULT_ROBOT_STATE
    unity_robot_state = bullet2unity_robot(bullet_state=bullet_robot_state)

    # Convert object state from bullet to unity.
    unity_object_states = bullet2unity_objects(
        bullet_state=bullet_state["objects"],
        bullet_shoulder_pos=const.ROBOT_SHOULDER_POS,
    )

    # Combine the robot and object states.
    unity_state = unity_robot_state + unity_object_states
    return unity_state


def bullet2unity_robot(bullet_state: Dict[str, float]) -> List[float]:
    """Converts robot state from bullet to unity.

    Args:
        bullet_state: The robot pose with the following format:
            {<joint_name>: <joint_angle>}
    
    Returns:
        unity_state: A list of joint angles, corresponding to the order of
            joints in `SEND_JOINT_NAMES`.
    """
    unity_state = []
    for joint_name in const.SEND_JOINT_NAMES:
        # Get the joint angle.
        unity_angle = bullet_state[joint_name]

        # Get pose for unity.
        unity_state.append(unity_angle)
    return unity_state


def bullet2unity_objects(
    bullet_state: Dict[int, Dict], bullet_shoulder_pos: List[float]
):
    """Convert object states from bullet to unity.
    
    Args:
        bullet_state: Object states, in dictionary with the following 
            format: {
                <oid>: {
                    "shape": <shape>,
                    "color": <color>,
                    "radius": <radius>,
                    "height": <height>,
                    "position": [x, y, z],
                    "orientation": [x, y, z, w],
                }
            }
        bullet_shoulder_pos: The bullet shoulder position of the robot.
    
    Returns:
        unity_state: A list representing the object state, in the following
            format: [
                <n_objects>,
                <oid>,
                <shape>, 
                <color>, 
                <x_size>, 
                <y_size>, 
                <z_size>, 
                <x_pos>, 
                <y_pos>, 
                <z_pos>,
                <x_rot>,
                <y_rot>,
                <z_rot>,
            ], where rotations are in Euler angles (degrees).
        otags: A list of object tags.
    """
    n_objects = len(bullet_state)
    unity_state = [n_objects]
    for oid, odict in bullet_state.items():
        shape = odict["shape"]
        color = odict["color"]
        radius = odict["radius"]
        height = odict["height"]
        bullet_position = odict["position"]
        bullet_orientation = odict["orientation"]

        # Convert the object size.
        width = radius * 2
        unity_size = bullet2unity_size(bullet_size=[width, width, height])

        # Convert the object position.
        bullet_rel_position = np.array(bullet_position) - np.array(
            bullet_shoulder_pos
        )
        unity_rel_position = bullet_to_unity_position(
            bullet_position=bullet_rel_position
        )

        # Convert the object orientation.
        unity_rotation = bullet_to_unity_rot(bullet_orn=bullet_orientation)

        # Create the state for the current object.
        otag = f"{oid:02}"
        ostate = (
            [otag, shape, color]
            + list(unity_size)
            + list(unity_rel_position)
            + list(unity_rotation)  # Euler angles (degrees)
        )
        unity_state += ostate
    return unity_state


def bullet2unity_size(bullet_size: List[float]) -> List[float]:
    """Converts XYZ size from bullet to unity.

    Args:
        bullet_size: A list of XYZ sizes, where X and Y are the width (for
            symmetric objects) and Z is the height.
        
    Returns:
        unity_size: The unity size.
    """
    # Swap Y and Z for unity.
    unity_size = bullet_size.copy()
    unity_size[1] = bullet_size[2]
    unity_size[2] = bullet_size[1]
    return unity_size


def quaternion_to_euler(quaternion: List[float], degrees: bool):
    r = R.from_quat(quaternion)
    angles = r.as_euler("xyz", degrees=degrees)
    return angles


def bullet_to_unity_position(bullet_position: List[float]):
    """Converts from bullet to unity position
    
    Args:
        bullet_position: The xyz position in bullet.
    
    Returns:
        unity_position: The xyz position in Unity.
    """
    unity_position = np.copy(bullet_position)
    x = bullet_position[0]
    y = bullet_position[1]
    z = bullet_position[2]
    # new_vector = swap_axes(new_vector, 1, 2)  # swap y and z
    # new_vector = swap_axes(new_vector, 0, 2)  # swap x and z
    # new_vector[2] *= -1  # Negate z
    unity_position = [y, z, -1 * x]
    return unity_position


def bullet_to_unity_rot(bullet_orn: List[float]) -> List[float]:
    """Converts bullet to unity rotation
    
    Args:
        bullet_orn: The bullet orientation, in quaternion [x, y, z, w] format.

    Returns:
        unity_rot: The unity xyz rotation in euler angles (degrees).
    """
    bullet_orn = quaternion_to_euler(bullet_orn, degrees=True)
    unity_rot = np.copy(bullet_orn)
    unity_rot = swap_axes(unity_rot, 0, 2)  # swap x and z
    unity_rot = swap_axes(unity_rot, 0, 1)  # swap x and y
    unity_rot[0] *= -1  # Negate x
    unity_rot[1] *= -1  # Negate y
    return unity_rot


def swap_axes(vector, axis1, axis2):
    """Swaps two axes in a vector."""
    new_vector = np.copy(vector)

    a1 = new_vector[axis1]
    a2 = new_vector[axis2]
    new_vector[axis1] = a2
    new_vector[axis2] = a1
    return new_vector


def radians_to_degrees(radians):
    return 180.0 / math.pi * radians
