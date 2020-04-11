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
import ns_vqa_dart.bullet.util as util


def bullet2unity_state(bullet_state: Dict, bullet_camera_targets):
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
        bullet_shoulder_pos=const.BULLET_SHOULDER_POS,
    )

    # Beginning is sid, and target camera start idx.
    targets_start_idx = 2 + len(unity_robot_state + unity_object_states)
    n_targets = len(bullet_camera_targets)
    unity_target_state = [n_targets]
    for tid, bullet_pos in bullet_camera_targets.items():
        bullet_rel_position = np.array(bullet_pos) - np.array(
            const.BULLET_SHOULDER_POS
        )
        unity_rel_position = bullet2unity_position(
            bullet_position=bullet_rel_position
        )
        unity_target_state += [tid] + unity_rel_position

    # Combine the robot and object states.
    unity_state = (
        [targets_start_idx]
        + unity_robot_state
        + unity_object_states
        + unity_target_state
    )
    return unity_state


# def compute_bworld2ucam_transformation(
#     uworld_cam_position: List[float], uworld_cam_orientation: List[float]
# ):
#     """Computes the transformation between bullet world coordinate frame to
#     unity camera coordinate frame.

#     Args:
#         uworld_cam_position: The camera position in unity world coordinate
#             frame.
#         uworld_cam_orientation: The camera orientation in unity world
#             coordinate frame.

#     Returns:
#         transformation: A 4x4 transformation matrix.
#     """
#
#     bworld2bshoulder
#     bullet2unity(position, up)
#     ushoulder2ucamera
#     return transformation


def bworld2ucam(
    bworld_position: List[float],
    bworld_orientation: List[float],
    uworld_cam_position: List[float],
    uworld_cam_orientation: List[float],
):
    """Converts bullet world position and orientation into unity camera 
    coordinate frame.

    Args:
        bworld_position: A xyz position in bullet world coordinate frame.
        bworld_orientation: A xyzw orientation in bullet world coordinate 
            frame.
        uworld_cam_position: The camera position in unity world coordinate 
            frame.
        uworld_cam_orientation: The camera orientation in unity world 
            coordinate frame.
    
    Returns:
        ucam_position: The input position, converted into unity camera 
            coordinate frame.
        ucam_euler: The input orientation, converted into unity camera
            coordinate frame and represented as xyz euler angles (degrees).
    """
    T_bw_bs, T_us_uc = compute_bullet2unity_transforms(
        uworld_cam_position=uworld_cam_position,
        uworld_cam_orientation=uworld_cam_orientation,
    )

    p_bw = bworld_position
    p_bs = util.apply_transform(xyz=p_bw, transformation=T_bw_bs)
    p_us = bullet2unity_position(bullet_position=p_bs)
    p_uc = util.apply_transform(xyz=p_us, transformation=T_us_uc)

    # Convert position from world coordinate frame to shoulder coordinate
    # frame.
    bshoulder_position = np.array(bworld_position) - np.array(
        const.BULLET_SHOULDER_POS
    )
    bshoulder_orientation = bworld_orientation

    # Convert from bullet to unity coordinates.
    # Note that there is no rotation between world and shoulder coordinate
    # frames.
    ushoulder_position = bullet2unity_position(
        bullet_position=bshoulder_position
    )
    ushoulder_euler = bullet2unity_euler(bullet_orn=bshoulder_orientation)

    # Convert the unity camera from world2cam into shoulder2cam.
    ushoulder2camera = util.create_transformation(
        position=np.array(uworld_cam_position)
        - np.array(const.UNITY_SHOULDER_POS),
        orientation=uworld_cam_orientation,
    )

    # Convert from shoulder to camera coordinate frame.
    ucam_position = util.apply_transform(
        xyz=ushoulder_position, transformation=ushoulder2camera
    )
    ucam_euler = util.apply_transform(
        xyz=ushoulder_euler, transformation=ushoulder2camera
    )
    return p_uc, ucam_euler


def ucam2bworld(
    ucam_position: List[float],
    ucam_up_vector: List[float],
    uworld_cam_position: List[float],
    uworld_cam_orientation: List[float],
):
    """Converts position and up vector from unity camera coordinate frame into
    bullet world coordinate frame.

    Args:
        ucam_position: The position in unity camera coordinate frame.
        ucam_up_vector: The up vector in unity camera coordinate frame.
        uworld_cam_position: The position of the unity camera in unity world
            coordinate frame.
        uworld_cam_orientation: The orientation of the unity camera in unity 
            world coordinate frame.
    
    Returns:
        bworld_position: The position in bullet world coordinate frame.
        bworld_up_vector: The up vector in bullet world coordinate frame.
    """
    T_bw_bs, T_us_uc = compute_bullet2unity_transforms(
        uworld_cam_position=uworld_cam_position,
        uworld_cam_orientation=uworld_cam_orientation,
    )

    p_uc = ucam_position
    p_us = util.apply_inv_transform(xyz=p_uc, transformation=T_us_uc)
    p_bs = unity2bullet_position(unity_position=p_us)
    p_bw = util.apply_inv_transform(xyz=p_bs, transformation=T_bw_bs)

    # Convert from unity camera coordinate frame to shoulder coordinate frame.
    # This is equivalent to the inverse transform of shoulder to camera.
    ushoulder2camera = util.create_transformation(
        position=np.array(uworld_cam_position)
        - np.array(const.UNITY_SHOULDER_POS),
        orientation=uworld_cam_orientation,
    )
    ushoulder_position = util.apply_inv_transform(
        xyz=ucam_position, transformation=ushoulder2camera
    )
    ushoulder_up_vector = util.apply_inv_transform(
        xyz=ucam_up_vector, transformation=ushoulder2camera
    )

    # Then, convert from unity to bullet coordinate system.
    bshoulder_position = unity2bullet_position(
        unity_position=ushoulder_position
    )
    bshoulder_up_vector = unity2bullet_up(unity_up=ushoulder_up_vector)

    # Finally, convert from bullet shoulder to bullet world coordinate frame.
    bworld_position = np.array(const.BULLET_SHOULDER_POS) + np.array(
        bshoulder_position
    )
    bworld_up_vector = bshoulder_up_vector
    return p_bw, bworld_up_vector


def compute_bullet2unity_transforms(
    uworld_cam_position: List[float], uworld_cam_orientation: List[float]
):
    T_bw_bs = util.create_transformation(
        position=const.BULLET_SHOULDER_POS, orientation=[0.0, 0.0, 0.0, 1.0]
    )
    T_uw_us = util.create_transformation(
        position=const.UNITY_SHOULDER_POS, orientation=[0.0, 0.0, 0.0, 1.0]
    )
    T_us_uw = np.linalg.inv(T_uw_us)
    T_uc_uw = util.create_transformation(
        position=uworld_cam_position, orientation=uworld_cam_orientation
    )
    T_us_uc = T_uc_uw.dot(T_us_uw)
    return T_bw_bs, T_us_uc


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
    bullet_state: Dict[int, Dict], bullet_shoulder_pos: List[float],
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
    for idx, odict in enumerate(bullet_state.values()):
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
        unity_rel_position = bullet2unity_position(
            bullet_position=bullet_rel_position
        )

        # Convert the object orientation.
        unity_rotation = bullet2unity_euler(bullet_orn=bullet_orientation)

        # Create the state for the current object.
        otag = f"{idx:02}"
        # look_at_flag = int(idx in look_at_idxs)
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


def bullet2unity_position(bullet_position: List[float]):
    """Converts from bullet to unity position
    
    Args:
        bullet_position: The xyz position in bullet.
    
    Returns:
        unity_position: The xyz position in Unity.
    """
    # unity_position = np.copy(bullet_position)
    # x = bullet_position[0]
    # y = bullet_position[1]
    # z = bullet_position[2]
    # new_vector = swap_axes(new_vector, 1, 2)  # swap y and z
    # new_vector = swap_axes(new_vector, 0, 2)  # swap x and z
    # new_vector[2] *= -1  # Negate z
    x, y, z = bullet_position
    unity_position = [y, z, -x]
    return unity_position


def unity2bullet_position(unity_position: List[float]):
    """Converts from unity to bullet position.

    Args:
        unity_position: The xyz position in unity coordinates.
    
    Returns:
        bullet_position: The xyz position in bullet coordinates.
    """
    x, y, z = unity_position
    bullet_position = [-z, x, y]
    return bullet_position


def bullet2unity_euler(bullet_orn: List[float]) -> List[float]:
    """Converts bullet to unity rotation
    
    Args:
        bullet_orn: The bullet orientation, in quaternion [x, y, z, w] format.

    Returns:
        unity_rot: The unity xyz rotation in euler angles (degrees).
    """
    bullet_euler = util.orientation_to_euler(orientation=bullet_orn)
    x, y, z = bullet_euler
    unity_euler = [-y, -z, x]
    # unity_rot = np.copy(bullet_euler)
    # unity_rot = swap_axes(unity_rot, 0, 2)  # swap x and z
    # unity_rot = swap_axes(unity_rot, 0, 1)  # swap x and y
    # unity_rot[0] *= -1  # Negate x
    # unity_rot[1] *= -1  # Negate y
    return unity_euler


def unity2bullet_up(unity_up: List[float]) -> List[float]:
    """Converts an up vector from unity coordinates into bullet coordinates.

    Args:
        unity_up: The up vector, in unity coordinates.

    Returns:
        bullet_up: The up vector, in bullet coordinates.
    """
    unity_euler = util.up_to_euler(up=unity_up)
    x, y, z = unity_euler
    bullet_euler = [z, -x, -y]
    bullet_up = util.euler_to_up(euler=bullet_euler)
    return bullet_up
