import copy
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
from ns_vqa_dart.bullet.seg import UNITY_OIDS


PLAN_TARGET_POSITION = [-0.06, 0.3, 0.0]
CAM_TARGET_Z = 0.23


def compute_bullet_camera_targets(
    version, send_image, save_image, stage, tx=None, ty=None, odicts=None, oidx=None,
):
    if stage == "plan":
        cam_target = PLAN_TARGET_POSITION
    elif stage == "place":
        if version == "v2":
            cam_target = (tx, ty, CAM_TARGET_Z)
        elif version == "v1":
            assert odicts is not None
            assert oidx is not None
            cam_target = get_object_camera_target(bullet_odicts=odicts, oidx=oidx)
    else:
        raise ValueError(f"Invalid stage: {stage}")

    # Set the camera target.
    bullet_camera_targets = create_bullet_camera_targets(
        position=cam_target, should_save=save_image, should_send=send_image,
    )
    return bullet_camera_targets


def create_bullet_camera_targets(
    position, should_save: bool, should_send: bool,
):
    """ Creates bullet camera targets.

    Args:
        camera_control: The method of camera control.
        bullet_odicts: The bullet object dictionaries. If the camera control
            method is `stack`, we assume that the destination object dictionary
            comes first.
    
    Returns:
        bullet_camera_targets: A dictionary of camera targets in the bullet
            world coordinate frame, with the following format:
            {
                <target_id: int>: {
                    "position": <List[float]>,
                    "should_save": <bool>,
                    "should_send": <bool>,
                }
            }
    """
    bullet_camera_targets = {
        0: {
            "position": position,
            "should_save": should_save,
            "should_send": should_send,
        }
    }
    return bullet_camera_targets


def get_object_camera_target(bullet_odicts: List[Dict], oidx: int):
    """Computes the position of the target for the camera to look at, for a
    given object index.

    Args:
        bullet_odicts: A list of object dictionaries.
        oidx: The object index to compute the target for.
    """
    # Make a copy because we are modifying.
    target_odict = copy.deepcopy(bullet_odicts[oidx])

    # The target position is computed as center-top position of the object,
    # computed by adding the height to the com position.
    position = target_odict["position"]
    position[2] += target_odict["height"] / 2
    return position


def bullet2unity_state(
    bullet_state: Dict,
    bullet_animation_target: List[float],
    bullet_camera_targets: Dict,
):
    """Converts a bullet state to a unity state.

    Args:
        bullet_state, with the format: 
            {
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
        bullet_animation_target: The target position for the animation of the
            head/neck.
        bullet_camera_targets: A dictionary of target positions that we want
            Unity to point the camera at, in the format:
            {
                <id>: {
                    "position": <List[float]>  # The xyz position, in bullet world coordinate frame.
                    "save": <bool>,  # Whether to save an image using the camera.
                    "send": <bool>,  # Whether to send the image over the websocket.
                }
            }
    
    Returns:
        unity_state: The Unity state, which is a list with the format: 
            [
                targets_start_idx,
                joint_angles[0],
                ...
                n_objects,
                objects[0].shape,
                objects[0].color,
                objects[0].size,
                objects[0].position,
                objects[0].rotation,
                ...
                animation_target_position,
                num_targets,
                tids[0],
                should_save[0],
                should_send[0],
                target_positions[0],
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
    unity_object_states = bullet2unity_objects(bullet_state=bullet_state["objects"])

    # Compute the target position in Unity coordinates.
    if bullet_animation_target:
        bullet_animation_target_shoulder = bullet_world2shoulder_position(
            pos_world=bullet_animation_target
        )
        unity_animation_target_position_shoulder = bullet2unity_position(
            bullet_position=bullet_animation_target_shoulder
        )
    else:
        unity_animation_target_position_shoulder = [None, None, None]

    # We offset by two because the first two elements of the unity message
    # are sid and target camera start idx.
    targets_start_idx = 2 + len(unity_robot_state + unity_object_states)
    unity_cam_targets = []
    for tid, target_info in bullet_camera_targets.items():
        bullet_pos = target_info["position"]
        bullet_pos_shoulder = bullet_world2shoulder_position(pos_world=bullet_pos)
        unity_rel_position = bullet2unity_position(bullet_position=bullet_pos_shoulder)
        unity_cam_targets += [
            tid,
            int(target_info["should_save"]),
            int(target_info["should_send"]),
        ] + unity_rel_position

    # Combine the robot and object states.
    unity_state = (
        [targets_start_idx]
        + unity_robot_state
        + unity_object_states
        + unity_animation_target_position_shoulder
        + [len(bullet_camera_targets)]
        + unity_cam_targets
    )
    return unity_state


def bworld2ucam(
    p_bw: List[float],
    up_bw: List[float],
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
    T_bw_bs, T_uw_uc, T_us_uc = compute_bullet2unity_transforms(
        uworld_cam_position=uworld_cam_position,
        uworld_cam_orientation=uworld_cam_orientation,
    )

    # Transform position.
    p_bs = util.apply_transform(xyz=p_bw, transformation=T_bw_bs)
    p_us = bullet2unity_position(bullet_position=p_bs)
    p_uc = util.apply_transform(xyz=p_us, transformation=T_us_uc)

    # Transform orientation.
    up_uw = bullet2unity_vec(bvec=up_bw)
    up_uc = util.apply_transform(xyz=up_uw, transformation=T_uw_uc)
    return p_uc, up_uc


def ucam2bworld(
    p_uc: List[float],
    up_uc: List[float],
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
    T_bw_bs, T_uw_uc, T_us_uc = compute_bullet2unity_transforms(
        uworld_cam_position=uworld_cam_position,
        uworld_cam_orientation=uworld_cam_orientation,
    )

    # Transform the position.
    p_us = util.apply_inv_transform(xyz=p_uc, transformation=T_us_uc)
    p_bs = unity2bullet_position(unity_position=p_us)
    p_bw = util.apply_inv_transform(xyz=p_bs, transformation=T_bw_bs)

    # Transform orientation.
    up_uw = util.apply_inv_transform(xyz=up_uc, transformation=T_uw_uc)
    up_bw = unity2bullet_vec(uvec=up_uw)
    return p_bw, up_bw


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
    T_uw_uc = util.create_transformation(
        position=uworld_cam_position, orientation=uworld_cam_orientation
    )
    T_us_uc = T_uw_uc.dot(T_us_uw)
    return T_bw_bs, T_uw_uc, T_us_uc


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


def bullet2unity_objects(bullet_state: Dict[int, Dict]):
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
        # The object id must be defined in the Unity RGB mapping. Otherwise, Unity will
        # not be able to encode the object segmentation in the segmentation image it
        # produces.
        if type(oid) == str and oid.startswith("h_"):
            otag = oid
        elif oid in UNITY_OIDS:
            otag = f"{oid:02}"
        elif oid not in UNITY_OIDS:
            raise ValueError(f"Object ID not supported by Unity: {oid}")

        shape = odict["shape"]
        color = odict["color"]
        radius = odict["radius"]
        height = odict["height"]
        bullet_position = odict["position"]
        if "orientation" in odict:
            bullet_orientation = odict["orientation"]
        else:
            bullet_orientation = util.up_to_orientation(up=odict["up_vector"])

        # Convert the object size.
        width = radius * 2
        unity_size = bullet2unity_size(bullet_size=[width, width, height])

        # Convert the object position.
        bullet_position_shoulder = bullet_world2shoulder_position(
            pos_world=bullet_position
        )
        unity_rel_position = bullet2unity_position(
            bullet_position=bullet_position_shoulder
        )

        # Convert the object orientation.
        unity_rotation = bullet2unity_euler(bullet_orn=bullet_orientation)

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


def bullet_world2shoulder_position(pos_world: List[float]) -> List[float]:
    """ Converts from bullet absolution position to position in shoulder 
    coordinates.

    Args:
        pos_world: The position in bullet world coordinates.

    Returns:
        pos_shoulder: The position bullet shoulder coordinates.
    """
    pos_shoulder = np.array(pos_world) - np.array(const.BULLET_SHOULDER_POS)
    return pos_shoulder


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


def bullet2unity_vec(bvec: List[float]) -> List[float]:
    """Converts an up vector from bullet to unity coordinates.

    Args:
        bullet_up: The up vector in bullet coordinates.

    Returns:
        unity_up: The up vector in unity coordinates.
    """
    # bullet_euler = util.up_to_euler(up=bullet_up)
    x, y, z = bvec
    uvec = [-y, -z, x]
    # unity_up = util.euler_to_up(euler=unity_euler)
    return uvec


def unity2bullet_vec(uvec: List[float]) -> List[float]:
    """Converts an up vector from unity coordinates into bullet coordinates.

    Args:
        unity_up: The up vector, in unity coordinates.

    Returns:
        bullet_up: The up vector, in bullet coordinates.
    """
    # unity_euler = util.up_to_euler(up=unity_up)
    x, y, z = uvec
    bvec = [z, -x, -y]
    # bullet_up = util.euler_to_up(euler=bullet_euler)
    return bvec
