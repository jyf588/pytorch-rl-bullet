"""Contains a class definition for bullet."""
import math
import numpy as np
import pybullet as p
import time

import bullet2unity.const as const


class Bullet:
    def __init__(self, args):
        self.fps = args.fps
        self.motion_type = args.motion_type
        self.move_agent = args.move_agent
        self.n_steps = args.n_steps
        self.rotate_objects = args.rotate_objects
        self.sin_amplitude = args.sin_amplitude
        self.sin_vshift = args.sin_vshift

        # "Connect" to bullet.
        cin = p.connect(p.SHARED_MEMORY)
        if cin < 0:
            cin = p.connect(p.GUI)

        # Load objects.
        _ = self.load_object(const.TABLE)
        self.object_ids = self.load_objects(const.OBJECTS)
        self.robot_id = self.load_object(const.ROBOT)

        # Set the initial robot pose.
        for jointIndex in range(p.getNumJoints(self.robot_id)):
            if args.pose == "grasp":
                p.resetJointState(
                    self.robot_id, jointIndex, const.JOINT_ANGLES[jointIndex]
                )
            elif args.pose == "rest":
                p.resetJointState(self.robot_id, jointIndex, 0.0)
            else:
                raise ValueError(f"Invalid pose: {args.start_pos}")

        cid0 = p.createConstraint(
            4,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000],
            [-0.300000, 0.500000, -1.250000],
            [0.000000, 0.000000, 0.000000, 1.000000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        )
        p.changeConstraint(cid0, maxForce=500.000000)
        p.setGravity(0.000000, 0.000000, 0.000000)

    def load_objects(self, objects_list):
        """Loads objects from their urdf files."""
        object_ids = []
        for obj_info in objects_list:
            object_ids.append(self.load_object(obj_info))
        return object_ids

    def load_object(self, obj_info):
        return p.loadURDF(*obj_info)

    def step(self, curtime):
        """Computes poses of objects and joints at each bullet timestep"""
        p.stepSimulation()
        time.sleep(1.0 / self.fps)

        joint_angles = self.get_robot_joint_angles(curtime)
        object_poses = self.get_object_poses(curtime)

        poses = joint_angles + object_poses
        return poses

    def get_robot_joint_angles(self, curtime):
        """Computes robot joint angles for the current time."""
        # Vary the joints if desired.
        if self.move_agent:
            for jointIndex in const.VARY_JOINTS:
                if self.motion_type == "sinusoidal":
                    degrees = (
                        math.sin(curtime * math.pi) * self.sin_amplitude
                        + self.sin_vshift
                    )
                    radians = math.pi / 180 * degrees
                elif self.motion_type == "circular":
                    radians = (curtime % (2 * math.pi)) - math.pi
                else:
                    raise ValueError(
                        f"Invalid motion type: {self.motion_type}"
                    )
                # Set the joint angles.
                p.resetJointState(self.robot_id, jointIndex, radians)

        # Filter on only the robot joints we want to send.
        joint_states = p.getJointStates(self.robot_id, const.SEND_JOINTS)
        joint_angles = [j[0] for j in joint_states]
        return joint_angles

    def get_object_poses(self, curtime):
        """Compute object poses for the current time."""
        # Set object 6DOF poses.
        if self.rotate_objects:
            x_rad = (curtime % (2 * math.pi)) - math.pi
            y_rad = (curtime % (2 * math.pi)) - math.pi
            z_rad = (curtime % (2 * math.pi)) - math.pi
        else:
            x_rad, y_rad, z_rad = 0.0, 0.0, 0.0
        bullet_rotation_rads = [x_rad, y_rad, z_rad]

        # Get the shoulder position for computing relative object positions
        # below.
        shoulder_pos = p.getLinkState(self.robot_id, 25)[4]

        # Convert object positions from bullet to unity coordinates.
        poses = []
        for obj_id in self.object_ids:
            position, quaternion = p.getBasePositionAndOrientation(obj_id)
            rel_position = np.array(position) - np.array(shoulder_pos)

            # Set new rotation in bullet
            quaternion = p.getQuaternionFromEuler(bullet_rotation_rads)
            p.resetBasePositionAndOrientation(obj_id, position, quaternion)

            # Compute values for unity
            rel_position = bullet_to_unity_pos(rel_position)
            unity_rotation = bullet_to_unity_rot(bullet_rotation_rads)

            poses += list(rel_position)
            poses += list(unity_rotation)
        return poses

    def disconnect(self):
        """Disconnect from the bullet physics server."""
        p.disconnect()


def bullet_to_unity_pos(vector):
    """Converts from bullet to unity position"""
    new_vector = np.copy(vector)
    new_vector = swap_axes(new_vector, 1, 2)  # swap y and z
    new_vector = swap_axes(new_vector, 0, 2)  # swap x and z
    new_vector[2] *= -1  # Negate z
    return new_vector


def bullet_to_unity_rot(vector):
    """Converts bullet to unity rotation"""
    new_vector = np.copy(vector)
    new_vector = swap_axes(new_vector, 0, 2)  # swap x and z
    new_vector = swap_axes(new_vector, 0, 1)  # swap x and y
    new_vector[0] *= -1  # Negate x
    new_vector[1] *= -1  # Negate y

    # Convert radians to degrees
    for i in range(len(new_vector)):
        new_vector[i] = radians_to_degrees(new_vector[i])
    return new_vector


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
