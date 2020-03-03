import json
import pybullet as p
from typing import *


class PoseSaver:
    def __init__(self, path: str):
        """Saves object and robot poses.

        Args:
            path: The JSON filepath to save poses to.
        """
        self.path = path
        self.poses = []

    def get_poses(self, oids: List[int], robot_id: int):
        """Saves poses for a robot and objects in a Bullet scene.

        Args:
            oids: A list of object IDs.
            robot_id: The ID of the robot.
        """
        # Get object poses.
        oid2pose = {}
        for oid in oids:
            position, orientation = p.getBasePositionAndOrientation(oid)
            oid2pose[oid] = {"position": position, "orientation": orientation}
        joint_name2angle = {}
        for joint_idx in range(p.getNumJoints(robot_id)):
            joint_name = p.getJointInfo(robot_id, joint_idx)[1].decode("utf-8")
            joint_angle = p.getJointState(
                bodyUniqueId=robot_id, jointIndex=joint_idx
            )[0]
            joint_name2angle[joint_name] = joint_angle
        self.poses.append({"objects": oid2pose, "robot": joint_name2angle})

    def save(self):
        """Saves the poses to a JSON file."""
        with open(self.path, "w") as f:
            json.dump(
                self.poses, f, sort_keys=True, indent=2, separators=(",", ": ")
            )
        print(f"Saved poses to: {self.path}")
