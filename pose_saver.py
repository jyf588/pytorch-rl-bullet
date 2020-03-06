import json
import pybullet as p
from typing import *


class PoseSaver:
    def __init__(self, path: str, oids: List[int], robot_id: int):
        """Saves object and robot poses.

        Args:
            path: The JSON filepath to save poses to.
            oids: A list of object IDs.
            robot_id: The ID of the robot.
        """
        self.path = path
        self.oids = oids
        self.robot_id = robot_id
        self.poses = []

    def get_poses(self):
        """Queries poses for the current bullet scene."""
        # Get object poses.
        oid2pose = {}
        for oid in self.oids:
            position, orientation = p.getBasePositionAndOrientation(oid)
            oid2pose[oid] = {"position": position, "orientation": orientation}
        joint_name2angle = {}
        for joint_idx in range(p.getNumJoints(self.robot_id)):
            joint_name = p.getJointInfo(self.robot_id, joint_idx)[1].decode(
                "utf-8"
            )
            joint_angle = p.getJointState(
                bodyUniqueId=self.robot_id, jointIndex=joint_idx
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
