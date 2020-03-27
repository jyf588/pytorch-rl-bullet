import copy
import json
import os
import pybullet as p
from typing import *

import my_pybullet_envs


class PoseSaver:
    def __init__(self, path: str, odicts: List[Dict], robot_id: int):
        """Saves object and robot poses.

        Args:
            path: The JSON filepath to save poses to.
            odicts: A list of object dictionaries to save poses for. Format:
                {
                    "id": Object ID.
                    "shape": Pybullet geometry enum.
                    "color": RGBA color.
                    "half_width": The radius.
                    "height": The height.
                }
            robot_id: The ID of the robot.
        """
        self.path = path
        self.robot_id = robot_id
        self.poses = []

        # Create the path's directory if it doesn't already exist.
        pathdir = os.path.dirname(self.path)
        os.makedirs(pathdir, exist_ok=True)

        self.oid2state = {}
        for odict in odicts:
            oid = odict["id"]

            shape = my_pybullet_envs.utils.GEOM2SHAPE[odict["shape"]]
            color = my_pybullet_envs.utils.RGBA2COLOR[tuple(odict["color"])]
            radius = odict["half_width"]
            height = odict["height"]

            self.oid2state[oid] = {
                "shape": shape,
                "color": color,
                "radius": radius,
                "height": height,
            }

    def get_poses(self):
        """Queries poses for the current bullet scene."""
        # Query the current object poses.
        for oid in self.oid2state.keys():
            position, orientation = p.getBasePositionAndOrientation(oid)
            self.oid2state[oid]["position"] = position
            self.oid2state[oid]["orientation"] = orientation

        # Query the current robot pose.
        joint_name2angle = {}
        for joint_idx in range(p.getNumJoints(self.robot_id)):
            joint_name = p.getJointInfo(self.robot_id, joint_idx)[1].decode(
                "utf-8"
            )
            joint_angle = p.getJointState(
                bodyUniqueId=self.robot_id, jointIndex=joint_idx
            )[0]
            joint_name2angle[joint_name] = joint_angle
        self.poses.append(
            {
                "objects": copy.deepcopy(self.oid2state),
                "robot": joint_name2angle,
            }
        )

    def save(self):
        """Saves the poses to a JSON file."""
        with open(self.path, "w") as f:
            json.dump(
                self.poses, f, sort_keys=True, indent=2, separators=(",", ": ")
            )
        print(f"Saved poses to: {self.path}")
