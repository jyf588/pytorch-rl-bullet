"""Saves bullet states into pickle files.

Generated output structure:
    <args.out_dir>/
        <sid>.p = {
            "objects": [
                {
                    "shape": <shape>,
                    "color": <color>,
                    "radius": <radius>,
                    "height": <height>,
                    "position": <position>,
                    "orientation": <orientation>
                },
                ...
            ],
            "robot": {
                <joint_name>: <joint_angle>,
                ...
            }
        }
"""

import copy
import json
import os
import pybullet as p
from typing import *

import my_pybullet_envs
import ns_vqa_dart.bullet.util as util


class StateSaver:
    def __init__(self, out_dir: str):
        """Saves object and robot poses.

        Args:
            out_dir: The JSON filepath to save poses to.            
        """
        self.out_dir = out_dir

        # Create the directory if it doesn't already exist.
        os.makedirs(out_dir, exist_ok=True)

        self.poses = []
        self.robot_id = None
        self.oid2state = []
        self.sid = 0

    def track(self, odicts: List[Dict], robot_id: int):
        """
        Tracks objects and robot.
        
        Args:
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
        self.robot_id = robot_id

        self.oid2attr = {}
        for odict in odicts:
            oid, new_odict = self.convert_odict(odict=odict)
            self.oid2attr[oid] = new_odict

    def convert_odict(self, odict: Dict):
        """Converts from Yifeng's object dictionary format for Michelle's 
        format.

        Args:
            odict: An object dictionary with the following format:
                {
                    "id": Object ID.
                    "shape": Pybullet geometry enum.
                    "color": RGBA color.
                    "half_width": The radius.
                    "height": The height.
                }
            
        Returns:
            oid: The object ID.
            new_odict: An object dictionary with the following format:
                {
                    "shape": <shape>,
                    "color": <color>,
                    "radius": <radius>,
                    "height": <height>,
                    "position": <position>,
                    "orientation": <orientation>
                }
        
        Note that if "color" is not supplied in the original input, the 
        resulting dictionary will not contain color.
        
        """
        oid = odict["id"]

        new_odict = {
            "shape": my_pybullet_envs.utils.GEOM2SHAPE[odict["shape"]],
            "radius": odict["half_width"],
            "height": odict["height"],
        }
        if "color" in odict:
            new_odict["color"] = my_pybullet_envs.utils.RGBA2COLOR[
                tuple(odict["color"])
            ]
        return oid, new_odict

    def save_state(self):
        """
        Queries bullet for the current state of tracked objects and robot.
        """
        object_states = self.get_object_states()
        robot_state = self.get_robot_state()

        # Combine in a state dictionary.
        state = {
            "objects": object_states,
            "robot": robot_state,
        }

        # Save into a pickle file.
        path = os.path.join(self.out_dir, f"{self.sid:06}.p")
        util.save_pickle(path=path, data=state)
        print(f"Saved poses to: {path}")
        self.sid += 1

    def get_object_states(self) -> List[Dict]:
        """Updates object states.
        
        Returns:
            object_states: A list of object state dictionaries with the format:
                [
                    {
                        "shape": <shape>,
                        "color": <color>,
                        "radius": <radius>,
                        "height": <height>,
                        "position": <position>,
                        "orientation": <orientation>
                    },
                    ...
                ]
        """
        object_states = []
        for oid, attr in self.oid2attr.items():
            state = copy.deepcopy(attr)
            position, orientation = p.getBasePositionAndOrientation(oid)
            state["position"] = position
            state["orientation"] = orientation
            object_states.append(state)
        return object_states

    def get_robot_state(self) -> Dict[str, float]:
        """Updates robot states.
        
        Returns:
            robot_state: A dictionary with the following format:
                {
                    <joint_name>: <joint_angle>
                }
        """
        robot_state = {}
        for joint_idx in range(p.getNumJoints(self.robot_id)):
            joint_name = p.getJointInfo(self.robot_id, joint_idx)[1].decode(
                "utf-8"
            )
            joint_angle = p.getJointState(
                bodyUniqueId=self.robot_id, jointIndex=joint_idx
            )[0]
            robot_state[joint_name] = joint_angle
        return robot_state
