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
import os
import pprint
import pybullet as p
from typing import *
import shutil
import sys

import my_pybullet_envs
from my_pybullet_envs import utils
from ns_vqa_dart.bullet import util


class StateSaver:
    def __init__(self, out_dir: str):
        """Saves object and robot poses.

        Args:
            out_dir: The JSON filepath to save poses to.            
        """
        self.out_dir = out_dir

        # Create the directory, deleting the existing directory contents if
        # requested.
        util.delete_and_create_dir(dir=out_dir)

        self.poses = []
        self.robot_id = None
        self.oid2state = []
        self.sid = 0
        self.oid2attr = {}

    def track(self, trial: int, odicts: Dict[int, Dict], robot_id: int):
        """
        Tracks objects and robot.
        
        Args:
            odicts: A dict of obj dicts keyed by bullet id. Format:
                {
                    <oid>: {
                        "shape": <shape>,
                        "color": <color>,
                        "radius": <radius>,
                        "height": <height>,
                        "position": <position>,
                        "orientation": <orientation>
                    }
                }
            robot_id: The ID of the robot.
        """
        self.robot_id = robot_id
        self.oid2attr = odicts
        self.trial = trial

    def save_state(self):
        """
        Queries bullet for the current state of tracked objects and robot.
        """
        object_states = self.get_object_states()
        robot_state = self.get_robot_state()

        # Combine in a state dictionary.
        state = {
            "trial": self.trial,
            "objects": object_states,
            "robot": robot_state,
        }

        # Save into a pickle file.
        path = os.path.join(self.out_dir, f"{self.sid:07}.p")
        my_pybullet_envs.utils.save_pickle(path=path, data=state)
        print(f"Saved poses to: {path}")
        print(f"trial: {self.trial}")
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
            try:
                position, orientation = p.getBasePositionAndOrientation(oid)
            except p.error as e:
                pprint.pprint(self.oid2attr)
                print(f"oid: {oid}")
                print(f"attr: {attr}")
                raise e
            state["position"] = list(position)
            state["orientation"] = list(orientation)
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
