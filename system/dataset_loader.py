"""Contains a class definition for bullet."""
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


class DatasetLoader:
    def __init__(self, states_dir: str, start_id: int, end_id: int):
        """
        Args:
            states_dir: The directory containing the states, with the following
                structure:
                <states_dir>/
                    <sid>.p = {
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
                    }
        """
        self.states_dir = states_dir
        self.end_id = end_id
        self.scene_counter = start_id

    def get_next_state(self) -> Tuple[str, Dict]:
        """Retrieves the next bullet state and converts to Unity state.
        
        Returns:
            state_id: The ID of the state.
            state: The Unity state, which is a list with the following
                format: {
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
                }
        """
        if self.scene_counter >= self.end_id:
            return None, None

        sid = f"{self.scene_counter:06}"

        # Loads the state for the current sid.
        state_path = os.path.join(self.states_dir, f"{sid}.p")
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.scene_counter += 1
        return sid, state
