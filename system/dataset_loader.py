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
from ns_vqa_dart.bullet import util


class DatasetLoader:
    def __init__(
        self, stage: str, states_dir: str, start_trial_incl: int, end_trial_incl: int
    ):
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
        self.stage = stage
        self.states_dir = states_dir
        self.start_trial_incl = start_trial_incl
        self.end_trial_incl = end_trial_incl

        self.idx = 0
        self.examples = self.collect_examples()

    def collect_examples(self):
        examples = []
        if self.stage == "plan":
            for f in sorted(os.listdir(self.states_dir)):
                example_id = f.split(".")[0]
                path = os.path.join(self.states_dir, f)
                e = (example_id, path)
                examples.append(e)
        elif self.stage == "place":
            n_trials = 0
            for t in sorted(os.listdir(self.states_dir)):
                if self.start_trial_incl <= int(t) <= self.end_trial_incl:
                    t_dir = os.path.join(self.states_dir, t)
                    trial_examples = []
                    for f in sorted(os.listdir(t_dir)):
                        sid = f.split(".")[0]
                        path = os.path.join(t_dir, f)
                        example_id = f"{t}_{sid}"
                        e = (example_id, path)
                        trial_examples.append(e)
                    examples += trial_examples
                    if len(trial_examples) > 0:
                        n_trials += 1
            print(f"Loaded {len(examples)} examples from {n_trials} trials.")
        return examples

    def get_next_state(self) -> Optional[Tuple]:
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
        # No more examples.
        if self.idx > len(self.examples):
            return None, None

        example_id, path = self.examples[self.idx]
        state = util.load_pickle(path)

        self.idx += 1
        return example_id, state
