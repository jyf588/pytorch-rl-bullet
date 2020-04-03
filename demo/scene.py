import copy
import numpy as np
from typing import *

import my_pybullet_envs.utils as utils


class SceneGenerator:
    def __init__(self, base_scene: List[Dict], mu: Optional[float] = None):
        """
        Args:
            base_scene: The scene to base generated scenes off of, in the 
                format below. Note that only shape, color, and position are
                currently required.
                [
                    {
                        "shape": <shape>,
                        "color": <color>,
                        "radius": <radius>,
                        "height": <height>,
                        "position": [x, y, z],
                        "orientation": [x, y, z, w],
                        "mass": <mass>,
                        "mu": <mu>
                    },
                    ...
                ]
            mu: The object friction.
        """
        self.base_scene = base_scene

        self.mu = mu

    def generate(self):
        scene = copy.deepcopy(self.base_scene)

        # Fill in additional attributes programmatically.
        for idx, odict in enumerate(scene):
            if "height" not in odict:
                odict["height"] = np.random.uniform(utils.H_MIN, utils.H_MAX)
            if "radius" not in odict:
                odict["radius"] = np.random.uniform(
                    utils.HALF_W_MIN, utils.HALF_W_MAX
                )
            if "mass" not in odict:
                odict["mass"] = np.random.uniform(
                    utils.MASS_MIN, utils.MASS_MAX
                )
            if "mu" not in odict:
                odict["mu"] = self.mu

            # Downsize radius for boxes.
            if odict["shape"] == "box":
                odict["radius"] *= 0.8

            # Set the z position to be half of the object's height.
            odict["position"][2] = odict["height"] / 2

            # Set to zero rotation.
            odict["orientation"] = [0.0, 0.0, 0.0, 1.0]

            # Store the updated object dictionary.
            scene[idx] = odict
        return scene
