import math
import random
import pybullet
import argparse
import numpy as np
from typing import *


class SceneGenerator:
    def __init__(self, seed: int, opt: Dict):
        """
        Args:
            seed: The random seed.
            opt: Various options.
        """
        self.opt = opt

        # Set the seed for random number generators.
        np.random.seed(seed)
        random.seed(seed)

    def generate_tabletop_objects(
        self,
        existing_odicts: Optional[List[Dict]] = None,
        unique_odicts: Optional[List[Dict]] = None,
    ) -> List:
        """Generates a single random scene.

        Note that box radius is downscaled by 0.8.
        
        Args:
            existing_odicts: A list of existing objects to include in the
                closeness tests so that new objects are not too close to the
                existing objects.
            unique_odicts: A list of object dictionaries that generated 
                objects need to be unique from.

        Returns:
            odicts: A list of newly generated objects. The object
                dictionaries provided in `existing_odicts` are excluded.
        """
        if existing_odicts is None:
            existing_odicts = []

        # Randomly select the number of objects to generate.
        n_objects = self.generate_n_objects()

        odicts: List[Dict] = []
        n_tries = 0
        # Generate `n_objects` objects, one by one.
        while len(odicts) < n_objects and n_tries < self.opt["max_retries"]:
            odict = self.generate_object()

            # Check if generated object is too close to others.
            other_odicts = odicts + existing_odicts
            if self.any_close(src_odict=odict, other_odicts=other_odicts):
                n_tries += 1
            # Check whether the generated object is unique from `unique_odicts`.
            elif unique_odicts is not None and self.found_matching_object(
                src_odict=odict, unique_odicts=unique_odicts
            ):
                n_tries += 1
            else:
                odicts.append(odict)
        return odicts

    def generate_object(self):
        """Generates a random object.

        Note that box radius is downscaled by 0.8.
        
        Returns:
            odict: The randomly generated object dictionary.
        """
        shape = random.choice(self.opt["shapes"])
        radius, height = self.generate_random_size(shape=shape)
        position = self.generate_random_xyz(height=height)

        odict = {
            "shape": shape,
            "color": random.choice(self.opt["colors"]),
            "radius": radius,
            "height": height,
            "position": position,
            "orientation": self.generate_orientation(shape=shape),
            "mass": self.sample_float(self.opt["mass"]),
            "mu": self.sample_float(self.opt["mu"]),
        }
        return odict

    def generate_n_objects(self) -> int:
        value_type = type(self.opt["n"])
        if value_type == int:
            return self.opt["n"]
        elif value_type in [tuple, list]:
            # `self.opt.n[1]` is exclusive while `random.randint` is
            # inclusive, so that's why we subtract one from the max.
            min_objs, max_objs = self.opt["n"]
            n_objects = random.randint(min_objs, max_objs - 1)
        else:
            raise ValueError(f"Invalid type for n_objects: {value_type}")
        return n_objects

    def generate_random_size(self, shape: str) -> Tuple[float, float]:
        """Generates a random radius and height. Note that box radius is 
        downscaled by 0.8.

        Args:
            shape: The shape we are generating a size for.

        Returns:
            radius: The radius of the object.
            height: The height of the object. This is 2*r for sphere.
        """
        radius = self.sample_float(self.opt["radius"])
        height = self.sample_float(self.opt["height"])
        if shape == "sphere":
            height *= 0.75
            radius = height / 2
        if shape == "box":
            radius *= 0.8
        return radius, height

    def generate_random_xyz(self, height: Optional[float] = None) -> List[float]:
        """Generates a random xyz based on axis bounds.

        Returns:
            xyz: The randomly generated xyz values.
        """
        xyz = []
        for axis_range in [self.opt["x_pos"], self.opt["y_pos"], self.opt["z_pos"]]:
            axis_value = self.sample_float(range=axis_range)
            xyz.append(axis_value)

        # Modify the z coordinate depending on whether the user wants
        # the z coordinate of the com or the base.
        if self.opt["position_mode"] == "com" and height is not None:
            xyz[2] += height / 2
        elif self.opt["position_mode"] == "base":
            pass
        else:
            raise ValueError(f'Invalid position mode: {self.opt["position_mode"]}')
        return xyz

    def generate_orientation(self, shape):
        if self.opt["z_rot"] == 0.0 or shape == "sphere":
            orientation = [0.0, 0.0, 0.0, 1.0]
        else:
            orientation = pybullet.getQuaternionFromEuler(
                [0.0, 0.0, self.sample_float(self.opt["z_rot"])]
            )
        return orientation

    def sample_float(self, range) -> float:
        if type(range) == float:
            return range
        elif type(range) in [tuple, list] and len(range) == 2:
            low, high = range
            return np.random.uniform(low=low, high=high)
        else:
            raise ValueError(f"Invalid range type: {type(range)}")

    def any_close(self, src_odict: Dict, other_odicts: List[Dict]) -> bool:
        """Checks if the source object is close to any of the other objects.

        Args:
            src_odict: The source object dictionary.
            other_odicts: The other object dictionaries.
        Returns:
            is_close: Whether the source object is close to any of the other 
                objects in xy space.
        """
        close_arr = [
            self.is_close(
                ax=src_odict["position"][0],
                ay=src_odict["position"][1],
                bx=other_odict["position"][0],
                by=other_odict["position"][1],
            )
            for other_odict in other_odicts
        ]
        is_close = any(close_arr)
        return is_close

    def is_close(self, ax: float, ay: float, bx: float, by: float) -> bool:
        """Checks whether two (x, y) points are within a certain threshold 
        distance of each other.

        Args:
            ax: The x position of the first point.
            ay: The y position of the first point.
            bx: The x position of the second point.
            by: The y position of the second point.
        
        Returns:
            Whether the distance between the two points is less than or equal
                to the threshold distance.
        """
        return (ax - bx) ** 2 + (ay - by) ** 2 <= self.opt["dist_thresh"] ** 2

    def found_matching_object(self, src_odict: Dict, unique_odicts: List[Dict]):
        """Checks whether we were able to find a matching object between the source
        object and the list of objects we need to be unique from.

        Returns:
            found_matching_object: Whether we found at least one object in `unique_odicts`
                that matches the description of the `src_odict`.
        """
        for unique_odict in unique_odicts:
            # By default, we assume the objects have the same description.
            same_description = True

            # Check to see if any of the attributes used to describe the objects differ.
            for attr in ["shape", "color"]:
                # If one of the attributes differ, we know they have unique descriptions.
                if src_odict[attr] != unique_odict[attr]:
                    same_description = False
                    break
            # We found an object with the same description as the source object.
            if same_description:
                return True
        # We were not able to find any matching objects.
        return False


def gen_rand_color(colors):
    c = random.choice(colors)
    return c
