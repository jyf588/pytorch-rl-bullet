"""A class that saves Unity data."""
import json
import os
from typing import *

import ns_vqa_dart.bullet.util as util


class UnitySaver:
    def __init__(self, out_dir: str, save_keys: List[str]):
        """
        Args:
            out_dir: The output directory to save data into.
            save_keys: A list of keys for which to save data for.
        """
        self.out_dir = out_dir
        self.save_keys = save_keys

        os.makedirs(out_dir, exist_ok=True)

    def save(self, msg_id: str, data: Dict):
        """Saves unity data. The generated output files format is the 
        following:

        <self.out_dir>/
            <msg_id>/
                <tag_id>.json = {
                    "camera_position": List[float],  # The position of the camera, in unity world coordinate frame.
                    "camera_orientation": List[float],  # The orientation of the camera (xyzw quaternion), in unity world coordinate frame.
                }
            

        Args:
            data: A dictionary containing unity data, in the format:
                {
                    <tag_id>:{
                        "camera_position": List[float],  # The position of the camera, in unity world coordinate frame.
                        "camera_orientation": List[float],  # The orientation of the camera (xyzw quaternion), in unity world coordinate frame.
                        "rgb": np.ndarray,  # The RGB image.
                        "seg_img": np.ndarray, # The segmentation, as a RGB image.
                    },
                    ...
                }
        
        Note that only `camera_position` and `camera_orientation` are currently
        being saved.
        """
        path = os.path.join(self.out_dir, f"{msg_id}.json")
        data_to_save = {}
        for tag_id, tag_data in data.items():
            data_to_save[tag_id] = {}
            for k in self.save_keys:
                data_to_save[tag_id][k] = data[tag_id][k]
        util.save_json(path=path, data=data_to_save)
