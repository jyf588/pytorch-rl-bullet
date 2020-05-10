import os
from typing import *

import exp.loader
import ns_vqa_dart.bullet.util as util


def save_scenes(scenes: List, exp: str, set_name: str):
    path = get_scenes_path(experiment=exp, set_name=set_name, create_dir=True)
    util.save_pickle(path=path, data=scenes)
    print(f"Saved scenes to: {path}.")


def load_scenes(exp: str, set_name: str):
    path = get_scenes_path(experiment=exp, set_name=set_name)
    scenes = util.load_pickle(path=path)
    return scenes


def get_scenes_path(
    experiment: str, set_name: str, create_dir: Optional[bool] = False
) -> str:
    set_dir = exp.loader.get_exp_set_dir(
        exp=experiment, set_name=set_name, create_dir=create_dir
    )
    path = os.path.join(set_dir, "scenes.p")
    return path
