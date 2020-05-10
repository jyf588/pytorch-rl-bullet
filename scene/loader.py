import os
from typing import *

import ns_vqa_dart.bullet.util as util


def save_scenes(scenes: List, experiment: str, set_name: str):
    path = get_scenes_path(experiment=experiment, set_name=set_name, create_dir=True)
    util.save_pickle(path=path, data=scenes)
    print(f"Saved scenes to: {path}.")


def load_scenes(experiment: str, set_name: str):
    path = get_scenes_path(experiment=experiment, set_name=set_name)
    scenes = util.load_pickle(path=path)
    return scenes


def get_scenes_path(
    experiment: str, set_name: str, create_dir: Optional[bool] = False
) -> str:
    set_dir = os.path.join(util.get_user_homedir(), "data/dash", experiment, set_name)
    if create_dir:
        util.delete_and_create_dir(set_dir)
    path = os.path.join(set_dir, "scenes.p")
    return path
