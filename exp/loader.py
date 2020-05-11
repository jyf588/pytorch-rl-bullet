import os
from typing import *

import ns_vqa_dart.bullet.util as util


KEY2EXT = {"cam": "json", "rgb": "png", "masks": "npy"}


def get_exp_set_dir(exp: str, set_name: str, create_dir=False) -> str:
    set_dir = os.path.join(util.get_user_homedir(), "data/dash", exp, set_name)
    if create_dir:
        util.delete_and_create_dir(set_dir)
    return set_dir


def get_frame_dir(
    exp: str, set_name: str, key: str, scene_id: str, create_dir=False
) -> str:
    scene_dir = os.path.join(
        util.get_user_homedir(), "data/dash", exp, set_name, key, f"{scene_id:04}"
    )
    if create_dir:
        util.delete_and_create_dir(scene_dir)
    return scene_dir


def get_scenes_path(exp: str, set_name: str) -> str:
    return os.path.join(get_exp_set_dir(exp=exp, set_name=set_name), "scenes.p")


def get_states_path(exp: str, set_name: str, scene_id: int) -> str:
    return os.path.join(
        get_exp_set_dir(exp=exp, set_name=set_name), "states", f"{scene_id:04}.p"
    )


def get_frame_paths(
    exp: str, set_name: str, scene_id: int, timestep: int, keys: List, create_dir: bool
) -> str:
    paths = []
    for k in keys:
        frame_dir = get_frame_dir(
            exp=exp, set_name=set_name, key=k, scene_id=scene_id, create_dir=create_dir
        )
        path = os.path.join(frame_dir, f"{timestep:04}.{KEY2EXT[k]}")
        paths.append(path)
    return paths
