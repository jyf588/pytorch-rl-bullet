import os
from typing import *

from exp.options import EXPERIMENT_OPTIONS
import ns_vqa_dart.bullet.util as util


KEY2EXT = {"cam": "json", "img": "png", "masks": "npy"}


class ExpLoader:
    def __init__(self, exp_name: str):
        self.set_names = list(EXPERIMENT_OPTIONS[exp_name].keys())


class SetLoader:
    def __init__(
        self, exp_name: str, set_name: str, root_dir: Optional[str] = "data/dash"
    ):
        self.exp_name = exp_name
        self.set_name = set_name
        self.root_dir = root_dir

        self.set_dir = self.construct_set_dir()

    def construct_set_dir(self):
        set_dir = os.path.join(
            util.get_user_homedir(), self.root_dir, self.exp_name, self.set_name
        )
        return set_dir

    def get_key_dir(self, key: str):
        key_dir = os.path.join(self.set_dir, key)
        return key_dir

    def get_scene2frames(self):
        scene2frames = {}
        key_dir = self.get_key_dir(key="img")
        for scene_id in sorted(os.listdir(key_dir)):
            frames = []
            for fname in sorted(os.listdir(os.path.join(key_dir, scene_id))):
                frame_id = int(fname.split(".")[0])
                frames.append(frame_id)
            scene2frames[int(scene_id)] = frames
        return scene2frames

    def get_frame_path(self, key: str, scene_id: int, frame_id: int):
        path = os.path.join(
            self.get_key_dir(key=key), f"{scene_id:04}", f"{frame_id:04}.{KEY2EXT[key]}"
        )
        return path

    def get_ids(self):
        ids = []
        for scene_id, frame_ids in self.get_scene2frames().items():
            for frame_id in frame_ids:
                ids.append((scene_id, frame_id))
        return ids

    def get_key2paths(self):
        k2paths = {}
        n_examples = None

        scene_frame_ids = self.get_ids()
        for k in ["cam", "masks", "img"]:
            paths = []
            for (scene_id, frame_id) in scene_frame_ids:
                path = self.get_frame_path(key=k, scene_id=scene_id, frame_id=frame_id)
                paths.append(path)
            k2paths[k] = paths
            if n_examples is None:
                n_examples = len(paths)
            else:
                assert len(paths) == n_examples
        return k2paths

    def __len__(self):
        return len(self.get_key2paths()["img"])


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
