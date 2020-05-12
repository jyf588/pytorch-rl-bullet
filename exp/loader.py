import os
from typing import *

from exp.options import EXPERIMENT_OPTIONS
import ns_vqa_dart.bullet.util as util


KEY2EXT = {"cam": "json", "img": "png", "masks": "npy"}


class ExpLoader:
    def __init__(self, exp_name: str):
        self.set_names = list(EXPERIMENT_OPTIONS[exp_name].keys())
        self.set_name2opt = EXPERIMENT_OPTIONS[exp_name]


class SetLoader:
    def __init__(
        self, exp_name: str, set_name: str, root_dir: Optional[str] = "data/dash"
    ):
        self.exp_name = exp_name
        self.set_name = set_name
        self.root_dir = root_dir

        self.set_dir = self.construct_set_dir()
        self.scenes_dir = os.path.join(self.set_dir, "scenes")
        self.states_root_dir = os.path.join(self.set_dir, "states")

    def construct_set_dir(self):
        set_dir = os.path.join(
            util.get_user_homedir(), self.root_dir, self.exp_name, self.set_name
        )
        return set_dir

    """Scene-related functions"""

    def get_scene_path(self, scene_id: str):
        path = os.path.join(self.scenes_dir, f"{scene_id}.p")
        return path

    def save_scenes(self, scenes: List):
        # Create the scenes directory.
        os.makedirs(self.scenes_dir)

        # Save scenes one by one.
        for idx, scene in enumerate(scenes):
            path = self.get_scene_path(scene_id=f"{idx:04}")
            util.save_pickle(path=path, data=scene)

    def get_scene_ids(self) -> List:
        scene_ids = sorted([name.split(".")[0] for name in os.listdir(self.scenes_dir)])
        return scene_ids

    def load_id2scene(self) -> Dict:
        id2scene = {}
        for scene_id in self.get_scene_ids():
            scene = util.load_pickle(path=self.get_scene_path(scene_id=scene_id))
            id2scene[scene_id] = scene
        return id2scene

    """State-related functions"""

    def get_scene_states_dir(self, scene_id: str):
        scene_states_dir = os.path.join(self.states_root_dir, scene_id)
        return scene_states_dir

    def create_states_dir_for_scene(self, scene_id: str):
        scene_states_dir = self.get_scene_states_dir(scene_id=scene_id)
        os.makedirs(scene_states_dir)

    def get_state_path(self, scene_id: str, timestep: int):
        path = os.path.join(
            self.get_scene_states_dir(scene_id=scene_id), f"{timestep:06}.p"
        )
        return path

    def save_state(self, scene_id: str, timestep: int, state: Dict):
        path = self.get_state_path(scene_id=scene_id, timestep=timestep)
        util.save_pickle(path=path, data=state)

    """Frame-level functions"""

    def get_key_dir(self, key: str):
        key_dir = os.path.join(self.set_dir, key)
        return key_dir

    def get_scene2frames(self):
        scene2frames = {}
        for scene_id in self.load_id2scene().keys():
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
