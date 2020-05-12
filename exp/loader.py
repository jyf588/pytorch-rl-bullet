import os
import imageio
import numpy as np
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

        self.opt = ExpLoader(exp_name=exp_name).set_name2opt[set_name]

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

    """Frame-level functions"""

    def get_key_dir(self, key: str):
        key_dir = os.path.join(self.set_dir, key)
        return key_dir

    def get_scene2frames(self):
        scene2frames = {}
        for scene_id in self.get_scene_ids():
            scene_loader = SceneLoader(
                exp_name=self.exp_name, set_name=self.set_name, scene_id=scene_id
            )
            timesteps = scene_loader.get_timesteps()
            scene2frames[int(scene_id)] = timesteps
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


class SceneLoader:
    def __init__(self, exp_name: str, set_name: str, scene_id: str):
        set_loader = SetLoader(exp_name=exp_name, set_name=set_name)
        self.states_dir = os.path.join(set_loader.set_dir, "states", scene_id)
        self.cam_dir = os.path.join(set_loader.set_dir, "cam", scene_id)
        self.rgb_dir = os.path.join(set_loader.set_dir, "rgb", scene_id)
        self.masks_dir = os.path.join(set_loader.set_dir, "masks", scene_id)
        self.detectron_masks_dir = os.path.join(
            set_loader.set_dir, "detectron_masks", scene_id
        )

    def get_timesteps(self):
        timesteps = []
        for fname in sorted(os.listdir(self.states_dir)):
            ts = int(fname.split(".")[0])
            timesteps.append(ts)
        return timesteps

    """State-related functions"""

    def get_state_path(self, timestep: int):
        path = os.path.join(self.states_dir, f"{timestep:06}.p")
        return path

    def save_state(self, scene_id: str, timestep: int, state: Dict):
        path = self.get_state_path(timestep=timestep)
        util.save_pickle(path=path, data=state)

    def load_state(self, timestep: int):
        path = self.get_state_path(timestep=timestep)
        state = util.load_pickle(path=path)
        return state

    def load_scene_states(self):
        ts_state_list = []
        timesteps = self.get_timesteps()
        for ts in timesteps:
            state = self.load_state(timestep=ts)
            ts_state_list.append((ts, state))
        return ts_state_list

    """Frame-related functions"""

    def get_cam_path(self, timestep: int):
        path = os.path.join(self.cam_dir, f"{timestep:06}.json")
        return path

    def get_rgb_path(self, timestep: int):
        path = os.path.join(self.rgb_dir, f"{timestep:06}.png")
        return path

    def get_masks_path(self, timestep: int):
        path = os.path.join(self.masks_dir, f"{timestep:06}.npy")
        return path

    def get_detectron_masks_path(self, timestep: int):
        path = os.path.join(self.detectron_masks_dir, f"{timestep:06}.npy")
        return path

    def save_cam(self, timestep: int, cam_dict: Dict):
        path = self.get_cam_path(timestep=timestep)
        util.save_json(path=path, data=cam_dict)

    def save_rgb(self, timestep: int, rgb: np.ndarray):
        path = self.get_rgb_path(timestep=timestep)
        imageio.imwrite(path, rgb)

    def save_masks(self, timestep: int, masks: np.ndarray):
        path = self.get_masks_path(timestep=timestep)
        np.save(path, masks)

    def save_detectron_masks(self, timestep: str, masks: np.ndarray):
        path = self.get_detectron_masks_path(timestep=timestep)
        np.save(path, masks)
