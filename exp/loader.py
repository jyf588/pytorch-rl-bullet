import os
import imageio
import numpy as np
from typing import *

import ns_vqa_dart.bullet.util as util
from exp.options import EXPERIMENT_OPTIONS
from ns_vqa_dart.bullet.seg import UNITY_OIDS


KEY2EXT = {"cam": "json", "img": "png", "masks": "npy"}


class ExpLoader:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name

        self.set_names = list(EXPERIMENT_OPTIONS[exp_name].keys())
        self.set_name2opt = EXPERIMENT_OPTIONS[exp_name]

    def get_idx2info(self):
        idx2info = {}
        idx = 0
        for set_name in self.set_names:
            set_loader = SetLoader(exp_name=self.exp_name, set_name=set_name)
            for scene_id in set_loader.get_scene_ids():
                scene_loader = SceneLoader(
                    exp_name=self.exp_name, set_name=set_name, scene_id=scene_id
                )
                for ts in scene_loader.get_timesteps():
                    for oid in scene_loader.get_oids(timestep=ts):
                        idx2info[idx] = (set_name, scene_id, ts, oid)
                        idx += 1
        return idx2info


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
        self.masks_root_dir = os.path.join(set_loader.set_dir, "masks", scene_id)
        self.detectron_masks_dir = os.path.join(
            set_loader.set_dir, "detectron_masks", scene_id
        )

    def get_timesteps(self):
        timesteps = []
        for fname in sorted(os.listdir(self.rgb_dir)):
            ts = int(fname.split(".")[0])
            timesteps.append(ts)
        return timesteps

    """State-related functions"""

    def get_state_path(self, timestep: int):
        path = os.path.join(self.states_dir, f"{timestep:06}.p")
        return path

    def save_state(self, scene_id: str, timestep: int, state: Dict):
        # Convert from Bullet object IDs to IDs we assign ourselves.
        new_object_states = {}

        # Here we assume a mapping that is determined by the order of the dictionary
        # values. The main reason that we reassign the IDs here is that Unity only
        # supports a few IDs; however, IDs can be arbitrarily large if several other
        # objects were loaded in the bullet scene before the objects were.
        for new_oid, odict in enumerate(state["objects"].values()):
            # Check that the new ID is within the set of supported Unity IDs.
            if new_oid in UNITY_OIDS:
                pass
            else:
                raise ValueError(f"New oid: {new_oid} is not supported.")

            new_object_states[new_oid] = odict
        state["objects"] = new_object_states

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

    def get_oids(self, timestep: int):
        oids = list(self.load_state(timestep=timestep)["objects"].keys())
        return oids

    def load_odict(self, timestep: int, oid: int):
        odict = self.load_state(timestep=timestep)["objects"][oid]
        return odict

    def get_masks_dir(self, timestep: int):
        masks_dir = os.path.join(self.masks_root_dir, f"{timestep:06}")
        return masks_dir

    def create_masks_dir(self, timestep: int):
        masks_dir = self.get_masks_dir(timestep=timestep)
        os.makedirs(masks_dir)

    def get_cam_path(self, timestep: int):
        path = os.path.join(self.cam_dir, f"{timestep:06}.json")
        return path

    def get_rgb_path(self, timestep: int):
        path = os.path.join(self.rgb_dir, f"{timestep:06}.png")
        return path

    def get_mask_path(self, timestep: int, oid: int):
        path = os.path.join(self.get_masks_dir(timestep=timestep), f"{oid:02}.npy")
        return path

    def get_mask_paths(self, timestep: int):
        paths = []
        masks_dir = self.get_masks_dir(timestep=timestep)
        for fname in sorted(os.listdir(masks_dir)):
            paths.append(os.path.join(masks_dir, fname))
        return paths

    def get_detectron_masks_path(self, timestep: int):
        path = os.path.join(self.detectron_masks_dir, f"{timestep:06}.npy")
        return path

    def save_cam(self, timestep: int, cam_dict: Dict):
        path = self.get_cam_path(timestep=timestep)
        util.save_json(path=path, data=cam_dict)

    def save_rgb(self, timestep: int, rgb: np.ndarray):
        path = self.get_rgb_path(timestep=timestep)
        imageio.imwrite(path, rgb)

    def save_mask(self, timestep: int, mask: np.ndarray, oid: int):
        path = self.get_mask_path(timestep=timestep, oid=oid)
        np.save(path, mask)

    def save_detectron_masks(self, timestep: str, masks: np.ndarray):
        path = self.get_detectron_masks_path(timestep=timestep)
        np.save(path, masks)

    def load_cam(self, timestep: int):
        path = self.get_cam_path(timestep=timestep)
        cam_dict = util.load_json(path=path)
        return cam_dict

    def load_rgb(self, timestep: int):
        path = self.get_rgb_path(timestep=timestep)
        rgb = imageio.imread(path)
        return rgb

    def load_mask(self, timestep: int, oid: int):
        path = self.get_mask_path(timestep=timestep, oid=oid)
        mask = np.load(path)
        return mask

    def load_detectron_masks(self, timestep: str):
        path = self.get_detectron_masks_path(timestep=timestep)
        masks = np.load(path)
        return masks
