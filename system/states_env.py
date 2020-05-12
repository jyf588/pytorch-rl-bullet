import os
import pprint
import imageio
import numpy as np
from typing import *

from exp.options import EXPERIMENT_OPTIONS
import exp.loader
import system.env
import scene.util
import ns_vqa_dart.bullet.seg
import ns_vqa_dart.bullet.util as util


class StatesEnv:
    def __init__(
        self,
        opt,
        experiment: str,
        set_name: str,
        scene_id: int,
        scene: List,
        task: str,
        place_dst_xy: Tuple,
    ):
        self.exp = experiment
        self.set_name = set_name
        self.scene_id = scene_id

        exp_options = EXPERIMENT_OPTIONS[self.exp][set_name]
        self.task = task
        self.stage = exp_options["stage"]

        # Get the relevant paths for the experiment / set.
        self.set_dir = exp.loader.get_exp_set_dir(exp=self.exp, set_name=set_name)
        # scenes_path = exp.loader.get_scenes_path(exp=self.exp, set_name=set_name)
        states_path = exp.loader.get_states_path(
            exp=self.exp, set_name=set_name, scene_id=scene_id
        )

        # scene = util.load_pickle(path=scenes_path)[scene_id]
        self.states = util.load_pickle(path=states_path)

        # Initialize the index and timestep.
        self.idx2timestep = {idx: ts for idx, ts in enumerate(self.states.keys())}
        self.idx = 0
        self.timestep = self.idx2timestep[self.idx]

        # Convert the scene for placing, and determine the placing destination.
        # self.place_dst_xy = None
        # if self.task == "place":
        #     scene, self.place_dst_xy, _ = scene.util.convert_scene_for_placing(
        #         opt=opt, scene=scene
        #     )

        # Get the initial observation.
        self.initial_obs = scene
        if opt.obs_noise:
            self.initial_obs = system.env.apply_obs_noise(opt=opt, obs=self.initial_obs)

        # Compute the task parameters.
        if self.task == "stack":
            self.src_idx = opt.scene_stack_src_idx
            self.dst_idx = opt.scene_stack_dst_idx
        elif self.task == "place":
            self.src_idx = opt.scene_place_src_idx
            self.dst_idx = None

    def step(self):
        self.idx += 1
        is_done = self.idx == len(self.states)
        if not is_done:
            self.timestep = self.idx2timestep[self.idx]
        success = True
        return is_done, success

    def get_state(self):
        ts = self.idx2timestep[self.idx]
        state = self.states[ts]
        return state

    def get_current_stage(self):
        return self.stage, self.idx

    def set_unity_data(self, data):
        image_dict = data[0]
        cam_dict = {
            "position": image_dict["camera_position"],
            "orientation": image_dict["camera_orientation"],
        }
        rgb = image_dict["rgb"]
        seg_img = image_dict["seg_img"]

        masks, _ = ns_vqa_dart.bullet.seg.seg_img_to_map(seg_img=seg_img)

        cam_path, rgb_path, masks_path = exp.loader.get_frame_paths(
            exp=self.exp,
            set_name=self.set_name,
            scene_id=self.scene_id,
            timestep=self.timestep,
            keys=["cam", "rgb", "masks"],
            create_dir=self.idx == 0,
        )
        util.save_json(path=cam_path, data=cam_dict)
        imageio.imwrite(rgb_path, rgb)
        np.save(masks_path, masks)

    def cleanup(self):
        pass
