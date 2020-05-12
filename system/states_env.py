import os
import copy
import pprint
import imageio
import numpy as np
from typing import *

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
        self.exp_name = experiment
        self.set_name = set_name
        self.scene_id = scene_id
        self.task = task

        self.set_loader = exp.loader.SetLoader(exp_name=experiment, set_name=set_name)
        self.stage = self.set_loader.opt["stage"]

        self.scene_loader = exp.loader.SceneLoader(
            exp_name=experiment, set_name=set_name, scene_id=scene_id
        )
        self.ts_state_list = self.scene_loader.load_scene_states()

        # Initialize the index and timestep.
        self.idx = 0
        self.timestep = self.ts_state_list[self.idx][0]

        # Get the initial observation.
        self.initial_obs = copy.deepcopy(scene)
        if opt.obs_noise:
            self.initial_obs = system.env.apply_obs_noise(opt=opt, obs=self.initial_obs)

        # Compute the task parameters.
        if self.task == "stack":
            self.src_idx = opt.scene_stack_src_idx
            self.dst_idx = opt.scene_stack_dst_idx
        elif self.task == "place":
            self.src_idx = opt.scene_place_src_idx
            self.dst_idx = None

        # Create the directories which we will save Unity data to.
        for directory in [
            self.scene_loader.cam_dir,
            self.scene_loader.rgb_dir,
            self.scene_loader.masks_root_dir,
        ]:
            os.makedirs(directory)

    def step(self):
        self.idx += 1
        is_done = self.idx == len(self.ts_state_list)
        if not is_done:
            self.timestep = self.ts_state_list[self.idx][0]
        success = True
        return is_done, success

    def get_state(self):
        _, state = self.ts_state_list[self.idx]
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

        masks, oids = ns_vqa_dart.bullet.seg.seg_img_to_map(seg_img=seg_img)

        self.scene_loader.save_cam(self.timestep, cam_dict)
        self.scene_loader.save_rgb(self.timestep, rgb)

        self.scene_loader.create_masks_dir(timestep=self.timestep)
        for mask, oidx in zip(masks, oids):
            self.scene_loader.save_mask(self.timestep, mask, oid)

    def cleanup(self):
        pass
