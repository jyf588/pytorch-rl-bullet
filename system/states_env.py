import os

import ns_vqa_dart.bullet.util as util


class StatesEnv:
    def __init__(self, opt, task: str, states_path: str, stage="place"):
        self.stage = stage
        self.states = util.load_pickle(path=states_path)
        self.idx2timestep = {idx: ts for idx, ts in enumerate(self.states.keys())}
        self.idx = 0
        self.timestep = self.idx2timestep[self.idx]

        self.place_dst_xy = None
        self.initial_obs = self.states[0]["objects"]
        if task == "stack":
            self.src_idx = opt.scene_stack_src_idx
            self.dst_idx = opt.scene_stack_dst_idx
        elif task == "place":
            self.src_idx = opt.scene_place_src_idx
            self.dst_idx = None

    def step(self):
        self.idx += 1
        self.timestep = self.idx2timestep[self.idx]

    def get_state(self):
        ts = self.idx2timestep[self.idx]
        state = self.states[ts]
        return state

    def get_current_stage(self):
        return self.stage, self.timestep
