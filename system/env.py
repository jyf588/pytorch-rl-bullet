import os
import cv2
import sys
import copy
import time
import torch
import pprint
import argparse
import numpy as np
import pybullet as p
from typing import *
from scipy.optimize import linear_sum_assignment

import exp.loader
import system.policy
from system import openrave
from system.bullet_world import BulletWorld
from system.vision_module import VisionModule
from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew

from NLP_module import NLPmod

import ns_vqa_dart.bullet.seg
import my_pybullet_envs.utils as utils
from ns_vqa_dart.bullet import dash_object, util
from ns_vqa_dart.scene_parse.detectron2.dash import DASHSegModule


class DemoEnvironment:
    def __init__(
        self,
        opt: argparse.Namespace,
        bullet_opt: argparse.Namespace,
        policy_opt: argparse.Namespace,
        shape2policy_dict: Dict,
        vision_opt: argparse.Namespace,
        exp_name: str,
        set_name: str,
        scene_id: int,
        scene: List[Dict],
        task: str,
        outputs_dir: str,
        command: Optional[str] = None,
        place_dst_xy: Optional[List[float]] = None,
        vision_models_dict: Dict = None,
    ):
        """
        Args:
            opt: Various demo configurations/options.
            trial: The trial number.
            scene: A list of object dictionaries defining the starting
                configuration of tabletop objects in the scene.
            task: The task to execute, either place or stack.
            command: The command that the robot should execute.
            place_dst_xy: The destination (x, y) location to place at. Used 
                only if `task` is `place`.
        """
        print("**********CREATING DEMO ENVIRONMENT**********")
        self.opt = opt
        self.bullet_opt = bullet_opt
        self.policy_opt = policy_opt
        self.shape2policy_dict = shape2policy_dict
        self.vision_opt = vision_opt
        self.exp_name = exp_name
        self.set_name = set_name
        self.scene_id = scene_id
        self.scene = scene
        self.task = task
        self.outputs_dir = outputs_dir
        self.command = command
        self.place_dst_xy = place_dst_xy
        self.vision_models_dict = vision_models_dict

        np.random.seed(self.opt.seed)

        self.timestep = 0
        self.planning_complete = False
        self.initial_obs = None
        self.obs = None
        self.w = None
        self.src_idx, self.dst_idx = None, None

        (
            self.stage2ts_bounds,
            self.stage2steps,
            self.n_total_steps,
        ) = self.compute_stages()
        self.stage, self.stage_ts = self.get_current_stage()

    def cleanup(self):
        # p.disconnect()
        # del self
        p.resetSimulation()

    def is_done(self) -> bool:
        """Determines whether we have finished the sequence.

        Returns:
            done: Whether the sequence has finished.
        """
        # If we executed all steps, we are done.
        return self.timestep == self.n_total_steps

    def check_success(self) -> bool:
        """Checks whether the current state configuration is considered successful."""
        if self.task == "place":
            src_idx = self.opt.scene_place_src_idx
        elif self.task == "stack":
            src_idx = self.opt.scene_stack_src_idx
        src_odict = list(self.get_state()["objects"].values())[src_idx]
        x, y, z = src_odict["position"]
        dst_x, dst_y, dst_z = self.dst_xyz
        z_check = z - dst_z > 0.05
        xy_check = (x - dst_x) ** 2 + (y - dst_y) ** 2 < 0.1 ** 2
        is_success = z_check and xy_check
        return is_success

    def compute_stages(self):
        if self.opt.enable_reaching:
            reach_start = 0
            reach_end = reach_start + self.policy_opt.n_plan_steps
            grasp_start = reach_end
        else:
            grasp_start = 0

        n_grasp = self.policy_opt.grasp_control_steps
        n_place = self.policy_opt.place_control_steps

        if not self.opt.use_control_skip:
            n_grasp *= self.opt.control_skip
            n_place *= self.opt.control_skip

        grasp_end = grasp_start + n_grasp
        transport_start = grasp_end
        transport_end = transport_start + self.policy_opt.n_plan_steps
        place_start = transport_end
        place_end = place_start + n_place

        stage2ts_bounds = {
            "grasp": (grasp_start, grasp_end),
            "transport": (transport_start, transport_end),
            "place": (place_start, place_end),
        }

        if self.opt.enable_reaching:
            stage2ts_bounds["reach"] = (reach_start, reach_end)
        if self.opt.enable_retract:
            retract_start = place_end
            retract_end = retract_start + self.policy_opt.n_plan_steps
            stage2ts_bounds["retract"] = (retract_start, retract_end)

        stage2steps = {}
        n_total_steps = 0
        for stage, (start_ts, end_ts) in stage2ts_bounds.items():
            n_stage_steps = end_ts - start_ts
            n_total_steps += n_stage_steps
            stage2steps[stage] = n_stage_steps
        return stage2ts_bounds, stage2steps, n_total_steps

    def get_current_stage(self) -> Tuple[str, int]:
        """Retrieves the current stage of the demo.

        Returns:
            stage: The stage of the demo.
            stage_s: The timestep of the current stage. 
        """
        if not self.planning_complete:
            return "plan", 0

        current_stage = None
        for stage, ts_bounds in self.stage2ts_bounds.items():
            start, end = ts_bounds
            if start <= self.timestep < end:
                stage_ts = self.timestep - start
                current_stage = stage
                break

        if current_stage is None:
            raise ValueError(f"No stage found for current timestep: {self.timestep}")
        return current_stage, stage_ts

    def stage_progress(self):
        completion_frac = self.stage_ts / self.stage2steps[self.stage]
        return completion_frac

    def step(self):
        """Policy performs a single action based on the current state.
        
        Returns:
            is_done: Whether we are done with the demo.
        """
        # print(f"Step info:")
        # print(f"\tTimestep: {self.timestep}")
        # print(f"\tStage: {stage}")
        # print(f"\tStage timestep: {stage_ts}")

        # By default we assume that the stepping succeeds.
        step_succeeded = True
        if self.stage == "plan":
            self.plan()
        elif self.stage == "reach":
            step_succeeded = self.execute_plan(stage=self.stage, stage_ts=self.stage_ts)
        elif self.stage == "grasp":
            self.grasp(stage_ts=self.stage_ts)
        elif self.stage == "transport":
            step_succeeded = self.execute_plan(stage=self.stage, stage_ts=self.stage_ts)
        elif self.stage == "place":
            self.place(stage_ts=self.stage_ts)
        elif self.stage == "retract":
            step_succeeded = self.execute_plan(
                stage=self.stage,
                stage_ts=self.stage_ts,
                restore_fingers=self.policy_opt.restore_fingers,
            )
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

        # We don't step in the planning stage.
        if self.stage != "plan":
            self.timestep += 1

        done = self.is_done()

        # Update the stage.
        if not done:
            self.stage, self.stage_ts = self.get_current_stage()

        # Compute whether we have finished the entire sequence.
        return done, step_succeeded

    def get_state(self):
        """Retrieves the current state of the bullet world.

        Returns:
            state: The state of the bullet environment, in the format:
                {
                    "objects": {
                        "<oid>": {
                            "shape": shape,
                            "color": color,
                            "radius": radius,
                            "height": height,
                            "position": [x, y, z],
                            "orientation": [x, y, z, w],
                        },
                        ...
                    },
                    "robot": {
                        "<joint_name>": <joint_angle>,
                        ...
                    }
                }
            Note that if no bullet world is created yet, oid is simply assigned
            using zero-indexed assignment.
        """
        # If there is no bullet world, we simply return the input scene.
        if self.w is None:
            state = {"objects": {idx: odict for idx, odict in enumerate(self.scene)}}
        else:
            state = self.w.get_state()

        # Additionally, compute the up vector from the orientation.
        for idx in state["objects"].keys():
            orn = state["objects"][idx]["orientation"]
            state["objects"][idx]["up_vector"] = util.orientation_to_up(orn)
        return state

    def plan(self):
        # First, get the current observation which we will store as the initial
        # observation for planning reach/transport and for grasping.
        self.initial_obs = self.get_observation()
        self.obs = copy.deepcopy(self.initial_obs)

        # Determine the source / target pbject(s) and destination position.
        task_params = self.compute_task_params(observation=self.initial_obs)
        (
            self.src_idx,
            self.dst_idx,
            self.dst_xyz,
            plan_xyz,
            self.policy_dst_xyz,
            self.is_sphere,
        ) = task_params

        # Extract policy-related variables according to the shape.
        policy_dict = self.shape2policy_dict[
            "sphere" if self.is_sphere else "universal"
        ]
        self.grasp_policy = policy_dict["grasp_policy"]
        self.place_policy = policy_dict["place_policy"]
        self.hidden_states = copy.deepcopy(policy_dict["hidden_states"])
        self.masks = copy.deepcopy(policy_dict["masks"])
        self.o_quat_pf_ave = policy_dict["o_quat_pf_ave"]
        self.p_pos_of_ave = policy_dict["p_pos_of_ave"]
        self.p_quat_of_ave = policy_dict["p_quat_of_ave"]

        # obtained from initial_obs
        self.src_xy = self.initial_obs[self.src_idx]["position"][:2]

        # Compute the goal arm poses for reaching and transport.
        self.q_reach_dst, self.q_transport_dst = self.compute_qs(
            src_xy=self.src_xy, dst_xyz=plan_xyz,
        )

        # Create the bullet world now that we've finished our imaginary
        # sessions.
        self.w = BulletWorld(
            opt=self.bullet_opt,
            policy_opt=self.policy_opt,
            p=p,
            scene=self.scene,
            visualize=self.opt.render_bullet,
            use_control_skip=self.opt.use_control_skip,
        )

        # If reaching is disabled, set the robot arm directly to the dstination
        # of reaching.
        if self.opt.enable_reaching:
            q_zero = [0.0] * len(self.q_reach_dst)
            self.w.robot_env.robot.reset_with_certain_arm_q(q_zero)
        else:
            self.w.robot_env.robot.reset_with_certain_arm_q(self.q_reach_dst)

        # Flag planning as complete.
        self.planning_complete = True

    def compute_task_params(self, observation: Dict):
        """Parses a language command in the context of an observation of a
        scene and computes the source and target objects and location for a 
        pick-and-place task.

        Args:
            command: The command to execute.
            observation: A list of object dictionaries defining the tabletop 
                objects in a scene.
        
        Returns:
            src_idx: The index of the source object.
            dst_idx: The index of the destination object.
            dst_xyz: The (x, y, z) location to end transport at / start placing.
        """
        dst_idx = None

        # In the absence of language, we determine the task parameters by assuming that
        # we know the attributes of the objects that are to be manipulated.
        if self.command is None:
            # We assume that the first object in the ground truth scene is the
            # source object.
            if self.opt.obs_mode == "gt":
                if self.task == "place":
                    src_idx = self.opt.scene_place_src_idx
                elif self.task == "stack":
                    src_idx = self.opt.scene_stack_src_idx
                    dst_idx = self.opt.scene_stack_dst_idx
            elif self.opt.obs_mode == "vision":
                # Compute the mapping from GT indexes to predicted indexes.
                gt2pred_idxs = match_objects(
                    src_odicts=self.scene, dst_odicts=observation
                )
                if self.task == "place":
                    src_idx = gt2pred_idxs[self.opt.scene_place_src_idx]
                elif self.task == "stack":
                    src_idx = gt2pred_idxs[self.opt.scene_stack_src_idx]
                    dst_idx = gt2pred_idxs[self.opt.scene_stack_dst_idx]
            else:
                raise ValueError(f"Invalid observation mode: {self.opt.obs_mode}.")
            if self.task == "place":
                assert self.place_dst_xy is not None
                dst_x, dst_y = self.place_dst_xy
            elif self.task == "stack":
                dst_x, dst_y, _ = observation[dst_idx]["position"]
        else:
            if self.task == "place":
                raise NotImplementedError
            elif self.task == "stack":
                # Zero-pad the scene's position with fourth dimension because that's
                # what the language module expects.
                language_input = copy.deepcopy(observation)
                for idx, odict in enumerate(language_input):
                    language_input[idx]["position"] = odict["position"] + [0.0]

                # Feed through the language module.
                src_idx, (dst_x, dst_y), dst_idx = NLPmod(
                    sentence=self.command, vision_output=language_input
                )

        # We currently only support placing spheres, not stacking them.
        if self.task == "stack":
            for oidx in [src_idx, dst_idx]:
                if observation[oidx]["shape"] == "sphere":
                    raise NotImplementedError

        is_sphere = observation[src_idx]["shape"] == "sphere"

        # Compute the destination z based on whether there is a destination
        # object that we are placing on top of (stacking).
        if dst_idx is None:
            z = 0.0
            policy_z = z
        else:
            z = observation[dst_idx]["height"]
            policy_z = z
            if not self.policy_opt.use_height:
                policy_z = utils.H_MAX

        dst_xyz = [dst_x, dst_y, z]
        plan_xyz = [dst_x, dst_y, z + utils.PLACE_START_CLEARANCE]
        policy_dst_xyz = [dst_x, dst_y, policy_z]
        return src_idx, dst_idx, dst_xyz, plan_xyz, policy_dst_xyz, is_sphere

    def compute_qs(self, src_xy: List[float], dst_xyz: List[float]) -> Tuple:
        """Computes the arm joint angles that will be used to determine the
        arm poses for:
            (1) After reaching / before grasping, and
            (2) After transport / before placing
        
        Args:
            src_xy: The (x, y) position that the robot should reach to.
            dst_xyz: The (x, y, z) position that the robot should finish
                transporting to.
        
        Returns:
            q_reach_dst: The arm joint angles that reaching should end at.
            q_transport_dst: The arm joint angles that transport should end at.
        """
        # Compute the destination arm pose for reaching.
        src_xyz = [src_xy[0], src_xy[1], 0.0]
        table_id = utils.create_table(self.bullet_opt.floor_mu)
        robot = InmoovShadowNew(
            init_noise=False, timestep=utils.TS, np_random=np.random,
        )
        q_reach_dst = utils.get_n_optimal_init_arm_qs(
            robot,
            utils.PALM_POS_OF_INIT,
            p.getQuaternionFromEuler(utils.PALM_EULER_OF_INIT),
            src_xyz,
            table_id,
            wrist_gain=3.0,
        )[0]
        p.resetSimulation()

        # Compute the destination arm pose for transport.
        table_id = utils.create_table(self.bullet_opt.floor_mu)
        robot = InmoovShadowNew(
            init_noise=False, timestep=utils.TS, np_random=np.random,
        )
        if self.is_sphere:
            _, desired_oquat = p.multiplyTransforms(
                [0, 0, 0],
                p.getQuaternionFromEuler(utils.PALM_EULER_OF_INIT),
                [0, 0, 0],
                self.o_quat_pf_ave,
            )
        else:
            desired_oquat = [0.0, 0.0, 0.0, 1.0]
        q_transport_dst = utils.get_n_optimal_init_arm_qs(
            robot,
            self.p_pos_of_ave,
            self.p_quat_of_ave,
            dst_xyz,
            table_id,
            desired_obj_quat=desired_oquat,
        )[0]
        p.resetSimulation()
        return q_reach_dst, q_transport_dst

    def compute_trajectory(self, stage: str) -> np.ndarray:
        q_start, q_end = None, None
        expected_src_base_z_post_placing = None
        odicts = self.initial_obs

        if stage == "reach":
            q_end = self.q_reach_dst
        elif stage == "transport":
            q_start = self.w.get_robot_q()[0]
            q_end = self.q_transport_dst
        elif stage == "retract":
            q_start = self.w.get_robot_q()[0]

            # Instead of using the initial observation, we use the latest
            # observation.
            if self.opt.obs_mode == "gt":
                odicts = self.get_observation()
            elif self.opt.obs_mode == "vision":
                odicts = self.last_pred_obs
            else:
                raise ValueError(f"Invalid observation mode: {self.opt.obs_mode}.")
            if self.task == "place":
                expected_src_base_z_post_placing = 0.0
            elif self.task == "stack":
                expected_src_base_z_post_placing = self.initial_obs[self.dst_idx][
                    "height"
                ]
        else:
            raise ValueError(f"Invalid stage: {stage}.")
        trajectory = openrave.compute_trajectory(
            odicts=odicts,
            target_idx=self.src_idx,
            q_start=q_start,
            q_end=q_end,
            stage=stage,
            src_base_z_post_placing=expected_src_base_z_post_placing,
            container_dir=self.opt.container_dir,
        )
        return trajectory

    def grasp(self, stage_ts: int):
        obs = system.policy.wrap_obs(
            self.get_grasp_observation(), is_cuda=self.opt.is_cuda
        )
        with torch.no_grad():
            _, action, _, self.hidden_states = self.grasp_policy.act(
                obs, self.hidden_states, self.masks, deterministic=self.bullet_opt.det
            )
        self.w.step_robot(
            action=system.policy.unwrap_action(action=action, is_cuda=self.opt.is_cuda),
            timestep=self.timestep,
        )
        self.masks.fill_(1.0)

    def place(self, stage_ts: int):
        if stage_ts == 0:
            self.w.robot_env.change_control_skip_scaling(
                c_skip=self.policy_opt.control_skip
            )

        # Get the current observation.
        obs = system.policy.wrap_obs(
            self.get_place_observation(), is_cuda=self.opt.is_cuda
        )
        with torch.no_grad():
            _, action, _, self.hidden_states = self.place_policy.act(
                obs, self.hidden_states, self.masks, deterministic=self.bullet_opt.det
            )

        self.w.step_robot(
            action=system.policy.unwrap_action(action=action, is_cuda=self.opt.is_cuda),
            timestep=self.timestep,
        )

    def execute_plan(
        self, stage: str, stage_ts: int, restore_fingers: Optional[bool] = False,
    ) -> bool:
        """
        Returns:
            success: Whether executing the current step of the plan succeeded.
        """
        # Get initial robot pose.
        if stage_ts == 0:
            self.trajectory = self.compute_trajectory(stage=stage)
            if len(self.trajectory) == 0:
                return False
            self.w.robot_env.robot.tar_arm_q = self.trajectory[-1]
            self.init_tar_fin_q = self.w.robot_env.robot.tar_fin_q
            self.last_tar_arm_q, self.init_fin_q = self.w.get_robot_q()

        if stage_ts > len(self.trajectory) - 1:
            tar_arm_q = self.trajectory[-1]
        else:
            tar_arm_q = self.trajectory[stage_ts]

        tar_arm_vel = (tar_arm_q - self.last_tar_arm_q) / self.bullet_opt.ts
        max_force = self.w.robot_env.robot.maxForce

        p.setJointMotorControlArray(
            bodyIndex=self.w.robot_env.robot.arm_id,
            jointIndices=self.w.robot_env.robot.arm_dofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(tar_arm_q),
            targetVelocities=list(tar_arm_vel),
            forces=[max_force * 5] * len(self.w.robot_env.robot.arm_dofs),
        )

        if restore_fingers and stage_ts >= len(self.trajectory) * 0.1:
            blending = np.clip(
                (stage_ts - len(self.trajectory) * 0.1) / (len(self.trajectory) * 0.6),
                0.0,
                1.0,
            )
            cur_fin_q = self.w.robot_env.robot.get_q_dq(
                self.w.robot_env.robot.fin_actdofs
            )[0]
            tar_fin_q = self.w.robot_env.robot.init_fin_q * blending + cur_fin_q * (
                1 - blending
            )
        else:
            # try to keep fin q close to init_fin_q (keep finger pose)
            # add at most offset 0.05 in init_tar_fin_q direction so that grasp is tight
            tar_fin_q = np.clip(
                self.init_tar_fin_q, self.init_fin_q - 0.05, self.init_fin_q + 0.05,
            )

        # clip to joint limit
        tar_fin_q = np.clip(
            tar_fin_q,
            self.w.robot_env.robot.ll[self.w.robot_env.robot.fin_actdofs],
            self.w.robot_env.robot.ul[self.w.robot_env.robot.fin_actdofs],
        )

        p.setJointMotorControlArray(
            bodyIndex=self.w.robot_env.robot.arm_id,
            jointIndices=self.w.robot_env.robot.fin_actdofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(tar_fin_q),
            forces=[max_force] * len(self.w.robot_env.robot.fin_actdofs),
        )
        p.setJointMotorControlArray(
            bodyIndex=self.w.robot_env.robot.arm_id,
            jointIndices=self.w.robot_env.robot.fin_zerodofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.0] * len(self.w.robot_env.robot.fin_zerodofs),
            forces=[max_force / 4.0] * len(self.w.robot_env.robot.fin_zerodofs),
        )

        self.last_tar_arm_q = tar_arm_q
        self.w.step(timestep=self.timestep)
        return True

    def get_grasp_observation(self) -> torch.Tensor:
        x, y = self.src_xy

        odict = self.initial_obs[self.src_idx]
        half_height = odict["height"] / 2
        is_box = odict["shape"] == "box"

        if self.policy_opt.use_height:
            obs = self.w.robot_env.get_robot_contact_txtytz_halfh_shape_obs_no_dup(
                x, y, 0.0, half_height, is_box
            )
        else:
            obs = self.w.robot_env.get_robot_contact_txty_shape_obs_no_dup(x, y, is_box)
        return obs

    def get_place_observation(self):
        # Update the observation only every `vision_delay` steps.
        if self.timestep % self.policy_opt.vision_delay == 0:
            self.obs = self.get_observation()

        # Compute the observation vector from object poses and placing position.
        tx, ty, tz = self.policy_dst_xyz  # this should be dst xy rather than src xy

        tdict = self.obs[self.src_idx]
        t_pos = tdict["position"]
        t_up = tdict["up_vector"]
        is_box = tdict["shape"] == "box"  # should not matter if use init_obs or obs

        if self.task == "stack":
            bdict = self.obs[self.dst_idx]
            b_pos = bdict["position"]
            b_up = bdict["up_vector"]
        elif self.task == "place":
            b_pos = [0.0, 0.0, 0.0]
            b_up = [0.0, 0.0, 1.0]

        if self.policy_opt.use_height:
            p_obs = self.w.robot_env.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
                tx=tx,
                ty=ty,
                tz=tz,
                half_h=tdict["height"] / 2,
                shape=is_box,
                t_pos=t_pos,
                t_up=t_up,
                b_pos=b_pos,
                b_up=b_up,
            )
        else:
            p_obs = self.w.robot_env.get_robot_contact_txty_shape_2obj6dUp_obs_nodup_from_up(
                tx=tx,
                ty=ty,
                shape=is_box,
                t_pos=t_pos,
                t_up=t_up,
                b_pos=b_pos,
                b_up=b_up,
            )
        return p_obs

    def get_observation(self):
        """Gets an observation from the current state.
        
        Args:
            observation_mode: The mode of observation, e.g. using ground truth 
                or vision.

        Returns:
            observation: A list of dictionaries storing object observations 
                for the current world state.
        """
        if self.opt.obs_mode == "gt":
            # The ground truth observation is simply the same as the true
            # state of the world.
            obs = list(self.get_state()["objects"].values())
            self.obs_to_render = obs
        elif self.opt.obs_mode == "vision":
            obs = self.get_vision_observation()
            self.obs_to_render = obs

            if self.vision_opt.use_gt_obs:
                obs = list(self.get_state()["objects"].values())
        else:
            raise ValueError("Unsupported observation mode: {self.opt.obs_mode}")
        if self.opt.obs_noise:
            obs = apply_obs_noise(opt=self.opt, obs=obs)
        return copy.deepcopy(obs)

    def get_vision_observation(self):
        """Computes the observation of the bullet world using the vision 
        module.
        
        Returns:
            obs: The vision observation.
        """
        # Select the vision module based on the current stage.
        if self.vision_opt.separate_vision_modules:
            if self.stage == "plan":
                vision_module = self.vision_models_dict["plan"]
            elif self.stage == "place":
                vision_module = self.vision_models_dict[self.task]
            else:
                raise ValueError(f"No vision module for stage: {self.stage}.")
        else:
            vision_module = self.vision_models_dict["combined"]

        # Retrieves the image, camera pose, and object segmentation masks.
        rgb, oid2mask, masks, cam_position, cam_orientation = self.get_images()

        # Either predict segmentations or use ground truth.
        if self.vision_opt.use_segmentation_module:
            masks = self.vision_models_dict["seg"].eval_example(bgr=rgb[:, :, ::-1])

        # Predict attributes for all the segmentations.
        pred, inputs_img = vision_module.predict(rgb=rgb, masks=masks)

        pred_odicts = []
        for y in pred:
            # Convert vectorized predictions to dictionary form. The
            # predicted pose is also transformed using camera information.
            odict = dash_object.y_vec_to_dict(
                y=list(y),
                coordinate_frame=self.vision_opt.coordinate_frame,
                cam_position=cam_position,
                cam_orientation=cam_orientation,
            )
            pred_odicts.append(odict)

        # We track predicted objects throughout time by matching the attributes
        # predicted at the current timestep with the object attributes from the
        # initial observation.
        if self.initial_obs is None:
            pred_obs = pred_odicts
            s2d_idxs = None
        else:
            pred_obs, s2d_idxs = self.match_predictions_with_initial_obs(
                pred_odicts=pred_odicts
            )

        # Store the computed observation for future computation.
        self.last_pred_obs = copy.deepcopy(pred_obs)

        # Save the predicted and ground truth object dictionaries.
        if self.vision_opt.save_predictions:
            frame_fname = f"{self.task}_{self.scene_id}_{self.timestep:06}"
            path = os.path.join(self.outputs_dir, f"{frame_fname}.p",)
            util.save_pickle(
                path=path,
                data={
                    "task": self.task,
                    "scene_id": self.scene_id,
                    "timestep": self.timestep,
                    "stage": self.stage,
                    "stage_ts": self.stage_ts,
                    "gt": {"oid2odict": self.get_state()["objects"],},
                    "pred": {"odicts": pred_obs},
                    "s2d_idxs": s2d_idxs,
                    "src_idx": self.src_idx,
                    "dst_idx": self.dst_idx,
                    "rgb": rgb,
                    "masks": masks,
                    "vision_inputs": inputs_img,
                },
            )
        return pred_obs

    def match_predictions_with_initial_obs(self, pred_odicts: List[Dict]) -> List:
        """Matches predictions with `self.initial_obs`.

        Args:
            pred_odicts: A list of predicted object dictionaries to match with
                `self.initial_obs`.

        Returns:
            pred_obs: A list of predicted object observations that is same 
                length as `self.initial_obs`. Each element in `pred_obs` 
                corresponds to the object predicted in `self.initial_obs`, and
                contains either:
                (1) The assigned object prediction from the current timestep based
                    on matching with the corresponding object from 
                    `self.initial_obs`, or
                (2) The most recent matched prediction.
        """
        # Verify that `self.initial_obs` is defined.
        assert self.initial_obs is not None

        # Compute assignments.
        s2d_idxs = match_objects(src_odicts=self.initial_obs, dst_odicts=pred_odicts)

        # If `self.initial_obs` has more elements than `pred_odicts`, then
        # there will be unassigned elements in `self.initial_obs`. We assign
        # unassigned elements with the last observation. To do this, we
        # initialize the `pred_obs` which will be returned with
        # `self.last_pred_obs`. Since we only override assigned elements, each
        # unassigned element will by default hold the last prediction.
        pred_obs = copy.deepcopy(self.last_pred_obs)

        # Override the pose predictions for the objects being manipulated.
        if self.task == "stack":
            place_object_idxs = [self.src_idx, self.dst_idx]
        elif self.task == "place":
            place_object_idxs = [self.src_idx]
        else:
            raise ValueError(f"Invalid task: {self.task}")
        for s in place_object_idxs:
            # We only update the place object if we found a corresponding
            # match in the predictions.
            if s in s2d_idxs:
                d = s2d_idxs[s]
                dst_odict = pred_odicts[d]
                for attr in ["position", "up_vector"]:
                    pred_obs[s][attr] = dst_odict[attr]
        return pred_obs, s2d_idxs

    def get_images(self) -> Tuple:
        """Retrieves the images that are input to the vision module.

        Returns:
            rgb: The RGB image of the object.
            masks: A numpy array of shape (N, H, W) of instance masks.
            camera_position: The position of the camera used to capture the
                image.
            camera_orientation: The orientation of the camera used to capture 
                the image.
        """
        if self.vision_opt.renderer == "unity":
            # Unity should have already called set_unity_data before this.
            rgb = self.unity_data[0]["rgb"]
            seg_img = self.unity_data[0]["seg_img"]

            camera_position = self.unity_data[0]["camera_position"]
            camera_orientation = self.unity_data[0]["camera_orientation"]

            # Optionally we can visualize unity images using OpenCV.
            if self.opt.visualize_unity:
                cv2.imshow("rgb", rgb[:, :, ::-1])
                cv2.imshow("seg", seg_img[:, :, ::-1])
                cv2.waitKey(5)
        else:
            raise ValueError(f"Invalid renderer: {self.vision_opt.renderer}")

        masks, oids = ns_vqa_dart.bullet.seg.seg_img_to_map(seg_img)
        oid2mask = {oid: mask for mask, oid in zip(masks, oids)}
        return rgb, oid2mask, masks, camera_position, camera_orientation

    def set_unity_data(self, data: Dict):
        """Sets the data received from Unity.

        Args:
            data: Unity data, in the format
                {
                    <oid>:{
                        "camera_position": <camera_position>,
                        "camera_orientation": <camera_orientation>,
                        "image": <image>,
                    },
                    ...
                }        
        """
        self.unity_data = data


def apply_obs_noise(opt, obs):
    noisy_obs = copy.deepcopy(obs)
    for idx in range(len(obs)):
        noisy_odict = noisy_obs[idx]
        noisy_odict["position"] = utils.perturb(
            np.random, noisy_odict["position"], r=opt.position_noise
        )
        noisy_odict["up_vector"] = utils.perturb(
            np.random, noisy_odict["up_vector"], r=opt.upv_noise
        )
        noisy_odict["height"] = utils.perturb_scalar(
            np.random, noisy_odict["height"], r=opt.height_noise
        )
        noisy_obs[idx] = noisy_odict
    return noisy_obs


def match_objects(src_odicts: List[Dict], dst_odicts: List[Dict]) -> Tuple:
    """
    Args:
        src_odicts: The source object dictionaries to match from.
        dst_odicts: The destination object dictionaries to match to.

    Returns:
        src2dst_idxs: A mapping from src to dst index assignments.
    """
    # Construct the cost matrix.
    cost = np.zeros((len(src_odicts), len(dst_odicts)))
    for src_idx, src_y in enumerate(src_odicts):
        for dst_idx, dst_y in enumerate(dst_odicts):
            cell = 0
            for attr in ["color", "shape"]:
                if src_y[attr] != dst_y[attr]:
                    cell += 1
            cost[src_idx][dst_idx] = cell

    # Compute assignments.
    src_idxs, dst_idxs = linear_sum_assignment(cost)
    src2dst_idxs = {s: d for s, d in zip(src_idxs, dst_idxs)}
    return src2dst_idxs
