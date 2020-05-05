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

import system.policy
from system import openrave
from system.bullet_world import BulletWorld
from system.vision_module import VisionModule
from system.seg_module import SegmentationModule
from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew
from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import (
    ImaginaryArmObjSession,
)
from my_pybullet_envs.inmoov_shadow_place_env_v9 import (
    InmoovShadowHandPlaceEnvV9,
)
from NLP_module import NLPmod

import ns_vqa_dart.bullet.seg
import my_pybullet_envs.utils as utils
from ns_vqa_dart.bullet.metrics import Metrics
from ns_vqa_dart.bullet import dash_object, gen_dataset, util


class DemoEnvironment:
    def __init__(
        self,
        opt: argparse.Namespace,
        trial: int,
        scene: List[Dict],
        task: str,
        command: str,
        observation_mode: str,
        visualize_bullet: bool,
        visualize_unity: bool,
        renderer: Optional[str] = None,
        place_dst_xy: Optional[List[float]] = None,
    ):
        """
        Args:
            opt: Various demo configurations/options.
            trial: The trial number.
            scene: A list of object dictionaries defining the starting
                configuration of tabletop objects in the scene, in the format:
                [
                    {
                        "shape": <shape>,
                        "color": <color>,
                        "position": <position>,
                        "orientation": <orientation>,
                        "radius": <radius>,
                        "height": <height>,
                    },
                    ...
                ].
                For placing, the first object in the scene is assumed to be the
                source object.
            task: The task to execute, either pick or place.
            command: The command that the robot should execute.
            observation_mode: Source of observation, which can either be from
                ground truth (`gt`) or vision (`vision`).
            visualize_bullet: Whether to visualize the demo in pybullet in 
                real-time in an OpenGL window.
            visualize_unity: Whether to visualize the unity image stream sent
                from unity to the current class in a live OpenCV window.
            renderer: The renderer to use to generate images, if 
                `observation_mode` is `vision`.
            place_dst_xy: The destination (x, y) location to place at. Used 
                only if `task` is `place`.
        """
        self.opt = opt
        self.trial = trial
        self.scene = scene
        self.task = task
        self.command = command
        self.observation_mode = observation_mode
        self.visualize_bullet = visualize_bullet
        self.visualize_unity = visualize_unity
        self.renderer = renderer
        self.place_dst_xy = place_dst_xy

        print("**********DEMO ENVIRONMENT**********")

        self.stage2ts_bounds, self.n_total_steps = self.compute_stages()

        # Whether we've finished planning.
        self.planning_complete = False
        self.initial_obs = None
        self.obs = None
        self.world = None

        # Initialize the vision module if we are using vision for our
        # observations.
        if self.observation_mode == "vision":
            if task == "stack":
                place_checkpoint_path = self.opt.stacking_checkpoint_path
            elif task == "place":
                place_checkpoint_path = self.opt.placing_checkpoint_path
            else:
                raise ValueError(f"Invalid task: {task}")
            self.planning_vision_module = VisionModule(
                load_checkpoint_path=self.opt.planning_checkpoint_path,
                debug_dir=self.opt.debug_dir,
            )
            self.placing_vision_module = VisionModule(
                load_checkpoint_path=place_checkpoint_path,
                debug_dir=self.opt.debug_dir,
            )

        # Initialize the segmentation module if requested.
        if self.opt.use_segmentation_module:
            self.segmentation_module = SegmentationModule(
                load_checkpoint_path=self.opt.segmentation_checkpoint_path,
                debug_dir=self.opt.debug_dir,
            )

        if visualize_bullet:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.imaginary_sess = ImaginaryArmObjSession()
        self.a = InmoovShadowHandPlaceEnvV9(
            renders=False, grasp_pi_name=self.opt.grasp_pi
        )

        self.timestep = 0

    def compute_stages(self):
        if self.opt.disable_reaching:
            reach_start = -1
            reach_end = -1
            grasp_start = 0
        else:
            reach_start = 0
            reach_end = reach_start + self.opt.n_plan_steps
            grasp_start = reach_end
        grasp_end = grasp_start + self.opt.n_grasp_steps
        transport_start = grasp_end
        transport_end = transport_start + self.opt.n_plan_steps
        place_start = transport_end
        place_end = place_start + self.opt.n_place_steps
        release_start = place_end
        release_end = release_start + self.opt.n_release_steps
        retract_start = release_end
        retract_end = retract_start + self.opt.n_plan_steps

        stage2ts_bounds = {
            "reach": (reach_start, reach_end),
            "grasp": (grasp_start, grasp_end),
            "transport": (transport_start, transport_end),
            "place": (place_start, place_end),
            "release": (release_start, release_end),
            "retract": (retract_start, retract_end),
        }

        n_total_steps = 0
        for start_ts, end_ts in stage2ts_bounds.values():
            n_total_steps += end_ts - start_ts
        return stage2ts_bounds, n_total_steps

    def plan(self):
        # First, get the current observation which we will store as the initial
        # observation for planning reach/transport and for grasping.
        self.initial_obs = self.get_observation(
            observation_mode=self.observation_mode, renderer=self.renderer
        )
        self.obs = copy.deepcopy(self.initial_obs)

        # Use language module to determine the source / target objects and
        # positions.
        self.src_idx, self.dst_idx, dst_xyz = self.parse_command(
            command=self.command, observation=self.initial_obs
        )

        # Compute the goal arm poses for reaching and transport.
        self.q_reach_dst, self.q_transport_dst = self.compute_qs(
            src_xy=self.initial_obs[self.src_idx]["position"][:2],
            dst_xyz=dst_xyz,
        )

        # Create the bullet world now that we've finished our imaginary
        # sessions.
        self.world = BulletWorld(
            opt=self.opt,
            p=p,
            scene=self.scene,
            visualize=self.visualize_bullet,
        )

        # If reaching is disabled, set the robot arm directly to the dstination
        # of reaching.
        if self.opt.disable_reaching:
            self.world.robot_env.robot.reset_with_certain_arm_q(
                self.q_reach_dst
            )

        # Flag planning as complete.
        self.planning_complete = True

    def parse_command(self, command: str, observation: Dict):
        """Parses a language command in the context of an observation of a
        scene and computes the source and target objects and location for a 
        pick-and-place task.

        Args:
            command: The command to execute.
            observation: A list of object dictionaries defining the tabletop 
                objects in a scene, in the format:
                [
                    {
                        "shape": shape,
                        "color": color,
                        "radius": radius,
                        "height": height,
                        "position": [x, y, z],
                        "orientation": [x, y, z, w],
                    },
                    ...
                ]
        
        Returns:
            src_idx: The index of the source object.
            dst_idx: The index of the destination object.
            dst_xyz: The (x, y, z) location to end transport at / start placing.
        """
        # Language is only supported for stacking for now.
        if self.task == "stack":
            # Zero-pad the scene's position with fourth dimension because that's
            # what the language module expects.
            language_input = copy.deepcopy(observation)
            for idx, odict in enumerate(language_input):
                language_input[idx]["position"] = odict["position"] + [0.0]

            # Feed through the language module.
            src_idx, (dst_x, dst_y), dst_idx = NLPmod(
                sentence=command, vision_output=language_input
            )
        elif self.task == "place":
            # We assume that the first object in the scene is the source
            # object.
            src_idx = 0
            dst_idx = None
            assert self.place_dst_xy is not None
            dst_x, dst_y = self.place_dst_xy
        else:
            raise ValueError(f"Invalid task: {self.task}.")

        # Compute the destination z based on whether there is a destination
        # object that we are placing on top of (stacking).
        if dst_idx is None:
            z = 0.0
        else:
            if self.opt.use_height:
                z = observation[dst_idx]["height"]
            else:
                z = utils.H_MAX

        dst_xyz = [dst_x, dst_y, z + utils.PLACE_START_CLEARANCE]
        return src_idx, dst_idx, dst_xyz

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
        if self.world is None:
            state = {
                "objects": {idx: odict for idx, odict in enumerate(self.scene)}
            }
        else:
            state = self.world.get_state()

        # Additionally, compute the up vector from the orientation.
        for idx in state["objects"].keys():
            orn = state["objects"][idx]["orientation"]
            state["objects"][idx]["up_vector"] = util.orientation_to_up(orn)
        return state

    def step(self):
        """Policy performs a single action based on the current state.
        
        Returns:
            is_done: Whether we are done with the demo.
        """
        stage, stage_ts = self.get_current_stage()

        # print(f"Step info:")
        # print(f"\tTimestep: {self.timestep}")
        # print(f"\tStage: {stage}")
        # print(f"\tStage timestep: {stage_ts}")

        if stage == "plan":
            self.plan()
        else:
            if stage == "reach":
                success = self.reach(stage_ts=stage_ts)
                if not success:
                    print(f"Reaching failed. Terminating early.")
                    return True
            elif stage == "grasp":
                self.grasp(stage_ts=stage_ts)
            elif stage == "transport":
                success = self.transport(stage_ts=stage_ts)
                if not success:
                    return True  # Let user know we're done.
            elif stage == "place":
                self.place(stage_ts=stage_ts)
            elif stage == "release":
                self.release()
            elif stage == "retract":
                self.retract(stage_ts=stage_ts)
            else:
                raise ValueError(f"Invalid stage: {stage}")
            self.timestep += 1

        # Compute whether we have finished the entire sequence.
        done = self.is_done()
        if done:
            p.disconnect()
            # if self.observation_mode == "vision":
            #     self.metrics.print()
        return done

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
            raise ValueError(
                f"No stage found for current timestep: {self.timestep}"
            )
        return current_stage, stage_ts

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
        table_id = utils.create_table(self.opt.floor_mu)
        robot = InmoovShadowNew(
            init_noise=False, timestep=utils.TS, np_random=np.random,
        )
        debug = utils.get_n_optimal_init_arm_qs(
            robot,
            utils.PALM_POS_OF_INIT,
            p.getQuaternionFromEuler(utils.PALM_EULER_OF_INIT),
            src_xyz,
            table_id,
            wrist_gain=3.0,
        )
        print(f"get_n_optimal_init_arm_qs: {debug}")
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
        # self.a.seed(self.opt.seed)
        table_id = utils.create_table(self.opt.floor_mu)
        (
            o_pos_pf_ave,
            o_quat_pf_ave,
            _,
        ) = utils.read_grasp_final_states_from_pickle(self.opt.grasp_pi)
        p_pos_of_ave, p_quat_of_ave = p.invertTransform(
            o_pos_pf_ave, o_quat_pf_ave
        )
        robot = InmoovShadowNew(
            init_noise=False, timestep=utils.TS, np_random=np.random,
        )
        q_transport_dst = utils.get_n_optimal_init_arm_qs(
            robot, p_pos_of_ave, p_quat_of_ave, dst_xyz, table_id
        )[0]
        p.resetSimulation()
        return q_reach_dst, q_transport_dst

    def compute_reach_trajectory(self) -> np.ndarray:
        """Computes the reaching trajectory.
        
        Returns:
            trajectory: The reaching trajectory of shape (n_steps, 7).
        """
        trajectory = openrave.compute_trajectory(
            odicts=self.initial_obs,
            target_idx=self.src_idx,
            q_start=None,
            q_end=self.q_reach_dst,
            stage="reach",
        )
        return trajectory

    def compute_transport_trajectory(self) -> np.ndarray:
        """Computes the transport trajectory.
        
        Returns:
            trajectory: The transport trajectory of shape (n_steps, 7).
        """
        q_src = self.world.get_robot_arm_q()
        q_dst = self.q_transport_dst

        trajectory = openrave.compute_trajectory(
            odicts=self.initial_obs,
            target_idx=self.src_idx,
            q_start=q_src,
            q_end=q_dst,
            stage="transport",
        )
        return trajectory

    def compute_retract_trajectory(self) -> np.ndarray:
        """Computes the retract trajectory.
        
        Returns:
            trajectory: The retract trajectory of shape (n_steps, 7).
        """
        if self.observation_mode == "gt":
            odicts = self.get_observation(
                observation_mode=self.observation_mode, renderer=self.renderer
            )
        elif self.observation_mode == "vision":
            odicts = self.last_pred_obs
        q_src = self.world.get_robot_arm_q()

        trajectory = openrave.compute_trajectory(
            odicts=odicts,
            target_idx=self.src_idx,
            q_start=q_src,
            q_end=None,
            stage="retract",
        )
        return trajectory

    def reach(self, stage_ts: int):
        if stage_ts == 0:
            self.reach_trajectory = self.compute_reach_trajectory()
            if len(self.reach_trajectory) == 0:
                return False
        self.execute_plan(trajectory=self.reach_trajectory, idx=stage_ts)
        return True

    def grasp(self, stage_ts: int):
        # Load the grasping actor critic model.
        if stage_ts == 0:
            (
                self.policy,
                _,
                self.hidden_states,
                self.masks,
            ) = system.policy.load(
                policy_dir=self.opt.grasp_dir,
                env_name=self.opt.grasp_env_name,
                is_cuda=self.opt.is_cuda,
            )
        obs = system.policy.wrap_obs(
            self.get_grasp_observation(), is_cuda=self.opt.is_cuda
        )
        with torch.no_grad():
            _, action, _, self.hidden_states = self.policy.act(
                obs, self.hidden_states, self.masks, deterministic=self.opt.det
            )
        self.world.step_robot(
            action=system.policy.unwrap_action(
                action=action, is_cuda=self.opt.is_cuda
            )
        )
        self.masks.fill_(1.0)

    def transport(self, stage_ts: int):
        if stage_ts == 0:
            self.transport_trajectory = self.compute_transport_trajectory()
            if len(self.transport_trajectory) == 0:
                return False
        self.execute_plan(trajectory=self.transport_trajectory, idx=stage_ts)
        return True

    def place(self, stage_ts: int):
        if stage_ts == 0:
            self.world.robot_env.change_control_skip_scaling(
                c_skip=self.opt.placing_control_skip
            )
            (
                self.policy,
                _,
                self.hidden_states,
                self.masks,
            ) = system.policy.load(
                policy_dir=self.opt.place_dir,
                env_name=self.opt.place_env_name,
                is_cuda=self.opt.is_cuda,
            )

        # Get the current observation.
        obs = system.policy.wrap_obs(
            self.get_place_observation(), is_cuda=self.opt.is_cuda
        )
        with torch.no_grad():
            _, action, _, self.hidden_states = self.policy.act(
                obs, self.hidden_states, self.masks, deterministic=self.opt.det
            )

        self.world.step_robot(
            action=system.policy.unwrap_action(
                action=action, is_cuda=self.opt.is_cuda
            )
        )

    def release(self):
        self.world.step()

    def retract(self, stage_ts: int):
        if stage_ts == 0:
            self.retract_trajectory = self.compute_retract_trajectory()
            if len(self.retract_trajectory) == 0:
                return False
        self.execute_plan(trajectory=self.retract_trajectory, idx=stage_ts)
        return True

    def execute_plan_original(self, trajectory: np.ndarray, idx: int):
        if idx > len(trajectory) - 1:
            tar_arm_q = trajectory[-1]
        else:
            tar_arm_q = trajectory[idx]
        self.world.robot_env.robot.tar_arm_q = tar_arm_q

        # Restore fingers.
        if self.opt.restore_fingers and idx >= len(trajectory) * 0.1:
            blending = np.clip(
                (idx - len(trajectory) * 0.1) / (len(trajectory) * 0.6),
                0.0,
                1.0,
            )
            cur_fin_q = self.world.robot_env.robot.get_q_dq(
                self.world.robot_env.robot.fin_actdofs
            )[0]
            tar_fin_q = (
                self.world.robot_env.robot.init_fin_q * blending
                + cur_fin_q * (1 - blending)
            )
            self.world.robot_env.robot.tar_fin_q = tar_fin_q

        self.world.robot_env.robot.apply_action([0.0] * 24)
        self.world.step()

    def execute_plan(self, trajectory: np.ndarray, idx: int):
        # Get initial robot pose.
        if idx == 0:
            self.world.robot_env.robot.tar_arm_q = trajectory[-1]
            init_tar_fin_q = self.world.robot_env.robot.tar_fin_q
            init_fin_q = self.world.robot_env.robot.get_q_dq(
                self.world.robot_env.robot.fin_actdofs
            )[0]
            self.last_tar_arm_q = self.world.robot_env.robot.get_q_dq(
                self.world.robot_env.robot.arm_dofs
            )[0]

        if idx > len(trajectory) - 1:
            tar_arm_q = trajectory[-1]
        else:
            tar_arm_q = trajectory[idx]

        tar_arm_vel = (tar_arm_q - self.last_tar_arm_q) / self.opt.ts
        max_force = self.world.robot_env.robot.maxForce

        p.setJointMotorControlArray(
            bodyIndex=self.world.robot_env.robot.arm_id,
            jointIndices=self.world.robot_env.robot.arm_dofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(tar_arm_q),
            targetVelocities=list(tar_arm_vel),
            forces=[max_force * 5] * len(self.world.robot_env.robot.arm_dofs),
        )

        if self.opt.restore_fingers and idx >= len(trajectory) * 0.1:
            blending = np.clip(
                (idx - len(trajectory) * 0.1) / (len(trajectory) * 0.6),
                0.0,
                1.0,
            )
            cur_fin_q = self.world.robot_env.robot.get_q_dq(
                self.world.robot_env.robot.fin_actdofs
            )[0]
            tar_fin_q = (
                self.world.robot_env.robot.init_fin_q * blending
                + cur_fin_q * (1 - blending)
            )
        else:
            # try to keep fin q close to init_fin_q (keep finger pose)
            # add at most offset 0.05 in init_tar_fin_q direction so that grasp is tight
            tar_fin_q = np.clip(
                init_tar_fin_q, init_fin_q - 0.05, init_fin_q + 0.05
            )

        # clip to joint limit
        tar_fin_q = np.clip(
            tar_fin_q,
            self.world.robot_env.robot.ll[
                self.world.robot_env.robot.fin_actdofs
            ],
            self.world.robot_env.robot.ul[
                self.world.robot_env.robot.fin_actdofs
            ],
        )

        p.setJointMotorControlArray(
            bodyIndex=self.world.robot_env.robot.arm_id,
            jointIndices=self.world.robot_env.robot.fin_actdofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(tar_fin_q),
            forces=[max_force] * len(self.world.robot_env.robot.fin_actdofs),
        )
        p.setJointMotorControlArray(
            bodyIndex=self.world.robot_env.robot.arm_id,
            jointIndices=self.world.robot_env.robot.fin_zerodofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.0]
            * len(self.world.robot_env.robot.fin_zerodofs),
            forces=[max_force / 4.0]
            * len(self.world.robot_env.robot.fin_zerodofs),
        )

        self.last_tar_arm_q = tar_arm_q

    def get_grasp_observation(self) -> torch.Tensor:
        odict = self.initial_obs[self.src_idx]
        position = odict["position"]
        half_height = odict["height"] / 2
        x, y = position[0], position[1]
        is_box = odict["shape"] == "box"
        if self.opt.use_height:
            obs = self.world.robot_env.get_robot_contact_txtytz_halfh_shape_obs_no_dup(
                x, y, 0.0, half_height, is_box
            )
        else:
            obs = self.world.robot_env.get_robot_contact_txty_shape_obs_no_dup(
                x, y, is_box
            )
        return obs

    def get_place_observation(self):
        # Update the observation only every `vision_delay` steps.
        if self.timestep % self.opt.vision_delay == 0:
            self.obs = self.get_observation(
                observation_mode=self.observation_mode, renderer=self.renderer
            )

        # Compute the observation vector from object poses and placing position.
        t_init_dict = self.scene[self.src_idx]
        x, y, _ = t_init_dict["position"]
        is_box = t_init_dict["shape"] == "box"

        if self.task == "stack":
            b_init_dict = self.scene[self.dst_idx]
            tz = b_init_dict["height"]
        else:
            tz = 0.0

        tdict = self.obs[self.src_idx]
        t_pos = tdict["position"]
        t_up = tdict["up_vector"]

        if self.task == "stack":
            bdict = self.obs[self.dst_idx]
            b_pos = bdict["position"]
            b_up = bdict["up_vector"]
        elif self.task == "place":
            b_pos = [0.0, 0.0, 0.0]
            b_up = [0.0, 0.0, 1.0]

        if self.opt.use_height:
            p_obs = self.world.robot_env.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
                tx=x,
                ty=y,
                tz=tz,
                half_h=tdict["height"] / 2,
                shape=is_box,
                t_pos=t_pos,
                t_up=t_up,
                b_pos=b_pos,
                b_up=b_up,
            )
        else:
            p_obs = self.world.robot_env.get_robot_contact_txty_shape_2obj6dUp_obs_nodup_from_up(
                tx=x,
                ty=y,
                shape=is_box,
                t_pos=t_pos,
                t_up=t_up,
                b_pos=b_pos,
                b_up=b_up,
            )
        return p_obs

    def get_observation(self, observation_mode: str, renderer: str):
        """Gets an observation from the current state.
        
        Args:
            observation_mode: The mode of observation, e.g. using ground truth 
                or vision.
            renderer: If we are using vision as our observation model, this
                specifies the renderer we are using to render images that are
                input to vision.

        Returns:
            observation: A list of dictionaries storing object observations 
                for the current world state, in the format: 
                [
                    {
                        "shape": shape,
                        "color": color,
                        "radius": radius,
                        "height": height,
                        "position": [x, y, z],
                        "orientation": [x, y, z, w],
                    },
                    ...
                ]
        """
        if observation_mode == "gt":
            # The ground truth observation is simply the same as the true
            # state of the world.
            obs = list(self.get_state()["objects"].values())
        elif observation_mode == "vision":
            obs = self.get_vision_observation(renderer=renderer)
        else:
            raise ValueError(
                "Unsupported observation mode: {self.observation_mode}"
            )
        return copy.deepcopy(obs)

    def get_vision_observation(self, renderer: str):
        """Computes the observation of the bullet world using the vision 
        module.

        Args:
            renderer: The renderer of the input images to the vision module.
        
        Returns:
            obs: The vision observation, in the format:
                [
                    {
                        "shape": shape,  # First prediction
                        "color": color,  # First prediction
                        "radius": radius,  # First prediction
                        "height": height,  # First prediction
                        "position": [x, y, z],  # Current prediction
                        "up_vector": [u1, u2, u3],  # Current prediction
                    },
                    ...
                ], 
        """
        # Select the vision module based on the current stage.
        stage, _ = self.get_current_stage()
        if stage == "plan":
            vision_module = self.planning_vision_module
        elif stage == "place":
            vision_module = self.placing_vision_module
        else:
            raise ValueError(f"No vision module for stage: {stage}.")

        # Retrieves the image, camera pose, and object segmentation masks.
        rgb, masks, cam_position, cam_orientation = self.get_images()

        # Predict attributes for all the segmentations.
        pred = vision_module.predict(
            rgb=rgb, masks=masks, debug_id=self.timestep
        )

        pred_odicts = []
        for y in pred:
            # Convert vectorized predictions to dictionary form. The
            # predicted pose is also transformed using camera information.
            odict = dash_object.y_vec_to_dict(
                y=list(y),
                coordinate_frame=self.opt.coordinate_frame,
                cam_position=cam_position,
                cam_orientation=cam_orientation,
            )
            pred_odicts.append(odict)

        # We track predicted objects throughout time by matching the attributes
        # predicted at the current timestep with the object attributes from the
        # initial observation.
        if self.initial_obs is None:
            pred_obs = pred_odicts
        else:
            pred_obs = self.match_predictions_with_initial_obs(
                pred_odicts=pred_odicts
            )
        # Store the computed observation for future computation.
        self.last_pred_obs = copy.deepcopy(pred_obs)

        # Save the predicted and ground truth object dictionaries.
        # if self.opt.save_predictions:
        #     path = os.path.join(
        #         self.opt.save_preds_dir,
        #         f"{self.trial:02}",
        #         f"{self.timestep:04}.p",
        #     )
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        #     util.save_pickle(
        #         path=path, data={"gt": gt_odicts, "pred": pred_odicts}
        #     )
        return pred_obs

    def match_predictions_with_initial_obs(
        self, pred_odicts: List[Dict]
    ) -> List:
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

        # Construct the cost matrix.
        cost = np.zeros((len(self.initial_obs), len(pred_odicts)))
        for src_idx, src_y in enumerate(self.initial_obs):
            for dst_idx, dst_y in enumerate(pred_odicts):
                cell = 0
                for attr in ["color", "shape"]:
                    if src_y[attr] != dst_y[attr]:
                        cell += 1
                cost[src_idx][dst_idx] = cell

        # Compute assignments.
        src_idxs, dst_idxs = linear_sum_assignment(cost)

        # If `self.initial_obs` has more elements than `pred_odicts`, then
        # there will be unassigned elements in `self.initial_obs`. We assign
        # unassigned elements with the last observation. To do this, we
        # initialize the `pred_obs` which will be returned with
        # `self.last_pred_obs`. Since we only override assigned elements, each
        # unassigned element will by default hold the last prediction.
        pred_obs = copy.deepcopy(self.last_pred_obs)

        # Override the pose predictions for the objects being manipulated.
        for src_idx, dst_idx in zip(src_idxs, dst_idxs):
            if src_idx in [self.src_idx, self.dst_idx]:
                dst_odict = pred_odicts[dst_idx]
                for attr in ["position", "up_vector"]:
                    pred_obs[src_idx][attr] = dst_odict[attr]
        return pred_obs

    def get_images(self) -> Tuple:
        """Retrieves the images that are input to the vision module.

        Args:
            oid: The object ID to retrieve images for.
            renderer: The renderer that the images are rendered with.
        
        Returns:
            rgb: The RGB image of the object.
            masks: A numpy array of shape (N, H, W) of instance masks.
            camera_position: The position of the camera used to capture the
                image.
            camera_orientation: The orientation of the camera used to capture 
                the image.
        """
        if self.renderer == "opengl":
            raise NotImplementedError
        elif self.renderer == "tiny_renderer":
            raise NotImplementedError
        elif self.renderer == "unity":
            # Unity should have already called set_unity_data before this.
            rgb = self.unity_data[0]["rgb"]
            seg_img = self.unity_data[0]["seg_img"]

            camera_position = self.unity_data[0]["camera_position"]
            camera_orientation = self.unity_data[0]["camera_orientation"]

            # Optionally we can visualize unity images using OpenCV.
            if self.visualize_unity:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                seg_bgr = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
                cv2.imshow("rgb", bgr)
                cv2.imshow("seg", seg_bgr)
                cv2.waitKey(5)
        else:
            raise ValueError(f"Invalid renderer: {self.renderer}")

        # Either predict segmentations or use ground truth.
        if self.opt.use_segmentation_module:
            masks = self.segmentation_module.predict(
                img=rgb, debug_id=self.timestep
            )
        else:
            # If using ground truth, convert the segmentation image into a
            # segmentation map.
            masks, _ = ns_vqa_dart.bullet.seg.seg_img_to_map(seg_img)
        return rgb, masks, camera_position, camera_orientation

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

    def is_done(self) -> bool:
        """Determines whether we have finished the sequence.

        Returns:
            done: Whether the sequence has finished.
        """
        # If we executed all steps, we are done.
        return self.timestep == self.n_total_steps
