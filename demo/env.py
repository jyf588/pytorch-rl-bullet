import argparse
import copy
import cv2
import numpy as np
import os
import pprint
import pybullet as p
import sys
import time
import torch
from typing import *

from demo import openrave
from demo.bullet_world import BulletWorld
import demo.policy
from demo.vision_module import VisionModule
from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import (
    ImaginaryArmObjSession,
)
from my_pybullet_envs.inmoov_shadow_place_env_v9 import (
    InmoovShadowHandPlaceEnvV9,
)
import my_pybullet_envs.utils as utils
from NLP_module import NLPmod
from ns_vqa_dart.bullet import dash_object, util
from ns_vqa_dart.bullet.metrics import Metrics


class DemoEnvironment:
    def __init__(
        self,
        opt: argparse.Namespace,
        scene: List[Dict],
        command: str,
        observation_mode: str,
        visualize_bullet: bool,
        visualize_unity: bool,
        renderer: Optional[str] = None,
        floor_mu: Optional[float] = 1.0,
    ):
        """
        Args:
            opt: Various demo configurations/options.
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
                ]
            command: The command that the robot should execute.
            observation_mode: Source of observation, which can either be from
                ground truth (`gt`) or vision (`vision`).
            visualize_bullet: Whether to visualize the demo in pybullet in 
                real-time in an OpenGL window.
            visualize_unity: Whether to visualize the unity image stream sent
                from unity to the current class in a live OpenCV window.
            renderer: The renderer to use to generate images, if 
                `observation_mode` is `vision`.
        """

        self.opt = opt
        self.scene = scene
        self.command = command
        self.observation_mode = observation_mode
        self.visualize_bullet = visualize_bullet
        self.visualize_unity = visualize_unity
        self.renderer = renderer

        print("**********DEMO ENVIRONMENT**********")

        # Whether we've finished planning.
        self.planning_complete = False
        self.initial_obs = None
        self.obs = None
        self.world = None

        # Initialize the vision module if we are using vision for our
        # observations.
        if self.observation_mode == "vision":
            self.vision_module = VisionModule()

            # Initialize a class for tracking and computing metrics for the
            # vision module's predictions.
            self.metrics = Metrics()
        if visualize_bullet:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.imaginary_sess = ImaginaryArmObjSession()
        self.a = InmoovShadowHandPlaceEnvV9(
            renders=False, grasp_pi_name=self.opt.grasp_pi
        )

        self.timestep = 0
        self.n_total_steps = (
            self.opt.n_plan_steps  # reach
            + self.opt.n_grasp_steps  # grasp
            + self.opt.n_plan_steps  # transport
            + self.opt.n_place_steps  # place
            + self.opt.n_release_steps  # release
        )

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
        p.resetSimulation()

        # Create the bullet world now that we've finished our imaginary
        # sessions.
        self.world = BulletWorld(
            opt=self.opt,
            p=p,
            scene=self.scene,
            visualize=self.visualize_bullet,
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
        # Zero-pad the scene's position with fourth dimension because that's
        # what the language module expects.
        language_input = copy.deepcopy(observation)
        print(f"scene:")
        pprint.pprint(self.scene)
        print(f"observation:")
        pprint.pprint(observation)
        for idx, odict in enumerate(language_input):
            print(f'odict position: {odict["position"]}')
            language_input[idx]["position"] = odict["position"] + [0.0]

        # Feed through the language module.
        src_idx, (dst_x, dst_y), dst_idx = NLPmod(
            sentence=command, vision_output=language_input
        )

        # Compute the destination z based on whether there is a destination
        # object that we are placing on top of (stacking).
        if dst_idx is None:
            z = 0.0
        else:
            z = observation[dst_idx]["height"]
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
            else:
                raise ValueError(f"Invalid stage: {stage}")
            self.timestep += 1

        # Compute whether we have finished the entire sequence.
        done = self.is_done()
        if done:
            p.disconnect()
            if self.observation_mode == "vision":
                self.metrics.print()
        return done

    def get_current_stage(self) -> Tuple[str, int]:
        """Retrieves the current stage of the demo.

        Returns:
            stage: The stage of the demo.
            stage_s: The timestep of the current stage. 
        """
        if not self.planning_complete:
            return "plan", 0

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

        stage2ts_bounds = {
            "reach": (reach_start, reach_end),
            "grasp": (grasp_start, grasp_end),
            "transport": (transport_start, transport_end),
            "place": (place_start, place_end),
            "release": (release_start, release_end),
        }
        current_stage = None
        for stage, ts_bounds in stage2ts_bounds.items():
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
        src_x, src_y = src_xy
        q_reach_dst = np.array(
            self.imaginary_sess.get_most_comfortable_q_and_refangle(
                src_x, src_y
            )[0]
        )

        # Compute the destination arm pose for transport.
        self.a.seed(self.opt.seed)
        table_id = p.loadURDF(
            os.path.join("my_pybullet_envs/assets/tabletop.urdf"),
            utils.TABLE_OFFSET,
            useFixedBase=1,
        )
        p_pos_of_ave, p_quat_of_ave = p.invertTransform(
            self.a.o_pos_pf_ave, self.a.o_quat_pf_ave
        )
        q_transport_dst = utils.get_n_optimal_init_arm_qs(
            self.a.robot, p_pos_of_ave, p_quat_of_ave, dst_xyz, table_id
        )[0]
        return q_reach_dst, q_transport_dst

    def compute_reach_trajectory(self) -> np.ndarray:
        """Computes the reaching trajectory.
        
        Returns:
            trajectory: The reaching trajectory of shape (200, 7).
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
            trajectory: The transport trajectory of shape (200, 7).
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
            self.policy, _, self.hidden_states, self.masks = demo.policy.load(
                policy_dir=self.opt.grasp_dir,
                env_name=self.opt.grasp_env_name,
                is_cuda=self.opt.is_cuda,
            )
        obs = demo.policy.wrap_obs(
            self.get_grasping_observation(), is_cuda=self.opt.is_cuda
        )
        with torch.no_grad():
            _, action, _, self.hidden_states = self.policy.act(
                obs, self.hidden_states, self.masks, deterministic=self.opt.det
            )
        self.world.step_robot(
            action=demo.policy.unwrap_action(
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
            self.policy, _, self.hidden_states, self.masks = demo.policy.load(
                policy_dir=self.opt.place_dir,
                env_name=self.opt.place_env_name,
                is_cuda=self.opt.is_cuda,
            )

        # Get the current observation.
        obs = demo.policy.wrap_obs(
            self.get_placing_observation(), is_cuda=self.opt.is_cuda
        )
        with torch.no_grad():
            _, action, _, self.hidden_states = self.policy.act(
                obs, self.hidden_states, self.masks, deterministic=self.opt.det
            )

        self.world.step_robot(
            action=demo.policy.unwrap_action(
                action=action, is_cuda=self.opt.is_cuda
            )
        )

    def release(self):
        self.world.step()

    def execute_plan(self, trajectory: np.ndarray, idx: int):
        if idx > len(trajectory) - 1:
            tar_arm_q = trajectory[-1]
        else:
            tar_arm_q = trajectory[idx]
        self.world.robot_env.robot.tar_arm_q = tar_arm_q
        self.world.robot_env.robot.apply_action([0.0] * 24)
        self.world.step()

    def get_grasping_observation(self) -> torch.Tensor:
        odict = self.initial_obs[self.src_idx]
        position = odict["position"]
        half_height = odict["height"] / 2
        x, y = position[0], position[1]
        # obs = self.world.robot_env.get_robot_contact_txty_halfh_obs_nodup(
        #     x, y, half_height
        # )
        obs = self.world.robot_env.get_robot_contact_txtytz_halfh_shape_obs_no_dup(
            x, y, 0.0, half_height, odict["shape"] == "box"
        )
        return obs

    def get_placing_observation(self):
        # Update the observation only every `vision_delay` steps.
        if self.timestep % self.opt.vision_delay == 0:
            self.obs = self.get_observation(
                observation_mode=self.observation_mode, renderer=self.renderer
            )

        # Compute the observation vector from object poses and placing position.
        tdict = self.obs[self.src_idx]
        bdict = self.obs[self.dst_idx]
        t_init_dict = self.scene[self.src_idx]
        x, y, z = t_init_dict["position"]
        is_box = t_init_dict["shape"] == "box"
        p_obs = self.world.robot_env.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
            x,
            y,
            z,
            tdict["height"] / 2,
            is_box,
            tdict["position"],
            tdict["up_vector"],
            bdict["position"],
            bdict["up_vector"],
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
        return obs

    def get_vision_observation(self, renderer: str):
        """Computes the observation of the bullet world using the vision 
        module.

        Args:
            renderer: The renderer of the input images to the vision module.
        
        Returns:
            obs: The observation, in the format: 
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
                }.
        """
        # The final visual observation, for now, is the same as the ground
        # truth state with the source object's pose predicted. So, we
        # initialize the observation with the true state as a starting point.
        obs = copy.deepcopy(
            self.get_observation(observation_mode="gt", renderer=self.renderer)
        )

        # Predict the object pose for the objects that we've "looked" at.
        for idx in range(len(obs)):
            rgb, seg_img = self.get_images(oid=idx, renderer=renderer)
            pred = self.vision_module.predict(
                oid=idx, rgb=rgb, seg_img=seg_img
            )

            # Convert vectorized predictions to dictionary form using camera
            # information.
            y_dict = dash_object.y_vec_to_dict(
                y=list(pred[0]),
                coordinate_frame="unity_camera",
                cam_position=self.unity_data[idx]["camera_position"],
                cam_orientation=self.unity_data[idx]["camera_orientation"],
            )

            gt_odicts = self.get_observation(
                observation_mode="gt", renderer=self.renderer
            )
            self.metrics.add_example(gt_dict=gt_odicts[idx], pred_dict=y_dict)

            # Update the position and up vector with predicted values.
            obs[idx]["position"] = y_dict["position"]
            obs[idx]["up_vector"] = y_dict["up_vector"]
        return obs

    def get_images(self, oid: int, renderer: str):
        """Retrieves the images that are input to the vision module.

        Args:
            oid: The object ID to retrieve images for.
            renderer: The renderer that the images are rendered with.
        
        Returns:
            rgb: The RGB image of the object.
            seg_rgb: The RGB segmentation image of the object.
        """
        if renderer == "opengl":
            raise NotImplementedError
        elif renderer == "tiny_renderer":
            raise NotImplementedError
        elif renderer == "unity":
            # Unity should have already called set_unity_data before this.
            rgb = self.unity_data[oid]["rgb"]
            seg_rgb = self.unity_data[oid]["seg_img"]

            # Optionally we can visualize unity images using OpenCV.
            if self.visualize_unity:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("rgb", bgr)
                cv2.imshow("seg", seg_bgr)
                cv2.waitKey(5)
        else:
            raise ValueError(f"Invalid renderer: {renderer}")
        return rgb, seg_rgb

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
