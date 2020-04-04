import argparse
import copy
import cv2
import numpy as np
import os
import pprint
import pybullet as p
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

        # Whether we've finished planning.
        self.planning_complete = False
        self.initial_obs = None

        # Initialize the vision module if we are using vision for our
        # observations.
        if self.observation_mode == "vision":
            self.vision_module = VisionModule()

        # Initialize the pybullet world with the initial state. We do this
        # during initialization because unity will be querying the state at
        # the beginning in order to render the first frame.
        self.world = BulletWorld(
            opt=self.opt, scene=self.scene, visualize=False
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
        print("Observation:")
        pprint.pprint(self.initial_obs["objects"])

        # Temporarily use ground truth state for the source object.
        # state = self.world.get_state()
        # self.initial_obs["objects"][2] = state["objects"][2]
        # self.initial_obs["objects"][3] = state["objects"][3]
        # self.initial_obs["objects"][4] = state["objects"][4]
        # self.initial_obs["objects"][5] = state["objects"][5]

        # Use language module to determine the source / target objects and
        # positions.
        self.src_idx, self.dst_idx, self.dst_xyz = self.parse_command(
            command=self.command, observation=self.initial_obs
        )
        print(f"src_idx: {self.src_idx}")
        print(f"dst_idx: {self.dst_idx}")

        # Disconnect from the world client because we are creating temporary
        # clients for planning. Then, we recreate the world client.
        self.world.bc.disconnect()
        self.q_reach_dst, self.q_transport_dst = self.compute_qs()
        self.world = BulletWorld(
            opt=self.opt, scene=self.scene, visualize=self.visualize_bullet,
        )
        self.src_oid = self.world.oids[self.src_idx]
        self.dst_oid = self.world.oids[self.dst_idx]

        # Flag planning as complete.
        self.planning_complete = True

    def parse_command(self, command: str, observation: Dict):
        """Parses a language command in the context of an observation of a
        scene and computes the source and target objects and location for a 
        pick-and-place task.

        Args:
            command: The command to execute.
            scene: A list of object dictionaries defining the tabletop objects 
            in a scene, in the format:
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
                    ...
                }
        """
        # Zero-pad the scene's position with fourth dimension because that's
        # what the language module expects.
        odicts = copy.deepcopy(list(observation["objects"].values()))
        for idx, odict in enumerate(odicts):
            odict["position"] = odict["position"] + [0.0]
            odicts[idx] = odict

        src_idx, dst_xy, dst_idx = NLPmod(
            sentence=command, vision_output=odicts
        )

        # Compute the destination z based on whether there is a destination
        # object that we are placing on top of (stacking).
        if dst_idx is None:
            dst_z = 0.0
        else:
            dst_z = odicts[dst_idx]["height"]
        dst_xyz = [dst_xy[0], dst_xy[1], dst_z]
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
        """
        state = self.world.get_state()
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

    def compute_qs(self):
        dst_x, dst_y, dst_z = self.dst_xyz
        dst_position = [dst_x, dst_y, dst_z + utils.PLACE_START_CLEARANCE]
        a = InmoovShadowHandPlaceEnvV9(
            renders=False, grasp_pi_name=self.opt.grasp_pi
        )
        a.seed(self.opt.seed)
        table_id = p.loadURDF(
            os.path.join("my_pybullet_envs/assets/tabletop.urdf"),
            utils.TABLE_OFFSET,
            useFixedBase=1,
        )
        p_pos_of_ave, p_quat_of_ave = p.invertTransform(
            a.o_pos_pf_ave, a.o_quat_pf_ave
        )
        q_transport_dst = utils.get_n_optimal_init_arm_qs(
            a.robot, p_pos_of_ave, p_quat_of_ave, dst_position, table_id
        )[0]
        return None, q_transport_dst

    def compute_reach_trajectory(self) -> np.ndarray:
        """Computes the reaching trajectory.
        
        Returns:
            trajectory: The reaching trajectory of shape (200, 7).

        Vision input:
            object_positions: [[ 0.20730351  0.39945856  0.          0.        ]
                [ 0.11214702 -0.06985024  0.          0.        ]]
            q_start: None
            q_end: [-0.15365159 -0.50185157 -0.02217072 -1.48444567 -0.15545888 -0.41534081
                0.        ]
            object_positions: [[ 0.20730351  0.39945856  0.          0.        ]
                [ 0.11360673 -0.07798449  0.          0.        ]
                [-0.01266057  0.11019371  0.          0.        ]]
            q_start: [-0.15783011 -1.06514825  0.88929165 -0.9382602  -0.66303376 -0.6086176
                -0.65333982]
            q_end: [-1.2406110168162194, -0.3081875742466146, -0.630957998641116, -1.145658779172018, -0.6684807223498881, -0.5182690168629324, -0.13594495023924133]

        
        GT input (4 objects):
            object_positions: [[ 0.2   0.4   0.    0.  ]
                [ 0.15  0.7   0.    0.  ]
                [ 0.1  -0.05  0.    0.  ]
                [ 0.    0.1   0.    0.  ]]
            q_start: None
            q_end: [-0.15365159 -0.50185157 -0.02217072 -1.48444567 -0.15545888 -0.41534081
                0.        ]
            trajectories (first 5):
                [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
                0.00000000e+00  0.00000000e+00  0.00000000e+00]
                [-1.99004696e-04 -6.49982355e-04 -2.87148218e-05 -1.92260729e-03
                -2.01345443e-04 -5.37936337e-04  0.00000000e+00]
                [-3.98009392e-04 -1.29996471e-03 -5.74296435e-05 -3.84521459e-03
                -4.02690887e-04 -1.07587267e-03  0.00000000e+00]
                [-5.97014089e-04 -1.94994706e-03 -8.61444653e-05 -5.76782188e-03
                -6.04036330e-04 -1.61380901e-03  0.00000000e+00]
                [-7.96018785e-04 -2.59992942e-03 -1.14859287e-04 -7.69042917e-03
                -8.05381774e-04 -2.15174535e-03  0.00000000e+00]]
            trajectories (last 5):
                [[-0.16326611 -0.50382429 -0.02956165 -1.48224766 -0.16329572 -0.41453275
                0.00605788]
                [-0.16086248 -0.50333111 -0.02771392 -1.48279716 -0.16133651 -0.41473476
                0.00454341]
                [-0.15845885 -0.50283793 -0.02586619 -1.48334666 -0.1593773  -0.41493678
                0.00302894]
                [-0.15605522 -0.50234475 -0.02401845 -1.48389617 -0.15741809 -0.41513879
                0.00151447]
                [-0.15365159 -0.50185157 -0.02217072 -1.48444567 -0.15545888 -0.41534081
                0.        ]]
        
        GT input (2 objects):
            object_positions: [[ 0.2   0.4   0.    0.  ]
                [ 0.1  -0.05  0.    0.  ]]
            q_start: None
            q_end: [-0.15365159 -0.50185157 -0.02217072 -1.48444567 -0.15545888 -0.41534081
                0.        ]
            trajectories (first 5):
                [[ 0.          0.          0.          0.          0.          0.
                0.        ]
                [ 0.00198177  0.00106869 -0.00158268 -0.00057203  0.00162862  0.00140415
                -0.00011614]
                [ 0.00396353  0.00213738 -0.00316537 -0.00114406  0.00325724  0.0028083
                -0.00023228]
                [ 0.0059453   0.00320607 -0.00474805 -0.00171609  0.00488586  0.00421245
                -0.00034842]
                [ 0.00792706  0.00427476 -0.00633074 -0.00228812  0.00651448  0.0056166
                -0.00046457]]
            trajectories (last 5):
                [[-1.60346544e-01 -4.98437998e-01 -3.12129875e-02 -1.48881469e+00
                -1.61395155e-01 -4.19566399e-01  1.47671953e-02]
                [-1.57659794e-01 -5.00024413e-01 -2.75020668e-02 -1.48676763e+00
                -1.59002763e-01 -4.17769626e-01  8.61846517e-03]
                [-1.54973045e-01 -5.01610827e-01 -2.37911461e-02 -1.48472056e+00
                -1.56610371e-01 -4.15972853e-01  2.46973506e-03]
                [-1.53999572e-01 -5.01891784e-01 -2.25407352e-02 -1.48440019e+00
                -1.55755132e-01 -4.15437553e-01  4.96828589e-04]
                [-1.53651585e-01 -5.01851570e-01 -2.21707225e-02 -1.48444567e+00
                -1.55458877e-01 -4.15340806e-01  0.00000000e+00]]
        """
        src_x, src_y, _ = self.scene[self.src_idx]["position"]
        q_reach_dst = np.array(
            ImaginaryArmObjSession().get_most_comfortable_q_and_refangle(
                src_x, src_y
            )[0]
        )

        trajectory = openrave.compute_trajectory(
            state=self.initial_obs,
            dst_oid=self.src_oid,
            q_start=None,
            q_end=q_reach_dst,
            stage="reach",
        )
        return trajectory

    def compute_transport_trajectory(self) -> np.ndarray:
        """Computes the transport trajectory.
        
        Returns:
            trajectory: The transport trajectory of shape (200, 7).
        """
        q_src = self.world.robot_env.robot.get_q_dq(
            self.world.robot_env.robot.arm_dofs
        )[0]
        q_dst = self.q_transport_dst

        trajectory = openrave.compute_trajectory(
            state=self.initial_obs,
            dst_oid=self.src_oid,
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
        obs = self.get_grasping_observation()
        obs = demo.policy.wrap_obs(obs, is_cuda=self.opt.is_cuda)
        with torch.no_grad():
            _, action, _, self.hidden_states = self.policy.act(
                obs,
                self.hidden_states,
                self.masks,
                deterministic=self.opt.det,
            )
        action = demo.policy.unwrap_action(
            action=action, is_cuda=self.opt.is_cuda
        )
        self.world.step_robot(action=action)

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
        obs = self.get_placing_observation()
        obs = demo.policy.wrap_obs(obs, is_cuda=self.opt.is_cuda)
        with torch.no_grad():
            _, action, _, self.hidden_states = self.policy.act(
                obs,
                self.hidden_states,
                self.masks,
                deterministic=self.opt.det,
            )

        action = demo.policy.unwrap_action(
            action=action, is_cuda=self.opt.is_cuda
        )
        self.world.step_robot(action=action)

    def release(self):
        self.world.step()

    def execute_plan(self, trajectory: np.ndarray, idx: int):
        self.world.robot_env.robot.tar_arm_q = trajectory[idx]
        self.world.robot_env.robot.apply_action([0.0] * 24)
        self.world.step()

    def get_grasping_observation(self) -> torch.Tensor:
        odict = self.initial_obs["objects"][self.src_oid]
        position = odict["position"]
        half_height = odict["height"] / 2
        x, y = position[0], position[1]
        obs = self.world.robot_env.get_robot_contact_txty_halfh_obs_nodup(
            x, y, half_height
        )
        return obs

    def get_placing_observation(self):
        # Update the observation only every `vision_delay` steps.
        if self.timestep % self.opt.vision_delay == 0:
            self.obs = self.get_observation(
                observation_mode=self.observation_mode, renderer=self.renderer
            )

        # Compute the observation vector from object poses and placing position.
        tdict = self.obs["objects"][self.src_oid]
        bdict = self.obs["objects"][self.dst_oid]
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
            observation_mode:
            renderer: 

        Returns:
            observation:
        """
        if self.observation_mode == "gt":
            # The ground truth observation is simply the same as the true
            # state of the world.
            obs = self.world.get_state()
        elif self.observation_mode == "vision":
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
        obs = self.world.get_state()

        # Predict the object pose for the objects that we've "looked" at.
        for oid in self.world.oids:
            rgb, seg_img = self.get_images(oid=oid, renderer=renderer)
            pred = self.vision_module.predict(
                oid=oid, rgb=rgb, seg_img=seg_img
            )

            # Convert vectorized predictions to dictionary form using camera
            # information.
            y_dict = dash_object.y_vec_to_dict(
                y=list(pred[0]),
                coordinate_frame="camera",
                cam_position=self.unity_data[oid]["camera_position"],
                cam_orientation=self.unity_data[oid]["camera_orientation"],
            )

            # Update the position and up vector with predicted values.
            obs["objects"][oid]["position"] = y_dict["position"]
            obs["objects"][oid]["up_vector"] = y_dict["up_vector"]
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
