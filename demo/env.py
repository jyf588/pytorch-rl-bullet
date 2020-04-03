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
from demo import policy
from demo.vision_module import VisionModule

from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import (
    ImaginaryArmObjSession,
)
from my_pybullet_envs.inmoov_shadow_demo_env_v4 import (
    InmoovShadowHandDemoEnvV4,
)
from my_pybullet_envs.inmoov_shadow_place_env_v9 import (
    InmoovShadowHandPlaceEnvV9,
)
import my_pybullet_envs.utils as utils

from ns_vqa_dart.bullet.renderer import BulletRenderer
from ns_vqa_dart.bullet import dash_object, util


class DemoEnvironment:
    def __init__(
        self,
        opt: argparse.Namespace,
        scene: List[Dict],
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
            observation_mode: Source of observation, which can either be from
                ground truth (`gt`) or vision (`vision`).
            visualize_bullet: Whether to visualize the demo in pybullet in 
                real-time in an OpenGL window.
            visualize_unity: Whether to visualize the unity image stream sent
                from unity to the current class in a live OpenCV window.
            renderer: The renderer to use to generate images, if 
                `observation_mode` is `vision`.
        """
        np.random.seed(11)

        self.opt = opt
        self.observation_mode = observation_mode
        self.visualize_bullet = visualize_bullet
        self.visualize_unity = visualize_unity
        self.renderer = renderer

        # Set the initial scene.
        self.scene = scene

        # self.state = copy.deepcopy(self.init_state)
        self.src_idx = 1
        self.dst_idx = 2

        # We need to do this at the beginning before the actual bullet world is
        # setup because it involves an imaginary session. Another option is to
        # do this in a separate session using direct rendering.
        # self.q_transport_dst = self.compute_q_transport_dst()

        self.q_reach_dst, self.q_transport_dst = self.compute_qs()

        # Initialize the pybullet world with the initial state.
        self.bc = self.create_bullet_client()
        (
            self.robot_env,
            self.oids,
            self.table_id,
        ) = self.initialize_bullet_world(odicts=self.scene)
        self.src_oid = self.oids[self.src_idx]
        self.dst_oid = self.oids[self.dst_idx]
        self.state = self.construct_state()

        print("constructed state:")
        pprint.pprint(self.state)

        # Initialize the vision module if we are using vision.
        if self.observation_mode == "vision":
            self.vision_module = VisionModule()

        self.reach_trajectory = None

        self.timestep = 0
        self.n_plan_steps = 200
        self.n_total_steps = (
            self.n_plan_steps * 2
            + self.opt.n_grasp_steps
            + self.opt.n_place_steps
            + self.opt.n_release_steps
        )

    def construct_state(self):
        """Constructs the state dictionary of the bullet environment.
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
        state = {"objects": {}, "robot": self.get_robot_state()}
        for idx in range(len(self.oids)):
            oid = self.oids[idx]
            odict = self.scene[idx]
            state["objects"][oid] = odict
        return state

    def update_state(self):
        for oid, odict in self.state["objects"].items():
            position = odict["position"]
            position, orientation = self.bc.getBasePositionAndOrientation(oid)
            odict["position"] = list(position)
            odict["orientation"] = list(orientation)
            odict["up_vector"] = util.orientation_to_up(
                orientation=orientation
            )
            self.state["objects"][oid] = odict
        self.state["robot"] = self.get_robot_state()

    def get_robot_state(self):
        """Gets the current robot state.
        
        Returns:
            robot_state: A dictionary with the following format:
                {
                    <joint_name>: <joint_angle>
                }
        """
        robot_state = {}
        robot_id = self.robot_env.robot.arm_id
        for joint_idx in range(self.bc.getNumJoints(robot_id)):
            joint_name = self.bc.getJointInfo(robot_id, joint_idx)[1].decode(
                "utf-8"
            )
            joint_angle = self.bc.getJointState(
                bodyUniqueId=robot_id, jointIndex=joint_idx
            )[0]
            robot_state[joint_name] = joint_angle
        return robot_state

    def create_bullet_client(self):
        """Creates the bullet client.

        Returns:
            bc: The bullet client, which uses either DIRECT or GUI for the
                renderering.
        """
        if self.visualize_bullet:
            mode = "gui"
        else:
            mode = "direct"
        bc = util.create_bullet_client(mode=mode)
        return bc

    def initialize_bullet_world(self, odicts: List[Dict]):
        """Initializes the bullet world by create a bullet client and loading 
        the initial state containing the robot, table, and tabletop objects.

        Args:
            odicts: Object dictionaries, with the format: 
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
            robot_env: The robot environment.
            table_id: The bullet ID of the tabletop.
        """
        # Set parameters of the bullet world.
        self.bc.setPhysicsEngineParameter(
            numSolverIterations=self.opt.bullet_contact_iter
        )
        self.bc.setPhysicsEngineParameter(
            deterministicOverlappingPairs=self.opt.det_contact
        )
        self.bc.setTimeStep(self.opt.ts)
        self.bc.setGravity(0, 0, -10)

        # Load robot, table, and objects.
        robot_env = self.load_robot()
        oids, table_id = self.load_table_and_objects(odicts=odicts)
        return robot_env, oids, table_id

    def load_robot(self):
        # Load the robot.
        robot_env = InmoovShadowHandDemoEnvV4(
            seed=self.opt.seed,
            init_noise=self.opt.init_noise,
            timestep=self.opt.ts,
            withVel=False,
            diffTar=True,
            robot_mu=self.opt.hand_mu,
            control_skip=self.opt.grasping_control_skip,
        )

        # Set the robot to zero pose?
        robot_env.robot.reset_with_certain_arm_q([0.0] * 7)
        return robot_env

    def load_table_and_objects(self, odicts: List[Dict]):
        """Loads a state into bullet.

        Args:
            odicts: Object dictionaries, with the format: 
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
            table_id: The bullet ID of the tabletop.
        """
        renderer = BulletRenderer(p=self.bc)
        oids = renderer.load_objects_from_state(
            ostates=odicts, position_mode="com"
        )

        # Load the tabletop.
        table_id = self.load_table()
        return oids, table_id

    def load_table(self):
        table_id = self.bc.loadURDF(
            os.path.join("my_pybullet_envs/assets/tabletop.urdf"),
            utils.TABLE_OFFSET,
            useFixedBase=1,
        )
        self.bc.changeVisualShape(
            table_id, -1, rgbaColor=utils.COLOR2RGBA["grey"]
        )
        self.bc.changeDynamics(table_id, -1, lateralFriction=self.opt.floor_mu)
        return table_id

    def compute_qs(self):
        src_x, src_y, _ = self.scene[self.src_idx]["position"]
        dst_x, dst_y, dst_z = self.scene[self.dst_idx]["position"]
        dst_position = [dst_x, dst_y, dst_z + utils.PLACE_START_CLEARANCE]

        sess = ImaginaryArmObjSession()

        q_reach_dst = np.array(
            sess.get_most_comfortable_q_and_refangle(src_x, src_y)[0]
        )

        print(f"q reach: {q_reach_dst}")

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
        return q_reach_dst, q_transport_dst

    def compute_reach_trajectory(self) -> np.ndarray:
        """Computes the reaching trajectory.
        
        Returns:
            trajectory: The reaching trajectory of shape (200, 7).
        """
        # sess = ImaginaryArmObjSession()
        # reach_position = self.state["objects"][self.src_oid]["position"]
        # x, y = reach_position[0], reach_position[1]
        # Qreach = np.array(sess.get_most_comfortable_q_and_refangle(x, y)[0])

        trajectory = openrave.compute_trajectory(
            state=self.state,
            dst_oid=self.src_oid,
            q_start=None,
            q_end=self.q_reach_dst,
            stage="reach",
        )
        # self.bc.resetSimulation()
        return trajectory

    def compute_q_transport_dst(self):
        a = InmoovShadowHandPlaceEnvV9(
            renders=False, grasp_pi_name=self.opt.grasp_pi
        )
        p_pos_of_ave, p_quat_of_ave = self.bc.invertTransform(
            a.o_pos_pf_ave, a.o_quat_pf_ave
        )
        odict = self.state["objects"][self.dst_oid]
        x, y, z = odict["position"]
        dst_position = [x, y, z + utils.PLACE_START_CLEARANCE]
        table_id = self.load_table()
        q_dst = utils.get_n_optimal_init_arm_qs(
            a.robot, p_pos_of_ave, p_quat_of_ave, dst_position, table_id
        )[0]
        return q_dst

    def compute_transport_trajectory(self) -> np.ndarray:
        """Computes the transport trajectory.
        
        Returns:
            trajectory: The transport trajectory of shape (200, 7).
        """
        q_src = self.robot_env.robot.get_q_dq(self.robot_env.robot.arm_dofs)[0]

        # print(f"q_transport_dst: {self.q_transport_dst}")
        # print(f"q_transport_dst type: {type(self.q_transport_dst)}")
        # print(f"q_transport_dst shape: {self.q_transport_dst.shape}")
        q_src = [
            -0.1950513,
            -0.79553950,
            0.55946438,
            -1.34713527,
            -0.50737202,
            -0.91808677,
            -0.58151554,
        ]
        q_dst = [
            -1.2203942305396256,
            -0.28347340556272493,
            -0.6584654379872827,
            -1.2869851602338127,
            -0.6849580878601577,
            -0.6234920041770489,
            -0.18889481363039415,
        ]

        trajectory = openrave.compute_trajectory(
            state=self.state,
            dst_oid=self.src_oid,
            q_start=q_src,
            # q_end=self.q_transport_dst,
            q_end=q_dst,
            stage="transport",
        )
        return trajectory

    def step(self):
        """Policy performs a single action based on the current state.
        
        Returns:
            res: If we are done, return -1. Otherwise, we return 0.
        """
        print(f"Step: timestep: {self.timestep}")
        reach_start = 0
        reach_end = reach_start + self.n_plan_steps
        grasp_start = reach_end
        grasp_end = grasp_start + self.opt.n_grasp_steps
        transport_start = grasp_end
        transport_end = transport_start + self.n_plan_steps
        place_start = transport_end
        place_end = place_start + self.opt.n_place_steps
        release_start = place_end
        release_end = release_start + self.opt.n_release_steps

        print(f"place_start: {place_start}")
        print(f"place_end: {place_end}")

        # Execute reaching.
        if reach_start <= self.timestep < reach_end:
            self.reach()
        # Execute grasping.
        elif grasp_start <= self.timestep < grasp_end:
            self.grasp()
        # Transport.
        elif transport_start <= self.timestep < transport_end:
            self.transport()
        # Place.
        elif place_start <= self.timestep < place_end:
            self.place()
        elif release_start <= self.timestep < release_end:
            self.release()
        else:
            raise ValueError(f"Invalid timestep: {self.timestep}")

        self.timestep += 1
        return self.is_done()

    def reach(self):
        print("reach")
        if self.reach_trajectory is None:
            self.reach_trajectory = self.compute_reach_trajectory()
        self.execute_plan(trajectory=self.reach_trajectory, idx=self.timestep)

    def grasp(self):
        print("grasp")
        # Load the grasping actor critic model.
        if self.timestep - self.n_plan_steps == 0:
            self.policy, _, self.hidden_states, self.masks = policy.load(
                policy_dir=self.opt.grasp_dir,
                env_name=self.opt.grasp_env_name,
                is_cuda=self.opt.is_cuda,
            )
        obs = self.get_grasping_observation()
        with torch.no_grad():
            _, action, _, self.hidden_states = self.policy.act(
                obs,
                self.hidden_states,
                self.masks,
                deterministic=self.opt.det,
            )
        self.robot_env.step(self.unwrap_action(act_tensor=action))

    def transport(self):
        print("transport")
        idx = self.timestep - self.n_plan_steps - self.opt.n_grasp_steps
        if idx == 0:
            self.transport_trajectory = self.compute_transport_trajectory()
        self.execute_plan(trajectory=self.transport_trajectory, idx=idx)

    def place(self):
        print("place")
        idx = self.timestep - self.n_plan_steps * 2 - self.opt.n_grasp_steps
        if idx == 0:
            self.robot_env.change_control_skip_scaling(
                c_skip=self.opt.placing_control_skip
            )
            self.policy, _, self.hidden_states, self.masks = policy.load(
                policy_dir=self.opt.place_dir,
                env_name=self.opt.place_env_name,
                is_cuda=self.opt.is_cuda,
            )

        # Get the current observation.
        obs = self.get_placing_observation()
        with torch.no_grad():
            value, action, _, self.hidden_states = self.policy.act(
                obs,
                self.hidden_states,
                self.masks,
                deterministic=self.opt.det,
            )

        self.robot_env.step(self.unwrap_action(action))

    def release(self):
        print("release")
        self.bc.stepSimulation()
        time.sleep(self.opt.ts)

    def execute_plan(self, trajectory: np.ndarray, idx: int):
        self.robot_env.robot.tar_arm_q = trajectory[idx]
        self.robot_env.robot.apply_action([0.0] * 24)
        self.bc.stepSimulation()
        time.sleep(self.opt.ts)

    def get_grasping_observation(self) -> torch.Tensor:
        odict = self.state["objects"][self.src_oid]
        position = odict["position"]
        half_height = odict["height"] / 2
        x, y = position[0], position[1]
        obs = self.robot_env.get_robot_contact_txty_halfh_obs_nodup(
            x, y, half_height
        )
        obs = self.wrap_obs(obs)
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
        p_obs = self.robot_env.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
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
        p_obs = self.wrap_obs(p_obs)
        return p_obs

    def wrap_obs(self, obs: np.ndarray) -> torch.Tensor:
        obs = torch.Tensor([obs])
        if self.opt.is_cuda:
            obs = obs.cuda()
        return obs

    def unwrap_action(self, act_tensor: torch.Tensor) -> np.ndarray:
        action = act_tensor.squeeze()
        action = action.cpu() if self.opt.is_cuda else action
        return action.numpy()

    def get_observation(self, observation_mode: str, renderer: str):
        """Gets the observation for the current timestep.
        
        Args:
            observation_mode:
            renderer: 

        Returns:
            observation:
        """
        self.update_state()
        obs = copy.deepcopy(self.state)
        if self.observation_mode == "gt":
            pass
        elif self.observation_mode == "vision":
            y_dict = self.get_observation_from_vision(renderer=renderer)
            src_odict = obs["objects"][self.src_oid]
            src_odict["position"] = y_dict["position"]
            src_odict["up_vector"] = y_dict["up_vector"]
            obs["objects"][self.src_oid] = src_odict
        else:
            raise ValueError(
                "Unsupported observation mode: {self.observation_mode}"
            )
        return obs

    def get_observation_from_vision(self, renderer: str):
        oid = 2
        rgb, seg_img = self.get_image(oid=oid, renderer=renderer)
        start = time.time()
        pred = self.vision_module.predict(oid=oid, rgb=rgb, seg_img=seg_img)
        print(f"Vision inference time: {time.time() - start}")

        # Convert vectorized predictions to dictionary form using camera
        # information.
        y_dict = dash_object.y_vec_to_dict(
            y=list(pred[0]),
            coordinate_frame="camera",
            cam_position=self.unity_data[oid]["camera_position"],
            cam_orientation=self.unity_data[oid]["camera_orientation"],
        )

        print(f"Vision predictions: {y_dict}")
        return y_dict

    def obs_dict_to_vec(self, obs_dict: Dict) -> np.ndarray:
        """Converts an observation dictionary to an observation vector."""
        obs_vec = np.zeros((15,))
        return obs_vec

    def get_image(self, oid: int, renderer: str):
        if renderer == "opengl":
            raise NotImplementedError
        elif renderer == "tiny_renderer":
            raise NotImplementedError
        elif renderer == "unity":
            # Unity should have already called set_unity_data before this.
            rgb = self.unity_data[oid]["rgb"]
            seg_rgb = self.unity_data[oid]["seg_img"]
            if self.visualize_unity:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("rgb", bgr)
                cv2.imshow("seg", seg_bgr)
                cv2.waitKey(5)
        else:
            raise ValueError(f"Invalid renderer: {renderer}")
        return rgb, seg_rgb

    def get_current_state(self):
        """Retrieves the current state of the bullet environment.

        Returns:
            state_id: The ID of the state.
            object_tags: A list of object tags to obtain Unity data for.
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
            }.
        """
        self.update_state()
        state_id = f"{self.timestep:06}"
        object_tags = [f"{oid:02}" for oid in self.state["objects"].keys()]
        look_at_oids = [2]
        return state_id, object_tags, self.state, look_at_oids

    def state2obs(self, state: Dict) -> np.ndarray:
        """
        Args:
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
            }.
        
        Returns:
            obs: The observation vector for the policy, with the following
                components:
                [
                    <position>,
                    ...
                ]
        """
        obs = []
        for odict in state["objects"].values():
            obs += odict["position"]
        return np.array(obs)

    def act(self, obs: np.ndarray):
        """Policy determines an action based on the observation, and the action
        is applied to the bullet environment.
        
        Args:
            obs: The observation vector for the policy, with the following
                components:
                [
                    <position>,
                    ...
                ]        
        """
        pass

    def set_unity_data(self, data: Dict):
        """Processes data received from Unity.

        Args:
            data: Unity data, in the format
                {
                    <otag>:{
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
