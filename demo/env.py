import argparse
import copy
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
import ns_vqa_dart.bullet.util as util


class DemoEnvironment:
    def __init__(
        self,
        opt: argparse.Namespace,
        observation_mode: str,
        visualize_bullet: bool,
        renderer: Optional[str] = None,
        floor_mu: Optional[float] = 1.0,
    ):
        """
        Args:
            opt: Various demo configurations/options.
            observation_mode: Source of observation, which can either be from
                ground truth (`gt`) or vision (`vision`).
            visualize_bullet: Whether to visualize the demo in pybullet in 
                real-time in an OpenGL window.
            renderer: The renderer to use to generate images, if 
                `observation_mode` is `vision`.
        """
        np.random.seed(11)

        self.opt = opt
        self.observation_mode = observation_mode
        self.visualize_bullet = visualize_bullet
        self.renderer = renderer

        # Get the initial state.
        self.init_state = self.get_initial_state()
        self.state = copy.deepcopy(self.init_state)
        self.src_oid = 1
        self.dst_oid = 2

        # We need to do this at the beginning before the actual bullet world is
        # setup because it involves an imaginary session. Another option is to
        # do this in a separate session using direct rendering.
        # self.q_transport_dst = self.compute_q_transport_dst()

        self.q_reach_dst, self.q_transport_dst = self.compute_qs()

        # Initialize the pybullet world with the initial state.
        self.bc = self.create_bullet_client()
        self.robot_env, self.table_id = self.initialize_bullet_world(
            state=self.state
        )

        # Initialize the vision module if we are using vision.
        if self.observation_mode == "vision":
            self.vision_module = VisionModule()

        self.timestep = 0
        self.n_plan_steps = 200
        self.n_total_steps = (
            self.n_plan_steps * 2
            + self.opt.n_grasp_steps
            + self.opt.n_place_steps
            + self.opt.n_release_steps
        )

    def get_initial_state(self) -> Dict:
        """Loads the initial state of the demo.
        
        Returns:
            state: The initial state of the demo, with the format: 
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
            }.
        """
        odicts = {
            0: {
                "shape": "box",
                "color": "yellow",
                "position": [0.15, 0.7, 0.09],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "radius": 0.03,
                "height": 0.18,
            },
            1: {
                "shape": "box",
                "color": "green",
                "position": [0.2, 0.4, 0.09],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "radius": 0.03,
                "height": 0.18,
            },
            2: {
                "shape": "cylinder",
                "color": "blue",
                "position": [0.1, -0.05, 0.09],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "radius": 0.03,
                "height": 0.18,
            },
            3: {
                "shape": "box",
                "color": "yellow",
                "position": [0.0, 0.1, 0.09],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "radius": 0.03,
                "height": 0.18,
            },
        }
        for oid, odict in odicts.items():
            odict["mass"] = np.random.uniform(utils.MASS_MIN, utils.MASS_MAX)
            odict["mu"] = self.opt.obj_mu
            odict["height"] = np.random.uniform(utils.H_MIN, utils.H_MAX)
            odict["radius"] = np.random.uniform(
                utils.HALF_W_MIN, utils.HALF_W_MAX
            )
            if odict["shape"] == "box":
                odict["radius"] *= 0.8
            odicts[oid] = odict
        state = {"objects": odicts}
        return state

    def update_state(self):
        for oid, odict in self.state["objects"].items():
            position, orientation = self.bc.getBasePositionAndOrientation(oid)
            odict["position"] = position
            odict["orientation"] = orientation
            odict["up_vector"] = util.orientation_to_up(
                orientation=orientation
            )
            self.state["objects"][oid] = odict

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

    def initialize_bullet_world(self, state: Dict):
        """Initializes the bullet world by create a bullet client and loading 
        the initial state containing the robot, table, and tabletop objects.

        Args:
            state: The state dictionary, with the format: 
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
                }
        
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
        table_id = self.load_table_and_objects(state=state)
        return robot_env, table_id

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

    def load_table_and_objects(self, state: Dict):
        """Loads a state into bullet.

        Args:
            state: The state dictionary, with the format: 
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
            }.
        
        Returns:
            table_id: The bullet ID of the tabletop.
        """
        renderer = BulletRenderer(p=self.bc)
        renderer.load_objects_from_state(
            ostates=list(state["objects"].values()), position_mode="com"
        )

        # Load the tabletop.
        table_id = self.load_table()
        return table_id

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
        src_x, src_y, _ = self.state["objects"][self.src_oid]["position"]
        dst_x, dst_y, dst_z = self.state["objects"][self.dst_oid]["position"]
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
        # obs = self.get_observation(
        #     observation_mode=self.observation_mode, renderer=self.renderer
        # )
        # self.act(obs)
        # self.bc.stepSimulation()

        self.timestep += 1
        return self.is_done()

    def reach(self):
        print("reach")
        if self.timestep == 0:
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
        # Update the state only every `vision_delay` steps.
        if self.timestep % self.opt.vision_delay == 0:
            self.update_state()

        # Compute the observation vector from object poses and placing position.
        tdict = self.state["objects"][self.src_oid]
        bdict = self.state["objects"][self.dst_oid]
        t_init_dict = self.init_state["objects"][self.src_oid]
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
        if observation_mode == "gt":
            observation = []
        elif observation_mode == "vision":
            observation = self.get_observation_from_vision(renderer=renderer)
        else:
            raise ValueError(f"Invalid observation mode: {observation_mode}")
        return observation

    def get_observation_from_vision(self, renderer: str):
        observation = []
        image = self.get_image(renderer=renderer)
        pred_dict = self.vision_module.predict(image=image)
        observation = self.obs_dict_to_vec(obs_dict=pred_dict)
        return observation

    def obs_dict_to_vec(self, obs_dict: Dict) -> np.ndarray:
        """Converts an observation dictionary to an observation vector."""
        obs_vec = np.zeros((15,))
        return obs_vec

    def get_image(self, renderer: str):
        if renderer == "opengl":
            raise NotImplementedError
        elif renderer == "tiny_renderer":
            raise NotImplementedError
        elif renderer == "unity":
            # Unity should have already called set_unity_data before this.
            image = self.data["image"]
        else:
            raise ValueError(f"Invalid renderer: {renderer}")
        return image

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
        state = self.state
        state_id = f"{self.timestep:06}"
        object_tags = [f"{oid:02}" for oid in state["objects"].keys()]
        return state_id, object_tags, state

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
        self.data = data

    def is_done(self) -> bool:
        """Determines whether we have finished the sequence.

        Returns:
            done: Whether the sequence has finished.
        """
        # If we executed all steps, we are done.
        return self.timestep == self.n_total_steps
