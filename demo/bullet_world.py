"""The class definition for a Bullet World which contains a robot, tabletop,
and tabletop objects."""
import argparse
import copy
import numpy as np
import os
import pybullet
import time
import torch
from typing import *

import demo.policy
from my_pybullet_envs.inmoov_shadow_demo_env_v4 import (
    InmoovShadowHandDemoEnvV4,
)
import my_pybullet_envs.utils as utils
from ns_vqa_dart.bullet.renderer import BulletRenderer
import ns_vqa_dart.bullet.util


class BulletWorld:
    def __init__(
        self,
        opt: argparse.Namespace,
        scene: List[Dict],
        visualize: bool,
        p: Optional = None,
    ):
        """
        Args:
            opt: Various options.
            scene: A scene description in the format:
                [
                    {
                        "shape": <shape>,
                        "color": <color>,
                        "radius": <radius>,
                        "height": <height>,
                        "position": [x, y, z],
                        "orientation": [x, y, z, w],
                        "mass": <mass>,
                        "mu": <mu>
                    },
                    ...
                ]
            visualize: Whether to visualize the world in OpenGL.
        """
        self.opt = opt
        self.scene = scene
        self.visualize = visualize

        if p is None:
            self.bc = self.create_bullet_client()
        else:
            self.bc = p
        self.set_parameters()

        # Load the robot, table, and objects into the scene.
        self.robot_env, self.oids = self.load_scene(scene=scene)

        # Construct the state dictionary.
        self.state = self.construct_state()

    def create_bullet_client(self):
        """Creates the bullet client for the world.

        Returns:
            bc: The bullet client.
        """
        mode = "gui" if self.visualize else "direct"
        bc = ns_vqa_dart.bullet.util.create_bullet_client(mode=mode)
        return bc

    def set_parameters(self):
        """Sets parameters of the bullet world."""
        self.bc.setPhysicsEngineParameter(
            numSolverIterations=self.opt.bullet_contact_iter
        )
        self.bc.setPhysicsEngineParameter(
            deterministicOverlappingPairs=self.opt.det_contact
        )
        self.bc.setTimeStep(self.opt.ts)
        self.bc.setGravity(0, 0, -10)

    def load_scene(self, scene: List):
        """Loads a bullet world scene.

        Args:
            scene: A scene description in the format:
                [
                    {
                        "shape": <shape>,
                        "color": <color>,
                        "radius": <radius>,
                        "height": <height>,
                        "position": [x, y, z],
                        "orientation": [x, y, z, w],
                        "mass": <mass>,
                        "mu": <mu>
                    },
                    ...
                ]
        
        Returns:
            robot_env: The robot environment.
            oids: A list of object IDs corresponding to the object order in the
                input scene.
        """
        robot_env = self.load_robot()
        self.load_table()
        oids = self.load_tabletop_objects(scene=scene)
        return robot_env, oids

    def load_robot(self):
        """Loads the robot into the world.

        Returns:
            robot_env: The robot environment.
        """
        # Load the robot.
        robot_env = InmoovShadowHandDemoEnvV4(
            # seed=self.opt.seed,
            init_noise=self.opt.init_noise,
            timestep=self.opt.ts,
            withVel=False,
            diffTar=True,
            robot_mu=self.opt.hand_mu,
            control_skip=self.opt.grasping_control_skip,
        )
        init_fin_q = np.array(
            [0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.1]
        )
        robot_env.change_init_fin_q(init_fin_q)

        # Set the robot to zero pose.
        robot_env.robot.reset_with_certain_arm_q([0.0] * 7)
        return robot_env

    def load_tabletop_objects(self, scene: List[Dict]):
        """Loads a state into bullet.

        Args:
            scene: A scene description in the format:
                [
                    {
                        "shape": <shape>,
                        "color": <color>,
                        "radius": <radius>,
                        "height": <height>,
                        "position": [x, y, z],
                        "orientation": [x, y, z, w],
                        "mass": <mass>,
                        "mu": <mu>
                    },
                    ...
                ]
        
        Returns:
            oids: A list of object IDs corresponding to the object order in the
                input scene.
        """
        renderer = BulletRenderer(p=self.bc)
        oids = renderer.load_objects_from_state(
            odicts=scene, position_mode="com"
        )
        return oids

    def load_table(self):
        """Loads the tabletop.

        Returns:
            table_id: The bullet ID of the tabletop object.
        """
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

    def construct_state(self):
        """Constructs the state dictionary of the bullet environment.

        Returns:
            state: The state of the bullet environment, in the format:
                {
                    "objects": {
                        "<oid>": {
                            "shape": <shape>,
                            "color": <color>,
                            "radius": <radius>,
                            "height": <height>,
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

    def get_state(self):
        """Retrieves a copy of the current state of the bullet environment. The
        reason we return a copy is because we don't want the class' state to be
        modified by the callers of this function.

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
        self.state["objects"] = self.get_object_states()
        self.state["robot"] = self.get_robot_state()
        return copy.deepcopy(self.state)

    def get_object_states(self) -> Dict:
        """Retrieves the states of the tabletop objects currently in the scene.

        Returns:
            oid2dict: The current object states, in the format:
                {
                    <oid>: {
                        <attr>: <value>,
                    },
                    ...
                }
        """
        oid2dict = {}
        for oid, odict in self.state["objects"].items():
            old_position = odict["position"]
            position, orientation = self.bc.getBasePositionAndOrientation(oid)
            odict["position"] = list(position)
            odict["orientation"] = list(orientation)
            odict["up_vector"] = ns_vqa_dart.bullet.util.orientation_to_up(
                orientation=orientation
            )
            oid2dict[oid] = odict
            # new_position = np.array(odict["position"])
            # old_position = np.array(old_position)
            # if not np.allclose(new_position, old_position, atol=1e-02):
            #     print(f"position: {new_position}")
            #     print(f"old_position: {old_position}")
            #     print(f"difference: {np.abs(new_position - old_position)}")
            #     debug = 5 / 0
            # assert odict["position"] == old_position
        return oid2dict

    def get_robot_state(self):
        """Retrieves the current robot state.
        
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

    def get_robot_arm_q(self):
        """Retrieves the arm joint angles.

        Returns:
            arm_q: Arm joint angles in the following joint order:
                r_shoulder_out_joint
                r_shoulder_lift_joint
                r_upper_arm_roll_joint
                r_elbow_flex_joint
                r_elbow_roll_joint
                rh_WRJ2
                rh_WRJ1
        """
        arm_q = self.robot_env.robot.get_q_dq(self.robot_env.robot.arm_dofs)[0]
        return arm_q

    def act(self):
        pass

    def step(self):
        self.bc.stepSimulation()
        time.sleep(self.opt.ts)

    def step_robot(self, action: np.ndarray):
        self.robot_env.step(action=action)
