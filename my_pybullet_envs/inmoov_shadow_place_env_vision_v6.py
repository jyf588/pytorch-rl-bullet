from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math
import pickle
import random

import os
import inspect

from ns_vqa_dart.bullet.dash_object import DashObject
from ns_vqa_dart.bullet.generate_placing import PlacingDatasetGenerator
from ns_vqa_dart.bullet.renderer import BulletRenderer

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)


class InmoovShadowHandPlaceEnvVisionV6(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(
        self,
        renders=False,
        init_noise=True,
        up=False,
        random_shape=False,
        random_size=True,
        default_box=True,  # if not random shape, false: cylinder as default
        place_floor=False,
        use_gt_6d=True,
        gt_only_init=False,
        grasp_pi_name=None,
        exclude_hard=True,
        gen_vision_dataset=True,
        dataset_dir="/home/michelle/datasets/placing",
    ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        # self.is_box = is_box
        # self.is_small = is_small
        self.random_shape = random_shape
        self.random_size = random_size
        self.default_box = default_box
        self.place_floor = place_floor
        self.use_gt_6d = use_gt_6d
        self.gt_only_init = gt_only_init
        self.exclude_hard = exclude_hard

        self.hard_orn_thres = 0.9
        self.obj_mass = 3.5
        self.half_height = -1  # dummy, to be overwritten

        # TODO: hardcoded here
        if grasp_pi_name is None:
            if not random_shape:
                if default_box:
                    self.grasp_pi_name = "0219_box_2"
                else:
                    self.grasp_pi_name = "0219_cyl_2"  # TODO: ball
            else:
                pass  # TODO
        else:
            self.grasp_pi_name = grasp_pi_name

        # self.half_obj_height = 0.065 if self.is_small else 0.09
        self.start_clearance = 0.14
        self.cand_angles = [
            0.0,
            3.14 / 3,
            6.28 / 3,
            3.14,
            -6.28 / 3,
            -3.14 / 3,
        ]  # TODO: finer grid?
        self.cand_quats = [
            p.getQuaternionFromEuler([0, 0, cand_angle])
            for cand_angle in self.cand_angles
        ]

        self._timeStep = 1.0 / 240.0

        # Rendering configurations.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.renderer = BulletRenderer(p=p)

        # Object-related configurations.
        self.colors = ["red", "blue", "yellow", "green"]
        self.obj_id = None
        self.bottom_obj_id = None
        self.btm_object = DashObject(
            shape="cylinder",
            color=random.choice(self.colors),
            radius=0.05,
            height=0.18,
            position=[0.0, 0.0, 0.0],  # To be overridden.
        )
        self.table_object = DashObject(
            shape="tabletop",
            color="grey",
            radius=None,
            height=None,
            position=[0.25, 0.2, 0.0],
        )

        # Vision-related configurations.
        self.gen_vision_dataset = gen_vision_dataset
        self.dataset = PlacingDatasetGenerator(
            p=p,
            dataset_dir=dataset_dir,
            camera_offset=[0.0, self.table_object.position[1], 0.0],
        )

        self.np_random = None
        self.robot = None
        self.viewer = None
        self.timer = 0

        self.frameSkip = 3
        self.action_scale = np.array(
            [0.004] * 7 + [0.008] * 17
        )  # shadow hand is 22-5=17dof

        self.tx = -1  # dummy
        self.ty = -1  # dummy
        self.tz = -1  # dummy
        self.desired_obj_pos_final = None

        self.saved_file = None
        with open(
            os.path.join(
                currentdir,
                "assets/place_init_dist/final_states_"
                + self.grasp_pi_name
                + ".pickle",
            ),
            "rb",
        ) as handle:
            self.saved_file = pickle.load(handle)
        assert self.saved_file is not None

        self.o_pos_pf_ave = self.saved_file["ave_obj_pos_in_palm"]
        self.o_quat_pf_ave = self.saved_file["ave_obj_quat_in_palm"]
        self.o_quat_pf_ave /= np.linalg.norm(
            self.o_quat_pf_ave
        )  # in case not normalized
        self.init_states = self.saved_file["init_states"]  # a list of dicts

        # print(self.o_pos_pf_ave)
        # print(self.o_quat_pf_ave)
        # print(self.init_states[10])
        # print(self.init_states[51])
        # print(self.init_states[89])

        self.seed(
            0
        )  # used once temporarily, will be overwritten outside by env
        self.robot = InmoovShadowNew(
            init_noise=False, timestep=self._timeStep, np_random=self.np_random
        )

        self.observation = self.getExtendedObservation()
        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * action_dim),
            high=np.array([+1.0] * action_dim),
        )
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567] * obs_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf * obs_dummy, high=np.inf * obs_dummy
        )
        #
        # input("press enter")

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(
            np.array(arr)
            + self.np_random.uniform(low=-r, high=r, size=len(arr))
        )

    def reset_robot_object_from_sample(self, state, arm_q):
        o_pos_pf = state["obj_pos_in_palm"]
        o_quat_pf = state["obj_quat_in_palm"]
        if self.init_noise:
            o_pos_pf = list(self.perturb(o_pos_pf, 0.005))
            o_quat_pf = list(self.perturb(o_quat_pf, 0.005))
        all_fin_q_init = state["all_fin_q"]
        tar_fin_q_init = state["fin_tar_q"]

        self.robot.reset_with_certain_arm_q_finger_states(
            arm_q, all_fin_q_init, tar_fin_q_init
        )

        p_pos, p_quat = self.robot.get_link_pos_quat(self.robot.ee_id)
        o_pos, o_quat = p.multiplyTransforms(
            p_pos, p_quat, o_pos_pf, o_quat_pf
        )

        z_axis, _ = p.multiplyTransforms(
            [0, 0, 0], o_quat, [0, 0, 1], [0, 0, 0, 1]
        )  # R_cl * unitz[0,0,1]
        rotMetric = np.array(z_axis).dot(np.array([0, 0, 1]))
        # print(rotMetric, rotMetric)
        if self.exclude_hard and rotMetric < self.hard_orn_thres:
            return False

        # Initializing the top object.
        geom = state["obj_shape"]
        self.is_box = True if geom == p.GEOM_BOX else False
        self.dim = state["obj_dim"]

        # Compute object's half height based on the object shape.
        if self.is_box:
            self.half_height = self.dim[-1]  # BOX last dim is half H.
        else:
            self.half_height = self.dim[-1] / 2.0  # CYL last dim is H.

        # `o_pos` is the position of the COM; compute the position of the base
        # because the renderer (below) expects base position.
        base_position = list(o_pos).copy()
        base_position[2] -= self.half_height

        # Define a DashObject for the top object.
        o = DashObject(
            shape="box" if self.is_box else "cylinder",
            color=random.choice(self.colors),
            radius=self.dim[0],
            height=self.half_height * 2,
            position=base_position,
            orientation=o_quat,
        )

        # Render the object, and store the object ID.
        self.obj_id = self.renderer.render_object(o=o, base_mass=self.obj_mass)
        o.oid = self.obj_id

        # Add the top object to the dataset.
        self.dataset.track_object(o)

        p.changeDynamics(self.obj_id, -1, lateralFriction=1.0)
        # self.obj_mass = p.getDynamicsInfo(self.obj_id, -1)[0]

        return True

    def get_optimal_init_arm_q(self, desired_obj_pos):
        # TODO: desired obj init pos -> should add clearance to z.
        arm_q = None
        cost = 1e30
        ref = np.array([0.0] * 3 + [-1.57] + [0.0] * 3)
        for ind, cand_quat in enumerate(self.cand_quats):
            p_pos_of_ave, p_quat_of_ave = p.invertTransform(
                self.o_pos_pf_ave, self.o_quat_pf_ave
            )
            p_pos, p_quat = p.multiplyTransforms(
                desired_obj_pos, cand_quat, p_pos_of_ave, p_quat_of_ave
            )
            cand_arm_q = self.robot.solve_arm_IK(p_pos, p_quat)
            if cand_arm_q is not None:
                this_cost = np.sum(
                    np.abs(np.array(cand_arm_q) - ref)
                )  # change to l1
                if this_cost < cost:
                    arm_q = cand_arm_q
                    cost = this_cost
        return arm_q

    def sample_valid_arm_q(self):
        self.tz = self.btm_object.height if not self.place_floor else 0.0
        while True:
            if self.up:
                self.tx = self.np_random.uniform(low=0, high=0.3)
                self.ty = self.np_random.uniform(low=-0.1, high=0.5)
                # self.tx = self.np_random.uniform(low=0, high=0.2)
                # self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            else:
                self.tx = 0.0
                self.ty = 0.0

            desired_obj_pos = [
                self.tx,
                self.ty,
                self.start_clearance + self.tz,
            ]
            self.desired_obj_pos_final = [
                self.tx,
                self.ty,
                self.half_height + self.tz,
            ]
            arm_q = self.get_optimal_init_arm_q(desired_obj_pos)
            if arm_q is None:
                continue
            else:
                return arm_q

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0
        if self.gen_vision_dataset:
            self.dataset.reset()

        self.robot = InmoovShadowNew(
            init_noise=False, timestep=self._timeStep, np_random=self.np_random
        )

        arm_q = self.sample_valid_arm_q()  # reset done during solving IK

        init_done = False
        while not init_done:
            init_state = self.sample_init_state()
            init_done = self.reset_robot_object_from_sample(init_state, arm_q)

        if self.place_floor:
            self.bottom_obj_id = self.load_table()
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
        else:
            # Load the table.
            self.floor_id = self.load_table()
            self.renderer.color_object(
                oid=self.floor_id, color=self.table_object.color
            )

            # Create the bottom object.
            com_position = np.array([self.tx, self.ty, self.tz / 2])
            if self.init_noise:  # Optionally add noise.
                com_position += np.append(
                    self.np_random.uniform(low=-0.015, high=0.015, size=2), 0
                )
            self.btm_object.color = random.choice(self.colors)

            # Render the object.
            self.bottom_obj_id = p.loadURDF(
                fileName=os.path.join(currentdir, "assets/cylinder.urdf"),
                basePosition=com_position,
                useFixedBase=0,
            )
            self.renderer.color_object(
                oid=self.bottom_obj_id, color=self.btm_object.color
            )
            self.btm_object.oid = self.bottom_obj_id

            # Convert from COM to base position.
            base_position = com_position.copy()
            base_position[2] = 0.0  # Change from center to base Z.
            self.btm_object.position = base_position

            self.dataset.track_object(self.btm_object)  # Add to the dataset.

            # Change dynamics.
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
            p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)

        self.robot.change_hand_friction(1.7)

        p.stepSimulation()  # TODO

        # init obj pose
        self.t_pos, self.t_orn = p.getBasePositionAndOrientation(self.obj_id)
        self.b_pos, self.b_orn = p.getBasePositionAndOrientation(
            self.bottom_obj_id
        )
        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    def load_table(self):
        oid = p.loadURDF(
            fileName=os.path.join(currentdir, "assets/tabletop.urdf"),
            basePosition=self.table_object.position,
            useFixedBase=1,
        )
        return oid

    def sample_init_state(self):
        ran_ind = int(
            self.np_random.uniform(low=0, high=len(self.init_states) - 0.1)
        )
        return self.init_states[ran_ind]

    def __del__(self):
        p.disconnect()

    def step(self, action):
        for _ in range(self.frameSkip):
            # action is not in -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)   # TODO
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        # if upright, reward
        # if no contact, reward
        # if use small force, reward

        reward = 0.0
        clPos, clQuat = p.getBasePositionAndOrientation(self.obj_id)
        clVels = p.getBaseVelocity(self.obj_id)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])

        # we only care about the upright(z) direction
        z_axis, _ = p.multiplyTransforms(
            [0, 0, 0], clQuat, [0, 0, 1], [0, 0, 0, 1]
        )  # R_cl * unitz[0,0,1]
        rotMetric = np.array(z_axis).dot(np.array([0, 0, 1]))

        # enlarge 0.15 -> 0.45
        # xyzMetric = 1 - (np.minimum(np.linalg.norm(np.array(self.desired_obj_pos_final) - np.array(clPos)), 0.45) / 0.15)
        # TODO:tmp change to xy metric, allow it to free drop
        xyzMetric = 1 - (
            np.minimum(
                np.linalg.norm(
                    np.array(self.desired_obj_pos_final[:2])
                    - np.array(clPos[:2])
                ),
                0.45,
            )
            / 0.15
        )
        linV_R = np.linalg.norm(clLinV)
        angV_R = np.linalg.norm(clAngV)
        velMetric = 1 - np.minimum(linV_R + angV_R / 2.0, 5.0) / 5.0

        reward += np.maximum(rotMetric * 20 - 15, 0.0)
        # print(np.maximum(rotMetric * 20 - 15, 0.))
        reward += xyzMetric * 5
        # print(xyzMetric * 5)
        reward += velMetric * 5
        # print(velMetric * 5)

        total_nf = 0
        cps_floor = p.getContactPoints(self.obj_id, self.bottom_obj_id, -1, -1)
        for cp in cps_floor:
            total_nf += cp[9]
        if np.abs(total_nf) > (
            self.obj_mass * 4.0
        ):  # mg        # TODO:tmp contact force hack
            meaningful_c = True
            reward += 5.0
        else:
            meaningful_c = False
        #     # reward += np.abs(total_nf) / 10.

        # not used when placing on floor
        btm_vels = p.getBaseVelocity(self.bottom_obj_id)
        btm_linv = np.array(btm_vels[0])
        btm_angv = np.array(btm_vels[1])
        reward += (
            np.maximum(
                -np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0
            )
            * 0.3
        )
        # print(np.maximum(-np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0) * 0.3)

        diff_norm = self.robot.get_norm_diff_tar()
        reward += 15.0 / (diff_norm + 1.0)
        # print(15. / (diff_norm + 1.))

        anyHandContact = False
        hand_r = 0
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(self.obj_id, self.robot.arm_id, -1, i)
            if len(cps) == 0:
                hand_r += 1.0  # the fewer links in contact, the better
            else:
                anyHandContact = True
        # print(hand_r)
        reward += hand_r - 15

        if (
            rotMetric > 0.9
            and xyzMetric > 0.8
            and velMetric > 0.8
            and meaningful_c
        ):  # close to placing
            reward += 5.0
            # print("upright")
            if not anyHandContact:
                reward += 20
                # print("no hand con")

        # print("r_total", reward)

        obs = self.getExtendedObservation()

        # Save the example.
        if self.gen_vision_dataset:
            self.dataset.generate_example()

        return obs, reward, False, {}

    def obj6DtoObs(self, o_pos, o_orn):
        objObs = []
        o_pos = np.array(o_pos)
        o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        objObs.extend(list(self.perturb(o_pos, r=0.005)))
        objObs.extend(list(self.perturb(o_pos, r=0.005)))
        objObs.extend(list(self.perturb(o_rotmat, r=0.005)))
        return objObs

    def obj6DtoObs_UpVec(self, o_pos, o_orn):
        objObs = []
        o_pos = np.array(o_pos)
        if self.up:  # TODO:tmp
            o_pos -= [self.tx, self.ty, 0]
        o_pos = o_pos * 3.0  # TODO:tmp, scale up
        o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        o_upv = [o_rotmat[2], o_rotmat[5], o_rotmat[8]]
        objObs.extend(list(self.perturb(o_pos, r=0.04)))
        objObs.extend(list(self.perturb(o_pos, r=0.04)))
        objObs.extend(list(self.perturb(o_upv, r=0.04)))
        objObs.extend(list(self.perturb(o_upv, r=0.04)))
        return objObs

    # change to tar pos fin pos diff
    # change to tx ty diff

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation(diff_tar=True)

        if self.use_gt_6d:
            if self.obj_id is None:
                self.observation.extend(
                    self.obj6DtoObs_UpVec([0, 0, 0], [0, 0, 0, 1])
                )  # TODO
            else:
                if self.gt_only_init:
                    clPos, clOrn = self.t_pos, self.t_orn
                else:
                    clPos, clOrn = p.getBasePositionAndOrientation(self.obj_id)
                self.observation.extend(
                    self.obj6DtoObs_UpVec(clPos, clOrn)
                )  # TODO

            # if stacking & real-time, include bottom 6D
            if not self.place_floor and not self.gt_only_init:
                if self.bottom_obj_id is None:
                    self.observation.extend(
                        self.obj6DtoObs_UpVec([0, 0, 0], [0, 0, 0, 1])
                    )  # TODO
                else:
                    clPos, clOrn = p.getBasePositionAndOrientation(
                        self.bottom_obj_id
                    )
                    self.observation.extend(
                        self.obj6DtoObs_UpVec(clPos, clOrn)
                    )  # TODO

        curContact = []
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
            con_this_link = False
            for cp in cps:
                if cp[1] != cp[2]:  # not self-collision of the robot
                    con_this_link = True
                    break
            if con_this_link:
                curContact.extend([1.0])
            else:
                curContact.extend([-1.0])
        self.observation.extend(curContact)

        if self.up:
            xy = np.array([self.tx, self.ty])  # TODO: tx, ty wrt world origin
            self.observation.extend(list(self.perturb(xy, r=0.01)))
            self.observation.extend(list(self.perturb(xy, r=0.01)))
            self.observation.extend(list(xy))
            # this is the vision module one also used for reset/planning

        if self.random_shape:
            shape_info = 1.0 if self.is_box else -1.0
            self.observation.extend([shape_info])

        if self.random_size:
            self.observation.extend(
                [
                    self.half_height * 4
                    + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                    self.half_height * 4
                    + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                    self.half_height * 4
                    + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                ]
            )
            # this is the true half_height, vision module one will be noisy

        # if self.lastContact is not None:
        #     self.observation.extend(self.lastContact)
        # else:   # first step
        #     self.observation.extend(curContact)
        # self.lastContact = curContact.copy()

        # print("obv", self.observation)
        # print("max", np.max(np.abs(np.array(self.observation))))
        # print("min", np.min(np.abs(np.array(self.observation))))

        return self.observation

    def seed(self, seed=None):
        random.seed(seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = (
                self.np_random
            )  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s
