from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import time
import numpy as np
import math
import pickle

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
asset_dir = os.path.join(currentdir, 'my_pybullet_envs/assets')

# episode length 400


class InmoovShadowHandPlaceVisualize():

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True,
                 is_box=False,       # box or cylinder
                 is_small=False,    # small or large
                 place_floor=True,     # place on floor or place on a large cylinder
                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        self.is_box = is_box
        self.is_small = is_small
        self.place_floor = place_floor

        # TODO: hardcoded here
        if self.is_box:
            if self.is_small:
                self.grasp_pi_name = '0120_box_s_1'
            else:
                self.grasp_pi_name = '0120_box_l_1'
        else:
            if self.is_small:
                self.grasp_pi_name = '0120_cyl_s_1'
            else:
                self.grasp_pi_name = '0120_cyl_l_0'

        self.half_obj_height = 0.065 if self.is_small else 0.09
        self.start_clearance = 0.14
        self.btm_obj_height = 0.18      # always place on larger one
        self.cand_angles = [0., 1.57, 3.14, -1.57]  # TODO: finer grid?
        self.cand_quats = [p.getQuaternionFromEuler([0, 0, cand_angle]) for cand_angle in self.cand_angles]

        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.np_random = None
        self.robot = None
        self.viewer = None
        self.timer = 0

        self.tx = -1    # dummy
        self.ty = -1    # dummy
        self.tz = -1    # dummy
        self.desired_obj_pos_final = None

        self.saved_file = None
        with open(os.path.join(asset_dir, 'place_init_dist/final_states_' + self.grasp_pi_name + '.pickle'),
                  'rb') as handle:
            self.saved_file = pickle.load(handle)
        assert self.saved_file is not None

        self.o_pos_pf_ave = self.saved_file['ave_obj_pos_in_palm']
        self.o_quat_pf_ave = self.saved_file['ave_obj_quat_in_palm']
        self.o_quat_pf_ave /= np.linalg.norm(self.o_quat_pf_ave)        # in case not normalized
        self.init_states = self.saved_file['init_states']  # a list of dicts

        # print(self.o_pos_pf_ave)
        # print(self.o_quat_pf_ave)
        # print(self.init_states[10])
        # print(self.init_states[51])
        # print(self.init_states[89])

        self.obj_id = None
        self.bottom_obj_id = None

        if self.np_random is None:
            self.seed(0)
        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep)
        if self.np_random is not None:
            self.robot.np_random = self.np_random

    def reset_robot_object_from_sample(self, state, arm_q):
        o_pos_pf = state['obj_pos_in_palm']
        o_quat_pf = state['obj_quat_in_palm']
        all_fin_q_init = state['all_fin_q']
        tar_fin_q_init = state['fin_tar_q']

        self.robot.reset_with_certain_arm_q_finger_states(arm_q, all_fin_q_init, tar_fin_q_init)

        p_pos, p_quat = self.robot.get_link_pos_quat(self.robot.ee_id)
        o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)

        if self.is_box:
            if self.is_small:
                self.obj_id = p.loadURDF(os.path.join(asset_dir, 'box_small.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
            else:
                self.obj_id = p.loadURDF(os.path.join(asset_dir, 'box.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
        else:
            if self.is_small:
                self.obj_id = p.loadURDF(os.path.join(asset_dir, 'cylinder_small.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
            else:
                self.obj_id = p.loadURDF(os.path.join(asset_dir, 'cylinder.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
        p.changeDynamics(self.obj_id, -1, lateralFriction=1.0)
        self.obj_mass = p.getDynamicsInfo(self.obj_id, -1)[0]

        return

    def get_optimal_init_arm_q(self, desired_obj_pos):
        arm_q = None
        cost = 1e30
        ref = np.array([0.] * 3 + [-1.57] + [0.] * 3)
        for ind, cand_quat in enumerate(self.cand_quats):
            p_pos_of_ave, p_quat_of_ave = p.invertTransform(self.o_pos_pf_ave, self.o_quat_pf_ave)
            p_pos, p_quat = p.multiplyTransforms(desired_obj_pos, cand_quat,
                                                 p_pos_of_ave, p_quat_of_ave)
            cand_arm_q = self.robot.solve_arm_IK(p_pos, p_quat)
            if cand_arm_q is not None:
                this_cost = np.sum(np.abs(np.array(cand_arm_q) - ref))  # change to l1
                if this_cost < cost:
                    arm_q = cand_arm_q
                    cost = this_cost
        return arm_q

    def sample_valid_arm_q(self):
        self.tz = self.btm_obj_height if not self.place_floor else 0.0
        while True:
            if self.up:
                self.tx = self.np_random.uniform(low=0, high=0.25)
                self.ty = self.np_random.uniform(low=-0.1, high=0.5)
                # self.tx = self.np_random.uniform(low=0, high=0.2)
                # self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            else:
                self.tx = 0.0
                self.ty = 0.0

            desired_obj_pos = [self.tx, self.ty, self.start_clearance + self.tz]
            self.desired_obj_pos_final = [self.tx, self.ty, self.half_obj_height + self.tz]
            arm_q = self.get_optimal_init_arm_q(desired_obj_pos)
            if arm_q is None:
                continue
            else:
                return arm_q

    def reset(self):
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0

        if self.np_random is None:
            self.seed(0)
        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep)
        if self.np_random is not None:
            self.robot.np_random = self.np_random

        arm_q = self.sample_valid_arm_q()   # reset done during solving IK
        init_state = self.sample_init_state()
        self.reset_robot_object_from_sample(init_state, arm_q)

        if self.place_floor:
            self.bottom_obj_id =p.loadURDF(os.path.join(asset_dir, 'tabletop.urdf'), [0.25, 0.2, 0.0],
                                             useFixedBase=1)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
        else:
            btm_xyz = np.array([self.tx, self.ty, self.tz/2.0])
            if self.init_noise:
                btm_xyz += np.append(self.np_random.uniform(low=-0.01, high=0.01, size=2), 0)
            self.bottom_obj_id = p.loadURDF(os.path.join(asset_dir, 'cylinder.urdf'),
                                            btm_xyz, useFixedBase=0)
            self.floor_id = p.loadURDF(os.path.join(asset_dir, 'tabletop.urdf'), [0.25, 0.2, 0.0],
                              useFixedBase=1)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
            p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)

        p.stepSimulation()      # TODO
        self.observation = self.getExtendedObservation()
        return np.array(self.observation)

    def sample_init_state(self):
        ran_ind = int(self.np_random.uniform(low=0, high=len(self.init_states) - 0.1))
        return self.init_states[ran_ind]

    def __del__(self):
        p.disconnect()

    # def getExtendedObservation(self):
    #     self.observation = self.robot.get_robot_observation()
    #
    #     self.use_gt_6d = True
    #
    #     if self.use_gt_6d:
    #         if self.obj_id is None:
    #             self.observation.extend([0.0]*(3+9+3))
    #         else:
    #             clPos, clOrn = p.getBasePositionAndOrientation(self.obj_id)
    #             clPos = np.array(clPos)
    #             clOrnMat = p.getMatrixFromQuaternion(clOrn)
    #             clOrnMat = np.array(clOrnMat)
    #
    #             self.observation.extend(list(clPos + self.np_random.uniform(low=-0.005, high=0.005, size=3)))
    #             self.observation.extend(list(clPos + self.np_random.uniform(low=-0.005, high=0.005, size=3)))
    #             self.observation.extend(list(clOrnMat + self.np_random.uniform(low=-0.005, high=0.005, size=9)))
    #         if not self.place_floor:
    #             if self.bottom_obj_id is None:
    #                 self.observation.extend([0.0] * (3 + 9 + 3))
    #             else:
    #                 clPos, clOrn = p.getBasePositionAndOrientation(self.bottom_obj_id)
    #                 clPos = np.array(clPos)
    #                 clOrnMat = p.getMatrixFromQuaternion(clOrn)
    #                 clOrnMat = np.array(clOrnMat)
    #
    #                 self.observation.extend(list(clPos + self.np_random.uniform(low=-0.005, high=0.005, size=3)))
    #                 self.observation.extend(list(clPos + self.np_random.uniform(low=-0.005, high=0.005, size=3)))
    #                 self.observation.extend(list(clOrnMat + self.np_random.uniform(low=-0.005, high=0.005, size=9)))
    #
    #     curContact = []
    #     for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
    #         cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
    #         con_this_link = False
    #         for cp in cps:
    #             if cp[1] != cp[2]:      # not self-collision of the robot
    #                 con_this_link = True
    #                 break
    #         if con_this_link:
    #             curContact.extend([1.0])
    #         else:
    #             curContact.extend([-1.0])
    #     self.observation.extend(curContact)
    #
    #     if self.up:
    #         xy = np.array([self.tx, self.ty])
    #         self.observation.extend(list(xy + self.np_random.uniform(low=-0.005, high=0.005, size=2)))
    #         self.observation.extend(list(xy + self.np_random.uniform(low=-0.005, high=0.005, size=2)))
    #         self.observation.extend(list(xy + self.np_random.uniform(low=-0.005, high=0.005, size=2)))
    #
    #     # if self.lastContact is not None:
    #     #     self.observation.extend(self.lastContact)
    #     # else:   # first step
    #     #     self.observation.extend(curContact)
    #     # self.lastContact = curContact.copy()
    #
    #     # print("obv", self.observation)
    #     # print("max", np.max(np.abs(np.array(self.observation))))
    #     # print("min", np.min(np.abs(np.array(self.observation))))
    #
    #     return self.observation

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random = np.random
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return seed

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s


if __name__ == "__main__":
    env = InmoovShadowHandPlaceVisualize(is_small=False, is_box=True, place_floor=False)
    env.seed(305)

    for _ in range(20):
        env.reset()
        input("press enter")

    p.disconnect()
