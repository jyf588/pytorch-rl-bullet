from .inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class InmoovShadowHandGraspEnvV3(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True,
                 is_box=True,
                 is_small=False,
                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        self.is_box = is_box
        self.is_small = is_small
        self.half_obj_height = 0.065 if self.is_small else 0.09

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

        self.final_states = []  # wont be cleared unless call clear function

        # TODO: tune this is not principled
        self.frameSkip = 3
        self.action_scale = np.array([0.004] * 7 + [0.008] * 17)  # shadow hand is 22-5=17dof

        self.tx = None
        self.ty = None
        self.tz = 0.0

        # !--    ref shadowhand_motor_simple_nomass.urdf-->
        # <!--    p.invertTransform([-0.18, 0.095, 0.11], p.getQuaternionFromEuler([1.8, -1.57, 0]))-->
        self.o_pos_pf_ave = [-0.10985665023326874, -0.15379363298416138, 0.1334318220615387]
        self.o_quat_pf_ave = [0.5541163086891174, -0.4393695592880249, 0.5536751747131348, -0.4397195875644684]

        self.reset()

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def __del__(self):
        p.disconnect()

    def get_optimal_init_arm_q(self, desired_obj_pos):
        # TODO: desired obj init pos -> should add clearance to z.
        arm_q = None
        cost = 1e30
        ref = np.array([0.] * 3 + [-1.57] + [0.] * 3)
        for ind, cand_quat in enumerate(self.cand_quats):
            p_pos_of_ave, p_quat_of_ave = p.invertTransform(self.o_pos_pf_ave, self.o_quat_pf_ave)
            p_pos, p_quat = p.multiplyTransforms(desired_obj_pos, cand_quat,
                                                 p_pos_of_ave, p_quat_of_ave)
            cand_arm_q = self.robot.solve_arm_IK(p_pos, p_quat)
            if cand_arm_q is not None:
                diff = np.abs(np.array(cand_arm_q) - ref)
                diff[-1] *= 2
                this_cost = np.sum(diff)  # change to l1
                if this_cost < cost:
                    arm_q = cand_arm_q
                    cost = this_cost
        return arm_q

    def sample_valid_arm_q(self):
        while True:
            if self.up:
                self.tx = self.np_random.uniform(low=0.05, high=0.3)  # sample xy
                self.ty = self.np_random.uniform(low=-0.2, high=0.6)
                # self.tx = self.np_random.uniform(low=0, high=0.2)
                # self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            else:
                self.tx = 0.0
                self.ty = 0.0

            desired_obj_pos = [self.tx, self.ty, self.tz]
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
            self.seed(0)    # used once temporarily, will be overwritten outside by env
        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep)
        if self.np_random is not None:
            self.robot.np_random = self.np_random

        arm_q = self.sample_valid_arm_q()   # reset done during solving IK
        self.robot.reset_with_certain_arm_q(arm_q)

        cylinderInitPos = [self.tx, self.ty, self.half_obj_height+0.001]
        cyl_init_pos = np.array(cylinderInitPos)
        if self.init_noise:
            cyl_init_pos += np.append(self.np_random.uniform(low=-0.015, high=0.015, size=2), 0)
        if self.is_box:
            if self.is_small:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/box_small.urdf'),
                                             cyl_init_pos, useFixedBase=0)
            else:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/box.urdf'),
                                             cyl_init_pos, useFixedBase=0)
        else:
            if self.is_small:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder_small.urdf'),
                                             cyl_init_pos, useFixedBase=0)
            else:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                             cyl_init_pos, useFixedBase=0)

        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'),
                                  [0, 0, 0], useFixedBase=1)
        p.changeDynamics(self.cylinderId, -1, lateralFriction=1.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=1.0)

        p.stepSimulation()  # TODO
        self.observation = self.getExtendedObservation()
        return np.array(self.observation)

    def step(self, action):
        if self.timer > 100*self.frameSkip:
            p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=0)
            # for i in range(-1, p.getNumJoints(self.robot.arm_id)):
            #     p.setCollisionFilterPair(self.floorId, self.robot.arm_id, -1, i, enableCollision=0)

        for _ in range(self.frameSkip):
            # action is in -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)   # TODO
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep)
            self.timer += 1

        reward = 3.0

        # rewards is height of target object
        clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
        dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(clPos)), 0.5)
        reward += -dist * 2.0
        reward += -np.minimum(np.linalg.norm(np.array([self.tx, self.ty, 0.1]) - np.array(clPos)), 0.4) * 4.0

        for i in self.robot.fin_tips[:4]:
            tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
            reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.5)  # 4 finger tips
        tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]      # thumb tip
        reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.5) * 5.0

        # for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, i)
        #     if len(cps) > 0:
        #         # print(len(cps))
        #         reward += 5.0   # the more links in contact, the better

        # cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, self.robot.ee_id)    # palm
        # if len(cps) > 0: reward += 4.0
        # for dof in np.copy(self.robot.fin_actdofs)[[0,1, 3,4, 6,7, 9,10]]:
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, dof)
        #     if len(cps) > 0:  reward += 2.0
        # for dof in np.copy(self.robot.fin_actdofs)[[2, 5, 8, 11]]:
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, dof)
        #     if len(cps) > 0:  reward += 3.5
        # for dof in self.robot.fin_actdofs[12:16]:         # thumb
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, dof)
        #     if len(cps) > 0:  reward += 10.0
        # cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, self.robot.fin_actdofs[16])    # thumb tip
        # if len(cps) > 0:  reward += 17.5

        cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, self.robot.ee_id)    # palm
        if len(cps) > 0: reward += 5.0
        f_bp = [0, 3, 6, 9, 12, 17]     # 3*4+5
        for ind_f in range(5):
            con = False
            # try onl reward distal and middle
            # for dof in self.robot.fin_actdofs[f_bp[ind_f]:f_bp[ind_f+1]]:
            for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 3):f_bp[ind_f + 1]]:
                cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, dof)
                if len(cps) > 0:  con = True
            if con:  reward += 5.0
            if con and ind_f == 4: reward += 20.0        # reward thumb even more

        clVels = p.getBaseVelocity(self.cylinderId)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])
        reward += np.maximum(-np.linalg.norm(clLinV) - np.linalg.norm(clAngV), -10.0) * 0.2

        if clPos[2] < -0.0 and self.timer > 300: # object dropped, do not penalize dropping when 0 gravity
            reward += -15.

        return self.getExtendedObservation(), reward, False, {}

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation()

        # clPos, clOrn = p.getBasePositionAndOrientation(self.cylinderId)
        # clPos = np.array(clPos)
        # clOrnMat = p.getMatrixFromQuaternion(clOrn)
        # clOrnMat = np.array(clOrnMat)
        #
        # self.observation.extend(list(clPos))
        # # self.observation.extend(list(clOrnMat))
        # self.observation.extend(list(clOrn))
        #
        # clVels = p.getBaseVelocity(self.cylinderId)
        # self.observation.extend(clVels[0])
        # self.observation.extend(clVels[1])

        # curContact = []
        # for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, i)
        #     if len(cps) > 0:
        #         curContact.extend([1.0])
        #         # print("touch!!!")
        #     else:
        #         curContact.extend([-1.0])
        # self.observation.extend(curContact)

        curContact = []
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
            con_this_link = False
            for cp in cps:
                if cp[1] != cp[2]:      # not self-collision of the robot
                    con_this_link = True
                    break
            if con_this_link:
                curContact.extend([1.0])
            else:
                curContact.extend([-1.0])
        self.observation.extend(curContact)

        if self.up:
            xy = np.array([self.tx, self.ty])   # TODO: tx, ty wrt world origin
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.005, high=0.005, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.005, high=0.005, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.005, high=0.005, size=2)))

        # if self.lastContact is not None:
        #     self.observation.extend(self.lastContact)
        # else:   # first step
        #     self.observation.extend(curContact)
        # self.lastContact = curContact.copy()

        # print("obv", self.observation)
        # print("max", np.max(np.abs(np.array(self.observation))))
        # print("min", np.min(np.abs(np.array(self.observation))))

        return self.observation

    def append_final_state(self):
        # output obj in palm frame (no need to output palm frame in world)
        # output finger q's, finger tar q's.
        # velocity will be assumed to be zero at the end of transporting phase
        # return a dict.
        obj_pos, obj_quat = p.getBasePositionAndOrientation(self.cylinderId)      # w2o
        hand_pos, hand_quat = self.robot.get_link_pos_quat(self.robot.ee_id)    # w2p
        inv_h_p, inv_h_q = p.invertTransform(hand_pos, hand_quat)       # p2w
        o_p_hf, o_q_hf = p.multiplyTransforms(inv_h_p, inv_h_q, obj_pos, obj_quat)  # p2w*w2o

        fin_q, _ = self.robot.get_q_dq(self.robot.all_findofs)

        state = {'obj_pos_in_palm': o_p_hf, 'obj_quat_in_palm': o_q_hf,
                 'all_fin_q': fin_q, 'fin_tar_q': self.robot.tar_fin_q}
        self.final_states.append(state)

    def clear_final_states(self):
        self.final_states = []

    def calc_average_obj_in_palm(self):
        count = len(self.final_states)
        o_pos_hf_sum = np.array([0., 0, 0])
        o_quat_hf_sum = np.array([0., 0, 0, 0])
        for dict in self.final_states:
            o_pos_hf_sum += np.array(dict['obj_pos_in_palm'])
            o_quat_hf_sum += np.array(dict['obj_quat_in_palm'])
        o_pos_hf_sum /= count
        o_quat_hf_sum /= count      # rough estimate of quat average
        return list(o_pos_hf_sum), list(o_quat_hf_sum)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]


    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s