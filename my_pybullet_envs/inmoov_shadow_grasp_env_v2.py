from .inmoov_shadow_hand_v2 import InmoovShadowNew
from .inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSession
from .inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSessionFlexWrist

import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# episode length 400

# TODO: txyz will be given by vision module. tz is zero for grasping, obj frame at bottom.


class InmoovShadowHandGraspEnvNew(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True,
                 is_box=False,
                 small=False,
                 using_comfortable=True,
                 using_comfortable_range=True):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        self.isBox = is_box
        self.small = small
        self.using_comfortable = using_comfortable
        self.using_comfortable_range = using_comfortable_range
        self.vary_angle_range = 0.6

        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)     # this session seems always 0
        self.np_random = None
        self.robot = None
        self.viewer = None

        self.final_states = []  # wont be cleared unless call clear function

        # TODO: tune this is not principled
        self.frameSkip = 3
        self.action_scale = np.array([0.004] * 7 + [0.008] * 17)  # shadow hand is 22-5=17dof

        self.tx = None
        self.ty = None

        self.reset()    # and update init

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def __del__(self):
        p.disconnect()
        # self.sess.__del__()

    def get_reset_poses_comfortable(self):
        # return/ sample one arm_q (or palm 6D, later), est. obj init,
        # during testing, another central Bullet session will be calc a bunch of arm_q given a obj init pos
        assert self.up

        sess = ImaginaryArmObjSession()
        cyl_init_pos = None
        arm_q = None
        while arm_q is None:
            if self.small:
                cyl_init_pos = [0, 0, 0.071]
            else:
                cyl_init_pos = [0, 0, 0.091]
            self.tx = self.np_random.uniform(low=0, high=0.25)
            self.ty = self.np_random.uniform(low=-0.1, high=0.5)
            # self.tx = 0.1
            # self.ty = 0.0
            cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])

            arm_q, _ = sess.get_most_comfortable_q_and_refangle(self.tx, self.ty)
        # print(arm_q)
        return arm_q, cyl_init_pos

    def get_reset_poses_comfortable_range(self):
        assert self.up
        assert self.using_comfortable

        sess = ImaginaryArmObjSessionFlexWrist()
        cyl_init_pos = None
        arm_q = None
        while arm_q is None:
            if self.small:
                cyl_init_pos = [0, 0, 0.071]
            else:
                cyl_init_pos = [0, 0, 0.091]
            self.tx = self.np_random.uniform(low=0, high=0.25)
            self.ty = self.np_random.uniform(low=-0.1, high=0.5)    # TODO 0.6?
            # self.tx = 0.1
            # self.ty = 0.0
            cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
            vary_angle = self.np_random.uniform(low=-self.vary_angle_range, high=self.vary_angle_range)
            arm_q = sess.sample_one_comfortable_q(self.tx, self.ty, vary_angle)
        # print(arm_q)
        return arm_q, cyl_init_pos

    def get_reset_poses_old(self):
        # old way. return (modify) the palm 6D and est. obj init
        init_palm_quat = p.getQuaternionFromEuler([1.8, -1.57, 0])
        if self.small:
            cyl_init_pos = [0, 0, 0.071]
            init_palm_pos = [-0.18, 0.095, 0.075]   # absorbed by imaginary session
        else:
            cyl_init_pos = [0, 0, 0.091]
            init_palm_pos = [-0.18, 0.095, 0.11]

        if self.up:
            self.tx = self.np_random.uniform(low=0, high=0.2)
            self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            # self.tx = 0.18
            # self.ty = -0.18
            cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
            init_palm_pos = np.array(init_palm_pos) + np.array([self.tx, self.ty, 0])
        return init_palm_pos, init_palm_quat, cyl_init_pos

    def reset(self):
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        if self.np_random is None:
            self.seed(0)    # used once temporarily, will be overwritten outside by env

        if self.using_comfortable:
            if self.using_comfortable_range:
                arm_q, cyl_init_pos = self.get_reset_poses_comfortable_range()
            else:
                arm_q, cyl_init_pos = self.get_reset_poses_comfortable()
        else:
            init_palm_pos, init_palm_quat, cyl_init_pos = self.get_reset_poses_old()

        # cyInit = np.array(cyl_init_pos)
        if self.init_noise:
            cyl_init_pos += np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)

        if self.isBox:
            if self.small:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/box_small.urdf'),
                                             cyl_init_pos, useFixedBase=0)
            else:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/box.urdf'),
                                             cyl_init_pos, useFixedBase=0)
        else:
            if self.small:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cyl_small.urdf'),
                                             cyl_init_pos, useFixedBase=0)
            else:
                self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                             cyl_init_pos, useFixedBase=0)

        # self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'),
        #                           [0, 0, 0], useFixedBase=1)
        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.25, 0.1, 0.0],
                              useFixedBase=1)  # TODO
        p.changeDynamics(self.cylinderId, -1, lateralFriction=1.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=1.0)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep)

        if self.np_random is not None:
            self.robot.np_random = self.np_random

        if self.using_comfortable:
            self.robot.reset_with_certain_arm_q(arm_q)
        else:
            self.robot.reset(list(init_palm_pos), init_palm_quat)       # reset at last to test collision
        #
        # tmp_id = p.loadURDF(os.path.join(currentdir,
        #                     "assets/inmoov_ros/inmoov_description/robots/inmoov_arm_v2_2_reaching_BB.urdf"),
        #                          [-0.30, 0.348, 0.272], p.getQuaternionFromEuler([0,0,0]),
        #                          # flags=p.URDF_USE_INERTIA_FROM_FILE,        # TODO
        #                          flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        #                                | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        #                          useFixedBase=1)
        # arm_dofs = [0, 1, 2, 3, 4, 6, 7]
        # for ind in range(len(arm_dofs)):
        #     p.resetJointState(tmp_id, arm_dofs[ind], arm_q[ind], 0.0)
        # input("press enter")

        self.timer = 0
        self.lastContact = None
        self.observation = self.getExtendedObservation()

        # # TODO
        # obj_p, obj_q = p.getBasePositionAndOrientation(self.cylinderId)
        # th_tip_p, th_tip_q = self.robot.get_link_pos_quat(self.robot.fin_tips[4])
        # inv_p, inv_q = p.invertTransform(th_tip_p, th_tip_q)
        # print(p.multiplyTransforms(inv_p, inv_q, obj_p, [0,0,0,1]))
        # # ((0.031040966510772705, -0.038185857236385345, 0.08783391863107681),
        # # (0.030863940715789795, -0.03799811005592346, 0.08814594149589539
        #
        # obj_p, obj_q = p.getBasePositionAndOrientation(self.cylinderId)
        # th_tip_p, th_tip_q = self.robot.get_link_pos_quat(self.robot.fin_tips[3])
        # inv_p, inv_q = p.invertTransform(th_tip_p, th_tip_q)
        # print(p.multiplyTransforms(inv_p, inv_q, obj_p, [0,0,0,1]))
        # # (0.02415177971124649, -0.05398867651820183, 0.0798909068107605)
        # # (0.023912787437438965, -0.05410301685333252, 0.07958468049764633)
        # input("press enter")

        return np.array(self.observation)

    def step(self, action):
        for _ in range(self.frameSkip):
            # action is in -1,1
            if action is not None:
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
            # for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 2):f_bp[ind_f + 1]]:
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

        if self.timer > 300:
            p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=0)
            # for i in range(-1, p.getNumJoints(self.robot.arm_id)):
            #     p.setCollisionFilterPair(self.floorId, self.robot.arm_id, -1, i, enableCollision=0)

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

        # curContact = []
        # for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, i)
        #     if len(cps) > 0:
        #         curContact.extend([1.0])
        #         # print("touch!!!")
        #     else:
        #         curContact.extend([-1.0])
        # self.observation.extend(curContact)

        if self.up:
            xy = np.array([self.tx, self.ty])
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))

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
        # print(state)
        # print(self.robot.get_joints_last_tau(self.robot.all_findofs))
        # self.robot.get_wrist_wrench()
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
        o_quat_hf_sum /= np.linalg.norm(o_quat_hf_sum)      # normalize quat
        return list(o_pos_hf_sum), list(o_quat_hf_sum)

    def calc_average_obj_in_palm_rot_invariant(self):
        count = len(self.final_states)
        o_pos_hf_sum = np.array([0., 0, 0])
        o_unitz_hf_sum = np.array([0., 0, 0])
        for dict in self.final_states:
            o_pos_hf_sum += np.array(dict['obj_pos_in_palm'])
            unitz_hf = p.multiplyTransforms([0, 0, 0], dict['obj_quat_in_palm'], [0, 0, 1], [0, 0, 0, 1])[0]
            o_unitz_hf_sum += np.array(unitz_hf)
        o_pos_hf_sum /= count
        o_unitz_hf_sum /= count      # rough estimate of unit z average
        o_unitz_hf_sum /= np.linalg.norm(o_unitz_hf_sum)      # normalize unit z vector

        x, y, z = o_unitz_hf_sum
        a1_solved = np.arcsin(-y)
        a2_solved = np.arctan2(x, z)
        # a3_solved is zero since equation has under-determined
        quat_solved = p.getQuaternionFromEuler([a1_solved, a2_solved, 0])

        uz_check = p.multiplyTransforms([0, 0, 0], quat_solved, [0, 0, 1], [0, 0, 0, 1])[0]
        assert np.linalg.norm(np.array(o_unitz_hf_sum) - np.array((uz_check))) < 1e-3

        return list(o_pos_hf_sum), list(quat_solved)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s