from .shadow_hand_velc_simple import ShadowHandVel

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# episode length 300 + 100

class ShadowHandGraspEnvVelC(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 collect_final_state=False,
                 init_noise=True,
                 act_noise=False):
        self.renders = renders
        self.collect_final_state = collect_final_state
        self.init_noise = init_noise
        self.act_noise = act_noise

        self._timeStep = 1. / 240.
        self.timer = 0
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.np_random = None
        self.robot = None

        # TODO: tune this is not principled
        self.action_scale = np.array([0.001] * 3 + [0.0015] * 3 + [0.004] * 17)

        self.frameSkip = 4

        self.cylinderInitPos = [-0.0, -0.0, 0.101]    # TODO 0.02 # initOri is identity      # 0.2/2

        self.sim_setup()

        self.lastContact = None

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

        self.viewer = None

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=200)

        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        if self.np_random is None:
            self.seed(0)    # used once temporarily, will be overwritten outside by env

        cyInit = np.array(self.cylinderInitPos)
        if self.init_noise:
            cyInit += np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)

        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder_heavier.urdf'),
                                     cyInit, useFixedBase=0)
        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
        p.changeDynamics(self.cylinderId, -1, lateralFriction=3.0, spinningFriction=1.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=3.0, spinningFriction=1.0)

        self.robot = ShadowHandVel(init_noise=self.init_noise, act_noise=self.act_noise,
                                   timestep=self._timeStep)

        if self.np_random is not None:
            self.robot.np_random = self.np_random

        self.robot.reset()      # call at last to prevent collision at init

    def step(self, action):

        reward = 3.0 * self.frameSkip
        for _ in range(self.frameSkip):
            if action is not None:
                # action = np.clip(np.array(action), -1., 1)  # TODO: action could go beyond [-1,1]
                # action = np.array(action)
                self.act = action
                # print(self.act[:3])
                self.robot.apply_action(self.act * self.action_scale)

            p.stepSimulation()
            # print("t", self.timer)
            # print("last wrist torque", self.robot.get_wrist_last_torque())
            # print("last finger torque", self.robot.get_fingers_last_torque())
            clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
            for i in range(self.robot.endEffectorId + 1, p.getNumJoints(self.robot.handId)):
                cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
                if len(cps) > 0:
                    reward += 5.0  # the more links in contact, the better
            clVels = p.getBaseVelocity(self.cylinderId)
            clLinV = np.array(clVels[0])
            clAngV = np.array(clVels[1])
            reward += np.maximum(-np.linalg.norm(clLinV) - np.linalg.norm(clAngV), -10.0) * 0.5
            # reward += -self.robot.get_wrist_torque_pen()
            if clPos[2] < -0.0 and self.timer > 300:  # object dropped, do not penalize dropping when 0 gravity
                reward += -13.
            if self.renders:
                time.sleep(self._timeStep)

            self.timer += 1
            if self.timer > 300 and not self.collect_final_state:
                # training
                p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=0)

        #
        # reward = 4.0 * self.frameSkip
        # for _ in range(self.frameSkip):
        #     p.stepSimulation()
        #
        #     # rewards is height of target object
        #     clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        #     handPos, _ = self.robot.get_palm_pos_orn()
        #
        #     dist = np.linalg.norm(np.array(handPos) - np.array(clPos))  # TODO
        #     if dist > 1: dist = 1
        #     if dist < 0.15: dist = 0.15
        #
        #     reward += -dist * 3.0
        #
        #     one_contact = False
        #     for i in range(self.robot.endEffectorId + 1, p.getNumJoints(self.robot.handId)):
        #         cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
        #         if len(cps) > 0:
        #             one_contact = True
        #             # print(len(cps))
        #             reward += 4.0  # the more links in contact, the better
        #     if one_contact:
        #         reward += 10
        #
        #     reward += -self.robot.get_wrist_torque_pen()
        #
        #     # # TODO: is position better?
        #     reward += np.maximum(-np.linalg.norm(np.array(clPos) - np.array(self.cylinderInitPos)), -0.2) * 5.0
        #     # clVels = p.getBaseVelocity(self.cylinderId)
        #     # # print(clVels)
        #     # clLinV = np.array(clVels[0])
        #     # clAngV = np.array(clVels[1])
        #     # reward += np.maximum(-np.linalg.norm(clLinV) - np.linalg.norm(clAngV), -10.0) * 0.5
        #
        #     # wm, _ = self.robot.get_wrist_q_dq()
        #     # wm = np.array([wm[0]+0.1, wm[1], wm[2], wm[3]/2, wm[4]/2, wm[5]/2])
        #     # reward += np.maximum(-np.linalg.norm(wm), -1.0) * 3.0
        #
        #     if clPos[2] < -0.0 and self.timer > 300:  # object dropped, do not penalize dropping when 0 gravity
        #         reward += -11.
        #
        #     if self.renders:
        #         time.sleep(self._timeStep)
        #
        #     self.timer += 1
        #     if self.timer > 300 and not self.collect_final_state:
        #         # training
        #         p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=0)

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
        #
        # if self.collect_final_state and self.timer > 300:
        #     # testing       # TODO
        #     self.observation[2] = 0.13
        #     self.observation[-5] -= (self.observation[2] - 0.13)

        curContact = []
        for i in range(self.robot.endEffectorId+1, p.getNumJoints(self.robot.handId)):   # ex palm aux
            cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
            # cps2 = p.getContactPoints(self.floorId, self.robot.handId, -1, i)
            if len(cps) > 0:
                curContact.extend([1.0])
            else:
                curContact.extend([-1.0])
        self.observation.extend(curContact)
        if self.lastContact is not None:
            self.observation.extend(self.lastContact)
        else:   # first step
            self.observation.extend(curContact)
        self.lastContact = curContact.copy()

        # print(self.observation)
        # input("press enetr")

        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def reset(self):
        self.sim_setup()

        self.timer = 0
        self.lastContact = None

        self.observation = self.getExtendedObservation()
        #
        # for j in range(-1, p.getNumJoints(self.robot.handId)):
        #     if j > 0 and j not in self.robot.activeDofs and j not in self.robot.lockDofs:   # i in [4,9,14,20,26]
        #         tipPos = p.getLinkState(self.robot.handId, j)[0]
        #         print(tipPos)

        # print("post-reset", self.observation)
        return np.array(self.observation)

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s