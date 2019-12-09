from .shadow_hand import ShadowHand

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# episode length 300 + 100

class ShadowHandGraspEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True):
        self.renders = renders
        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.np_random = None
        self.robot = None

        # TODO: tune this is not principled
        self.action_scale = np.array([0.004] * 3 + [0.004] * 3 + [0.01] * 17)  # shadow hand is 22-5=17dof

        self.frameSkip = 1

        self.cylinderInitPos = [0, 0, 0.101]    # initOri is identity      # 0.2/2

        self.robotInitBasePos = np.array(np.array([-0.17, 0.07, 0.13]))  # TODO: note, diff for different model

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
        self.timer = 0

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        if self.np_random is None:
            self.seed(0)    # used once temporarily, will be overwritten outside by env

        cyInit = np.array(self.cylinderInitPos) + np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)
        # cyInit = np.array(self.cylinderInitPos)
        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), cyInit, useFixedBase=0)
        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
        p.changeDynamics(self.cylinderId, -1, lateralFriction=3.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=3.0)
        self.robot = ShadowHand(self.robotInitBasePos)

        if self.np_random is not None:
            self.robot.np_random = self.np_random

        self.robot.reset()      # call at last to prevent collision at init

    def step(self, action):
        # action is in -1,1
        if action is not None:
            # print("act")
            # print(np.array(action))
            # print(np.array(action) * self.action_scale)
            # action = np.clip(np.array(action), -1, 1)   # TODO
            self.act = np.array(action)
            # if self.timer > 300:
            #     self.act[:6] = np.array([0.] * 6)
            #     self.act[:3] = np.array([0.5] * 3)
            self.robot.apply_action(self.act * self.action_scale)

        for _ in range(self.frameSkip):
            p.stepSimulation()

        # rewards is height of target object
        clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        handPos, handQuat = p.getBasePositionAndOrientation(self.robot.handId)
        handOri = p.getEulerFromQuaternion(handQuat)

        dist = np.linalg.norm(np.array(handPos) - self.robotInitBasePos - np.array(clPos))  # TODO
        if dist > 1: dist = 1
        if dist < 0.15: dist = 0.15

        reward = 3.0
        reward += -dist * 3.0

        for i in range(-1, p.getNumJoints(self.robot.handId)):
            cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
            if len(cps) > 0:
                # print(len(cps))
                reward += 5.0   # the more links in contact, the better

            # if i > 0 and i not in self.robot.activeDofs and i not in self.robot.lockDofs:   # i in [4,9,14,20,26]
            #     tipPos = p.getLinkState(self.robot.handId, i)[0]
            #     # print(tipPos)
            #     reward += -np.minimum(np.linalg.norm(np.array(tipPos) - np.array(clPos)), 0.5) * 1.0

        clVels = p.getBaseVelocity(self.cylinderId)
        # print(clVels)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])
        reward += np.maximum(-np.linalg.norm(clLinV) - np.linalg.norm(clAngV), -10.0) * 0.5

        if clPos[2] < -0.0 and self.timer > 300: # object dropped, do not penalize dropping when 0 gravity
            reward += -9.

        if self.renders:
            time.sleep(self._timeStep)

        self.timer += 1
        if self.timer > 300:
            p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=0)
            for i in range(-1, p.getNumJoints(self.robot.handId)):
                p.setCollisionFilterPair(self.floorId, self.robot.handId, -1, i, enableCollision=0)

        return self.getExtendedObservation(), reward, False, {}

    def getExtendedObservation(self):
        # TODO: odd
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py#L132
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
        for i in range(-1, p.getNumJoints(self.robot.handId)):
            cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
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

        # print("obv", self.observation)
        # print("max", np.max(np.abs(np.array(self.observation))))
        # print("min", np.min(np.abs(np.array(self.observation))))

        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def reset(self):
        self.sim_setup()
        #
        # cyInit = np.array(self.cylinderInitPos) + np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)
        # # cyInit = np.array(self.cylinderInitPos)
        # p.resetBasePositionAndOrientation(self.cylinderId,
        #                                   cyInit,
        #                                   p.getQuaternionFromEuler([0, 0, 0]))
        # p.changeDynamics(self.cylinderId, -1, lateralFriction=3.0)
        # p.changeDynamics(self.floorId, -1, lateralFriction=3.0)
        # p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=1)
        # for i in range(-1, p.getNumJoints(self.robot.handId)):
        #     p.setCollisionFilterPair(self.floorId, self.robot.handId, -1, i, enableCollision=1)
        # p.stepSimulation()
        self.timer = 0
        self.lastContact = None

        self.observation = self.getExtendedObservation()
        # print("post-reset", self.observation)
        return np.array(self.observation)

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s