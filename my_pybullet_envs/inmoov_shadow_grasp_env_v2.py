from .inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# episode length 400


class InmoovShadowHandGraspEnvNew(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up

        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.np_random = None
        self.robot = None
        self.lastContact = None

        self.viewer = None
        self.timer = 0

        # TODO: tune this is not principled
        self.action_scale = np.array([0.004] * 7 + [0.01] * 17)  # shadow hand is 22-5=17dof

        self.frameSkip = 1

        self.tx = None
        self.ty = None

        self.sim_setup()

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        if self.np_random is None:
            self.seed(0)    # used once temporarily, will be overwritten outside by env

        self.cylinderInitPos = [0, 0, 0.101]
        self.robotInitPalmPos = [-0.18, 0.105, 0.11]

        if self.up:
            self.tx = self.np_random.uniform(low=0, high=0.2)
            self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            self.cylinderInitPos = np.array(self.cylinderInitPos) + np.array([self.tx, self.ty, 0])
            self.robotInitPalmPos = np.array(self.robotInitPalmPos) + np.array([self.tx, self.ty, 0])

        cyInit = np.array(self.cylinderInitPos)
        if self.init_noise:
            cyInit += np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)

        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder_heavier.urdf'),
                                     cyInit, useFixedBase=0)
        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'),
                                  [0, 0, 0], useFixedBase=1)
        p.changeDynamics(self.cylinderId, -1, lateralFriction=1.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=1.0)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep)

        if self.np_random is not None:
            self.robot.np_random = self.np_random

        self.robot.reset(list(self.robotInitPalmPos) + [1.8, -1.57, 0])

    def step(self, action):
        # action is in -1,1
        if action is not None:
            # action = np.clip(np.array(action), -1, 1)   # TODO
            self.act = action
            self.robot.apply_action(self.act * self.action_scale)

        for _ in range(self.frameSkip):
            p.stepSimulation()

        # rewards is height of target object
        clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        handPos, handQuat = self.robot.get_link_pos_quat(self.robot.ee_id)

        dist = np.linalg.norm(np.array(handPos) - np.array(self.robotInitPalmPos) - np.array(clPos))
        if dist > 1: dist = 1
        if dist < 0.1: dist = 0.1

        reward = 3.0
        reward += -dist * 3.0

        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, i)
            if len(cps) > 0:
                # print(len(cps))
                reward += 5.0   # the more links in contact, the better

            # if i > 0 and i not in self.robot.activeDofs and i not in self.robot.lockDofs:   # i in [4,9,14,20,26]
            #     tipPos = p.getLinkState(self.robot.handId, i)[0]
            #     # print(tipPos)
            #     reward += -np.minimum(np.linalg.norm(np.array(tipPos) - np.array(clPos)), 0.5) * 1.0

        clVels = p.getBaseVelocity(self.cylinderId)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])
        reward += np.maximum(-np.linalg.norm(clLinV) - np.linalg.norm(clAngV), -10.0) * 0.2

        if clPos[2] < -0.0 and self.timer > 300: # object dropped, do not penalize dropping when 0 gravity
            reward += -9.

        if self.renders:
            time.sleep(self._timeStep)

        self.timer += 1
        if self.timer > 300:
            p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=0)
            for i in range(-1, p.getNumJoints(self.robot.arm_id)):
                p.setCollisionFilterPair(self.floorId, self.robot.arm_id, -1, i, enableCollision=0)

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
            cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, i)
            if len(cps) > 0:
                curContact.extend([1.0])
                # print("touch!!!")
            else:
                curContact.extend([-1.0])
        self.observation.extend(curContact)

        if self.up:
            xy = np.array(self.cylinderInitPos)[:2]
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
        return np.array(self.observation)

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s