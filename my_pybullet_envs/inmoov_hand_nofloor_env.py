from .inmoov_hand import InmoovHand

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# episode length 400

# TODO: about rotate policy, need to obtain rotating angle from motion planning

class InmoovHandGraspEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True):
        self.renders = renders
        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # TODO: tune this is not principled
        self.action_scale = np.array([0.005] * 3 + [0.005] * 3 + [0.015] * 17)

        self.frameSkip = 1

        self.cylinderInitPos = [0, 0, 0.105]    # initOri is identity

        # self.robotInitBasePos = np.array([-0.12, -0.00, 0.1])  # TODO: note, diff for different model
        self.robotInitBasePos = np.array([-0.15, 0.08, 0.1])

        self.sim_setup()

        # self.seed()    # TODO

        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        action_dim = len(self.action_scale)
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

        self.viewer = None
        self.timer = 0

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10.)

        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), self.cylinderInitPos, useFixedBase=0)     # 0.2/2
        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
        p.changeDynamics(self.cylinderId, -1, lateralFriction=3.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=3.0)
        self.robot = InmoovHand(self.robotInitBasePos)
        self.seed(0)    # TODO
        self.robot.reset()

    def step(self, action):
        # action is in -1,1
        if action is not None:
            # print("act")
            # print(np.array(action))
            # print(np.array(action) * self.action_scale)
            # action = np.clip(np.array(action), -1, 1)   # TODO
            self.robot.apply_action(np.array(action) * self.action_scale)

        for _ in range(self.frameSkip):
            p.stepSimulation()

        # rewards is height of target object
        clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        # closestPoints = p.getClosestPoints(self.cylinderId, self.robot.handId, 1000., -1, -1)
        # invClPos, invClQuat = p.invertTransform(clPos, clQuat)
        #
        handPos, handQuat = p.getBasePositionAndOrientation(self.robot.handId)
        handOri = p.getEulerFromQuaternion(handQuat)
        #
        dist = np.linalg.norm(np.array(handPos) - np.array([-0.05, 0.08, 0.1]) - np.array(clPos))
        if dist > 1: dist = 1
        if dist < 0.15: dist = 0.15

        reward = 3.0
        reward += -dist * 3.0

        # reward += -self.robot.get_three_finger_deviation() * 0.1

        for i in range(-1, p.getNumJoints(self.robot.handId)):
            cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
            if len(cps) > 0:
                # print(len(cps))
                reward += 5.0   # the more links in contact, the better

            # if i > 0 and i in [3, 7, 11, 16, 22]:   # TODO: hardcoded
            #     # this is fingertip
            #     tipPos = p.getLinkState(self.robot.handId, i)[0]
            #     reward += -np.minimum(np.linalg.norm(np.array(tipPos) - np.array(clPos)), 0.5) * 2.0

        clVels = p.getBaseVelocity(self.cylinderId)
        # print(clVels)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])
        reward += np.maximum(-np.linalg.norm(clLinV) - np.linalg.norm(clAngV), -10.0) * 0.3

        if clPos[2] < -0.2 and self.timer > 300: # object dropped, do not penalize dropping when 0 gravity
            reward += -7.

        if self.renders:
            time.sleep(self._timeStep)

        self.timer += 1
        if self.timer > 300:
            # p.setGravity(0, 0, -10)  # turn on gravity
            p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=0)

        return self.getExtendedObservation(), reward, False, {}

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation()

        # TODO: no vel as well
        clPos, clOrn = p.getBasePositionAndOrientation(self.cylinderId)
        clPos = np.array(clPos)
        # clPos = np.array(clPos) + self.np_random.uniform(low=-0.005, high=0.005, size=3)
        clOrnMat = p.getMatrixFromQuaternion(clOrn)
        clOrnMat = np.array(clOrnMat)
        # clOrnMat = np.array(clOrnMat) + self.np_random.uniform(low=-0.02, high=0.02, size=9)
        self.observation.extend(list(clPos))
        self.observation.extend(list(clOrnMat))

        clVels = p.getBaseVelocity(self.cylinderId)
        self.observation.extend(clVels[0])
        self.observation.extend(clVels[1])

        for i in range(-1, p.getNumJoints(self.robot.handId)):
            cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
            if len(cps) > 0:
                self.observation.extend([1.0])
            else:
                self.observation.extend([-1.0])

        # print("obv", self.observation)

        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def reset(self):
        # print("pre-reset", self.getExtendedObservation())
        # input("press enter")

        # self.sim_setup()

        self.robot.reset()
        # p.setGravity(0, 0, 0)
        cyInit = np.array(self.cylinderInitPos)
        # cyInit = np.array(self.cylinderInitPos) + np.append(self.np_random.uniform(low=-0.03, high=0.03, size=2), 0)
        p.resetBasePositionAndOrientation(self.cylinderId,
                                          cyInit,
                                          p.getQuaternionFromEuler([0, 0, 0]))
        p.changeDynamics(self.cylinderId, -1, lateralFriction=3.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=3.0)
        p.setCollisionFilterPair(self.cylinderId, self.floorId, -1, -1, enableCollision=1)
        p.stepSimulation()
        self.timer = 0

        # if self.sim_reset_counter > 0 and (self.sim_reset_counter % self.sim_full_restart_freq) == 0:
        # if False:
        #     # deepMimic reset
        #     # ran_ind = int(self.np_random.uniform(low=0, high=150-0.1))
        #     # self.cylinderInitPos[2] = self.obj_ref_z[ran_ind] + 0.02    # TODO: add height might not be necessary
        #     # self.robotInitBasePos[2] = self.hand_ref_z[ran_ind]
        #
        #     self.sim_setup()
        #     # TODO: seems save and restore states are necessary. but why?
        # else:
        #     self.robot.reset()
        #     # cyInit = np.array(self.cylinderInitPos) + np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)
        #     # cyInit = np.array(self.cylinderInitPos)
        #     # p.resetBasePositionAndOrientation(self.cylinderId,
        #     #                                   cyInit,
        #     #                                   p.getQuaternionFromEuler([0, 0, 0]))
        #     # p.stepSimulation()        # TODO
        # self.sim_reset_counter += 1

        self.observation = self.getExtendedObservation()
        # print("post-reset", self.observation)
        return np.array(self.observation)

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s