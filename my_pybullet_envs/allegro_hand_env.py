from .allegro_hand import AllegroHand

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class AllegroHandPickEnv(gym.Env):
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
        self.action_scale = np.array([0.005] * 3 + [0.02] * 3 + [0.015] * 16)

        self.sim_reset_counter = 0
        self.sim_full_restart_freq = 200      # TODO: test this
        self.frameSkip = 1

        self.cylinderInitPos = [0, 0, 0.105]    # initOri is identity

        # TODO: do we really need action/obv space?
        self.sim_setup()

        self.seed()

        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        action_dim = len(self.action_scale)
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        # TODO: 10.0?
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

        #
        self.viewer = None

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0])
        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), self.cylinderInitPos)     # 0.2/2
        p.changeDynamics(self.cylinderId, -1, lateralFriction=3.0, rollingFriction=1.0)
        self.robot = AllegroHand()
        p.stepSimulation()

    def step(self, action):
        # action is in -1,1
        if action is not None:
            # print("act")
            # print(np.array(action))
            # print(np.array(action) * self.action_scale)
            self.robot.apply_action(np.array(action) * self.action_scale)

        for _ in range(self.frameSkip):
            p.stepSimulation()

        # rewards is height of target object
        clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        # closestPoints = p.getClosestPoints(self.cylinderId, self.robot.handId, 1000., -1, -1)

        handPos, _ = p.getBasePositionAndOrientation(self.robot.handId)

        dist = np.linalg.norm(np.array(handPos) - np.array(clPos))
        if dist > 1: dist = 1
        if dist < 0.15: dist = 0.15

        # reward = -1000

        reward = 0.5
        reward += -dist
        # numPt = len(closestPoints)
        # if numPt > 0:
        #     # reward += -closestPoints[0][8] * 10  # [8] is the distance, positive for separation
        #     reward += -dist * 10
        if 0.2 < clPos[2] < 0.8:
            reward += 7.0
            # print("successfully grasped!!!")
            # print("self._envStepCounter")
            # print(self._envStepCounter)
            # print("self._envStepCounter")
            # print(self._envStepCounter)
            # print("reward")
            # print(reward)
        # print("reward")
        # print(reward)

        if self.renders:
            time.sleep(self._timeStep)

        return self.getExtendedObservation(), reward, False, {}

    def getExtendedObservation(self):
        # TODO: odd
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py#L132
        self.observation = self.robot.get_robot_observation()

        clPos, clOrn = p.getBasePositionAndOrientation(self.cylinderId)

        clPos = np.array(clPos) + self.np_random.uniform(low=-0.005, high=0.005, size=3)

        clOrnMat = p.getMatrixFromQuaternion(clOrn)
        clOrnMat = np.array(clOrnMat) + self.np_random.uniform(low=-0.02, high=0.02, size=9)
        # dir0 = [clOrnMat[0], clOrnMat[3], clOrnMat[6]]
        # dir1 = [clOrnMat[1], clOrnMat[4], clOrnMat[7]]
        # dir2 = [clOrnMat[2], clOrnMat[5], clOrnMat[8]]
        self.observation.extend(list(clPos))
        self.observation.extend(list(clOrnMat))

        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def reset(self):
        # if self.sim_reset_counter > 0 and (self.sim_reset_counter % self.sim_full_restart_freq) == 0:
        if False:
            self.sim_setup()
        else:
            self.robot.reset()
            cyInit = np.array(self.cylinderInitPos) + np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)
            p.resetBasePositionAndOrientation(self.cylinderId,
                                              cyInit,
                                              p.getQuaternionFromEuler([0, 0, 0]))
            p.stepSimulation()
        self.sim_reset_counter += 1

        self.observation = self.getExtendedObservation()
        return np.array(self.observation)
