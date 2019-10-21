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
                 renders=False):
        self.renders = renders
        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # TODO: tune this is not principled
        self.action_scale = np.array([0.005] * 3 + [0.01] * 3 + [0.015] * 16)

        # self.sim_reset_counter = 0
        # self.sim_full_restart_freq = 200      # TODO: test this
        self.frameSkip = 1

        self.cylinderInitPos = [0, 0, 0.105]    # initOri is identity
        # self.robotInitBasePos = np.array(np.array([-0.08, -0.05, 0.1]))  # TODO: note, diff for different model
        self.robotInitBasePos = np.array(np.array([-0.23, -0.00, 0.1]))  # TODO: note, diff for different model

        # TODO: do we really need action/obv space?
        self.sim_setup()

        # self.obj_ref_z = np.array(list(np.linspace(0.105, 0.105, num=150)) +
        #                             list(np.linspace(0.1, 0.5, num=100)))
        # self.hand_ref_z = np.array(list(np.linspace(0.5, 0.1, num=120)) + list(np.linspace(0.1, 0.1, num=30)) +
        #                              list(np.linspace(0.1, 0.5, num=100)))
        # self.ref_len = 250

        # self.seed()    # TODO

        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        action_dim = len(self.action_scale)
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

        self.viewer = None

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), self.cylinderInitPos, useFixedBase=0)     # 0.2/2
        p.changeDynamics(self.cylinderId, -1, lateralFriction=3.0, rollingFriction=1.0)
        self.robot = AllegroHand(self.robotInitBasePos)
        self.robot.reset()
        p.stepSimulation()    # TODO

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
        clPos, clQuat = p.getBasePositionAndOrientation(self.cylinderId)
        # closestPoints = p.getClosestPoints(self.cylinderId, self.robot.handId, 1000., -1, -1)
        # invClPos, invClQuat = p.invertTransform(clPos, clQuat)
        #
        handPos, handQuat = p.getBasePositionAndOrientation(self.robot.handId)
        handOri = p.getEulerFromQuaternion(handQuat)
        #
        dist = np.linalg.norm(np.array(handPos) - np.array(clPos))
        if dist > 1: dist = 1
        if dist < 0.15: dist = 0.15

        # reward = -1000

        # TODO: deepmimic
        # why target q is not part of state, only phase is?
        # pretty sure we should init from various phases
        # should we reward obj traj? should we reward hand traj?

        reward = 0.0
        reward += -dist

        # reward -= np.linalg.norm(np.array(handPos[:2]) - np.array([-0.08, 0])) * 2.0
        # reward -= np.linalg.norm(np.array(clPos[:2])) * 10.0
        # reward -= np.linalg.norm(np.array(handPos[2]) - np.array(clPos[2])) * 5.0
        # reward -= np.linalg.norm(np.array(handOri) - np.array([1.57, 0, 0])) * 1.0
        #
        # reward -= self.robot.get_finger_dist_from_init() * 3.0
        # # numPt = len(closestPoints)
        # # if numPt > 0:
        # #     # reward += -closestPoints[0][8] * 10  # [8] is the distance, positive for separation
        # #     reward += -dist * 10
        #
        # cps = p.getContactPoints(self.cylinderId, self.robot.handId)
        # counters = [0]*4
        # for cp in cps:
        #     # print(cp[5], cp[6])
        #     # print(cp[9])
        #     if abs(cp[9]) < 10.0:
        #         continue
        #     if cp[6][0] > clPos[0] and cp[6][1] > clPos[1]:
        #         counters[0] += 1
        #     if cp[6][0] > clPos[0] and cp[6][1] < clPos[1]:
        #         counters[1] += 1
        #     if cp[6][0] < clPos[0] and cp[6][1] > clPos[1]:
        #         counters[2] += 1
        #     if cp[6][0] < clPos[0] and cp[6][1] < clPos[1]:
        #         counters[3] += 1
        #
        # good_grasp = False
        # if counters[0] > 0 and counters[3] > 0:
        #     good_grasp = True
        # if counters[1] > 0 and counters[2] > 0:
        #     good_grasp = True
        #
        # if good_grasp:
        #     reward += 7.0
        #
        # if 0.3 < clPos[2] < 0.8:
        #     reward += 15.0

        if 0.3 < clPos[2] < 0.8:
            reward += 15.0 - 15.0*np.linalg.norm(clPos - np.array([0, 0, 0.6]))

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

        # TODO: no vel as well
        clPos, clOrn = p.getBasePositionAndOrientation(self.cylinderId)
        clPos = np.array(clPos)
        # clPos = np.array(clPos) + self.np_random.uniform(low=-0.005, high=0.005, size=3)
        clOrnMat = p.getMatrixFromQuaternion(clOrn)
        clOrnMat = np.array(clOrnMat)
        # clOrnMat = np.array(clOrnMat) + self.np_random.uniform(low=-0.02, high=0.02, size=9)
        # dir0 = [clOrnMat[0], clOrnMat[3], clOrnMat[6]]
        # dir1 = [clOrnMat[1], clOrnMat[4], clOrnMat[7]]
        # dir2 = [clOrnMat[2], clOrnMat[5], clOrnMat[8]]
        self.observation.extend(list(clPos))
        self.observation.extend(list(clOrnMat))

        clVels = p.getBaseVelocity(self.cylinderId)
        self.observation.extend(clVels[0])
        self.observation.extend(clVels[1])

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
        cyInit = np.array(self.cylinderInitPos)
        p.resetBasePositionAndOrientation(self.cylinderId,
                                          cyInit,
                                          p.getQuaternionFromEuler([0, 0, 0]))
        p.stepSimulation()

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