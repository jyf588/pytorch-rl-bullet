# output a lot of q, dq, obj(xyz/ori), dxyz?
# perturb a little
# p.simulation()

# should we do training on hand only or hand+arm
# depends on if IK is the problem or ID is the problem
# really strange, Reaching seems no problem, then probably is IK?

# later: change init xyz of wrist to right above object?


from .shadow_hand import ShadowHand

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math
import pickle

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# episode length 500

class ShadowHandPlaceEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True):
        self.renders = renders
        self._timeStep = 1. / 480.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.np_random = None

        # TODO: now DIFFERENT from grasping
        self.action_scale = np.array([0.002] * 3 + [0.002] * 3 + [0.01] * 17)  # shadow hand is 22-5=17dof

        self.frameSkip = 2

        self.cylinderInitPos = [0.1, 0, 0.105]    # TODO:dummy

        self.robotInitBasePos = np.array(np.array([-0.17, 0.07, 0.1]))  # TODO:dummy

        self.sim_setup()

        self.lastContact = None

        # self.seed()    # TODO

        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        action_dim = len(self.action_scale)
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

        self.viewer = None

        self.timer = 0

        self.save_qs = None
        with open('/home/yifengj/pytorch-rl-bullet/final_states.pickle', 'rb') as handle:   # TODO hardcoded
            self.save_qs = pickle.load(handle)
        assert self.save_qs is not None

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -5)

        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), self.cylinderInitPos, useFixedBase=0)     # 0.2/2
        # self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/bottom_object.urdf'), [0.0, 0, -0.05],
                             useFixedBase=1)        # TODO: add noise later
        p.changeDynamics(self.cylinderId, -1, lateralFriction=1.0)      # TODO: 3.0/1.0
        p.changeDynamics(self.floorId, -1, lateralFriction=1.0)
        self.robot = ShadowHand(self.robotInitBasePos, base_ll=np.array([-0.1, -0.1, -0.2]), base_ul=np.array([0.1, 0.1, 0.1]))
        if self.np_random is not None:
            self.robot.np_random = self.np_random
        if self.np_random is None:
            self.seed(0)
        self.robot.reset()

        # TODO: leave this here for now and only modify env.reset()

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

        reward = 0.
        clPos, clQuat = p.getBasePositionAndOrientation(self.cylinderId)
        clVels = p.getBaseVelocity(self.cylinderId)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])

        # handPos, handQuat = p.getBasePositionAndOrientation(self.robot.handId)
        # handOri = p.getEulerFromQuaternion(handQuat)
        #
        # if clPos[2] < -0.0: # object dropped, do not penalize dropping when 0 gravity
        #     reward += -1.
        #
        # if clPos[2] < 0.05 and self.timer > 300: # object dropped, do not penalize dropping when 0 gravity
        #     reward += -6.

        # cps = p.getContactPoints(bodyA=self.cylinderId, bodyB=self.floorId)
        # print(cps)
        # if len(cps) >= 3:
        #     reward += 1.
        #     print("good1")
        #     cps_hand = p.getContactPoints(bodyA=self.robot.handId)
        #     if len(cps_hand) == 0:
        #         reward += 1.
        #         print("good2")

        rotMetric = clQuat[3]*clQuat[3]     # 0 worse, 1/-1^2=1 means upright good
        xyzMetric = 1 - (np.minimum(np.linalg.norm(np.array([0, 0, 0.1]) - np.array(clPos)), 0.15) / 0.15)
        linV_R = np.linalg.norm(clLinV)
        angV_R = np.linalg.norm(clAngV)
        velMetric = 1 - np.minimum(linV_R + angV_R / 2.0, 5.0) / 5.0

        # TODO: add contact wrench info, getConstraintState? MOst of the force are used to combat own gravity
        if rotMetric > 0.8 and xyzMetric > 0.8 and velMetric > 0.8:     # close to placing
            for i in range(-1, p.getNumJoints(self.robot.handId)):
                cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
                if len(cps) == 0:
                    reward += 0.5   # the fewer links in contact, the better
        # else:
        #     for i in range(-1, p.getNumJoints(self.robot.handId)):
        #         cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
        #         if len(cps) > 0:
        #             reward += 0.5   # the more links in contact, the better

        reward += rotMetric * 5
        reward += xyzMetric * 5
        reward += velMetric * 5

        # # rewards is height of target object
        # clPos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        # handPos, handQuat = p.getBasePositionAndOrientation(self.robot.handId)
        # handOri = p.getEulerFromQuaternion(handQuat)
        #
        # dist = np.linalg.norm(np.array(handPos) - self.robotInitBasePos - np.array(clPos))  # TODO
        # if dist > 1: dist = 1
        # if dist < 0.15: dist = 0.15
        #
        # reward = 3.0
        # reward += -dist * 3.0
        #
        # for i in range(-1, p.getNumJoints(self.robot.handId)):
        #     cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
        #     if len(cps) > 0:
        #         # print(len(cps))
        #         reward += 5.0   # the more links in contact, the better
        #
        #     # if i > 0 and i not in self.robot.activeDofs and i not in self.robot.lockDofs:   # i in [4,9,14,20,26]
        #     #     tipPos = p.getLinkState(self.robot.handId, i)[0]
        #     #     # print(tipPos)
        #     #     reward += -np.minimum(np.linalg.norm(np.array(tipPos) - np.array(clPos)), 0.5) * 1.0
        #
        # clVels = p.getBaseVelocity(self.cylinderId)
        # # print(clVels)
        # clLinV = np.array(clVels[0])
        # clAngV = np.array(clVels[1])
        #
        # if clPos[2] < -0.2 and self.timer > 300: # object dropped, do not penalize dropping when 0 gravity
        #     reward += -7.

        if self.renders:
            time.sleep(self._timeStep)

        self.timer += 1
        if self.timer == 300:
            for i in range(-1, p.getNumJoints(self.robot.handId)):
                p.setCollisionFilterPair(self.cylinderId, self.robot.handId, -1, i, enableCollision=0)
            for _ in range(400):
                p.stepSimulation()
                if self.renders:
                    time.sleep(self._timeStep)
            clPosNow, _ = p.getBasePositionAndOrientation(self.cylinderId)
            if clPosNow[2] > 0.05:
                reward += 1000

        return self.getExtendedObservation(), reward, False, {}

    def getExtendedObservation(self):
        # TODO: odd
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py#L132
        self.observation = self.robot.get_robot_observation()

        clPos, clOrn = p.getBasePositionAndOrientation(self.cylinderId)
        self.observation.extend(list(clPos))
        self.observation.extend(list(clOrn))

        clVels = p.getBaseVelocity(self.cylinderId)
        self.observation.extend(clVels[0])
        self.observation.extend(clVels[1])

        # TODO: delete these for now (finger contact not seems important for releasing)
        # curContact = []
        # for i in range(-1, p.getNumJoints(self.robot.handId)):
        #     cps = p.getContactPoints(self.cylinderId, self.robot.handId, -1, i)
        #     if len(cps) > 0:
        #         curContact.extend([1.0])
        #     else:
        #         curContact.extend([-1.0])
        # self.observation.extend(curContact)
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
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def reset(self):

        # self.sim_setup()    # TODO: maybe we should do this

        ran_ind = int(self.np_random.uniform(low=0, high=len(self.save_qs) - 0.1))
        save_q = self.save_qs[ran_ind]

        # TODO: for now make problem simpler by recentering wrist&cylinder pose
        save_q[0] -= save_q[-12]
        save_q[1] -= save_q[-11]
        save_q[-12] = 0
        save_q[-11] = 0

        self.robot.reset_to_q(save_q[:-12])

        cyInit = save_q[-12:-9]
        cyOri = save_q[-9:-6]
        cyQuat = p.getQuaternionFromEuler(cyOri)
        cyLinVel = save_q[-6:-3]
        cyAngVel = save_q[-3:]
        p.resetBasePositionAndOrientation(self.cylinderId, cyInit, cyQuat)
        p.resetBaseVelocity(self.cylinderId, cyLinVel, cyAngVel)

        for i in range(-1, p.getNumJoints(self.robot.handId)):
            p.setCollisionFilterPair(self.cylinderId, self.robot.handId, -1, i, enableCollision=1)
        p.stepSimulation()
        self.timer = 0
        self.lastContact = None

        self.observation = self.getExtendedObservation()
        # print("post-reset", self.observation)
        return np.array(self.observation)

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s