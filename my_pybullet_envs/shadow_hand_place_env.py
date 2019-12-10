# output a lot of q, dq, obj(xyz/ori), dxyz?
# perturb a little
# p.simulation()

# should we do training on hand only or hand+arm
# depends on if IK is the problem or ID is the problem
# really strange, Reaching seems no problem, then probably is IK?

# later: change init xyz of wrist to right above object?
# later: remove cylinder info
# later: add box position noise

# should we warm-start from grasping policy?
# should not max out wrist force
# wrist force should be a informative measure, if hitting below object, then should stop.

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

# episode length 300 + 1

class ShadowHandPlaceEnv(gym.Env):
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

        # TODO: now DIFFERENT from grasping
        self.action_scale = np.array([0.004] * 3 + [0.004] * 3 + [0.01] * 17)  # shadow hand is 22-5=17dof

        self.frameSkip = 1

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

        self.save_qs = None
        with open('/home/yifengj/pytorch-rl-bullet/final_states_1209.pickle', 'rb') as handle:   # TODO hardcoded
            self.save_qs = pickle.load(handle)
        assert self.save_qs is not None

    def __del__(self):
        p.disconnect()

    def sim_setup(self):
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        if self.np_random is None:
            self.seed(0)    # used once, will be overwritten outside by env

        self.cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), useFixedBase=0)     # 0.2/2
        # self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/bottom_object.urdf'), [0.0, 0, -0.05],
                             useFixedBase=1)        # TODO: add noise later
        p.changeDynamics(self.cylinderId, -1, lateralFriction=1.0)      # TODO: 3.0/1.0
        p.changeDynamics(self.floorId, -1, lateralFriction=1.0)
        self.robot = ShadowHand(base_ll=np.array([-0.1, -0.1, -0.2]), base_ul=np.array([0.1, 0.1, 0.1]))

        if self.np_random is not None:
            self.robot.np_random = self.np_random

        self.robot.reset()  # TODO: leave this here for now and only modify env.reset()

    def step(self, action):
        # action is in -1,1
        if action is not None:
            # print("act")
            # print(np.array(action))
            # print(np.array(action) * self.action_scale)
            # action = np.clip(np.array(action), -1, 1)   # TODO
            self.act = np.array(action)
            self.robot.apply_action(self.act * self.action_scale)

        for _ in range(self.frameSkip):
            p.stepSimulation()

        reward = 0.
        clPos, clQuat = p.getBasePositionAndOrientation(self.cylinderId)
        clVels = p.getBaseVelocity(self.cylinderId)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])

        rotMetric = clQuat[3]*clQuat[3]     # 0 worse, 1/-1^2=1 means upright good
        xyzMetric = 1 - (np.minimum(np.linalg.norm(np.array([0, 0, 0.1]) - np.array(clPos)), 0.15) / 0.15)
        linV_R = np.linalg.norm(clLinV)
        angV_R = np.linalg.norm(clAngV)
        velMetric = 1 - np.minimum(linV_R + angV_R / 2.0, 5.0) / 5.0

        total_nf = 0
        cps_floor = p.getContactPoints(self.cylinderId, self.floorId, -1, -1)
        for cp in cps_floor:
            total_nf += cp[9]
        if np.abs(total_nf) > 70:       # TODO: 5kg * g=10
            meaningful_c = True
        else:
            meaningful_c = False
        # print(total_nf, meaningful_c)

        if rotMetric > 0.8 and xyzMetric > 0.8 and velMetric > 0.8 and meaningful_c:     # close to placing
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

        if self.renders:
            time.sleep(self._timeStep)

        self.timer += 1
        succeed = False
        obs = self.getExtendedObservation()
        if self.timer == 300:
            # this is slightly different from mountain car's sparse reward,
            # where you are only rewarded when reaching a certain state
            # this is saying you must be at certain state at certain time (after test)
            for i in range(-1, p.getNumJoints(self.robot.handId)):
                p.setCollisionFilterPair(self.cylinderId, self.robot.handId, -1, i, enableCollision=0)
            for _ in range(300):
                p.stepSimulation()
                if self.renders:
                    time.sleep(self._timeStep)
            clPosNow, _ = p.getBasePositionAndOrientation(self.cylinderId)
            if clPosNow[2] > 0.05:
                # succeed = True
                reward += 2000

        return obs, reward, False, {'s': succeed}

    def getExtendedObservation(self):
        # TODO: odd
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py#L132
        self.observation = self.robot.get_robot_observation()

        # clPos, clOrn = p.getBasePositionAndOrientation(self.cylinderId)
        # self.observation.extend(list(clPos))
        # self.observation.extend(list(clOrn))
        #
        # clVels = p.getBaseVelocity(self.cylinderId)
        # self.observation.extend(clVels[0])
        # self.observation.extend(clVels[1])

        # TODO: add contact wrench info, getConstraintState? MOst of the force are used to combat own gravity
        cf = np.array(p.getConstraintState(self.robot.cid))
        cf[:3] /= (self.robot.maxForce * 5)
        cf[3:] /= (self.robot.maxForce * 1)     # just in case there is not state normalization in ppo
        # TODO: would like to see cf increase as obj touch floor,
        # but if cf is mainly determined by control itself
        self.observation.extend(cf)

        self.observation.extend(self.act)

        # TODO: delete these for now (finger contact not seems important for releasing)
        # TODO: infact, finger torques might be more useful
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
        self.sim_setup()    # TODO: maybe we should do this

        initDone = False

        while not initDone:
            ran_ind = int(self.np_random.uniform(low=0, high=len(self.save_qs) - 0.1))
            save_q = self.save_qs[ran_ind]

            # TODO: for now make problem simpler by recentering wrist&cylinder pose
            save_q[0] -= save_q[-12]
            save_q[1] -= save_q[-11]
            save_q[-12] = 0
            save_q[-11] = 0

            self.robot.reset_to_q(save_q[:-12], needCorrection=False)

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

            cps = p.getContactPoints(bodyA=self.floorId)

            if len(cps) == 0:
                initDone = True
            # else:
            #     print("bad init q")

        self.timer = 0
        self.lastContact = None

        self.observation = self.getExtendedObservation()
        # print("post-reset", self.observation)
        return np.array(self.observation)

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s