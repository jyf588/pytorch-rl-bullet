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

# episode length 400

# TODO: txyz will be given by vision module. tz is zero for grasping, obj frame at bottom.

class InmoovShadowHandDemoEnvNew(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 init_noise=False,
                 timestep=1./240):
        self.renders = True
        self.init_noise = init_noise
        self._timeStep = timestep

        self.np_random = None
        self.robot = None
        self.viewer = None

        # TODO: tune this is not principled
        self.frameSkip = 3
        self.action_scale = np.array([0.004] * 7 + [0.008] * 17)  # shadow hand is 22-5=17dof

        self.tx = -1.
        self.ty = -1.   # dummy

        user_answer = input("withVel?").lower().strip()     # TODO:tmp
        if user_answer == "1":
            self.withVel = True
        elif user_answer == "0":
            self.withVel = False
        else:
            self.withVel = None
        # self.withVel = input("withVel?")
        # self.withVel = bool(self.withVel)

        self.obj_id = None      # TODO:tmp , none if unknown

        if self.np_random is None:
            self.seed(0)  # used once temporarily, will be overwritten outside by env

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep)

        if self.np_random is not None:
            self.robot.np_random = self.np_random

        self.observation = self.getExtendedObservation()

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def __del__(self):
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        # p.disconnect()
        # # self.sess.__del__()

    def assign_estimated_obj_pos(self, x, y):
        self.tx = x
        self.ty = y

    def update_obj_id(self, obj_id):    # TODO:tmp
        self.obj_id = obj_id
        self.observation = self.getExtendedObservation()
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def reset(self):

        # if self.np_random is None:
        #     self.seed(0)    # used once temporarily, will be overwritten outside by env
        #
        # if self.robot is not None:
        #     if p.isConnected(0):
        #         p.removeBody(self.robot.arm_id)
        #         self.robot.arm_id = -123
        #     self.robot = None
        # self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep)
        #
        # if self.np_random is not None:
        #     self.robot.np_random = self.np_random

        # self.robot.reset_with_certain_arm_q([0.0]*len(self.robot.arm_dofs))

        # delete this will just fail
        # self.robot.reset_with_certain_arm_q([-7.60999597e-01, 3.05809706e-02, -5.82112526e-01,
        #                                 -1.40855264e+00, -6.49374902e-01, -2.42410664e-01,
        #                                 0.00000000e+00])

        self.timer = 0

        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    def step(self, action):
        for _ in range(self.frameSkip):
            # action is in not -1,1
            if action is not None:
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep)
            self.timer += 1

        return self.getExtendedObservation(), 0.0, False, {}

    def getExtendedObservation(self):
        # print(self.withVel)
        self.observation = self.robot.get_robot_observation(self.withVel)

        if self.obj_id is not None:     # TODO:tmp
            clPos, clOrn = p.getBasePositionAndOrientation(self.obj_id)
            clPos = np.array(clPos)
            clOrnMat = p.getMatrixFromQuaternion(clOrn)
            clOrnMat = np.array(clOrnMat)

            self.observation.extend(list(clPos + self.np_random.uniform(low=-0.001, high=0.001, size=3)))
            self.observation.extend(list(clPos + self.np_random.uniform(low=-0.001, high=0.001, size=3)))
            self.observation.extend(list(clOrnMat + self.np_random.uniform(low=-0.001, high=0.001, size=9)))
            # self.observation.extend(list(clPos ))
            # self.observation.extend(list(clPos ))
            # self.observation.extend(list(clOrnMat ))

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
        #     cps = p.getContactPoints(2, self.robot.arm_id, -1, i)
        #     if len(cps) > 0:
        #         curContact.extend([1.0])
        #         # print("touch!!!")
        #     else:
        #         curContact.extend([-1.0])
        # self.observation.extend(curContact)

        # TODO: from now on, assume always Univeral Arm Policy
        xy = np.array([self.tx, self.ty])
        self.observation.extend(list(xy + self.np_random.uniform(low=-0.001, high=0.001, size=2)))
        self.observation.extend(list(xy + self.np_random.uniform(low=-0.001, high=0.001, size=2)))
        self.observation.extend(list(xy + self.np_random.uniform(low=-0.001, high=0.001, size=2)))
        # self.observation.extend(list(xy))
        # self.observation.extend(list(xy))
        # self.observation.extend(list(xy))

        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s