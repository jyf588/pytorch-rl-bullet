from .inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import time
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Note: we directly call this env without going through the gym wrapper.

class InmoovShadowHandDemoEnvV3():
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 init_noise=False,
                 noisy_obs=True,
                 timestep=1./240,
                 withVel=False,
                 seed=0):

        self.init_noise = init_noise
        self.noisy_obs = noisy_obs
        self._timeStep = timestep
        self.withVel = withVel
        self.renders = True
        self.timer = 0
        self.np_random = None
        self.robot = None
        self.viewer = None

        self.noisy_txty = 0.005
        self.noisy_obj_6d = 0.001   # TODO

        self.frameSkip = 3
        self.action_scale = np.array([0.004] * 7 + [0.008] * 17)  # shadow hand is 22-5=17dof

        self.seed(seed)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep, np_random=self.np_random)

    def __del__(self):
        pass    # TODO?
        # p.resetSimulation()
        # # p.setPhysicsEngineParameter(numSolverIterations=200)
        # p.setTimeStep(self._timeStep)
        # p.setGravity(0, 0, -10)
        # # p.disconnect()
        # # # self.sess.__del__()

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        if self.noisy_obs:
            return np.copy(np.array(arr) + self.np_random.uniform(low=-r, high=r, size=len(arr)))
        else:
            return np.array(arr)

    def obj6DtoObs(self, o_pos, o_quat):
        objObs = []
        o_pos = np.array(o_pos)
        o_rotmat = np.array(p.getMatrixFromQuaternion(o_quat))
        objObs.extend(list(self.perturb(o_pos, r=self.noisy_obj_6d)))
        objObs.extend(list(self.perturb(o_pos, r=self.noisy_obj_6d)))
        objObs.extend(list(self.perturb(o_rotmat, r=self.noisy_obj_6d)))
        return objObs

    def reset(self):
        self.timer = 0

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

    def get_robot_contact_obs(self):
        self.observation = self.robot.get_robot_observation(self.withVel)

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

        return self.observation

    def get_robot_contact_txty_obs(self, tx, ty):   # if we also know tx, ty from vision/reasoning
        self.get_robot_contact_obs()

        xy = np.array([tx, ty])
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))

        return self.observation

    def get_robot_obj6d_contact_txty_obs(self, tx, ty, t_pos, t_quat):
        # TODO: the ordering is not ideal, should append obj6d as last
        self.observation = self.robot.get_robot_observation(self.withVel)

        self.observation.extend(self.obj6DtoObs(t_pos, t_quat))

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

        xy = np.array([tx, ty])
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))

        return self.observation

    def get_robot_2obj6d_contact_txty_obs(self, tx, ty, t_pos, t_quat, b_pos, b_quat):
        # TODO: the ordering is not ideal, should append obj6d as last
        self.observation = self.robot.get_robot_observation(self.withVel)

        self.observation.extend(self.obj6DtoObs(t_pos, t_quat))
        self.observation.extend(self.obj6DtoObs(b_pos, b_quat))

        curContact = []
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
            con_this_link = False
            for cp in cps:
                if cp[1] != cp[2]:  # not self-collision of the robot
                    con_this_link = True
                    break
            if con_this_link:
                curContact.extend([1.0])
            else:
                curContact.extend([-1.0])
        self.observation.extend(curContact)

        xy = np.array([tx, ty])
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))

        return self.observation

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random = np.random
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return seed

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s