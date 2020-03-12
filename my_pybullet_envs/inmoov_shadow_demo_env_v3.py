from .inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import time
import numpy as np
import math

import os
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)

# Note: we directly call this env without going through the gym wrapper.


class InmoovShadowHandDemoEnvV3:
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(
        self,
        init_noise=False,
        noisy_obs=False,
        timestep=1.0 / 240,
        withVel=False,
        seed=0,
        control_skip=3,
    ):

        self.init_noise = init_noise
        self.noisy_obs = noisy_obs
        self._timeStep = timestep
        self.withVel = withVel
        self.diffTar = False
        self.renders = True
        self.timer = 0
        self.np_random = None
        self.robot = None
        self.viewer = None

        self.noisy_txty = 0.005
        self.noisy_obj_6d = 0.001  # TODO

        self.control_skip = control_skip
        # shadow hand is 22-5=17dof
        self.action_scale = np.array(
            [0.012 / self.control_skip] * 7 + [0.024 / self.control_skip] * 17
        )

        self.seed(seed)

        self.robot = InmoovShadowNew(
            init_noise=self.init_noise,
            timestep=self._timeStep,
            np_random=self.np_random,
        )

    def __del__(self):
        pass  # TODO?
        # p.resetSimulation()
        # # p.setPhysicsEngineParameter(numSolverIterations=200)
        # p.setTimeStep(self._timeStep)
        # p.setGravity(0, 0, -10)
        # # p.disconnect()
        # # # self.sess.__del__()

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        if self.noisy_obs:
            return np.copy(
                np.array(arr)
                + self.np_random.uniform(low=-r, high=r, size=len(arr))
            )
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

    def obj6DtoObs_UpVec(self, o_pos, o_orn, tx, ty):
        objObs = []
        o_pos = np.array(o_pos)
        o_pos -= [tx, ty, 0]
        o_pos = o_pos * 3.0
        o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        o_upv = [o_rotmat[2], o_rotmat[5], o_rotmat[8]]
        objObs.extend(list(self.perturb(o_pos, r=0.02)))
        objObs.extend(list(self.perturb(o_pos, r=0.02)))
        objObs.extend(list(self.perturb(o_upv, r=0.01)))
        objObs.extend(list(self.perturb(o_upv, r=0.01)))
        return objObs

    def obj_pos_and_up_to_obs(self, o_pos, o_upv, tx, ty):
        objObs = []
        o_pos = np.array(o_pos)
        o_pos -= [tx, ty, 0]
        o_pos = o_pos * 3.0
        objObs.extend(list(self.perturb(o_pos, r=0.02)))
        objObs.extend(list(self.perturb(o_pos, r=0.02)))
        objObs.extend(list(self.perturb(o_upv, r=0.01)))
        objObs.extend(list(self.perturb(o_upv, r=0.01)))
        return objObs

    def reset(self):  # deprecated
        self.timer = 0

    def step(self, action):
        for _ in range(self.control_skip):
            # action is in not -1,1
            if action is not None:
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.6)
            self.timer += 1

    def get_robot_contact_obs(self):
        self.observation = self.robot.get_robot_observation(
            self.withVel, self.diffTar
        )

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

        return self.observation

    def get_robot_contact_txty_obs(
        self, tx, ty
    ):  # if we also know tx, ty from vision/reasoning
        self.get_robot_contact_obs()

        # xy = np.array([tx, ty])
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        xy = np.array([tx, ty])
        self.observation.extend(list(self.perturb(xy, r=0.01)))
        self.observation.extend(list(self.perturb(xy, r=0.01)))
        self.observation.extend(list(xy))

        return self.observation

    def get_robot_contact_txty_halfh_obs(self, tx, ty, half_height):
        self.observation = self.get_robot_contact_txty_obs(tx, ty)
        self.observation.extend(
            [
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
            ]
        )  # TODO
        return self.observation

    def get_robot_contact_txty_halfh_obs_nodup(self, tx, ty, half_h):
        self.get_robot_contact_obs()
        self.observation.extend([tx, ty])
        self.observation.extend([tx, ty])
        self.observation.extend([half_h])
        return self.observation

    def get_robot_contact_txty_halfh_2obj6dUp_obs_nodup(
        self, tx, ty, half_h, t_pos, t_quat, b_pos, b_quat
    ):
        self.get_robot_contact_txty_halfh_obs_nodup(tx, ty, half_h)
        self.observation.extend(self.obj6DtoObs_UpVec(t_pos, t_quat, tx, ty))
        self.observation.extend(self.obj6DtoObs_UpVec(b_pos, b_quat, tx, ty))
        return self.observation

    def get_robot_contact_txty_halfh_2obj6dUp_obs_nodup_from_up(
        self, tx, ty, half_h, t_pos, t_up, b_pos, b_up
    ):
        self.get_robot_contact_txty_halfh_obs_nodup(tx, ty, half_h)
        self.observation.extend(
            self.obj_pos_and_up_to_obs(t_pos, t_up, tx, ty)
        )
        self.observation.extend(
            self.obj_pos_and_up_to_obs(b_pos, b_up, tx, ty)
        )
        return self.observation

    def get_robot_obj6d_contact_txty_obs(self, tx, ty, t_pos, t_quat):
        # TODO: the ordering is not ideal, should append obj6d as last
        self.observation = self.robot.get_robot_observation(
            self.withVel, self.diffTar
        )

        self.observation.extend(self.obj6DtoObs(t_pos, t_quat))

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

    def get_robot_2obj6d_contact_txty_obs(
        self, tx, ty, t_pos, t_quat, b_pos, b_quat
    ):
        # TODO: the ordering is not ideal, should append obj6d as last
        self.observation = self.robot.get_robot_observation(
            self.withVel, self.diffTar
        )

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

    def get_robot_2obj6dUp_contact_txty_halfh_obs(
        self, tx, ty, t_pos, t_quat, b_pos, b_quat, half_height
    ):
        """
        Args:
            tx: Target x position.
            ty: Target y position.
            t_pos: The xyz position of the top object.
            t_quat: The orientation of the top object, in xyzw quaternion 
                format.
            b_pos: The xyz position of the bottom object.
            b_quat: The orientation of the bottom object, in xyzw quaternion 
                format.
            half_height: Half of the height of the top object.
        """
        self.observation = self.robot.get_robot_observation(
            self.withVel, self.diffTar
        )

        self.observation.extend(self.obj6DtoObs_UpVec(t_pos, t_quat, tx, ty))
        self.observation.extend(self.obj6DtoObs_UpVec(b_pos, b_quat, tx, ty))

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

        # xy = np.array([tx, ty])
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        xy = np.array([tx, ty])
        self.observation.extend(list(self.perturb(xy, r=0.01)))
        self.observation.extend(list(self.perturb(xy, r=0.01)))
        self.observation.extend(list(xy))

        self.observation.extend(
            [
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
            ]
        )  # TODO

        return self.observation

    def get_robot_2obj6dUp_contact_txty_halfh_obs_from_up(
        self, tx, ty, t_pos, t_up, b_pos, b_up, half_height
    ):
        """Note that this differs from the 
        get_robot_2obj6dUp_contact_txty_halfh_obs() function in that up vectors
        are provided, instead of orientation. The reasoning for this is that we
        want to support vision, which predicts up vectors directly.

        Args:
            tx: Target x position.
            ty: Target y position.
            t_pos: The xyz position of the top object.
            t_up: The up vector of the top object.
            b_pos: The xyz position of the bottom object.
            b_up: The up vector of the bottom object.
            half_height: Half of the height of the top object.
        """
        self.observation = self.robot.get_robot_observation(
            self.withVel, self.diffTar
        )

        self.observation.extend(
            self.obj_pos_and_up_to_obs(t_pos, t_up, tx, ty)
        )
        self.observation.extend(
            self.obj_pos_and_up_to_obs(b_pos, b_up, tx, ty)
        )

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

        # xy = np.array([tx, ty])
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        # self.observation.extend(list(self.perturb(xy, r=self.noisy_txty)))
        xy = np.array([tx, ty])
        self.observation.extend(list(self.perturb(xy, r=0.01)))
        self.observation.extend(list(self.perturb(xy, r=0.01)))
        self.observation.extend(list(xy))

        self.observation.extend(
            [
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
                half_height * 4
                + self.np_random.uniform(low=-0.02, high=0.02) * 2,
            ]
        )  # TODO

        return self.observation

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random = np.random
        if self.robot is not None:
            self.robot.np_random = (
                self.np_random
            )  # use the same np_randomizer for robot as for env
        return seed

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s
