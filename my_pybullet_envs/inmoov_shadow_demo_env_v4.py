from .inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import time
import numpy as np
import os
import inspect

from . import utils

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)

# Note: we directly call this env without going through the gym wrapper.


class InmoovShadowHandDemoEnvV4:
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(
        self,
        renders=True,
        init_noise=True,
        timestep=1.0 / 240,
        withVel=False,
        diffTar=True,
        control_skip=3,
        robot_mu=1.0,
        np_random=np.random,
        sleep=True,
    ):
        self.renders = renders
        self.init_noise = init_noise
        self._timeStep = timestep
        self.withVel = withVel
        self.diffTar = diffTar
        self.sleep = sleep

        self.timer = 0
        self.robot = None
        self.viewer = None
        self.control_skip = None
        self.action_scale = None
        self.change_control_skip_scaling(control_skip)

        self.robot = InmoovShadowNew(
            init_noise=self.init_noise,
            timestep=self._timeStep,
            np_random=np_random,
        )
        self.robot.change_hand_friction(robot_mu)

    def change_control_skip_scaling(self, c_skip, arm_scale=0.012, fin_scale=0.024):
        self.control_skip = c_skip
        # shadow hand is 22-5=17dof
        self.action_scale = np.array(
            [arm_scale / self.control_skip] * 7 + [fin_scale / self.control_skip] * 17
        )

    def change_init_fin_q(self, init_fin_q):
        self.robot.init_fin_q = np.copy(init_fin_q)

    def reset(self):  # deprecated
        self.timer = 0

    def step(self, action):
        for _ in range(self.control_skip):
            # action is in not -1,1
            if action is not None:
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders and self.sleep:
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

    def get_robot_contact_txty_halfh_obs_nodup(self, tx, ty, half_h):
        self.get_robot_contact_obs()
        self.observation.extend([tx, ty])
        self.observation.extend([tx, ty])
        self.observation.extend([half_h])
        return self.observation

    def get_robot_contact_txtytz_halfh_obs_nodup(self, tx, ty, tz, half_h):
        self.get_robot_contact_obs()
        self.observation.extend([tx, ty, tz])
        self.observation.extend([tx, ty, tz])
        self.observation.extend([half_h])
        return self.observation

    def get_robot_contact_txtytz_halfh_shape_obs_no_dup(self, tx, ty, tz, half_h, t_is_box):
        self.get_robot_contact_txtytz_halfh_obs_nodup(tx, ty, tz, half_h)

        if t_is_box:
            shape_info = [1, -1, -1]
        else:
            shape_info = [-1, 1, -1]
        self.observation.extend(shape_info)

        return self.observation

    def get_robot_contact_txty_shape_obs_no_dup(self, tx, ty, t_is_box):
        self.get_robot_contact_obs()
        self.observation.extend([tx, ty])
        self.observation.extend([tx, ty])

        if t_is_box:
            shape_info = [1, -1, -1]
        else:
            shape_info = [-1, 1, -1]
        self.observation.extend(shape_info)

        return self.observation

    def get_robot_contact_txty_halfh_2obj6dUp_obs_nodup_from_up(
        self, tx, ty, half_h, t_pos, t_up, b_pos, b_up
    ):
        """
        Args:
            tx: Target x position.
            ty: Target y position.
            half_h: Half of the height of the top object.
            t_pos: The xyz position of the top object.
            t_up: The up vector of the top object, normalized
            b_pos: The xyz position of the bottom object.
            b_up: The up vector of the bottom object, normalized
        """
        self.get_robot_contact_txty_halfh_obs_nodup(tx, ty, half_h)
        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(t_pos, t_up, tx, ty)
        )
        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(b_pos, b_up, tx, ty)
        )
        return self.observation

    def get_robot_contact_txtytz_halfh_2obj6dUp_obs_nodup_from_up(
        self, tx, ty, tz, half_h, t_pos, t_up, b_pos, b_up
    ):
        """
        Args:
            tx: Target x position.
            ty: Target y position.
            tz: Target z position (height of bottom obj, or 0 if table)
            half_h: Half of the height of the top object.
            t_pos: The xyz position of the top object.
            t_up: The up vector of the top object, normalized
            b_pos: The xyz position of the bottom object.
            b_up: The up vector of the bottom object, normalized
        """
        self.get_robot_contact_txtytz_halfh_obs_nodup(tx, ty, tz, half_h)
        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(t_pos, t_up, tx, ty)
        )
        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(b_pos, b_up, tx, ty)
        )
        return self.observation

    def get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
        self, tx, ty, tz, half_h, t_is_box, t_pos, t_up, b_pos, b_up
    ):
        """
        Args:
            tx: Target x position.
            ty: Target y position.
            tz: Target z position (height of bottom obj, or 0 if table)
            half_h: Half of the height of the top object.
            t_is_box: is top object box or cylinder (todo, ball)
            t_pos: The xyz position of the top object.
            t_up: The up vector of the top object, normalized
            b_pos: The xyz position of the bottom object.
            b_up: The up vector of the bottom object, normalized
        """
        # self.get_robot_contact_txtytz_halfh_obs_nodup(tx, ty, tz, half_h)
        #
        # if t_is_box:
        #     shape_info = [1, -1, -1]
        # else:
        #     shape_info = [-1, 1, -1]
        # self.observation.extend(shape_info)

        self.get_robot_contact_txtytz_halfh_shape_obs_no_dup(tx, ty, tz, half_h, t_is_box)

        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(t_pos, t_up, tx, ty)
        )
        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(b_pos, b_up, tx, ty)
        )
        return self.observation

    def get_robot_contact_txty_shape_2obj6dUp_obs_nodup_from_up(
        self, tx, ty, t_is_box, t_pos, t_up, b_pos, b_up
    ):
        """
        Args:
            tx: Target x position.
            ty: Target y position.
            t_is_box: is top object box or cylinder (todo, ball)
            t_pos: The xyz position of the top object.
            t_up: The up vector of the top object, normalized
            b_pos: The xyz position of the bottom object.
            b_up: The up vector of the bottom object, normalized
        """

        self.get_robot_contact_txty_shape_obs_no_dup(tx, ty, t_is_box)

        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(t_pos, t_up, tx, ty)
        )
        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(b_pos, b_up, tx, ty)
        )
        return self.observation

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s
