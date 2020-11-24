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

    def change_control_skip_scaling(
        self, c_skip, arm_scale=0.009, fin_scale=0.024
    ):
        self.control_skip = c_skip
        # shadow hand is 22-5=17dof
        self.action_scale = np.array(
            [arm_scale / self.control_skip] * 7
            + [fin_scale / self.control_skip] * 17
        )

    def change_init_fin_q(self, init_fin_q):
        self.robot.init_fin_q = np.copy(init_fin_q)

    def reset(self):  # deprecated
        self.timer = 0

    # def place_step(self, action, table_id, top_id, plaing):
    #     reward = 0.0
    #     top_pos, top_quat = p.getBasePositionAndOrientation(top_id)
    #     top_vels = p.getBaseVelocity(top_id)
    #     top_lin_v = np.array(top_vels[0])
    #     top_ang_v = np.array(top_vels[1])
    #
    #     # we only care about the upright(z) direction
    #     z_axis, _ = p.multiplyTransforms(
    #         [0, 0, 0], top_quat, [0, 0, 1], [0, 0, 0, 1]
    #     )  # R_cl * unitz[0,0,1]
    #     rot_metric = np.array(z_axis).dot(np.array([0, 0, 1]))
    #
    #     xyz_metric = 1 - (
    #             np.minimum(
    #                 np.linalg.norm(
    #                     np.array(self.desired_obj_pos_final) - np.array(top_pos)
    #                 ),
    #                 0.15,
    #             )
    #             / 0.15
    #     )
    #     lin_v_r = np.linalg.norm(top_lin_v)
    #     # print("lin_v", lin_v_r)
    #     ang_v_r = np.linalg.norm(top_ang_v)
    #     # print("ang_v", ang_v_r)
    #     vel_metric = 1 - np.minimum(lin_v_r * 4.0 + ang_v_r, 5.0) / 5.0
    #
    #     reward += np.maximum(rot_metric * 20 - 15, 0.0)
    #     # print(np.maximum(rot_metric * 20 - 15, 0.))
    #     reward += xyz_metric * 5
    #     # print(xyz_metric * 5)
    #     reward += vel_metric * 5
    #     # print(vel_metric * 5)
    #     # print("upright", reward)
    #
    #     diff_norm = self.robot.get_norm_diff_tar()  # TODO: necessary?
    #     reward += 10.0 / (diff_norm + 1.0)
    #     # # print(10. / (diff_norm + 1.))
    #
    #     any_hand_contact = False
    #     hand_r = 0
    #     for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
    #         cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
    #         con_this_link = False
    #         for cp in cps:
    #             if cp[1] != cp[2]:  # not self-collision of the robot
    #                 con_this_link = True
    #                 break
    #         if con_this_link:
    #             any_hand_contact = True
    #         else:
    #             hand_r += 0.5
    #     reward += hand_r - 7
    #     # print("no contact", hand_r - 7.0)
    #
    #     reward -= self.robot.get_4_finger_deviation() * 0.3
    #
    #     #
    #     # if self.timer == 99 * self.control_skip:
    #     #     print(rot_metric, xyz_metric, vel_metric, any_hand_contact)
    #     #     # print(any_hand_contact)
    #
    #     if (
    #             rot_metric > 0.9
    #             and xyz_metric > 0.6
    #             and vel_metric > 0.6
    #             # and meaningful_c
    #     ):  # close to placing
    #         reward += 5.0
    #         # print("upright")
    #         if not any_hand_contact:
    #             reward += 20.0
    #             # print("no hand con")
    #
    # def grasp_step(self, action, table_id, top_id, g_tx, g_ty, np_random):
    #     if self.timer == self.test_start * self.control_skip:
    #         self.force_global = [np_random.uniform(-100, 100),
    #                              np_random.uniform(-100, 100),
    #                              -200.]
    #
    #     if self.timer > self.test_start * self.control_skip:
    #         p.setCollisionFilterPair(top_id, table_id, -1, -1, enableCollision=0)
    #         _, quat = p.getBasePositionAndOrientation(top_id)
    #         _, quat_inv = p.invertTransform([0, 0, 0], quat)
    #         force_local, _ = p.multiplyTransforms([0, 0, 0], quat_inv, self.force_global, [0, 0, 0, 1])
    #         p.applyExternalForce(top_id, -1, force_local, [0, 0, 0], flags=p.LINK_FRAME)
    #     for _ in range(self.control_skip):
    #         self.step_sim(action=action)
    #     reward = 0.0
    #
    #     # rewards is height of target object
    #     top_pos, _ = p.getBasePositionAndOrientation(top_id)
    #
    #     top_xy_ideal = np.array([g_tx, g_ty])
    #     xy_dist = np.linalg.norm(top_xy_ideal - np.array(top_pos[:2]))
    #     reward += -np.minimum(xy_dist, 0.4) * 6.0
    #
    #     vel_palm = np.linalg.norm(self.robot.get_link_v_w(self.robot.ee_id)[0])
    #     reward += -vel_palm * 1.0
    #     # vel_palm = self.robot.get_link_v_w(self.robot.ee_id)[0]
    #     # reward += -np.linalg.norm(np.array(prev_v) - np.array(vel_palm)) * 4.0
    #
    #     for i in self.robot.fin_tips[:4]:
    #         tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
    #         reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(top_pos)), 0.5)  # 4 finger tips
    #     tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]  # thumb tip
    #     reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(top_pos)), 0.5) * 5.0
    #     palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
    #     dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(top_pos)), 0.5)
    #     reward += -dist * 2.0
    #
    #     cps = p.getContactPoints(top_id, self.robot.arm_id, -1, self.robot.ee_id)  # palm
    #     if len(cps) > 0:
    #         reward += 5.0
    #     f_bp = [0, 3, 6, 9, 12, 17]  # 3*4+5
    #     for ind_f in range(5):
    #         con = False
    #         # for dof in self.robot.fin_actdofs[f_bp[ind_f]:f_bp[ind_f+1]]:
    #         # for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 2):f_bp[ind_f + 1]]:
    #         for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 3):f_bp[ind_f + 1]]:
    #             cps = p.getContactPoints(top_id, self.robot.arm_id, -1, dof)
    #             if len(cps) > 0:
    #                 con = True
    #         if con:
    #             reward += 5.0
    #         if con and ind_f == 4:
    #             reward += 20.0  # reward thumb even more
    #
    #     reward -= self.robot.get_4_finger_deviation() * 1.5
    #
    #     # object dropped during testing
    #     tz_act = 0. # grasp object on the table
    #     if top_pos[2] < (tz_act + 0.06) and self.timer > self.test_start * self.control_skip:
    #         reward += -15.
    #     return reward

    def step(self, action, is_sphere):
        """Applies the provided robot action and steps the simulation 
        `control_skip` times.
        """
        for _ in range(self.control_skip):
            self.step_sim(action=action, is_sphere=is_sphere)

    def step_sim(self, action, is_sphere):
        """Applies the provided robot action and takes one simulation step."""
        if action is not None:
            self.act = action
            self.robot.apply_action(self.act * self.action_scale, is_sphere)
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

    def get_robot_contact_txtytz_halfh_shape_obs_no_dup(
        self, tx, ty, tz, half_h, shape
    ):
        # TODO: shape: 1 box, 0 cyl, -1 sph

        self.get_robot_contact_txtytz_halfh_obs_nodup(tx, ty, tz, half_h)

        shape_info = []
        if shape == 1:
            shape_info = [1, -1, -1]
        elif shape == 0:
            shape_info = [-1, 1, -1]
        elif shape == -1:
            shape_info = [-1, -1, 1]
        else:
            assert False and "not implemented"
        self.observation.extend(shape_info)

        return self.observation

    def get_robot_contact_txty_shape_obs_no_dup(self, tx, ty, shape):
        # TODO: shape: 1 box, 0 cyl, -1 sph
        self.get_robot_contact_obs()
        self.observation.extend([tx, ty])
        self.observation.extend([tx, ty])

        shape_info = []
        if shape == 1:
            shape_info = [1, -1, -1]
        elif shape == 0:
            shape_info = [-1, 1, -1]
        elif shape == -1:
            shape_info = [-1, -1, 1]
        else:
            assert False and "not implemented"
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
        self, tx, ty, tz, half_h, shape, t_pos, t_up, b_pos, b_up
    ):
        """
        Args:
            tx: Target x position.
            ty: Target y position.
            tz: Target z position (height of bottom obj, or 0 if table)
            half_h: Half of the height of the top object.
            shape: TODO: shape: 1 box, 0 cyl, -1 sph
            t_pos: The xyz position of the top object.
            t_up: The up vector of the top object, normalized
            b_pos: The xyz position of the bottom object.
            b_up: The up vector of the bottom object, normalized
        """
        # self.get_robot_contact_txtytz_halfh_obs_nodup(tx, ty, tz, half_h)
        #
        # if shape:
        #     shape_info = [1, -1, -1]
        # else:
        #     shape_info = [-1, 1, -1]
        # self.observation.extend(shape_info)

        self.get_robot_contact_txtytz_halfh_shape_obs_no_dup(
            tx, ty, tz, half_h, shape
        )

        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(t_pos, t_up, tx, ty)
        )
        self.observation.extend(
            utils.obj_pos_and_upv_to_obs(b_pos, b_up, tx, ty)
        )
        return self.observation

    def get_robot_contact_txty_shape_2obj6dUp_obs_nodup_from_up(
        self, tx, ty, shape, t_pos, t_up, b_pos, b_up
    ):
        """
        Args:
            tx: Target x position.
            ty: Target y position.
            shape: TODO: shape: 1 box, 0 cyl, -1 sph
            t_pos: The xyz position of the top object.
            t_up: The up vector of the top object, normalized
            b_pos: The xyz position of the bottom object.
            b_up: The up vector of the bottom object, normalized
        """

        self.get_robot_contact_txty_shape_obs_no_dup(tx, ty, shape)

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
