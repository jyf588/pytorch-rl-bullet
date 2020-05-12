from .inmoov_shadow_hand_v2 import InmoovShadowNew
from . import utils

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class InmoovShadowHandEteEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True,

                 random_top_shape=False,
                 det_top_shape_ind=0,  # if not random shape, 1 box, 0 cyl, -1 sphere,

                 grasp_floor=True,  # always grasp from floor

                 control_skip=6,
                 obs_noise=True,

                 has_test_phase=True,
                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up

        self.random_top_shape = random_top_shape
        self.det_top_shape_ind = det_top_shape_ind

        self.grasp_floor = grasp_floor

        self.obs_noise = obs_noise

        self.has_test_phase = has_test_phase
        self.test_start = 80        # longer than grasping only, TODO

        self.test_end = 95      # grasp test end

        # self.n_best_cand = int(n_best_cand)

        # dummy, to be overwritten
        self.top_obj = {'id': None,
                        'mass': -1,
                        'mu': -1,
                        'shape': utils.SHAPE_IND_MAP[0],
                        'half_width': -1,
                        'height': -1}
        self.btm_obj = {'id': None,
                        'mass': -1,
                        'mu': -1,
                        'shape': utils.SHAPE_IND_MAP[0],
                        'half_width': -1,
                        'height': -1}
        self.table_id = None

        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)     # this session seems always 0
        self.np_random = None
        self.robot = None
        self.seed(0)  # used once temporarily, will be overwritten outside by env
        self.viewer = None
        self.timer = 0

        self.final_states = []  # wont be cleared unless call clear function

        self.control_skip = int(control_skip)
        # shadow hand is 22-5=17dof
        self.init_act_scale = np.array([0.04 / self.control_skip] * 7 + [0.024 / self.control_skip] * 17)
        self.action_scale = self.init_act_scale

        self.tx = -1    # dummy
        self.ty = -1    # dummy
        self.tz = -1    # dummy
        self.tx_act = -1    # dummy
        self.ty_act = -1    # dummy
        self.tz_act = -1    # dummy

        self.reset()    # and update init obs

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def sample_valid_arm_q(self):
        while True:
            if self.init_noise:
                self.tx, self.ty, self.tz, self.tx_act, self.ty_act, self.tz_act = \
                    utils.sample_tx_ty_tz(self.np_random, self.up, self.grasp_floor, 0.02, 0.02)
            else:
                self.tx, self.ty, self.tz, self.tx_act, self.ty_act, self.tz_act = \
                    utils.sample_tx_ty_tz(self.np_random, self.up, self.grasp_floor, 0.0, 0.0)

            return

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0

        assert self.grasp_floor

        self.grasp_stage = True

        mu_f = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
        self.table_id = utils.create_table(mu_f)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep, np_random=self.np_random)

        self.sample_valid_arm_q()       # reset obj init position, bad naming
        # arm_q = self.sample_valid_arm_q()
        self.robot.reset_with_certain_arm_q([-1.47]+[0.0]+[-0.7]+[0.0]+[-0.7]+[0.0]*2)

        # self.robot.reset_with_certain_arm_q([0.0] * 7)

        to = self.top_obj
        shape_ind = self.np_random.randint(2) if self.random_top_shape else self.det_top_shape_ind
        to['shape'] = utils.SHAPE_IND_MAP[shape_ind]
        to['mass'] = self.np_random.uniform(utils.MASS_MIN, utils.MASS_MAX)
        to['mu'] = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
        to['half_width'] = self.np_random.uniform(utils.HALF_W_MIN, utils.HALF_W_MAX)
        to['height'] = self.np_random.uniform(utils.H_MIN, utils.H_MAX)
        if to['shape'] == p.GEOM_BOX:
            to['half_width'] *= 0.8
        elif to['shape'] == p.GEOM_SPHERE:
            to['height'] *= 0.75
            to['half_width'] = None

        top_xyz = np.array([self.tx_act, self.ty_act, self.tz_act + to['height'] / 2.0])
        top_quat = p.getQuaternionFromEuler([0., 0., self.np_random.uniform(low=0, high=2.0 * math.pi)])
        to['id'] = utils.create_sym_prim_shape_helper(to, top_xyz, top_quat)

        # # note, one-time (same for all frames) noise from init vision module
        # if self.obs_noise:
        #     self.half_height_est = utils.perturb_scalar(self.np_random, self.top_obj['height']/2.0, 0.01)
        # else:
        #     self.half_height_est = self.top_obj['height']/2.0

        p.stepSimulation()  # TODO

        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    # def step():
    # if t<95:
    # keep the current
    # if t>95:
    # if not in hand done
    # calc move and place reward

    def step_reach_grasp(self, action):
        bottom_id = self.table_id if self.grasp_floor else self.btm_obj['id']

        if self.has_test_phase:
            if self.timer == self.test_start * self.control_skip:
                self.force_global = [self.np_random.uniform(-100, 100),
                                     self.np_random.uniform(-100, 100),
                                     -200.]

            if self.timer > self.test_start * self.control_skip:
                p.setCollisionFilterPair(self.top_obj['id'], bottom_id, -1, -1, enableCollision=0)
                _, quat = p.getBasePositionAndOrientation(self.top_obj['id'])
                _, quat_inv = p.invertTransform([0, 0, 0], quat)
                force_local, _ = p.multiplyTransforms([0, 0, 0], quat_inv, self.force_global, [0, 0, 0, 1])
                p.applyExternalForce(self.top_obj['id'], -1, force_local, [0, 0, 0], flags=p.LINK_FRAME)

        for _ in range(self.control_skip):
            # action is in not -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)
                self.act = action
                act_array = self.act * self.action_scale

                self.robot.apply_action(act_array)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        # TODO: seem to affect most
        top_pos, _ = p.getBasePositionAndOrientation(self.top_obj['id'])
        top_pos_1 = [self.tx_act, self.ty_act, self.tz_act + self.top_obj['height'] / 2.0]

        reward = 0.0

        palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
        dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(top_pos_1)), 1.0)

        # TODO: grasp does not have this term, add this term to reaching
        tip_pos = []  # 4 finger tips & thumb tip
        tip_dists = []
        dist_diffs = []
        for i in self.robot.fin_tips[:5]:
            cur_tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
            tip_pos.append(cur_tip_pos)
            cur_tip_dist = np.linalg.norm(np.array(cur_tip_pos) - np.array(top_pos_1))
            tip_dists.append(cur_tip_dist)
            dist_diffs.append(dist - cur_tip_dist)  # this number should be larger, finger closer to obj center

        # make this always negative so always beneficial to enter stage 2
        reward += (np.sum(dist_diffs[:-1]) + dist_diffs[-1] * 5.0) * 10.0

        if dist > 0.3:
            # only this for phase one
            reward += -dist * 30.0
            # this term is naturally bounded
        else:
            # good enough, phase two, focus on other things
            reward += -dist * 3.0

            # the following should just be phase two
            # and always positive
            # so it does not hurt to enter phase two

            # # TODO: add back with 2.0/12.0
            # top_xy_ideal = np.array([self.tx_act, self.ty_act])
            # xy_dist = np.linalg.norm(top_xy_ideal - np.array(top_pos[:2]))
            # reward += 5.0 - np.minimum(xy_dist, 0.4) * 8.0     # TODO: make this always positive

            for i in range(4):
                reward += 0.5 - np.minimum(tip_dists[i], 0.5)  # 4 fin tips
            reward += 2.5 - np.minimum(tip_dists[4], 0.5) * 5.0  # thumb tip

            cps = p.getContactPoints(self.top_obj['id'], self.robot.arm_id, -1, self.robot.ee_id)  # palm
            if len(cps) > 0:
                reward += 10.0
            f_bp = [0, 3, 6, 9, 12, 17]  # 3*4+5
            for ind_f in range(5):
                con = False
                for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 3):f_bp[ind_f + 1]]:
                    cps = p.getContactPoints(self.top_obj['id'], self.robot.arm_id, -1, dof)
                    if len(cps) > 0:
                        con = True
                if con:
                    reward += 5.0
                if con and ind_f == 4:
                    reward += 20.0  # reward thumb even more

        # reward -= self.robot.get_4_finger_deviation() * 1.5 # TODO: and for placing

        # "phase three", test phase, it is there no matter enter phase two or not
        # object dropped during testing
        if top_pos[2] < (self.tz_act + 0.04) and self.timer > self.test_start * self.control_skip:
            reward += -15.

        return reward, False    # not done

    def step_move_place(self, action):

        for _ in range(self.control_skip):
            # action is in not -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)
                self.act = action
                act_array = self.act * self.action_scale

                self.robot.apply_action(act_array)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        reward = 0.0

        top_pos, top_quat = p.getBasePositionAndOrientation(self.top_obj['id'])
        top_vels = p.getBaseVelocity(self.top_obj['id'])
        top_lin_v = np.array(top_vels[0])
        top_ang_v = np.array(top_vels[1])

        dist = np.linalg.norm(np.array(self.desired_obj_pos_final) - np.array(top_pos))
        xyz_metric = 1 - (
            np.minimum(
                dist,
                1.0,
            )
            / 1.0
        )
        reward += xyz_metric * 30.0      # when dist < 0.15, will be almost equivalent as term in placing

        if dist < 0.15:
            # good enough, phase place, focus on other things

            # we only care about the upright(z) direction
            z_axis, _ = p.multiplyTransforms(
                [0, 0, 0], top_quat, [0, 0, 1], [0, 0, 0, 1]
            )  # R_cl * unitz[0,0,1]
            rot_metric = np.array(z_axis).dot(np.array([0, 0, 1]))

            lin_v_r = np.linalg.norm(top_lin_v)
            # print("lin_v", lin_v_r)
            ang_v_r = np.linalg.norm(top_ang_v)
            # print("ang_v", ang_v_r)
            vel_metric = 1 - np.minimum(lin_v_r * 4.0 + ang_v_r, 5.0) / 5.0

            reward += np.maximum(rot_metric * 20 - 15, 0.0)
            # print(np.maximum(rot_metric * 20 - 15, 0.))

            reward += vel_metric * 5
            # print(vel_metric * 5)
            # print("upright", reward)

            diff_norm = self.robot.get_norm_diff_tar()
            reward += 10.0 / (diff_norm + 1.0)
            # # print(10. / (diff_norm + 1.))

            any_hand_contact = False
            hand_r = 0
            for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
                cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
                con_this_link = False
                for cp in cps:
                    if cp[1] != cp[2]:  # not self-collision of the robot
                        con_this_link = True
                        break
                if con_this_link:
                    any_hand_contact = True
                else:
                    hand_r += 0.5
            reward += hand_r

            # reward -= self.robot.get_4_finger_deviation() * 0.3   # TODO: make sure all terms positive
            #
            # if self.timer == 99 * self.control_skip:
            #     print(rot_metric, xyz_metric, vel_metric, any_hand_contact)
            #     # print(any_hand_contact)

            if (
                rot_metric > 0.9
                and xyz_metric > 0.6
                and vel_metric > 0.6
                # and meaningful_c
            ):  # close to placing
                reward += 5.0
                # print("upright")
                if not any_hand_contact:
                    reward += 20.0
                    # print("no hand con")
        # print("p", reward)
        return reward, False

    def step(self, action):
        bottom_id = self.table_id if self.grasp_floor else self.btm_obj['id']
        top_pos, _ = p.getBasePositionAndOrientation(self.top_obj['id'])

        if self.timer <= self.test_end * self.control_skip:
            reward, done = self.step_reach_grasp(action)
        else:
            if top_pos[2] > (self.tz_act + 0.04):
                # enter moving stage
                # sample tx ty again as destination

                if self.grasp_stage:
                    self.sample_valid_arm_q()
                    self.desired_obj_pos_final = [
                        self.tx_act,
                        self.ty_act,
                        self.top_obj["height"] / 2.0 + self.tz_act,
                    ]
                    p.setCollisionFilterPair(self.top_obj['id'], bottom_id, -1, -1, enableCollision=1)

                self.grasp_stage = False
                reward, done = self.step_move_place(action)

            else:
                reward = -30
                done = True

        #     print(self.robot.get_q_dq(range(29, 34))[0])

        return self.getExtendedObservation(), reward, done, {}

    def obj6DtoObs_UpVec(self, o_pos, o_orn, is_sph=False):
        o_pos = np.array(o_pos)
        if is_sph:
            o_orn = [0.0, 0, 0, 1]
        o_upv = utils.quat_to_upv(o_orn)

        if self.obs_noise:
            o_pos = utils.perturb(self.np_random, o_pos, r=0.02)
            o_upv = utils.perturb(self.np_random, o_upv, r=0.03)
            obj_obs = utils.obj_pos_and_upv_to_obs(
                o_pos, o_upv, self.tx, self.ty
            )
        else:
            o_pos = o_pos
            o_upv = o_upv
            obj_obs = utils.obj_pos_and_upv_to_obs(
                o_pos, o_upv, self.tx_act, self.ty_act
            )

        return obj_obs

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation(diff_tar=True)

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

        xy = np.array([self.tx, self.ty])
        self.observation.extend(list(xy))
        if self.obs_noise:
            self.observation.extend(list(xy))
        else:
            self.observation.extend([self.tx_act, self.ty_act])

        # btm obj shape is not important.
        if self.top_obj['shape'] == p.GEOM_BOX:
            shape_info = [1, -1, -1]
        elif self.top_obj['shape'] == p.GEOM_CYLINDER:
            shape_info = [-1, 1, -1]
        elif self.top_obj['shape'] == p.GEOM_SPHERE:
            shape_info = [-1, -1, 1]
        else:
            assert False
        self.observation.extend(shape_info)

        top_pos, top_orn = p.getBasePositionAndOrientation(self.top_obj['id'])
        self.observation.extend(
            self.obj6DtoObs_UpVec(top_pos, top_orn)
        )

        stage = 1. if self.grasp_stage else -1.
        self.observation.extend([stage + self.np_random.uniform(low=-0.01, high=0.01),
                                 stage + self.np_random.uniform(low=-0.01, high=0.01),
                                 stage])

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
