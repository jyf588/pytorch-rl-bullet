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


class InmoovShadowHandGraspEnvV6(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True,

                 random_top_shape=True,
                 det_top_shape_ind=1,  # if not random shape, 1 box, 0 cyl, -1 sphere,

                 cotrain_onstack_grasp=True,
                 grasp_floor=True,  # if not cotrain, is grasp from stack or grasp on table

                 control_skip=6,
                 obs_noise=True,

                 n_best_cand=2,

                 has_test_phase=True,
                 warm_start_phase=False,
                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up

        self.warm_start = warm_start_phase

        self.random_top_shape = random_top_shape
        self.det_top_shape_ind = det_top_shape_ind

        self.cotrain_onstack_grasp = cotrain_onstack_grasp
        self.grasp_floor = grasp_floor

        self.obs_noise = obs_noise

        self.has_test_phase = has_test_phase
        self.test_start = 50

        self.n_best_cand = int(n_best_cand)

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
        self.action_scale = np.array([0.009 / self.control_skip] * 7 + [0.024 / self.control_skip] * 17)

        self.p_pos_of_init = utils.PALM_POS_OF_INIT
        self.p_quat_of_init = p.getQuaternionFromEuler(utils.PALM_EULER_OF_INIT)

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

            desired_obj_pos = [self.tx, self.ty, self.tz]    # used for planning
            # obj frame (o_pos) in the COM of obj.

            arm_qs = utils.get_n_optimal_init_arm_qs(self.robot, self.p_pos_of_init, self.p_quat_of_init,
                                                     desired_obj_pos, self.table_id, n=self.n_best_cand,
                                                     wrist_gain=3.0)    # TODO
            if len(arm_qs) == 0:
                continue
            else:
                arm_q = arm_qs[self.np_random.randint(len(arm_qs))]
                return arm_q

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0

        # if self.cotrain_onstack_grasp:
        #     self.grasp_floor = self.np_random.randint(10) > 5
        if self.warm_start:
            if self.cotrain_onstack_grasp:
                self.grasp_floor = self.np_random.randint(10) >= 7  # 30%, TODO
        else:
            if self.cotrain_onstack_grasp:
                self.grasp_floor = self.np_random.randint(10) >= 6  # 40%, TODO

        mu_f = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
        self.table_id = utils.create_table(mu_f)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep, np_random=self.np_random)

        arm_q = self.sample_valid_arm_q()
        self.robot.reset_with_certain_arm_q(arm_q)

        if not self.grasp_floor:
            bo = self.btm_obj       # reference for safety
            bo['shape'] = utils.SHAPE_IND_MAP[self.np_random.randint(2)]      # btm cyl or box
            bo['half_width'] = self.np_random.uniform(utils.HALF_W_MIN_BTM, utils.HALF_W_MAX)
            if bo['shape'] == p.GEOM_BOX:
                bo['half_width'] *= 0.8
            bo['height'] = self.tz_act
            bo['mass'] = self.np_random.uniform(utils.MASS_MIN, utils.MASS_MAX)
            bo['mu'] = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)

            btm_xy = utils.perturb(self.np_random, [self.tx_act, self.ty_act], 0.015)
            btm_xyz = list(np.array(list(btm_xy) + [self.tz_act / 2.0]))

            btm_quat = p.getQuaternionFromEuler([0., 0., self.np_random.uniform(low=0, high=2.0 * math.pi)])
            bo['id'] = utils.create_sym_prim_shape_helper(bo, btm_xyz, btm_quat)

            if not self.warm_start:
                # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/constraint.py#L11
                _ = p.createConstraint(bo['id'], -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                       childFramePosition=btm_xyz,
                                       childFrameOrientation=btm_quat)

        to = self.top_obj
        shape_ind = self.np_random.randint(2) if self.random_top_shape else self.det_top_shape_ind
        to['shape'] = utils.SHAPE_IND_MAP[shape_ind]
        to['mass'] = self.np_random.uniform(utils.MASS_MIN, utils.MASS_MAX)
        to['mu'] = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
        to['half_width'] = self.np_random.uniform(utils.HALF_W_MIN, utils.HALF_W_MAX)
        if to['shape'] == p.GEOM_BOX:
            to['half_width'] *= 0.8
        elif to['shape'] == p.GEOM_SPHERE:
            to['height'] *= 0.75
        to['height'] = self.np_random.uniform(utils.H_MIN, utils.H_MAX)

        top_xyz = np.array([self.tx_act, self.ty_act, self.tz_act + to['height'] / 2.0])
        top_quat = p.getQuaternionFromEuler([0., 0., self.np_random.uniform(low=0, high=2.0 * math.pi)])
        to['id'] = utils.create_sym_prim_shape_helper(to, top_xyz, top_quat)

        # note, one-time (same for all frames) noise from init vision module
        if self.obs_noise:
            self.half_height_est = utils.perturb_scalar(self.np_random, self.top_obj['height']/2.0, 0.01)
        else:
            self.half_height_est = self.top_obj['height']/2.0

        p.stepSimulation()  # TODO

        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    def step(self, action):

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
                # action = np.clip(np.array(action), -1, 1)   # TODO
                self.act = action
                act_array = self.act * self.action_scale

                self.robot.apply_action(act_array)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        reward = 0.0

        # diff_norm = self.robot.get_norm_diff_tar_arm() * 10
        # reward += np.maximum(2. - diff_norm, 0)
        # # print(reward)

        # rewards is height of target object
        top_pos, _ = p.getBasePositionAndOrientation(self.top_obj['id'])

        top_xy_ideal = np.array([self.tx_act, self.ty_act])
        xy_dist = np.linalg.norm(top_xy_ideal - np.array(top_pos[:2]))
        reward += -np.minimum(xy_dist, 0.4) * 12.0

        for i in self.robot.fin_tips[:4]:
            tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
            reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(top_pos)), 0.5)  # 4 finger tips
        tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]      # thumb tip
        reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(top_pos)), 0.5) * 5.0
        palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
        dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(top_pos)), 0.5)
        reward += -dist * 2.0

        rot_metric = None
        if self.warm_start:
            # not used when grasp from floor
            _, btm_quat = p.getBasePositionAndOrientation(bottom_id)

            btm_vels = p.getBaseVelocity(bottom_id)
            btm_linv = np.array(btm_vels[0])
            btm_angv = np.array(btm_vels[1])
            reward += np.maximum(-np.linalg.norm(btm_linv) * 4.0 - np.linalg.norm(btm_angv), -5.0)

            z_axis, _ = p.multiplyTransforms(
                [0, 0, 0], btm_quat, [0, 0, 1], [0, 0, 0, 1]
            )  # R_cl * unitz[0,0,1]
            rot_metric = np.array(z_axis).dot(np.array([0, 0, 1]))
            reward += np.maximum(rot_metric * 20 - 15, 0.0)

        cps = p.getContactPoints(self.top_obj['id'], self.robot.arm_id, -1, self.robot.ee_id)    # palm
        if len(cps) > 0:
            reward += 5.0
        f_bp = [0, 3, 6, 9, 12, 17]     # 3*4+5
        for ind_f in range(5):
            con = False
            # for dof in self.robot.fin_actdofs[f_bp[ind_f]:f_bp[ind_f+1]]:
            # for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 2):f_bp[ind_f + 1]]:
            for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 3):f_bp[ind_f + 1]]:
                cps = p.getContactPoints(self.top_obj['id'], self.robot.arm_id, -1, dof)
                if len(cps) > 0:
                    con = True
            if con:
                reward += 5.0
            if con and ind_f == 4:
                reward += 20.0        # reward thumb even more

        reward -= self.robot.get_4_finger_deviation() * 1.5

        # object dropped during testing
        if top_pos[2] < (self.tz_act + 0.04) and self.timer > self.test_start * self.control_skip:
            reward += -15.

        # # add a final sparse reward
        # if self.timer == 65 * self.control_skip:        # TODO: hardcoded horizon
        #     if xy_dist < 0.05:
        #         # print("top good")
        #         reward += 100
        #     if self.warm_start and rot_metric > 0.9:
        #         # print("btm good")
        #         reward += 200

        #     print(self.robot.get_q_dq(range(29, 34))[0])

        return self.getExtendedObservation(), reward, False, {}

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

        xyz = np.array([self.tx, self.ty, self.tz])
        self.observation.extend(list(xyz))
        if self.obs_noise:
            self.observation.extend(list(xyz))
        else:
            self.observation.extend([self.tx_act, self.ty_act, self.tz_act])

        # top height info, btm height info included in tz
        self.observation.extend([self.half_height_est])

        # btm obj shape is not important.
        if self.random_top_shape:
            if self.top_obj['shape'] == p.GEOM_BOX:
                shape_info = [1, -1, -1]
            elif self.top_obj['shape'] == p.GEOM_CYLINDER:
                shape_info = [-1, 1, -1]
            elif self.top_obj['shape'] == p.GEOM_SPHERE:
                shape_info = [-1, -1, 1]
            else:
                shape_info = [-1, -1, -1]
            self.observation.extend(shape_info)

        return self.observation

    def append_final_state(self):
        # output obj in palm frame (no need to output palm frame in world)
        # output finger q's, finger tar q's.
        # velocity will be assumed to be zero at the end of transporting phase
        # return a dict.

        assert not self.has_test_phase

        obj_pos, obj_quat = p.getBasePositionAndOrientation(self.top_obj['id'])      # w2o
        hand_pos, hand_quat = self.robot.get_link_pos_quat(self.robot.ee_id)    # w2p
        inv_h_p, inv_h_q = p.invertTransform(hand_pos, hand_quat)       # p2w
        o_p_hf, o_q_hf = p.multiplyTransforms(inv_h_p, inv_h_q, obj_pos, obj_quat)  # p2w*w2o

        unitz_hf = p.multiplyTransforms([0, 0, 0], o_q_hf, [0, 0, 1], [0, 0, 0, 1])[0]
        # TODO: a heuritics that if obj up_vec points outside palm, then probably holding bottom & bad
        if unitz_hf[1] < -0.3:
            return
        else:
            fin_q, _ = self.robot.get_q_dq(self.robot.all_findofs)
            shape = self.top_obj['shape']
            dim = utils.to_bullet_dimension(shape, self.top_obj['half_width'], self.top_obj['height'])

            state = {'obj_pos_in_palm': o_p_hf, 'obj_quat_in_palm': o_q_hf,
                     'all_fin_q': fin_q, 'fin_tar_q': self.robot.tar_fin_q,
                     'obj_dim': dim, 'obj_shape': shape}
            # print(state)
            # print(self.robot.get_joints_last_tau(self.robot.all_findofs))
            # self.robot.get_wrist_wrench()
            self.final_states.append(state)

    def clear_final_states(self):
        self.final_states = []

    def calc_average_obj_in_palm(self):
        assert not self.has_test_phase
        count = len(self.final_states)
        o_pos_hf_sum = np.array([0., 0, 0])
        o_quat_hf_sum = np.array([0., 0, 0, 0])
        for state_dict in self.final_states:
            o_pos_hf_sum += np.array(state_dict['obj_pos_in_palm'])
            o_quat_hf_sum += np.array(state_dict['obj_quat_in_palm'])
        o_pos_hf_sum /= count
        o_quat_hf_sum /= count      # rough estimate of quat average
        o_quat_hf_sum /= np.linalg.norm(o_quat_hf_sum)      # normalize quat
        return list(o_pos_hf_sum), list(o_quat_hf_sum)

    def calc_average_obj_in_palm_rot_invariant(self):
        assert not self.has_test_phase
        count = len(self.final_states)
        o_pos_hf_sum = np.array([0., 0, 0])
        o_unitz_hf_sum = np.array([0., 0, 0])
        for state_dict in self.final_states:
            o_pos_hf_sum += np.array(state_dict['obj_pos_in_palm'])
            unitz_hf = p.multiplyTransforms([0, 0, 0], state_dict['obj_quat_in_palm'], [0, 0, 1], [0, 0, 0, 1])[0]
            o_unitz_hf_sum += np.array(unitz_hf)
        o_pos_hf_sum /= count
        o_unitz_hf_sum /= count      # rough estimate of unit z average
        o_unitz_hf_sum /= np.linalg.norm(o_unitz_hf_sum)      # normalize unit z vector

        x, y, z = o_unitz_hf_sum
        a1_solved = np.arcsin(-y)
        a2_solved = np.arctan2(x, z)
        # a3_solved is zero since equation has under-determined
        quat_solved = p.getQuaternionFromEuler([a1_solved, a2_solved, 0])

        uz_check = p.multiplyTransforms([0, 0, 0], quat_solved, [0, 0, 1], [0, 0, 0, 1])[0]
        assert np.linalg.norm(np.array(o_unitz_hf_sum) - np.array(uz_check)) < 1e-3

        return list(o_pos_hf_sum), list(quat_solved)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        if self.robot is not None:
            self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s
