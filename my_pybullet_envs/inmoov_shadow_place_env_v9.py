from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew

from my_pybullet_envs import utils

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math
import pickle

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


# multiple placing init pose & place on floor. / remove collision with floor init.

# easier way:
# 1. load floor first
# 2. permute the list of arm candidates  (some IK infeasible, filtered out)
# 3. reset arm (hand using init grasp pose)
# 4. if collide with floor, filter out, return a list (max 2) of arm q ranked by optimality
# If none exist, skip this txty/sample.
# init_done = False
# while not init_done:

class InmoovShadowHandPlaceEnvV9(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=False,
                 init_noise=True,   # variation during reset
                 up=True,

                 random_top_shape=True,
                 det_top_shape_ind=1,  # if not random shape, 1 means always box

                 cotrain_stack_place=True,
                 place_floor=True,     # if not cotrain, is stack or place-on-floor
                 grasp_pi_name=None,
                 exclude_hard=False,

                 use_gt_6d=True,
                 gt_only_init=False,
                 vision_skip=2,

                 control_skip=6,
                 obs_noise=False,    # noisy (imperfect) observation

                 n_best_cand=2,

                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up

        self.random_top_shape = random_top_shape

        self.cotrain_stack_place = cotrain_stack_place
        self.place_floor = place_floor
        self.exclude_hard = exclude_hard
        self.hard_orn_thres = 0.85

        self.use_gt_6d = use_gt_6d
        self.gt_only_init = gt_only_init

        self.obs_noise = obs_noise

        self.vision_skip = vision_skip
        self.vision_counter = 0

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

        # TODO: hardcoded here
        if grasp_pi_name is None:
            if not self.random_top_shape:
                if det_top_shape_ind:
                    self.grasp_pi_name = '0311_box_2_n_20_50'
                    self.is_box = True
                else:
                    self.grasp_pi_name = '0311_cyl_2_n_20_50'
                    self.is_box = False
            else:
                self.grasp_pi_name = "0313_2_n_25_45"
                self.is_box = False     # dummy, 2b overwritten
        else:
            self.grasp_pi_name = grasp_pi_name
            self.is_box = False  # dummy, 2b overwritten


        self.start_clearance = 0.14

        self.cand_angles = [0., 3.14 / 4, 6.28 / 4, 9.42 / 4, 3.14,
                            -9.42 / 4, -6.28 / 4, -3.14 / 4]  # TODO: finer grid? 8 grid now.
        self.cand_quats = [p.getQuaternionFromEuler([0, 0, cand_angle]) for cand_angle in self.cand_angles]

        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.np_random = None
        self.robot = None
        self.viewer = None
        self.timer = 0

        self.control_skip = int(control_skip)
        # shadow hand is 22-5=17dof
        self.action_scale = np.array([0.009 / self.control_skip] * 7 + [0.024 / self.control_skip] * 17)

        self.tx = -1    # dummy
        self.ty = -1    # dummy
        self.tz = -1    # dummy
        self.tx_act = -1    # dummy
        self.ty_act = -1    # dummy
        self.tz_act = -1    # dummy

        self.desired_obj_pos_final = None

        self.saved_file = None
        with open(os.path.join(currentdir, 'assets/place_init_dist/final_states_' + self.grasp_pi_name + '.pickle'),
                  'rb') as handle:
            self.saved_file = pickle.load(handle)
        assert self.saved_file is not None

        self.o_pos_pf_ave = self.saved_file['ave_obj_pos_in_palm']
        self.o_quat_pf_ave = self.saved_file['ave_obj_quat_in_palm']
        self.o_quat_pf_ave /= np.linalg.norm(self.o_quat_pf_ave)        # in case not normalized
        self.init_states = self.saved_file['init_states']  # a list of dicts

        # print(self.o_pos_pf_ave)
        # print(self.o_quat_pf_ave)
        # print(self.init_states[10])
        # print(self.init_states[51])
        # print(self.init_states[89])

        self.seed(0)    # used once temporarily, will be overwritten outside by env
        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep, np_random=self.np_random)

        self.observation = self.getExtendedObservation()
        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def reset_robot_top_object_from_sample(self, arm_q):
        while True:
            ran_ind = int(self.np_random.uniform(low=0, high=len(self.init_states) - 0.1))
            state = self.init_states[ran_ind]

            o_pos_pf = state['obj_pos_in_palm']
            o_quat_pf = state['obj_quat_in_palm']
            if self.init_noise:
                o_pos_pf = list(utils.perturb(self.np_random, o_pos_pf, 0.005))
                o_quat_pf = list(utils.perturb(self.np_random, o_quat_pf, 0.005))
            all_fin_q_init = state['all_fin_q']
            tar_fin_q_init = state['fin_tar_q']

            self.robot.reset_with_certain_arm_q_finger_states(arm_q, all_fin_q_init, tar_fin_q_init)

            p_pos, p_quat = self.robot.get_link_pos_quat(self.robot.ee_id)
            o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)

            z_axis, _ = p.multiplyTransforms([0, 0, 0], o_quat, [0, 0, 1], [0, 0, 0, 1])  # R_cl * unitz[0,0,1]
            rot_metric = np.array(z_axis).dot(np.array([0, 0, 1]))
            # print(rotMetric, rotMetric)
            if self.exclude_hard and rot_metric < self.hard_orn_thres:
                continue
            else:
                to = self.top_obj
                to['shape'] = state['obj_shape']
                to['mass'] = self.np_random.uniform(utils.MASS_MIN, utils.MASS_MAX)
                to['mu'] = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
                to['half_width'], to['height'] = \
                    utils.from_bullet_dimension(state['obj_shape'], state['obj_dim'])
                to['id'] = utils.create_prim_shape(to['mass'], to['shape'],
                                                   state['obj_dim'], to['mu'],
                                                   o_pos, o_quat)

                # only used for reward calc.
                self.desired_obj_pos_final = [self.tx_act, self.ty_act,
                                              to['height']/2.0 + self.tz_act]

                self.t_pos, self.t_orn = o_pos, o_quat
                self.last_t_pos, self.last_t_orn = o_pos, o_quat

                return

    def get_n_optimal_init_arm_qs(self, desired_obj_pos, n=2):
        # uses (self.o_pos_pf_ave, self.o_quat_pf_ave), so set mean stats to load properly
        arm_qs_costs = []
        ref = np.array([0.] * 3 + [-1.57] + [0.] * 3)
        for ind, cand_quat in enumerate(self.cand_quats):
            p_pos_of_ave, p_quat_of_ave = p.invertTransform(self.o_pos_pf_ave, self.o_quat_pf_ave)
            p_pos, p_quat = p.multiplyTransforms(desired_obj_pos, cand_quat,
                                                 p_pos_of_ave, p_quat_of_ave)
            cand_arm_q = self.robot.solve_arm_IK(p_pos, p_quat)
            if cand_arm_q is not None:
                cps = []
                if self.place_floor:
                    p.stepSimulation()      # TODO
                    cps = p.getContactPoints(bodyA=self.robot.arm_id, bodyB=self.table_id)
                if len(cps) == 0:
                    diff = np.array(cand_arm_q) - ref
                    cand_cost = np.sum(np.abs(diff))  # change to l1
                    arm_qs_costs.append((cand_arm_q, cand_cost))
        arm_qs_costs_sorted = sorted(arm_qs_costs, key=lambda x: x[1])[:n]  # fine if length < n
        return [arm_q_cost[0] for arm_q_cost in arm_qs_costs_sorted]

    def sample_valid_arm_q(self):
        while True:
            self.sample_tx_ty_tz()
            desired_obj_pos = [self.tx, self.ty, self.start_clearance + self.tz]    # used for planning
            arm_qs = self.get_n_optimal_init_arm_qs(desired_obj_pos, n=self.n_best_cand)
            if len(arm_qs) == 0:
                continue
            else:
                arm_q = arm_qs[self.np_random.randint(len(arm_qs))]
                return arm_q

    def sample_tx_ty_tz(self):
        # tx ty tz can be out of arm reach
        # tx ty used for planning, and are part of the robot obs
        # tx_act, ty_act are the actual btm obj x y
        # tz_act is the actual bottom obj height
        # tz used for planning and robot obs

        if self.up:
            self.tx = self.np_random.uniform(low=utils.TX_MIN, high=utils.TX_MAX)
            self.ty = self.np_random.uniform(low=utils.TY_MIN, high=utils.TY_MAX)
        else:
            self.tx = 0.0
            self.ty = 0.0

        # TODO: noise large enough?
        self.tx_act = utils.perturb_scalar(self.np_random, self.tx, 0.015) if self.init_noise else self.tx
        self.ty_act = utils.perturb_scalar(self.np_random, self.ty, 0.015) if self.init_noise else self.ty

        if self.place_floor:
            self.tz_act = 0
            self.tz = 0
        else:
            self.tz_act = self.np_random.uniform(utils.H_MIN, utils.H_MAX)
            self.tz = utils.perturb_scalar(self.np_random, self.tz_act, 0.02) if self.init_noise else self.tz_act

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0
        self.vision_counter = 0

        if self.cotrain_stack_place:
            self.place_floor = self.np_random.randint(10) > 6   # 30%

        self.table_id = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), utils.TABLE_OFFSET,
                                   useFixedBase=1)
        mu_f = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
        p.changeDynamics(self.table_id, -1, lateralFriction=mu_f)

        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep, np_random=self.np_random)

        arm_q = self.sample_valid_arm_q()   # reset done during solving IK

        if not self.place_floor:
            bo = self.btm_obj
            bo['shape'] = utils.SHAPE_IND_MAP[self.np_random.randint(2)]      # btm cyl or box
            bo['half_width'] = self.np_random.uniform(utils.HALF_W_MIN_BTM, utils.HALF_W_MAX)
            if bo['shape'] == p.GEOM_BOX:
                bo['half_width'] *= 0.8
            bo['height'] = self.tz_act
            bo['mass'] = self.np_random.uniform(utils.MASS_MIN, utils.MASS_MAX)
            bo['mu'] = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)

            btm_xyz = np.array([self.tx_act, self.ty_act, self.tz_act / 2.0])
            btm_quat = p.getQuaternionFromEuler([0., 0., self.np_random.uniform(low=0, high=2.0 * math.pi)])
            bo['id'] = utils.create_sym_prim_shape_helper(bo, btm_xyz, btm_quat)

            self.b_pos, self.b_orn = btm_xyz, btm_quat
            self.last_b_pos, self.last_b_orn = btm_xyz, btm_quat

        self.reset_robot_top_object_from_sample(arm_q)

        p.stepSimulation()      # TODO

        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    def __del__(self):
        p.disconnect()

    def step(self, action):
        for _ in range(self.control_skip):
            # action is not in -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)   # TODO
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        # if upright, reward
        # if no contact, reward
        # if use small force, reward

        bottom_id = self.table_id if self.place_floor else self.btm_obj['id']

        reward = 0.
        top_pos, top_quat = p.getBasePositionAndOrientation(self.top_obj['id'])
        top_vels = p.getBaseVelocity(self.top_obj['id'])
        top_lin_v = np.array(top_vels[0])
        top_ang_v = np.array(top_vels[1])

        # we only care about the upright(z) direction
        z_axis, _ = p.multiplyTransforms([0, 0, 0], top_quat, [0, 0, 1], [0, 0, 0, 1])          # R_cl * unitz[0,0,1]
        rot_metric = np.array(z_axis).dot(np.array([0, 0, 1]))

        # enlarge 0.15 -> 0.45
        # xyz_metric = 1 - (np.minimum(np.linalg.norm(np.array(self.desired_obj_pos_final) - np.array(clPos)), 0.45) / 0.15)
        # TODO:tmp change to xy metric, allow it to free drop
        # TODO: xyz metric should use gt value.
        xyz_metric = 1 - (np.minimum(np.linalg.norm(np.array(self.desired_obj_pos_final[:2]) - np.array(top_pos[:2])), 0.45) / 0.15)
        lin_v_r = np.linalg.norm(top_lin_v)
        ang_v_r = np.linalg.norm(top_ang_v)
        vel_metric = 1 - np.minimum(lin_v_r + ang_v_r / 2.0, 5.0) / 5.0

        reward += np.maximum(rot_metric * 20 - 15, 0.)
        # print(np.maximum(rot_metric * 20 - 15, 0.))
        reward += xyz_metric * 5
        # print(xyz_metric * 5)
        reward += vel_metric * 5
        # print(vel_metric * 5)

        total_nf = 0
        cps_floor = p.getContactPoints(self.top_obj['id'], bottom_id, -1, -1)
        for cp in cps_floor:
            total_nf += cp[9]
        if np.abs(total_nf) > (self.top_obj['mass'] * 4.):       # mg        # TODO:tmp contact force hack
            meaningful_c = True
            reward += 5.0
        else:
            meaningful_c = False
        #     # reward += np.abs(total_nf) / 10.

        # not used when placing on floor
        btm_vels = p.getBaseVelocity(bottom_id)
        btm_linv = np.array(btm_vels[0])
        btm_angv = np.array(btm_vels[1])
        reward += np.maximum(-np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0) * 0.3
        # print(np.maximum(-np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0) * 0.3)

        diff_norm = self.robot.get_norm_diff_tar()
        reward += 15. / (diff_norm + 1.)
        # print(15. / (diff_norm + 1.))

        any_hand_contact = False
        hand_r = 0
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(self.top_obj['id'], self.robot.arm_id, -1, i)
            if len(cps) == 0:
                hand_r += 1.0   # the fewer links in contact, the better
            else:
                any_hand_contact = True
        # print(hand_r)
        reward += (hand_r - 15)

        if rot_metric > 0.9 and xyz_metric > 0.8 and vel_metric > 0.8 and meaningful_c:     # close to placing
            reward += 5.0
            # print("upright")
            if not any_hand_contact:
                reward += 20
                # print("no hand con")

        # print("r_total", reward)

        obs = self.getExtendedObservation()

        return obs, reward, False, {}

    def obj6DtoObs_UpVec(self, o_pos, o_orn):
        objObs = []
        o_pos = np.array(o_pos)

        # center o_pos
        if self.obs_noise:
            o_pos -= [self.tx, self.ty, 0]
        else:
            o_pos -= [self.tx_act, self.ty_act, 0]

        # TODO: scale up since we do not have obs normalization
        if self.obs_noise:
            o_pos = utils.perturb(self.np_random, o_pos, r=0.02) * 3.0
        else:
            o_pos = o_pos * 3.0

        o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        o_upv = [o_rotmat[2], o_rotmat[5], o_rotmat[8]]
        if self.obs_noise:
            o_upv = utils.perturb(self.np_random, o_upv, r=0.03)
        else:
            o_upv = o_upv

        objObs.extend(o_pos)
        objObs.extend(o_upv)

        return objObs

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

        # note, per-frame noise here
        if self.obs_noise:
            half_height_est = utils.perturb_scalar(self.np_random, self.top_obj['height']/2.0, 0.01)
        else:
            half_height_est = self.top_obj['height']/2.0
        self.observation.extend([half_height_est])

        # TODO: ball
        if self.random_top_shape:
            if self.top_obj['shape'] == p.GEOM_BOX:
                shape_info = [1, -1, -1]
            else:
                shape_info = [-1, 1, -1]
            self.observation.extend(shape_info)

        if self.use_gt_6d:
            self.vision_counter += 1
            if self.top_obj['id'] is None:
                self.observation.extend(self.obj6DtoObs_UpVec([0.,0,0], [0.,0,0,1]))
            else:
                if self.gt_only_init:
                    clPos, clOrn = self.t_pos, self.t_orn
                else:
                    # model both delayed and low-freq vision input
                    # every vision_skip steps, update cur 6D
                    # but feed policy with last-time updated 6D
                    if self.vision_counter % self.vision_skip == 0:
                        self.last_t_pos, self.last_t_orn = self.t_pos, self.t_orn
                        self.t_pos, self.t_orn = p.getBasePositionAndOrientation(self.top_obj['id'])
                    clPos, clOrn = self.last_t_pos, self.last_t_orn

                    # clPos, clOrn = p.getBasePositionAndOrientation(self.obj_id)

                    # print("feed into", clPos, clOrn)
                    # clPos_act, clOrn_act = p.getBasePositionAndOrientation(self.obj_id)
                    # print("act",  clPos_act, clOrn_act)

                self.observation.extend(self.obj6DtoObs_UpVec(clPos, clOrn))

            # if not self.place_floor and not self.gt_only_init:  # if stacking & real-time, include bottom 6D
            if self.btm_obj['id'] is None or self.gt_only_init: # TODO
                self.observation.extend(self.obj6DtoObs_UpVec([0.,0,0], [0.,0,0,1]))
            else:
                # model both delayed and low-freq vision input
                # every vision_skip steps, update cur 6D
                # but feed policy with last-time updated 6D
                if self.vision_counter % self.vision_skip == 0:
                    self.last_b_pos, self.last_b_orn = self.b_pos, self.b_orn
                    self.b_pos, self.b_orn = p.getBasePositionAndOrientation(self.btm_obj['id'])
                clPos, clOrn = self.last_b_pos, self.last_b_orn

                # print("b feed into", clPos, clOrn)
                # clPos_act, clOrn_act = p.getBasePositionAndOrientation(self.bottom_obj_id)
                # print("b act", clPos_act, clOrn_act)

                # clPos, clOrn = p.getBasePositionAndOrientation(self.bottom_obj_id)

                self.observation.extend(self.obj6DtoObs_UpVec(clPos, clOrn))

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


if __name__ == "__main__":
    env = InmoovShadowHandPlaceEnvV9()
    p.setPhysicsEngineParameter(numSolverIterations=200)
    env.seed(303)
    for _ in range(20):
        env.reset()
        env.robot.tar_fin_q = env.robot.get_q_dq(env.robot.fin_actdofs)[0]
        for test_t in range(300):
            thumb_pose = [-0.84771132,  0.60768666, -0.13419822,  0.52214954,  0.25141182]
            open_up_q = np.array([0.0, 0.0, 0.0] * 4 + thumb_pose)
            devi = open_up_q - env.robot.get_q_dq(env.robot.fin_actdofs)[0]
            if test_t < 200:
                env.robot.apply_action(np.array([0.0] * 7 + list(devi / 150.)))
            p.stepSimulation()
            # input("press enter")
            if env.renders:
                time.sleep(env._timeStep * 2.0)
        print(env.robot.get_q_dq(env.robot.fin_actdofs))
    # input("press enter")
    p.disconnect()
