from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew

import pybullet as p
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math
import pickle

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class InmoovShadowHandPlaceEnvV7(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=False,
                 init_noise=True,   # variation during reset
                 up=True,
                 random_shape=False,
                 random_size=True,
                 default_box=False,  # if not random shape, false: cylinder as default
                 place_floor=False,
                 use_gt_6d=True,
                 gt_only_init=False,
                 grasp_pi_name=None,
                 exclude_hard=True,
                 vision_skip=1,
                 control_skip=3,
                 obs_noise=False    # noisy (imperfect) observation
                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        self.random_shape = random_shape
        self.random_size = random_size
        self.default_box = default_box
        self.place_floor = place_floor
        self.use_gt_6d = use_gt_6d
        self.gt_only_init = gt_only_init
        self.exclude_hard = exclude_hard
        self.obs_noise = obs_noise

        self.vision_skip = vision_skip
        self.vision_counter = 0

        self.hard_orn_thres = 0.9
        self.obj_mass = 3.5
        self.half_height = -1   # dummy, to be overwritten

        # TODO: hardcoded here
        if grasp_pi_name is None:
            if not random_shape:
                if default_box:
                    self.grasp_pi_name = '0302_box_20_n_80_99'
                else:
                    self.grasp_pi_name = '0302_cyl_4_n_80_100'   # TODO: ball
            else:
                pass    # TODO
        else:
            self.grasp_pi_name = grasp_pi_name

        # self.half_obj_height = 0.065 if self.is_small else 0.09
        self.start_clearance = 0.14
        self.btm_obj_height = 0.18      # always place on larger one
        self.cand_angles = [0., 3.14/3, 6.28/3, 3.14, -6.28/3, -3.14/3]  # TODO: finer grid?
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
        self.tx_act = -1
        self.ty_act = -1
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

        self.obj_id = None
        self.bottom_obj_id = None

        self.seed(0)    # used once temporarily, will be overwritten outside by env
        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep, np_random=self.np_random)

        self.observation = self.getExtendedObservation()
        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)
        #
        # input("press enter")

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(np.array(arr) + self.np_random.uniform(low=-r, high=r, size=len(arr)))

    def create_prim_2_grasp(self, shape, dim, init_xyz, init_quat=(0,0,0,1)):
        # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
        # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder
        # init_xyz vec3 of obj location
        visual_shape_id = None
        collision_shape_id = None
        if shape == p.GEOM_BOX:
            visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim)
            collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=dim)
        elif shape == p.GEOM_CYLINDER:
            # visual_shape_id = p.createVisualShape(shapeType=shape, radius=dim[0], length=dim[1])
            visual_shape_id = p.createVisualShape(shape, dim[0], [1,1,1], dim[1])
            # collision_shape_id = p.createCollisionShape(shapeType=shape, radius=dim[0], length=dim[1])
            collision_shape_id = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
        elif shape == p.GEOM_SPHERE:
            pass
            # visual_shape_id = p.createVisualShape(shape, radius=dim[0])
            # collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

        sid = p.createMultiBody(baseMass=self.obj_mass, baseInertialFramePosition=[0, 0, 0],
                               baseCollisionShapeIndex=collision_shape_id,
                               baseVisualShapeIndex=visual_shape_id,
                               basePosition=init_xyz, baseOrientation=init_quat)
        return sid

    def reset_robot_object_from_sample(self, state, arm_q):
        o_pos_pf = state['obj_pos_in_palm']
        o_quat_pf = state['obj_quat_in_palm']
        if self.init_noise:
            o_pos_pf = list(self.perturb(o_pos_pf, 0.005))
            o_quat_pf = list(self.perturb(o_quat_pf, 0.005))
        all_fin_q_init = state['all_fin_q']
        tar_fin_q_init = state['fin_tar_q']

        self.robot.reset_with_certain_arm_q_finger_states(arm_q, all_fin_q_init, tar_fin_q_init)

        p_pos, p_quat = self.robot.get_link_pos_quat(self.robot.ee_id)
        o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)

        z_axis, _ = p.multiplyTransforms([0, 0, 0], o_quat, [0, 0, 1], [0, 0, 0, 1])  # R_cl * unitz[0,0,1]
        rotMetric = np.array(z_axis).dot(np.array([0, 0, 1]))
        # print(rotMetric, rotMetric)
        if self.exclude_hard and rotMetric < self.hard_orn_thres: return False

        self.is_box = True if state['obj_shape'] == p.GEOM_BOX else False   # TODO: ball
        self.dim = state['obj_dim']
        self.obj_id = self.create_prim_2_grasp(state['obj_shape'], self.dim, o_pos, o_quat)

        if self.is_box:
            self.half_height = self.dim[-1]
        else:
            self.half_height = self.dim[-1] / 2.0    # TODO: ball

        mu_obj = self.np_random.uniform(0.8, 1.2)
        p.changeDynamics(self.obj_id, -1, lateralFriction=mu_obj)

        return True

    def get_optimal_init_arm_q(self, desired_obj_pos):
        # TODO: desired obj init pos -> should add clearance to z.
        # uses (self.o_pos_pf_ave, self.o_quat_pf_ave), so set mean stats to load properly
        arm_q = None
        cost = 1e30
        ref = np.array([0.] * 3 + [-1.57] + [0.] * 3)
        for ind, cand_quat in enumerate(self.cand_quats):
            p_pos_of_ave, p_quat_of_ave = p.invertTransform(self.o_pos_pf_ave, self.o_quat_pf_ave)
            p_pos, p_quat = p.multiplyTransforms(desired_obj_pos, cand_quat,
                                                 p_pos_of_ave, p_quat_of_ave)
            cand_arm_q = self.robot.solve_arm_IK(p_pos, p_quat)
            if cand_arm_q is not None:
                this_cost = np.sum(np.abs(np.array(cand_arm_q) - ref))  # change to l1
                if this_cost < cost:
                    arm_q = cand_arm_q
                    cost = this_cost
        return arm_q

    def sample_valid_arm_q(self):
        self.tz = self.btm_obj_height if not self.place_floor else 0.0
        while True:
            if self.up:
                self.tx = self.np_random.uniform(low=0, high=0.3)
                self.ty = self.np_random.uniform(low=-0.1, high=0.5)
                # self.tx = self.np_random.uniform(low=0, high=0.2)
                # self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            else:
                self.tx = 0.0
                self.ty = 0.0

            desired_obj_pos = [self.tx, self.ty, self.start_clearance + self.tz]
            self.desired_obj_pos_final = [self.tx, self.ty, self.half_height + self.tz]
            arm_q = self.get_optimal_init_arm_q(desired_obj_pos)
            if arm_q is None:
                continue
            else:
                return arm_q

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0
        self.vision_counter = 0

        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep, np_random=self.np_random)

        arm_q = self.sample_valid_arm_q()   # reset done during solving IK

        init_done = False
        while not init_done:
            init_state = self.sample_init_state()
            init_done = self.reset_robot_object_from_sample(init_state, arm_q)

        if self.place_floor:
            self.bottom_obj_id =p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.2, 0.2, 0.0],
                                             useFixedBase=1)
            mu_f = self.np_random.uniform(0.8, 1.2)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=mu_f)
        else:
            self.tx_act = self.tx
            self.ty_act = self.ty
            if self.init_noise:
                self.tx_act += self.np_random.uniform(low=-0.015, high=0.015)
                self.ty_act += self.np_random.uniform(low=-0.015, high=0.015)
            btm_xyz = np.array([self.tx_act, self.ty_act, self.tz/2.0])

            self.bottom_obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                            btm_xyz, useFixedBase=0)
            self.floor_id = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.2, 0.2, 0.0],
                              useFixedBase=1)
            mu_f = self.np_random.uniform(0.8, 1.2)
            mu_b = self.np_random.uniform(0.8, 1.2)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=mu_b)
            p.changeDynamics(self.floor_id, -1, lateralFriction=mu_f)

        p.stepSimulation()      # TODO

        # init obj pose
        self.t_pos, self.t_orn = p.getBasePositionAndOrientation(self.obj_id)
        self.last_t_pos, self.last_t_orn = p.getBasePositionAndOrientation(self.obj_id)
        self.b_pos, self.b_orn = p.getBasePositionAndOrientation(self.bottom_obj_id)
        self.last_b_pos, self.last_b_orn = p.getBasePositionAndOrientation(self.bottom_obj_id)
        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    def sample_init_state(self):
        ran_ind = int(self.np_random.uniform(low=0, high=len(self.init_states) - 0.1))
        return self.init_states[ran_ind]

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

        reward = 0.
        clPos, clQuat = p.getBasePositionAndOrientation(self.obj_id)
        clVels = p.getBaseVelocity(self.obj_id)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])

        # we only care about the upright(z) direction
        z_axis, _ = p.multiplyTransforms([0, 0, 0], clQuat, [0, 0, 1], [0, 0, 0, 1])          # R_cl * unitz[0,0,1]
        rotMetric = np.array(z_axis).dot(np.array([0, 0, 1]))

        # enlarge 0.15 -> 0.45
        # xyzMetric = 1 - (np.minimum(np.linalg.norm(np.array(self.desired_obj_pos_final) - np.array(clPos)), 0.45) / 0.15)
        # TODO:tmp change to xy metric, allow it to free drop
        xyzMetric = 1 - (np.minimum(np.linalg.norm(np.array(self.desired_obj_pos_final[:2]) - np.array(clPos[:2])), 0.45) / 0.15)
        linV_R = np.linalg.norm(clLinV)
        angV_R = np.linalg.norm(clAngV)
        velMetric = 1 - np.minimum(linV_R + angV_R / 2.0, 5.0) / 5.0

        reward += np.maximum(rotMetric * 20 - 15, 0.)
        # print(np.maximum(rotMetric * 20 - 15, 0.))
        reward += xyzMetric * 5
        # print(xyzMetric * 5)
        reward += velMetric * 5
        # print(velMetric * 5)

        total_nf = 0
        cps_floor = p.getContactPoints(self.obj_id, self.bottom_obj_id, -1, -1)
        for cp in cps_floor:
            total_nf += cp[9]
        if np.abs(total_nf) > (self.obj_mass*4.):       # mg        # TODO:tmp contact force hack
            meaningful_c = True
            reward += 5.0
        else:
            meaningful_c = False
        #     # reward += np.abs(total_nf) / 10.

        # not used when placing on floor
        btm_vels = p.getBaseVelocity(self.bottom_obj_id)
        btm_linv = np.array(btm_vels[0])
        btm_angv = np.array(btm_vels[1])
        reward += np.maximum(-np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0) * 0.3
        # print(np.maximum(-np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0) * 0.3)

        diff_norm = self.robot.get_norm_diff_tar()
        reward += 15. / (diff_norm + 1.)
        # print(15. / (diff_norm + 1.))

        anyHandContact = False
        hand_r = 0
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(self.obj_id, self.robot.arm_id, -1, i)
            if len(cps) == 0:
                hand_r += 1.0   # the fewer links in contact, the better
            else:
                anyHandContact = True
        # print(hand_r)
        reward += (hand_r - 15)

        if rotMetric > 0.9 and xyzMetric > 0.8 and velMetric > 0.8 and meaningful_c:     # close to placing
            reward += 5.0
            # print("upright")
            if not anyHandContact:
                reward += 20
                # print("no hand con")

        # print("r_total", reward)

        obs = self.getExtendedObservation()

        return obs, reward, False, {}

    # def obj6DtoObs(self, o_pos, o_orn):
    #     objObs = []
    #     o_pos = np.array(o_pos)
    #     o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
    #     objObs.extend(list(self.perturb(o_pos, r=0.005)))
    #     objObs.extend(list(self.perturb(o_pos, r=0.005)))
    #     objObs.extend(list(self.perturb(o_rotmat, r=0.005)))
    #     return objObs

    def obj6DtoObs_UpVec(self, o_pos, o_orn):
        objObs = []
        o_pos = np.array(o_pos)
        if self.up:                         # TODO: center o_pos
            if self.obs_noise:
                o_pos -= [self.tx, self.ty, 0]
            else:
                o_pos -= [self.tx_act, self.ty_act, 0]

        # TODO: scale up since we do not have obs normalization
        if self.obs_noise:
            o_pos = self.perturb(o_pos, r=0.02) * 3.0
        else:
            o_pos = o_pos * 3.0

        o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        o_upv = [o_rotmat[2], o_rotmat[5], o_rotmat[8]]
        if self.obs_noise:
            o_upv = self.perturb(o_upv, r=0.03)
        else:
            o_upv = o_upv

        if self.obs_noise:
            objObs.extend(list(self.perturb(o_pos, r=0.09)))
            objObs.extend(list(self.perturb(o_pos, r=0.09)))
            objObs.extend(list(self.perturb(o_upv, r=0.03)))
            objObs.extend(list(self.perturb(o_upv, r=0.03)))
        else:
            objObs.extend(list(self.perturb(o_pos, r=0.02)))
            objObs.extend(list(self.perturb(o_pos, r=0.02)))
            objObs.extend(list(self.perturb(o_upv, r=0.01)))
            objObs.extend(list(self.perturb(o_upv, r=0.01)))

        # o_pos = o_pos * 3.0
        # o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        # o_upv = [o_rotmat[2], o_rotmat[5], o_rotmat[8]]
        #
        # objObs.extend(list(self.perturb(o_pos, r=0.10)))
        # objObs.extend(list(self.perturb(o_pos, r=0.10)))
        # objObs.extend(list(self.perturb(o_upv, r=0.04)))
        # objObs.extend(list(self.perturb(o_upv, r=0.04)))
        return objObs

    # change to tar pos fin pos diff
    # change to tx ty diff

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

        if self.up:
            xy = np.array([self.tx, self.ty])
            self.observation.extend(list(xy))
            if self.obs_noise:
                self.observation.extend(list(xy))
            else:
                self.observation.extend([self.tx_act, self.ty_act])

        if self.random_size:
            if self.obs_noise:
                self.half_height_est = self.half_height + self.np_random.uniform(low=-0.01, high=0.01)
            else:
                self.half_height_est = self.half_height
            self.observation.extend([self.half_height_est])

        # TODO: if random_shape

        if self.use_gt_6d:
            self.vision_counter += 1
            if self.obj_id is None:
                self.observation.extend(self.obj6DtoObs_UpVec([0,0,0], [0,0,0,1]))  # TODO
            else:
                if self.gt_only_init:
                    clPos, clOrn = self.t_pos, self.t_orn
                else:
                    # # model both delayed and low-freq vision input
                    # # every vision_skip steps, update cur 6D
                    # # but feed policy with last-time updated 6D
                    # if self.vision_counter % self.vision_skip == 0:
                    #     self.last_t_pos, self.last_t_orn = self.t_pos, self.t_orn
                    #     self.t_pos, self.t_orn = p.getBasePositionAndOrientation(self.obj_id)
                    # clPos, clOrn = self.last_t_pos, self.last_t_orn

                    clPos, clOrn = p.getBasePositionAndOrientation(self.obj_id)

                    # print("feed into", clPos, clOrn)
                    # clPos_act, clOrn_act = p.getBasePositionAndOrientation(self.obj_id)
                    # print("act",  clPos_act, clOrn_act)

                self.observation.extend(self.obj6DtoObs_UpVec(clPos, clOrn))    # TODO
            if not self.place_floor and not self.gt_only_init:  # if stacking & real-time, include bottom 6D
                if self.bottom_obj_id is None:
                    self.observation.extend(self.obj6DtoObs_UpVec([0,0,0], [0,0,0,1]))    # TODO
                else:
                    # # model both delayed and low-freq vision input
                    # # every vision_skip steps, update cur 6D
                    # # but feed policy with last-time updated 6D
                    # if self.vision_counter % self.vision_skip == 0:
                    #     self.last_b_pos, self.last_b_orn = self.b_pos, self.b_orn
                    #     self.b_pos, self.b_orn = p.getBasePositionAndOrientation(self.bottom_obj_id)
                    # clPos, clOrn = self.last_b_pos, self.last_b_orn

                    # print("b feed into", clPos, clOrn)
                    # clPos_act, clOrn_act = p.getBasePositionAndOrientation(self.bottom_obj_id)
                    # print("b act", clPos_act, clOrn_act)

                    clPos, clOrn = p.getBasePositionAndOrientation(self.bottom_obj_id)

                    self.observation.extend(self.obj6DtoObs_UpVec(clPos, clOrn))  # TODO

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
    env = InmoovShadowHandPlaceEnvV7()
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
