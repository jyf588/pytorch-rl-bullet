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

# episode length 400

class InmoovShadowHandPlaceEnvV4(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=False,
                 init_noise=True,
                 up=True,
                 is_box=True,
                 is_small=False,
                 place_floor=False,
                 use_gt_6d=True,
                 gt_only_init=False,
                 grasp_pi_name=None
                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        self.is_box = is_box
        self.is_small = is_small
        self.place_floor = place_floor
        self.use_gt_6d = use_gt_6d
        self.gt_only_init = gt_only_init

        # TODO: hardcoded here
        if grasp_pi_name is None:
            if self.is_box:
                if self.is_small:
                    self.grasp_pi_name = '0120_box_s_1'
                else:
                    self.grasp_pi_name = '0120_box_l_1'
            else:
                if self.is_small:
                    self.grasp_pi_name = '0120_cyl_s_1'
                else:
                    self.grasp_pi_name = '0120_cyl_l_0'
        else:
            self.grasp_pi_name = grasp_pi_name


        self.half_obj_height = 0.065 if self.is_small else 0.09
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

        # TODO: tune this is not principled
        self.frameSkip = 3
        self.action_scale = np.array([0.004] * 7 + [0.008] * 17)  # shadow hand is 22-5=17dof

        self.tx = -1    # dummy
        self.ty = -1    # dummy
        self.tz = -1    # dummy
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
        if rotMetric < 0.9: return False

        if self.is_box:
            if self.is_small:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/box_small.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
            else:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/box.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
        else:
            if self.is_small:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cylinder_small.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
            else:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
        p.changeDynamics(self.obj_id, -1, lateralFriction=1.0)
        self.obj_mass = p.getDynamicsInfo(self.obj_id, -1)[0]

        return True

    def get_optimal_init_arm_q(self, desired_obj_pos):
        # TODO: desired obj init pos -> should add clearance to z.
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
                self.tx = self.np_random.uniform(low=0, high=0.25)
                self.ty = self.np_random.uniform(low=-0.1, high=0.5)
                # self.tx = self.np_random.uniform(low=0, high=0.2)
                # self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            else:
                self.tx = 0.0
                self.ty = 0.0

            desired_obj_pos = [self.tx, self.ty, self.start_clearance + self.tz]
            self.desired_obj_pos_final = [self.tx, self.ty, self.half_obj_height + self.tz]
            arm_q = self.get_optimal_init_arm_q(desired_obj_pos)
            if arm_q is None:
                continue
            else:
                return arm_q

    def reset(self):
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0

        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep, np_random=self.np_random)

        arm_q = self.sample_valid_arm_q()   # reset done during solving IK

        init_done = False
        while not init_done:
            init_state = self.sample_init_state()
            init_done = self.reset_robot_object_from_sample(init_state, arm_q)

        if self.place_floor:
            self.bottom_obj_id =p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.25, 0.2, 0.0],
                                             useFixedBase=1)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
        else:
            btm_xyz = np.array([self.tx, self.ty, self.tz/2.0])
            if self.init_noise:
                btm_xyz += np.append(self.np_random.uniform(low=-0.01, high=0.01, size=2), 0)
            self.bottom_obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                            btm_xyz, useFixedBase=0)
            self.floor_id = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.25, 0.2, 0.0],
                              useFixedBase=1)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
            p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)

        p.stepSimulation()      # TODO

        # init obj pose
        self.t_pos, self.t_orn = p.getBasePositionAndOrientation(self.obj_id)
        self.b_pos, self.b_orn = p.getBasePositionAndOrientation(self.bottom_obj_id)
        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    def sample_init_state(self):
        ran_ind = int(self.np_random.uniform(low=0, high=len(self.init_states) - 0.1))
        return self.init_states[ran_ind]

    def __del__(self):
        p.disconnect()

    def step(self, action):
        for _ in range(self.frameSkip):
            # action is not in -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)   # TODO
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            # if self.renders:
            #     time.sleep(self._timeStep * 1.0)
            self.timer += 1

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

        reward += rotMetric * 5
        reward += xyzMetric * 5
        reward += velMetric * 5

        total_nf = 0
        cps_floor = p.getContactPoints(self.obj_id, self.bottom_obj_id, -1, -1)
        for cp in cps_floor:
            total_nf += cp[9]
        if np.abs(total_nf) > (self.obj_mass*9.):       # mg
            # meaningful_c = True
            reward += 5.0
        # else:
        #     meaningful_c = False
        #     # reward += np.abs(total_nf) / 10.

        # not used when placing on floor
        btm_vels = p.getBaseVelocity(self.bottom_obj_id)
        btm_linv = np.array(btm_vels[0])
        btm_angv = np.array(btm_vels[1])
        reward += np.maximum(-np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0) * 0.3

        if rotMetric > 0.9 and xyzMetric > 0.8 and velMetric > 0.8:     # close to placing
            # print("close enough", self.timer)
            for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
                cps = p.getContactPoints(self.obj_id, self.robot.arm_id, -1, i)
                if len(cps) == 0:
                    reward += 0.5   # the fewer links in contact, the better
            palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
            dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(clPos)), 0.3)
            reward += dist * 10.0       # palm away from obj
            for i in self.robot.fin_tips[:4]:
                tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
                reward += np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.25) * 2.5  # 4 finger tips
            tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]  # thumb tip
            reward += np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.25) * 5.0   # away form obj

        # succeed = False
        obs = self.getExtendedObservation()     # call last obs before test period
        if self.timer == 300:
            # this is slightly different from mountain car's sparse reward,
            # where you are only rewarded when reaching a certain state
            # this is saying you must be at certain state at certain time (after test)
            # for i in range(-1, p.getNumJoints(self.robot.arm_id)):
            #     p.setCollisionFilterPair(self.obj_id, self.robot.arm_id, -1, i, enableCollision=0)
            #     p.setCollisionFilterPair(self.bottom_obj_id, self.robot.arm_id, -1, i, enableCollision=0)
            self.execute_release_traj()

            total_nf = 0
            cps_floor = p.getContactPoints(self.obj_id, self.bottom_obj_id, -1, -1)
            for cp in cps_floor:
                total_nf += cp[9]
            if np.abs(total_nf) > (self.obj_mass * 6.):  # mg
                meaningful_c = True
            else:
                meaningful_c = False
            _, upOrnNow = p.getBasePositionAndOrientation(self.obj_id)
            # btmPosNow, _ = p.getBasePositionAndOrientation(self.bottom_obj_id)
            z_axis, _ = p.multiplyTransforms([0, 0, 0], upOrnNow, [0, 0, 1], [0, 0, 0, 1])  # R_cl * unitz[0,0,1]
            rotMetric = np.array(z_axis).dot(np.array([0, 0, 1]))
            # dist = np.linalg.norm(np.array(btmPosNow) + [0., 0., self.tz/2+self.half_obj_height] - upPosNow)  # TODO
            # dist = np.linalg.norm(np.array(self.desired_obj_pos_final) - np.array(upPosNow))
            # reward += 3000 * np.exp(-(dist/0.03)**2)

            if meaningful_c and rotMetric > 0.6:        # TODO:tmp is this good for floor placing as well?
                # succeed = True
                reward += 3000

        return obs, reward, False, {}

    def execute_release_traj(self):

        cur_q = self.robot.get_q_dq(self.robot.fin_actdofs)[0]
        self.robot.tar_fin_q = cur_q
        for test_t in range(170):
            thumb_pose = list(cur_q[-5:])     # do not modify thumb
            open_up_q = np.array([0.1, 0.1, 0.1] * 4 + thumb_pose)
            devi = open_up_q - cur_q
            if test_t < 150:
                self.robot.apply_action(np.array([0.0] * 7 + list(devi / 150.)))
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep)

        self.robot.tar_arm_q = self.robot.get_q_dq(self.robot.arm_dofs)[0]
        self.robot.tar_fin_q = self.robot.get_q_dq(self.robot.fin_actdofs)[0]
        tar_wrist_xyz = np.array(self.robot.get_link_pos_quat(self.robot.ee_id)[0])
        dir = tar_wrist_xyz[:2] - [self.tx, self.ty]
        dir = dir / np.linalg.norm(dir)
        dir = np.array(list(dir) + [0.0])
        ik_q = None
        for test_t in range(170):
            if test_t < 150:
                tar_wrist_xyz += 0.001 * dir
                ik_q = p.calculateInverseKinematics(self.robot.arm_id, self.robot.ee_id, list(tar_wrist_xyz))
            self.robot.tar_arm_q = np.array(ik_q[:len(self.robot.arm_dofs)])
            self.robot.apply_action(np.array([0.0] * len(self.action_scale)))
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep)

    def obj6DtoObs(self, o_pos, o_orn):
        objObs = []
        o_pos = np.array(o_pos)
        o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        objObs.extend(list(self.perturb(o_pos, r=0.005)))
        objObs.extend(list(self.perturb(o_pos, r=0.005)))
        objObs.extend(list(self.perturb(o_rotmat, r=0.005)))
        return objObs

    def obj6DtoObs_UpVec(self, o_pos, o_orn):
        objObs = []
        o_pos = np.array(o_pos)
        o_rotmat = np.array(p.getMatrixFromQuaternion(o_orn))
        o_upv = [o_rotmat[2], o_rotmat[5], o_rotmat[8]]
        objObs.extend(list(self.perturb(o_pos, r=0.005)))
        objObs.extend(list(self.perturb(o_pos, r=0.005)))
        objObs.extend(list(self.perturb(o_upv, r=0.005)))
        objObs.extend(list(self.perturb(o_upv, r=0.005)))
        return objObs

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation()

        if self.use_gt_6d:
            if self.obj_id is None:
                self.observation.extend(self.obj6DtoObs_UpVec([0,0,0], [0,0,0,1]))  # TODO
            else:
                if self.gt_only_init:
                    clPos, clOrn = self.t_pos, self.t_orn
                else:
                    clPos, clOrn = p.getBasePositionAndOrientation(self.obj_id)
                self.observation.extend(self.obj6DtoObs_UpVec(clPos, clOrn))    # TODO
            if not self.place_floor and not self.gt_only_init:  # if stacking & real-time, include bottom 6D
                if self.bottom_obj_id is None:
                    self.observation.extend(self.obj6DtoObs_UpVec([0,0,0], [0,0,0,1]))    # TODO
                else:
                    clPos, clOrn = p.getBasePositionAndOrientation(self.bottom_obj_id)
                    self.observation.extend(self.obj6DtoObs_UpVec(clPos, clOrn))  # TODO

        #
        # clVels = p.getBaseVelocity(self.cylinderId)
        # self.observation.extend(clVels[0])
        # self.observation.extend(clVels[1])

        # # somehow wrist sensor is very noisy, maybe not useful as obs
        # cf = np.array(self.robot.get_wrist_wrench())
        # cf[:3] /= (self.robot.maxForce * 3)
        # cf[3:] /= (self.robot.maxForce * 0.5)     # just in case there is not state normalization in ppo
        #
        # # print("wf", cf)

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
            xy = np.array([self.tx, self.ty])   # TODO: tx, ty wrt world origin
            self.observation.extend(list(self.perturb(xy, r=0.005)))
            self.observation.extend(list(self.perturb(xy, r=0.005)))
            self.observation.extend(list(self.perturb(xy, r=0.005)))

        # if self.lastContact is not None:
        #     self.observation.extend(self.lastContact)
        # else:   # first step
        #     self.observation.extend(curContact)
        # self.lastContact = curContact.copy()

        # print("obv", self.observation)
        # print("max", np.max(np.abs(np.array(self.observation))))
        # print("min", np.min(np.abs(np.array(self.observation))))

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
    env = InmoovShadowHandPlaceEnvV4()
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
