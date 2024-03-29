# road map:
# collect a lot of final (hand state+ obj state) from Pi_G
# from the collected file, Calculate average \bar{T} palm->obj
#   calculate average of quaternions: just average the components as an estimate
# this average will be used to calc palm init pose for placing,
# such that on average obj init pose will be what we desire
# During training of placing,
# sample desired_obj_pos uniformly on the table in world frame (i.e. (T w->do))
#   issue: need to sample both the case of placing on table and placing on
#   another obj, z different
# sample (hand state+obj state)  from the file, calculate T palm->obj of sample
# reset palm pos T w->p = T w->do * \bar{T} o->p, find init q with IK
# reset obj pos T w->o = T w->p * T p->o


# Only To-->p is important in every data point, only need to collect that.
# finger q & finger tar q are invariant.

# store and copy finger tar pose at last step of grasping as well.
# during transporting keep the same tar pose as action.

# TODO: should we assume we have a policy that can "weld" the object?
# TODO: small objs


from .inmoov_shadow_hand_v2 import InmoovShadowNew
from .inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSession
from .inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSessionFlexWrist
from .inmoov_arm_obj_imaginary_sessions import URDFWriter

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


class InmoovShadowHandPlaceEnvNew(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True,
                 is_box=True,
                 is_small=False,
                 using_comfortable=False):
        self.renders = renders
        self.init_noise = init_noise        # bottom object noise
        self.up = up
        self.is_box = is_box
        self.using_comfortable = using_comfortable
        # self.grasp_pi_name = '0114_cyl_s_1'     # see experiment scripts, used to locate fin state pickle&create URDF
        self.grasp_pi_name = '0112_box'     # TODO    # see experiment scripts, used to locate fin state pickle&create URDF

        self.is_small = is_small

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

        self.tx = None
        self.ty = None
        self.tz = None
        self.desired_obj_pos_final = None

        self.btm_height = 0.18  # TODO: also need to cover put on floor later. / assume large cyl for now.
        self.start_height = 0.16         # TODO: 0.16 enough clearance? 0.13 small

        self.obs_noise = 1e-3

        self.saved_file = None

        with open(os.path.join(currentdir, 'assets/place_init_dist/final_states_'+self.grasp_pi_name+'.pickle'), 'rb') as handle:
            self.saved_file = pickle.load(handle)

        assert self.saved_file is not None
        self.init_states = self.saved_file['init_states']  # a list of dicts
        self.o_pos_pf_ave = self.saved_file['ave_obj_pos_in_palm']
        if using_comfortable:
            o_quat_pf_ave_ri = self.saved_file['ave_obj_quat_in_palm_rot_ivr']
            writer = URDFWriter()
            new_file = 'inmoov_arm_v2_2_obj_placing_' + self.grasp_pi_name + '.urdf'
            writer.add_obj(self.o_pos_pf_ave, o_quat_pf_ave_ri, new_file)
            pass
        else:
            self.o_quat_pf_ave = self.saved_file['ave_obj_quat_in_palm']
            self.o_quat_pf_ave /= np.linalg.norm(self.o_quat_pf_ave)        # in case not normalized

        # print(self.o_pos_pf_ave)
        # print(self.o_quat_pf_ave)
        # print(self.init_states[10])
        # print(self.init_states[51])
        # print(self.init_states[89])

        self.reset()    # and update init obs
        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)
        #
        # input("press enter")

    def get_reset_poses_comfortable(self):
        # return arm_q, & o_pos, o_quat, all_fin_q_init, tar_fin_q_init
        assert self.up
        sess = ImaginaryArmObjSession(filename='inmoov_arm_v2_2_obj_placing_'+self.grasp_pi_name+'.urdf')

        arm_q = None
        while arm_q is None:
            self.tz = self.btm_height
            if self.up:
                self.tx = self.np_random.uniform(low=0, high=0.25)
                self.ty = self.np_random.uniform(low=-0.1, high=0.5)
            else:
                self.tx = 0.0
                self.ty = 0.0
            desired_obj_pos = [self.tx, self.ty, self.start_height + self.tz]

            arm_q, _ = sess.get_most_comfortable_q_and_refangle_xz(desired_obj_pos[0],
                                                                   desired_obj_pos[1],
                                                                   desired_obj_pos[2])
        p_pos, p_quat = sess.get_link_pos_quat(sess.ee_id-1)

        init_state = self.sample_init_state()
        o_pos_pf = init_state['obj_pos_in_palm']
        o_quat_pf = init_state['obj_quat_in_palm']
        all_fin_q_init = init_state['all_fin_q']
        tar_fin_q_init = init_state['fin_tar_q']

        o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)

        return arm_q, o_pos, o_quat, all_fin_q_init, tar_fin_q_init

    def get_reset_poses_old(self, desired_obj_quat):
        # TODO, assume tx ty sampled already
        desired_obj_pos = [self.tx, self.ty, self.start_height + self.tz]

        p_pos_of_ave, p_quat_of_ave = p.invertTransform(self.o_pos_pf_ave, self.o_quat_pf_ave)
        p_pos, p_quat = p.multiplyTransforms(desired_obj_pos, desired_obj_quat,
                                             p_pos_of_ave, p_quat_of_ave)

        init_state = self.sample_init_state()
        o_pos_pf = init_state['obj_pos_in_palm']
        o_quat_pf = init_state['obj_quat_in_palm']
        all_fin_q_init = init_state['all_fin_q']
        tar_fin_q_init = init_state['fin_tar_q']

        o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)

        return p_pos, p_quat, o_pos, o_quat, all_fin_q_init, tar_fin_q_init

    def reset(self):
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        if self.np_random is None:
            self.seed(0)    # used once temporarily, will be overwritten outside by env
        self.robot = InmoovShadowNew(init_noise=False, timestep=self._timeStep)
        if self.np_random is not None:
            self.robot.np_random = self.np_random

        self.timer = 0

        arm_q = None
        if self.using_comfortable:
            arm_q, o_pos, o_quat, all_fin_q_init, tar_fin_q_init = self.get_reset_poses_comfortable()
        else:
            cand_angles = [0., 1.57, 3.14, -1.57]       # TODO: finer grid?
            cand_quats = [p.getQuaternionFromEuler([0, 0, cand_angle]) for cand_angle in cand_angles]
            ref = np.array([0.] * 3 + [-1.57] + [0.] * 3)
            self.tz = self.btm_height

            done = False
            cand_states = None
            while not done:
                if self.up:
                    self.tx = self.np_random.uniform(low=0, high=0.25)          # sample xy
                    self.ty = self.np_random.uniform(low=-0.1, high=0.5)
                    # self.tx = self.np_random.uniform(low=0, high=0.2)
                    # self.ty = self.np_random.uniform(low=-0.2, high=0.0)
                else:
                    self.tx = 0.0
                    self.ty = 0.0
                cand_states = [self.get_reset_poses_old(cand_quat) for cand_quat in cand_quats]
                cost = 1e30
                min_ind = None
                for ind, cand_state in enumerate(cand_states):
                    cand_arm_q = self.robot.solve_arm_IK(cand_state[0], cand_state[1])
                    if cand_arm_q is not None:
                        done = True
                        this_cost = np.linalg.norm(np.array(cand_arm_q) - ref)
                        if this_cost < cost:
                            min_ind = ind
                            arm_q = cand_arm_q
                            cost = this_cost
                # print(self.tx, self.ty, done, min_ind)
            _, _, o_pos, o_quat, all_fin_q_init, tar_fin_q_init = cand_states[min_ind]

        self.floor_id = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'),
                                   [0, 0, 0], useFixedBase=1)

        btm_xyz = np.array([self.tx, self.ty, self.tz/2.0])
        # if self.init_noise:
        #     btm_xyz += np.append(self.np_random.uniform(low=-0.01, high=0.01, size=2), 0)     # TODO
        self.bottom_obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                        btm_xyz, useFixedBase=0)

        if self.is_box:
            if self.is_small:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/box_small.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
            else:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/box.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
        else:
            if self.is_small:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cyl_small.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
            else:
                self.obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                         o_pos, o_quat, useFixedBase=0)
        self.obj_mass = p.getDynamicsInfo(self.obj_id, -1)[0]

        self.robot.reset_with_certain_arm_q_finger_states(arm_q, all_fin_q_init, tar_fin_q_init)

        p.changeDynamics(self.obj_id, -1, lateralFriction=1.0)
        p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
        p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)

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
            if self.renders:
                time.sleep(self._timeStep)
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
        desired_obj_pos_final = [self.tx, self.ty, 0.08 + self.tz]
        xyzMetric = 1 - (np.minimum(np.linalg.norm(np.array(desired_obj_pos_final) - np.array(clPos)), 0.45) / 0.10)
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
            meaningful_c = True
        else:
            meaningful_c = False

        btm_vels = p.getBaseVelocity(self.bottom_obj_id)
        btm_linv = np.array(btm_vels[0])
        btm_angv = np.array(btm_vels[1])
        reward += np.maximum(-np.linalg.norm(btm_linv) - np.linalg.norm(btm_angv), -10.0) * 0.3

        # print("nf", np.abs(total_nf))

        if rotMetric > 0.9 and xyzMetric > 0.8 and velMetric > 0.8 and meaningful_c:     # close to placing
            # print("close enough", self.timer)
            for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
                cps = p.getContactPoints(self.obj_id, self.robot.arm_id, -1, i)
                if len(cps) == 0:
                    reward += 0.5   # the fewer links in contact, the better
            # palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
            # dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(clPos)), 0.3)
            # reward += dist * 10.0
            # for i in self.robot.fin_tips[:4]:
            #     tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
            #     reward += np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.25)  # 4 finger tips
            # tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]  # thumb tip
            # reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.25) * 5.0

        # succeed = False
        obs = self.getExtendedObservation()     # call last obs before test period
        if self.timer == 300:
            # this is slightly different from mountain car's sparse reward,
            # where you are only rewarded when reaching a certain state
            # this is saying you must be at certain state at certain time (after test)
            for i in range(-1, p.getNumJoints(self.robot.arm_id)):
                p.setCollisionFilterPair(self.obj_id, self.robot.arm_id, -1, i, enableCollision=0)
                p.setCollisionFilterPair(self.bottom_obj_id, self.robot.arm_id, -1, i, enableCollision=0)
            for test_t in range(300):
                open_up_q = np.array([0.2, 0.2, 0.2] * 4 + [-0.4, 1.9, -0.0, 0.5, 0.0])
                devi = open_up_q - self.robot.get_q_dq(self.robot.fin_actdofs)[0]
                self.robot.apply_action(np.array([0.0]*7+list(devi/150.)))
                p.stepSimulation()
                if self.renders:
                    time.sleep(self._timeStep)
            clPosNow, _ = p.getBasePositionAndOrientation(self.obj_id)
            # dist = np.linalg.norm(np.array(self.desired_obj_pos_final) - np.array(clPosNow))
            if clPosNow[2] > self.tz:
                # succeed = True
                reward += 2000
                # print("good")
            # else:
            #     print("bad")

        return obs, reward, False, {}

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation()

        # clPos, clOrn = p.getBasePositionAndOrientation(self.cylinderId)
        # clPos = np.array(clPos)
        # clOrnMat = p.getMatrixFromQuaternion(clOrn)
        # clOrnMat = np.array(clOrnMat)
        #
        # self.observation.extend(list(clPos))
        # # self.observation.extend(list(clOrnMat))
        # self.observation.extend(list(clOrn))
        #
        # clVels = p.getBaseVelocity(self.cylinderId)
        # self.observation.extend(clVels[0])
        # self.observation.extend(clVels[1])

        # # TODO: somehow wrist sensor is very noisy, maybe not useful as obs
        # cf = np.array(self.robot.get_wrist_wrench())
        # cf[:3] /= (self.robot.maxForce * 3)
        # cf[3:] /= (self.robot.maxForce * 0.5)     # just in case there is not state normalization in ppo
        #
        # # print("wf", cf)

        curContact = []
        for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
            cps = p.getContactPoints(self.obj_id, self.robot.arm_id, -1, i)
            if len(cps) > 0:
                curContact.extend([1.0])
                # print("touch!!!")
            else:
                curContact.extend([-1.0])
        self.observation.extend(curContact)

        # curContact = []
        # for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
        #     cps = p.getContactPoints(bodyA=self.robot.arm_id, linkIndexA=i)
        #     con_this_link = False
        #     for cp in cps:
        #         if cp[1] != cp[2]:      # not self-collision of the robot
        #             con_this_link = True
        #             break
        #     if con_this_link:
        #         curContact.extend([1.0])
        #     else:
        #         curContact.extend([-1.0])
        # self.observation.extend(curContact)

        if self.up:
            xy = np.array([self.tx, self.ty])   # TODO: tx, ty wrt world origin
            self.observation.extend(list(xy + self.np_random.uniform(low=-self.obs_noise, high=-self.obs_noise, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-self.obs_noise, high=-self.obs_noise, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-self.obs_noise, high=-self.obs_noise, size=2)))

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
