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


# TODO: txyz will be given by vision module. tz is zero/btm height for grasping, obj frame at bottom.

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
                 ):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up

        self.random_top_shape = random_top_shape
        self.det_top_shape_ind = det_top_shape_ind

        self.cotrain_onstack_grasp = cotrain_onstack_grasp
        self.grasp_floor = grasp_floor

        self.obs_noise = obs_noise

        self.has_test_phase = has_test_phase
        self.test_start = 50

        self.n_best_cand = int(n_best_cand)

        # self.vary_angle_range = 0.6
        # self.obj_mass = -1  # dummy, 2b overwritten
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

        self.p_pos_of_init = [-0.18, 0.095, 0.11]
        self.p_quat_of_init = p.getQuaternionFromEuler([1.8, -1.57, 0])

        # if self.up:
        #     self.tx = None  # assigned later
        #     self.ty = None
        #     self.tx_act = None
        #     self.ty_act = None
        # else:
        #     self.tx = 0.
        #     self.ty = 0.    # constants
        #     self.tx_act = 0.
        #     self.ty_act = 0.
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
                                                     desired_obj_pos, self.table_id, n=self.n_best_cand)
            if len(arm_qs) == 0:
                continue
            else:
                arm_q = arm_qs[self.np_random.randint(len(arm_qs))]
                return arm_q

    # def __del__(self):
    #     p.disconnect()
    #     # self.sess.__del__()
    #
    # def get_reset_poses_comfortable(self):
    #     # return/ sample one arm_q (or palm 6D, later), est. obj init,
    #     # during testing, another central Bullet session will be calc a bunch of arm_q given a obj init pos
    #     assert self.up
    #
    #     sess = ImaginaryArmObjSession()
    #     cyl_init_pos = None
    #     arm_q = None
    #     while arm_q is None:
    #         cyl_init_pos = [0, 0, self.half_height + 0.001]     # obj rest on floor
    #         # if self.small:
    #         #     cyl_init_pos = [0, 0, 0.0651]
    #         # else:
    #         #     cyl_init_pos = [0, 0, 0.091]
    #         self.tx = self.np_random.uniform(low=-0.1, high=0.3)
    #         self.ty = self.np_random.uniform(low=-0.15, high=0.55)
    #         # self.tx = 0.14
    #         # self.ty = 0.3
    #         cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
    #
    #         arm_q, _ = sess.get_most_comfortable_q_and_refangle(self.tx, self.ty)
    #     # print(arm_q)
    #     return arm_q, cyl_init_pos
    #
    # def get_reset_poses_comfortable_range(self):
    #     assert self.up
    #     assert self.using_comfortable
    #
    #     sess = ImaginaryArmObjSessionFlexWrist()
    #     cyl_init_pos = None
    #     arm_q = None
    #     while arm_q is None:
    #         cyl_init_pos = [0, 0, self.half_height + 0.001]          # obj rest on floor
    #         # if self.small:
    #         #     cyl_init_pos = [0, 0, 0.0651]
    #         # else:
    #         #     cyl_init_pos = [0, 0, 0.091]
    #         self.tx = self.np_random.uniform(low=-0.1, high=0.3)
    #         self.ty = self.np_random.uniform(low=-0.15, high=0.55)
    #         # self.tx = 0.1
    #         # self.ty = 0.0
    #         cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
    #         vary_angle = self.np_random.uniform(low=-self.vary_angle_range, high=self.vary_angle_range)
    #         arm_q = sess.sample_one_comfortable_q(self.tx, self.ty, vary_angle)
    #     # print(arm_q)
    #     return arm_q, cyl_init_pos
    #
    # def get_reset_poses_old(self):
    #     # old way. return (modify) the palm 6D and est. obj init
    #     init_palm_quat = p.getQuaternionFromEuler([1.8, -1.57, 0])
    #
    #     init_palm_pos = [-0.18, 0.095, 0.11]
    #     cyl_init_pos = [0, 0, self.half_height+0.001]
    #     # if self.small:
    #     #     cyl_init_pos = [0, 0, 0.0651]
    #     #     init_palm_pos = [-0.18, 0.095, 0.075]   # absorbed by imaginary session
    #     # else:
    #     #     cyl_init_pos = [0, 0, 0.091]
    #     #     init_palm_pos = [-0.18, 0.095, 0.11]
    #
    #     if self.up:
    #         self.tx = self.np_random.uniform(low=0, high=0.2)
    #         self.ty = self.np_random.uniform(low=-0.2, high=0.0)
    #         # self.tx = 0.18
    #         # self.ty = -0.18
    #         cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
    #         init_palm_pos = np.array(init_palm_pos) + np.array([self.tx, self.ty, 0])
    #     return init_palm_pos, init_palm_quat, cyl_init_pos
    #
    # def create_prim_2_grasp(self, shape, dim, init_xyz, init_quat=(0,0,0,1)):
    #     # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    #     # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder
    #     # init_xyz vec3 of obj location
    #
    #     self.obj_mass = self.np_random.uniform(1.0, 5.0)
    #
    #     visual_shape_id = None
    #     collision_shape_id = None
    #     if shape == p.GEOM_BOX:
    #         visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim)
    #         collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=dim)
    #     elif shape == p.GEOM_CYLINDER:
    #         # visual_shape_id = p.createVisualShape(shapeType=shape, radius=dim[0], length=dim[1])
    #         visual_shape_id = p.createVisualShape(shape, dim[0], [1,1,1], dim[1])
    #         # collision_shape_id = p.createCollisionShape(shapeType=shape, radius=dim[0], length=dim[1])
    #         collision_shape_id = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    #     elif shape == p.GEOM_SPHERE:
    #         visual_shape_id = p.createVisualShape(shape, radius=dim[0])
    #         collision_shape_id = p.createCollisionShape(shape, radius=dim[0])
    #
    #     sid = p.createMultiBody(baseMass=self.obj_mass, baseInertialFramePosition=[0, 0, 0],
    #                            baseCollisionShapeIndex=collision_shape_id,
    #                            baseVisualShapeIndex=visual_shape_id,
    #                            basePosition=init_xyz, baseOrientation=init_quat)
    #     return sid

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self.timer = 0

        if self.cotrain_onstack_grasp:
            self.grasp_floor = self.np_random.randint(10) > 5   # 40%, TODO

        self.table_id = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), utils.TABLE_OFFSET,
                                   useFixedBase=1)
        mu_f = self.np_random.uniform(utils.MU_MIN, utils.MU_MAX)
        p.changeDynamics(self.table_id, -1, lateralFriction=mu_f)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep, np_random=self.np_random)

        arm_q = self.sample_valid_arm_q()  # reset done during solving IK
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

            btm_xyz = np.array([self.tx_act, self.ty_act, self.tz_act / 2.0])
            btm_quat = p.getQuaternionFromEuler([0., 0., self.np_random.uniform(low=0, high=2.0 * math.pi)])
            bo['id'] = utils.create_sym_prim_shape_helper(bo, btm_xyz, btm_quat)

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

        top_xyz = np.array([self.tx, self.ty, self.tz_act + to['height']/2.0])
        top_quat = p.getQuaternionFromEuler([0., 0., self.np_random.uniform(low=0, high=2.0 * math.pi)])
        to['id'] = utils.create_sym_prim_shape_helper(to, top_xyz, top_quat)

        # note, one-time (same for all frames) noise from init vision module
        if self.obs_noise:
            self.half_height_est = utils.perturb_scalar(self.np_random, self.top_obj['height']/2.0, 0.01)
        else:
            self.half_height_est = self.top_obj['height']/2.0

        # # self.isBox = bool(self.np_random.randint(2)) if self.random_shape else self.default_box
        #
        # # self.shape_ind = self.np_random.randint(3) - 1 if self.random_shape else self.default_box
        # self.shape_ind = self.np_random.randint(2) if self.random_top_shape else self.det_top_shape_ind
        # # TODO: only random between box or cyl for now
        #
        # self.half_height = self.np_random.uniform(low=0.055, high=0.09) if self.random_size else 0.07
        # self.half_width = self.np_random.uniform(low=0.03, high=0.05) if self.random_size else 0.04  # aka radius
        # # self.half_height is the true half_height, vision module one will be noisy
        # # Note: only estimate once before grasping.
        # if self.obs_noise:
        #     self.half_height_est = self.half_height + self.np_random.uniform(low=-0.01, high=0.01)
        # else:
        #     self.half_height_est = self.half_height
        #
        # if self.shape_ind == -1:
        #     self.half_height *= 0.75        # TODO
        #
        # if self.using_comfortable:
        #     if self.using_comfortable_range:
        #         arm_q, obj_init_xyz = self.get_reset_poses_comfortable_range()
        #     else:
        #         arm_q, obj_init_xyz = self.get_reset_poses_comfortable()
        # else:
        #     init_palm_pos, init_palm_quat, obj_init_xyz = self.get_reset_poses_old()
        #
        # self.tx_act = self.tx
        # self.ty_act = self.ty
        # if self.init_noise:
        #     noise = np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)
        #     self.tx_act += noise[0]
        #     self.ty_act += noise[1]
        #     obj_init_xyz += noise
        #
        # if self.shape_ind == 1:
        #     self.dim = [self.half_width*0.8, self.half_width*0.8, self.half_height]    # TODO
        #     if not self.box_rot:
        #         self.obj_id = self.create_prim_2_grasp(p.GEOM_BOX, self.dim, obj_init_xyz)
        #     else:
        #         quat = p.getQuaternionFromEuler([0., 0., self.np_random.uniform(low=0, high=2.0*math.pi)])
        #         self.obj_id = self.create_prim_2_grasp(p.GEOM_BOX, self.dim, obj_init_xyz, quat)
        # elif self.shape_ind == 0:
        #     self.dim = [self.half_width, self.half_height*2.0]
        #     self.obj_id = self.create_prim_2_grasp(p.GEOM_CYLINDER, self.dim, obj_init_xyz)
        # elif self.shape_ind == -1:
        #     self.dim = [self.half_height]
        #     self.obj_id = self.create_prim_2_grasp(p.GEOM_SPHERE, self.dim, obj_init_xyz)
        #
        # # self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.1, 0.2, 0.0],
        # #                       useFixedBase=1)
        #
        # mu_obj = self.np_random.uniform(0.8, 1.2)
        # # mu_f = self.np_random.uniform(0.8, 1.2)
        # p.changeDynamics(self.obj_id, -1, lateralFriction=mu_obj)
        # # p.changeDynamics(self.floorId, -1, lateralFriction=mu_f)
        #
        # # self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep, np_random=self.np_random,
        # #                              conservative_clip=False)
        #
        # if self.using_comfortable:
        #     self.robot.reset_with_certain_arm_q(arm_q)
        # else:
        #     self.robot.reset(list(init_palm_pos), init_palm_quat)       # reset at last to test collision
        # self.lastContact = None

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
                # for i in range(-1, p.getNumJoints(self.robot.arm_id)):
                #     p.setCollisionFilterPair(self.floorId, self.robot.arm_id, -1, i, enableCollision=0)
                _, quat = p.getBasePositionAndOrientation(self.top_obj['id'])
                _, quat_inv = p.invertTransform([0, 0, 0], quat)
                force_local, _ = p.multiplyTransforms([0, 0, 0], quat_inv, self.force_global, [0, 0, 0, 1])
                p.applyExternalForce(self.top_obj['id'], -1, force_local, [0, 0, 0], flags=p.LINK_FRAME)

        for _ in range(self.control_skip):
            # action is in -1,1
            if action is not None:
                # action = np.clip(np.array(action), -1, 1)   # TODO
                self.act = action
                act_array = self.act * self.action_scale

                self.robot.apply_action(act_array)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        reward = 3.0

        # rewards is height of target object
        top_pos, _ = p.getBasePositionAndOrientation(self.top_obj['id'])
        # palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
        # dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(top_pos)), 0.5)
        # reward += -dist * 2.0

        top_xyz_ideal = np.array([self.tx, self.ty, self.tz_act + self.top_obj['height'] / 2.0 + 0.05])
        reward += -np.minimum(np.linalg.norm(top_xyz_ideal - np.array(top_pos)), 0.4) * 12.0    # TODO

        for i in self.robot.fin_tips[:4]:
            tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
            reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(top_pos)), 0.5)  # 4 finger tips
        tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]      # thumb tip
        reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(top_pos)), 0.5) * 5.0

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

        reward -= self.robot.get_4_finger_deviation() * 0.5

        if top_pos[2] < self.tz_act:    # object dropped
            reward += -15.

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
            # if self.shape_ind == 1:
            #     shape = p.GEOM_BOX
            # elif self.shape_ind == 0:
            #     shape = p.GEOM_CYLINDER
            # elif self.shape_ind == -1:
            #     shape = p.GEOM_SPHERE
            # else:
            #     shape = None
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
