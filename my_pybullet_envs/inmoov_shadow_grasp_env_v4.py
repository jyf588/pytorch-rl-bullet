from .inmoov_shadow_hand_v2 import InmoovShadowNew
from .inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSession
from .inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSessionFlexWrist

import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# episode length 400

# TODO: txyz will be given by vision module. tz is zero for grasping, obj frame at bottom.


class InmoovShadowHandGraspEnvV4(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=True,
                 random_shape=False,
                 random_size=True,
                 default_box=1,      # if not random shape, 1 box, 0 cyl, -1 sphere, bad legacy naming
                 using_comfortable=True,
                 using_comfortable_range=False):
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        self.random_shape = random_shape
        self.random_size = random_size
        self.default_box = default_box
        self.using_comfortable = using_comfortable
        self.using_comfortable_range = using_comfortable_range

        self.vary_angle_range = 0.6
        self.obj_mass = 3.5

        self._timeStep = 1. / 240.
        if self.renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)     # this session seems always 0
        self.np_random = None
        self.robot = None
        self.seed(0)  # used once temporarily, will be overwritten outside by env
        self.robot = None
        self.viewer = None

        self.final_states = []  # wont be cleared unless call clear function

        self.frameSkip = 3
        self.action_scale = np.array([0.004] * 7 + [0.008] * 17)  # shadow hand is 22-5=17dof

        if self.up:
            self.tx = None  # assigned later
            self.ty = None
        else:
            self.tx = 0.
            self.ty = 0.    # constants

        self.reset()    # and update init

        action_dim = len(self.action_scale)
        self.act = self.action_scale * 0.0
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(self.observation)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def __del__(self):
        p.disconnect()
        # self.sess.__del__()

    def get_reset_poses_comfortable(self):
        # return/ sample one arm_q (or palm 6D, later), est. obj init,
        # during testing, another central Bullet session will be calc a bunch of arm_q given a obj init pos
        assert self.up

        sess = ImaginaryArmObjSession()
        cyl_init_pos = None
        arm_q = None
        while arm_q is None:
            cyl_init_pos = [0, 0, self.half_height + 0.001]     # obj rest on floor
            # if self.small:
            #     cyl_init_pos = [0, 0, 0.0651]
            # else:
            #     cyl_init_pos = [0, 0, 0.091]
            self.tx = self.np_random.uniform(low=0, high=0.3)
            self.ty = self.np_random.uniform(low=-0.1, high=0.5)
            # self.tx = 0.14
            # self.ty = 0.3
            cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])

            arm_q, _ = sess.get_most_comfortable_q_and_refangle(self.tx, self.ty)
        # print(arm_q)
        return arm_q, cyl_init_pos

    def get_reset_poses_comfortable_range(self):
        assert self.up
        assert self.using_comfortable

        sess = ImaginaryArmObjSessionFlexWrist()
        cyl_init_pos = None
        arm_q = None
        while arm_q is None:
            cyl_init_pos = [0, 0, self.half_height + 0.001]          # obj rest on floor
            # if self.small:
            #     cyl_init_pos = [0, 0, 0.0651]
            # else:
            #     cyl_init_pos = [0, 0, 0.091]
            self.tx = self.np_random.uniform(low=0, high=0.3)
            self.ty = self.np_random.uniform(low=-0.1, high=0.5)
            # self.tx = 0.1
            # self.ty = 0.0
            cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
            vary_angle = self.np_random.uniform(low=-self.vary_angle_range, high=self.vary_angle_range)
            arm_q = sess.sample_one_comfortable_q(self.tx, self.ty, vary_angle)
        # print(arm_q)
        return arm_q, cyl_init_pos

    def get_reset_poses_old(self):
        # old way. return (modify) the palm 6D and est. obj init
        init_palm_quat = p.getQuaternionFromEuler([1.8, -1.57, 0])

        init_palm_pos = [-0.18, 0.095, 0.11]
        cyl_init_pos = [0, 0, self.half_height+0.001]
        # if self.small:
        #     cyl_init_pos = [0, 0, 0.0651]
        #     init_palm_pos = [-0.18, 0.095, 0.075]   # absorbed by imaginary session
        # else:
        #     cyl_init_pos = [0, 0, 0.091]
        #     init_palm_pos = [-0.18, 0.095, 0.11]

        if self.up:
            self.tx = self.np_random.uniform(low=0, high=0.2)
            self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            # self.tx = 0.18
            # self.ty = -0.18
            cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
            init_palm_pos = np.array(init_palm_pos) + np.array([self.tx, self.ty, 0])
        return init_palm_pos, init_palm_quat, cyl_init_pos

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
            visual_shape_id = p.createVisualShape(shape, radius=dim[0])
            collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

        sid = p.createMultiBody(baseMass=self.obj_mass, baseInertialFramePosition=[0, 0, 0],
                               baseCollisionShapeIndex=collision_shape_id,
                               baseVisualShapeIndex=visual_shape_id,
                               basePosition=init_xyz, baseOrientation=init_quat)
        return sid

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        # self.isBox = bool(self.np_random.randint(2)) if self.random_shape else self.default_box

        self.shape_ind = self.np_random.randint(3) - 1 if self.random_shape else self.default_box
        self.half_height = self.np_random.uniform(low=0.055, high=0.09) if self.random_size else 0.07
        self.half_width = self.np_random.uniform(low=0.03, high=0.05) if self.random_size else 0.04  # aka radius
        if self.shape_ind == -1:
            self.half_height *= 0.75        # TODO

        if self.using_comfortable:
            if self.using_comfortable_range:
                arm_q, obj_init_xyz = self.get_reset_poses_comfortable_range()
            else:
                arm_q, obj_init_xyz = self.get_reset_poses_comfortable()
        else:
            init_palm_pos, init_palm_quat, obj_init_xyz = self.get_reset_poses_old()

        if self.init_noise:
            obj_init_xyz += np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)

        if self.shape_ind == 1:
            self.dim = [self.half_width*0.8, self.half_width*0.8, self.half_height]    # TODO
            self.obj_id = self.create_prim_2_grasp(p.GEOM_BOX, self.dim, obj_init_xyz)
        elif self.shape_ind == 0:
            self.dim = [self.half_width, self.half_height*2.0]
            self.obj_id = self.create_prim_2_grasp(p.GEOM_CYLINDER, self.dim, obj_init_xyz)
        elif self.shape_ind == -1:
            self.dim = [self.half_height]
            self.obj_id = self.create_prim_2_grasp(p.GEOM_SPHERE, self.dim, obj_init_xyz)

        self.floorId = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.2, 0.2, 0.0],
                              useFixedBase=1)
        p.changeDynamics(self.obj_id, -1, lateralFriction=1.0)
        p.changeDynamics(self.floorId, -1, lateralFriction=1.0)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep, np_random=self.np_random,
                                     conservative_clip=False)

        if self.using_comfortable:
            self.robot.reset_with_certain_arm_q(arm_q)
        else:
            self.robot.reset(list(init_palm_pos), init_palm_quat)       # reset at last to test collision

        self.timer = 0
        self.lastContact = None
        self.observation = self.getExtendedObservation()

        return np.array(self.observation)

    def step(self, action):
        if self.timer > 100*self.frameSkip:
            p.setCollisionFilterPair(self.obj_id, self.floorId, -1, -1, enableCollision=0)
            # for i in range(-1, p.getNumJoints(self.robot.arm_id)):
            #     p.setCollisionFilterPair(self.floorId, self.robot.arm_id, -1, i, enableCollision=0)

        for _ in range(self.frameSkip):
            # action is in -1,1
            if action is not None:
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep * 0.5)
            self.timer += 1

        reward = 3.0

        # rewards is height of target object
        clPos, _ = p.getBasePositionAndOrientation(self.obj_id)
        palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
        dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(clPos)), 0.5)
        reward += -dist * 2.0
        reward += -np.minimum(np.linalg.norm(np.array([self.tx, self.ty, 0.1]) - np.array(clPos)), 0.4) * 4.0

        for i in self.robot.fin_tips[:4]:
            tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
            reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.5)  # 4 finger tips
        tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]      # thumb tip
        reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.5) * 5.0

        cps = p.getContactPoints(self.obj_id, self.robot.arm_id, -1, self.robot.ee_id)    # palm
        if len(cps) > 0: reward += 5.0
        f_bp = [0, 3, 6, 9, 12, 17]     # 3*4+5
        for ind_f in range(5):
            con = False
            # try onl reward distal and middle
            # for dof in self.robot.fin_actdofs[f_bp[ind_f]:f_bp[ind_f+1]]:
            # for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 2):f_bp[ind_f + 1]]:
            for dof in self.robot.fin_actdofs[(f_bp[ind_f + 1] - 3):f_bp[ind_f + 1]]:
                cps = p.getContactPoints(self.obj_id, self.robot.arm_id, -1, dof)
                if len(cps) > 0:  con = True
            if con:  reward += 5.0
            if con and ind_f == 4: reward += 20.0        # reward thumb even more

        clVels = p.getBaseVelocity(self.obj_id)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])
        reward += np.maximum(-np.linalg.norm(clLinV) - np.linalg.norm(clAngV), -10.0) * 0.2
        #
        # if self.timer == 300:
        #     self.append_final_state()

        if clPos[2] < -0.0 and self.timer > 300: # object dropped, do not penalize dropping when 0 gravity
            reward += -15.

        return self.getExtendedObservation(), reward, False, {}

    def getExtendedObservation(self):
        self.observation = self.robot.get_robot_observation(diff_tar=True)

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

        # curContact = []
        # for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, i)
        #     if len(cps) > 0:
        #         curContact.extend([1.0])
        #         # print("touch!!!")
        #     else:
        #         curContact.extend([-1.0])
        # self.observation.extend(curContact)

        if self.up:
            xy = np.array([self.tx, self.ty])
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))
            self.observation.extend(list(xy))
            # this is the vision module one also used for reset/planning

        if self.random_shape:
            if self.shape_ind == 1:
                shape_info = [1, -1, -1]
            elif self.shape_ind == 0:
                shape_info = [-1, 1, -1]
            elif self.shape_ind == -1:
                shape_info = [-1, -1, 1]
            else:
                shape_info = [-1, -1, -1]
            self.observation.extend(shape_info)

        if self.random_size:
            self.observation.extend([self.half_height*4 + self.np_random.uniform(low=-0.02, high=0.02)*2,
                                     self.half_height*4 + self.np_random.uniform(low=-0.02, high=0.02)*2,
                                     self.half_height*4 + self.np_random.uniform(low=-0.02, high=0.02)*2])
            # this is the true half_height, vision module one will be noisy

        # self.observation.extend([self.timer/300 + self.np_random.uniform(low=-0.01, high=0.01),
        #                          self.timer/300 + self.np_random.uniform(low=-0.01, high=0.01),
        #                          self.timer/300])

        return self.observation

    def append_final_state(self):
        # output obj in palm frame (no need to output palm frame in world)
        # output finger q's, finger tar q's.
        # velocity will be assumed to be zero at the end of transporting phase
        # return a dict.
        obj_pos, obj_quat = p.getBasePositionAndOrientation(self.obj_id)      # w2o
        hand_pos, hand_quat = self.robot.get_link_pos_quat(self.robot.ee_id)    # w2p
        inv_h_p, inv_h_q = p.invertTransform(hand_pos, hand_quat)       # p2w
        o_p_hf, o_q_hf = p.multiplyTransforms(inv_h_p, inv_h_q, obj_pos, obj_quat)  # p2w*w2o

        fin_q, _ = self.robot.get_q_dq(self.robot.all_findofs)

        # also store the shape info here

        unitz_hf = p.multiplyTransforms([0, 0, 0], o_q_hf, [0, 0, 1], [0, 0, 0, 1])[0]
        # TODO: a heuritics that if obj up_vec points outside palm, then probably holding bottom & bad
        if unitz_hf[1] < -0.3:
            return
        else:
            if self.shape_ind == 1:
                shape = p.GEOM_BOX
            elif self.shape_ind == 0:
                shape = p.GEOM_CYLINDER
            elif self.shape_ind == -1:
                shape = p.GEOM_SPHERE
            else:
                shape = None
            state = {'obj_pos_in_palm': o_p_hf, 'obj_quat_in_palm': o_q_hf,
                     'all_fin_q': fin_q, 'fin_tar_q': self.robot.tar_fin_q,
                     'obj_dim': self.dim, 'obj_shape': shape}   # TODO: ball
            # print(state)
            # print(self.robot.get_joints_last_tau(self.robot.all_findofs))
            # self.robot.get_wrist_wrench()
            self.final_states.append(state)

    def clear_final_states(self):
        self.final_states = []

    def calc_average_obj_in_palm(self):
        count = len(self.final_states)
        o_pos_hf_sum = np.array([0., 0, 0])
        o_quat_hf_sum = np.array([0., 0, 0, 0])
        for dict in self.final_states:
            o_pos_hf_sum += np.array(dict['obj_pos_in_palm'])
            o_quat_hf_sum += np.array(dict['obj_quat_in_palm'])
        o_pos_hf_sum /= count
        o_quat_hf_sum /= count      # rough estimate of quat average
        o_quat_hf_sum /= np.linalg.norm(o_quat_hf_sum)      # normalize quat
        return list(o_pos_hf_sum), list(o_quat_hf_sum)

    def calc_average_obj_in_palm_rot_invariant(self):
        count = len(self.final_states)
        o_pos_hf_sum = np.array([0., 0, 0])
        o_unitz_hf_sum = np.array([0., 0, 0])
        for dict in self.final_states:
            o_pos_hf_sum += np.array(dict['obj_pos_in_palm'])
            unitz_hf = p.multiplyTransforms([0, 0, 0], dict['obj_quat_in_palm'], [0, 0, 1], [0, 0, 0, 1])[0]
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
        assert np.linalg.norm(np.array(o_unitz_hf_sum) - np.array((uz_check))) < 1e-3

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