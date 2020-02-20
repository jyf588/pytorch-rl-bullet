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

# TODO: txyz will be given by vision module. tz is zero for grasping, obj frame at bottom.


class InmoovShadowHandGraspPlaceEnvV1(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 renders=True,
                 init_noise=True,
                 up=False,
                 random_shape=False,
                 random_size=False,
                 using_comfortable=False,
                 using_comfortable_range=False,
                 cotrain_place=True,
                 place_floor=False,
                 use_gt_6d=True,
                 gt_only_init=False,
                 ):
        # TODO: use obj 6D later
        self.renders = renders
        self.init_noise = init_noise
        self.up = up
        self.random_shape = random_shape
        self.random_size = random_size
        self.using_comfortable = using_comfortable
        self.using_comfortable_range = using_comfortable_range
        self.vary_angle_range = 0.6
        self.cotrain_place = cotrain_place
        self.obj_mass = 3.5
        self.place_floor = place_floor
        self.use_gt_6d = use_gt_6d
        self.gt_only_init = gt_only_init

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
            self.tx = None
            self.ty = None
        else:
            self.tx = 0.
            self.ty = 0.    # constants
        self.tz = 0         # TODO: variable z
        self.btm_obj_height = 0.18  # TODO: hard coded here
        self.grasp_phase = True

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
            cyl_init_pos = [0, 0, self.half_height + 0.001]
            # if self.small:
            #     cyl_init_pos = [0, 0, 0.0651]
            # else:
            #     cyl_init_pos = [0, 0, 0.091]
            self.tx = self.np_random.uniform(low=0, high=0.25)
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
            cyl_init_pos = [0, 0, self.half_height + 0.001]
            # if self.small:
            #     cyl_init_pos = [0, 0, 0.0651]
            # else:
            #     cyl_init_pos = [0, 0, 0.091]
            self.tx = self.np_random.uniform(low=0, high=0.25)
            self.ty = self.np_random.uniform(low=-0.1, high=0.5)    # TODO 0.6?
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
        if self.up:
            self.tx = self.np_random.uniform(low=0, high=0.2)
            self.ty = self.np_random.uniform(low=-0.2, high=0.0)
            cyl_init_pos = np.array(cyl_init_pos) + np.array([self.tx, self.ty, 0])
            init_palm_pos = np.array(init_palm_pos) + np.array([self.tx, self.ty, 0])
        return init_palm_pos, init_palm_quat, cyl_init_pos

    def create_prim_2_grasp(self, shape, dim, init_xyz):
        # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
        # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder
        # init_xyz vec3 of obj location
        id = None
        if shape == p.GEOM_BOX:
            visualShapeId = p.createVisualShape(shapeType=shape, halfExtents=dim)
            collisionShapeId = p.createCollisionShape(shapeType=shape, halfExtents=dim)
            id = p.createMultiBody(baseMass=self.obj_mass, baseInertialFramePosition=[0, 0, 0],
                              baseCollisionShapeIndex=collisionShapeId,
                              baseVisualShapeIndex=visualShapeId,
                              basePosition=init_xyz)
        elif shape == p.GEOM_CYLINDER:
            # visualShapeId = p.createVisualShape(shapeType=shape, radius=dim[0], length=dim[1])
            visualShapeId = p.createVisualShape(shape, dim[0], [1,1,1], dim[1])
            # collisionShapeId = p.createCollisionShape(shapeType=shape, radius=dim[0], length=dim[1])
            collisionShapeId = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
            id = p.createMultiBody(baseMass=self.obj_mass, baseInertialFramePosition=[0, 0, 0],
                              baseCollisionShapeIndex=collisionShapeId,
                              baseVisualShapeIndex=visualShapeId,
                              basePosition=init_xyz)
        elif shape == p.GEOM_SPHERE:
            pass        # TODO
        return id

    def reset(self):
        p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)

        self.isBox = bool(self.np_random.randint(2)) if self.random_shape else False
        self.half_height = self.np_random.uniform(low=0.055, high=0.09) if self.random_size else 0.07
        self.half_width = self.np_random.uniform(low=0.03, high=0.05) if self.random_size else 0.04  # aka radius

        if self.using_comfortable:
            if self.using_comfortable_range:
                arm_q, obj_init_xyz = self.get_reset_poses_comfortable_range()
            else:
                arm_q, obj_init_xyz = self.get_reset_poses_comfortable()
        else:
            init_palm_pos, init_palm_quat, obj_init_xyz = self.get_reset_poses_old()

        if self.init_noise:
            obj_init_xyz += np.append(self.np_random.uniform(low=-0.02, high=0.02, size=2), 0)

        if self.isBox:
            dim = [self.half_width*0.8, self.half_width*0.8, self.half_height]     # TODO
            self.obj_id = self.create_prim_2_grasp(p.GEOM_BOX, dim, obj_init_xyz)
        else:
            dim = [self.half_width, self.half_height*2.0]
            self.obj_id = self.create_prim_2_grasp(p.GEOM_CYLINDER, dim, obj_init_xyz)
        p.changeDynamics(self.obj_id, -1, lateralFriction=1.0)

        if self.place_floor:
            self.bottom_obj_id =p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'), [0.25, 0.2, 0.0],
                                             useFixedBase=1)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
        else:
            btm_xyz = np.array([self.tx, self.ty, -self.btm_obj_height/2.0])
            if self.init_noise:
                btm_xyz += np.append(self.np_random.uniform(low=-0.01, high=0.01, size=2), 0)
            self.bottom_obj_id = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'),
                                            btm_xyz, useFixedBase=0)
            self.floor_id = p.loadURDF(os.path.join(currentdir, 'assets/tabletop.urdf'),
                                       [0.25, 0.2, -self.btm_obj_height],
                                       useFixedBase=1)
            p.changeDynamics(self.bottom_obj_id, -1, lateralFriction=1.0)
            p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)

        self.robot = InmoovShadowNew(init_noise=self.init_noise, timestep=self._timeStep, np_random=self.np_random,
                                     conservative_clip=False)

        if self.using_comfortable:
            self.robot.reset_with_certain_arm_q(arm_q)
        else:
            self.robot.reset(list(init_palm_pos), init_palm_quat)       # reset at last to test collision

        self.desired_obj_pos_final = [self.tx, self.ty, self.half_height + self.tz]

        self.timer = 0
        self.grasp_phase = True
        # init obj pose
        self.t_pos, self.t_orn = p.getBasePositionAndOrientation(self.obj_id)
        self.b_pos, self.b_orn = p.getBasePositionAndOrientation(self.bottom_obj_id)
        self.observation = self.getExtendedObservation()
        return np.array(self.observation)

    def calc_reward_grasp_phase(self):
        reward = 3.0

        # rewards is height of target object
        clPos, _ = p.getBasePositionAndOrientation(self.obj_id)
        palm_com_pos = p.getLinkState(self.robot.arm_id, self.robot.ee_id)[0]
        dist = np.minimum(np.linalg.norm(np.array(palm_com_pos) - np.array(clPos)), 0.5)
        reward += -dist * 2.0
        reward += -np.minimum(np.linalg.norm(np.array([self.tx, self.ty, 0.1]) - np.array(clPos)), 0.4) * 6.0

        for i in self.robot.fin_tips[:4]:
            tip_pos = p.getLinkState(self.robot.arm_id, i)[0]
            reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.5)  # 4 finger tips
        tip_pos = p.getLinkState(self.robot.arm_id, self.robot.fin_tips[4])[0]      # thumb tip
        reward += -np.minimum(np.linalg.norm(np.array(tip_pos) - np.array(clPos)), 0.5) * 5.0

        # for i in range(self.robot.ee_id, p.getNumJoints(self.robot.arm_id)):
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, i)
        #     if len(cps) > 0:
        #         # print(len(cps))
        #         reward += 5.0   # the more links in contact, the better

        # cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, self.robot.ee_id)    # palm
        # if len(cps) > 0: reward += 4.0
        # for dof in np.copy(self.robot.fin_actdofs)[[0,1, 3,4, 6,7, 9,10]]:
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, dof)
        #     if len(cps) > 0:  reward += 2.0
        # for dof in np.copy(self.robot.fin_actdofs)[[2, 5, 8, 11]]:
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, dof)
        #     if len(cps) > 0:  reward += 3.5
        # for dof in self.robot.fin_actdofs[12:16]:         # thumb
        #     cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, dof)
        #     if len(cps) > 0:  reward += 10.0
        # cps = p.getContactPoints(self.cylinderId, self.robot.arm_id, -1, self.robot.fin_actdofs[16])    # thumb tip
        # if len(cps) > 0:  reward += 17.5

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

        if clPos[2] < -0.0 and self.timer > 300:    # object dropped
            reward += -15.
        return reward

    def calc_reward_place_phase_old(self):
        reward = 0.
        clPos, clQuat = p.getBasePositionAndOrientation(self.obj_id)
        clVels = p.getBaseVelocity(self.obj_id)
        clLinV = np.array(clVels[0])
        clAngV = np.array(clVels[1])

        z_axis, _ = p.multiplyTransforms([0, 0, 0], clQuat, [0, 0, 1], [0, 0, 0, 1])          # R_cl * unitz[0,0,1]
        rotMetric = np.array(z_axis).dot(np.array([0, 0, 1]))

        # TODO:tmp change to xy metric, allow it to free drop
        xyzMetric = 1 - (np.minimum(np.linalg.norm(np.array(self.desired_obj_pos_final[:2]) - np.array(clPos[:2])), 0.2) / 0.2)
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

        return reward

    def calc_reward_place_phase(self):
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
        if np.abs(total_nf) > (self.obj_mass*7.):       # mg        # TODO:tmp contact force hack
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
        return reward

    def step(self, action):
        if self.grasp_phase:
            if self.timer == 100 * self.frameSkip:
                p.setCollisionFilterPair(self.obj_id, self.bottom_obj_id, -1, -1, enableCollision=0)
                # for i in range(-1, p.getNumJoints(self.robot.arm_id)):
                #     p.setCollisionFilterPair(self.bottom_obj_id, self.robot.arm_id, -1, i, enableCollision=0)
        if not self.grasp_phase:
            if self.timer == 0:
                p.setCollisionFilterPair(self.obj_id, self.bottom_obj_id, -1, -1, enableCollision=1)
                # for i in range(-1, p.getNumJoints(self.robot.arm_id)):
                #     p.setCollisionFilterPair(self.bottom_obj_id, self.robot.arm_id, -1, i, enableCollision=1)

        for _ in range(self.frameSkip):
            # action is not in -1,1
            if action is not None:
                self.act = action
                self.robot.apply_action(self.act * self.action_scale)
            p.stepSimulation()
            if self.renders:
                time.sleep(self._timeStep)
            self.timer += 1

        done = False

        if self.grasp_phase:
            r = self.calc_reward_grasp_phase()
            obs = self.getExtendedObservation()
            if self.timer == 140 * self.frameSkip:
                if not self.cotrain_place:
                    done = True
                else:
                    obj_xyz, _ = p.getBasePositionAndOrientation(self.obj_id)
                    cp_obj = p.getContactPoints(bodyA=self.obj_id)
                    # print(obj_xyz)
                    # print(len(cp_obj))
                    # if True:
                    # if len(cp_obj) > 0 and obj_xyz[2] > 0.0:
                    if len(cp_obj) > 0 and obj_xyz[2] > self.half_height:      # contact with hand   # TODO: assume skinny obj
                        # continue train placing
                        self.grasp_phase = False
                        self.timer = 0
                        done = False
                        r += 200        # TODO: next stage bonus
                    else:
                        done = True
        else:
            # placing stage
            r = self.calc_reward_place_phase()
            obs = self.getExtendedObservation()  # call last obs before test period
            if self.timer == 100 * self.frameSkip:
                done = True     # TODO: or just set tl=240

        return obs, r, done, {}

    # def execute_release_traj(self):
    #
    #     cur_q = self.robot.get_q_dq(self.robot.fin_actdofs)[0]
    #     self.robot.tar_fin_q = cur_q
    #     for test_t in range(120):
    #         thumb_pose = list(cur_q[-5:])     # do not modify thumb
    #         open_up_q = np.array([0.1, 0.1, 0.1] * 4 + thumb_pose)
    #         devi = open_up_q - cur_q
    #         if test_t < 100:
    #             self.robot.apply_action(np.array([0.0] * 7 + list(devi / 100.)))
    #         p.stepSimulation()
    #         if self.renders:
    #             time.sleep(self._timeStep)
    #
    #     self.robot.tar_arm_q = self.robot.get_q_dq(self.robot.arm_dofs)[0]
    #     self.robot.tar_fin_q = self.robot.get_q_dq(self.robot.fin_actdofs)[0]
    #     tar_wrist_xyz = np.array(self.robot.get_link_pos_quat(self.robot.ee_id)[0])
    #     obj_xyz, _ = p.getBasePositionAndOrientation(self.obj_id)   # TODO:tmp
    #     dir = tar_wrist_xyz[:2] - [obj_xyz[0], obj_xyz[1]]
    #     dir = dir / np.linalg.norm(dir)
    #     dir = np.array(list(dir) + [0.0])
    #     ik_q = None
    #     for test_t in range(200):
    #         if test_t < 180:
    #             tar_wrist_xyz += 0.0006 * dir
    #             ik_q = p.calculateInverseKinematics(self.robot.arm_id, self.robot.ee_id, list(tar_wrist_xyz))
    #         self.robot.tar_arm_q = np.array(ik_q[:len(self.robot.arm_dofs)])
    #         self.robot.apply_action(np.array([0.0] * len(self.action_scale)))
    #         p.stepSimulation()
    #         if self.renders:
    #             time.sleep(self._timeStep)

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(np.array(arr) + self.np_random.uniform(low=-r, high=r, size=len(arr)))

    def obj6DtoObs_UpVec(self, o_pos, o_orn):
        objObs = []
        o_pos = np.array(o_pos)
        if self.up:                         # TODO:tmp
            o_pos -= [self.tx, self.ty, 0]
        o_pos = o_pos * 3.0       # TODO:tmp, scale up
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
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))
            self.observation.extend(list(xy + self.np_random.uniform(low=-0.01, high=0.01, size=2)))

        if self.random_shape:
            shape_info = 1. if self.isBox else -1.
            self.observation.extend([shape_info])

        stage = 1. if self.grasp_phase else -1.
        if self.cotrain_place:
            self.observation.extend([stage + self.np_random.uniform(low=-0.01, high=0.01),
                                     stage + self.np_random.uniform(low=-0.01, high=0.01),
                                     stage])

        self.observation.extend([self.timer/300.*stage + self.np_random.uniform(low=-0.01, high=0.01),
                                 self.timer/300.*stage + self.np_random.uniform(low=-0.01, high=0.01),
                                 self.timer/300.*stage])

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

        state = {'obj_pos_in_palm': o_p_hf, 'obj_quat_in_palm': o_q_hf,
                 'all_fin_q': fin_q, 'fin_tar_q': self.robot.tar_fin_q}
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