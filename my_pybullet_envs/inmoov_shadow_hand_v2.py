# (0, b'r_shoulder_out_joint', 0) -1.57079632679 1.57079632679
# (1, b'r_shoulder_lift_joint', 0) -1.57079632679 1.57079632679
# (2, b'r_upper_arm_roll_joint', 0) -1.57079632679 1.57079632679
# (3, b'r_elbow_flex_joint', 0) -3.14159265359 0.0
# (4, b'r_elbow_roll_joint', 0) -1.57079632679 1.57079632679
# (5, b'r_wrist_roll_joint', 4) 0.0 -1.0
# (6, b'rh_WRJ2', 0) -1.0471975512 1.0471975512
# (7, b'rh_WRJ1', 0) -1.57079632679 1.57079632679
# (8, b'rh_FFJ4', 0) -0.349065850399 0.349065850399
# (9, b'rh_FFJ3', 0) 0.0 1.57079632679
# (10, b'rh_FFJ2', 0) 0.0 1.57079632679
# (11, b'rh_FFJ1', 0) 0.0 1.57079632679
# (12, b'rh_FFtip', 4) 0.0 -1.0
# (13, b'rh_MFJ4', 0) -0.349065850399 0.349065850399
# (14, b'rh_MFJ3', 0) 0.0 1.57079632679
# (15, b'rh_MFJ2', 0) 0.0 1.57079632679
# (16, b'rh_MFJ1', 0) 0.0 1.57079632679
# (17, b'rh_MFtip', 4) 0.0 -1.0
# (18, b'rh_RFJ4', 0) -0.349065850399 0.349065850399
# (19, b'rh_RFJ3', 0) 0.0 1.57079632679
# (20, b'rh_RFJ2', 0) 0.0 1.57079632679
# (21, b'rh_RFJ1', 0) 0.0 1.57079632679
# (22, b'rh_RFtip', 4) 0.0 -1.0
# (23, b'rh_LFJ5', 4) 0.0 -1.0
# (24, b'rh_LFJ4', 0) -0.349065850399 0.349065850399
# (25, b'rh_LFJ3', 0) 0.0 1.57079632679
# (26, b'rh_LFJ2', 0) 0.0 1.57079632679
# (27, b'rh_LFJ1', 0) 0.0 1.57079632679
# (28, b'rh_LFtip', 4) 0.0 -1.0
# (29, b'rh_THJ5', 0) -1.0471975512 1.0471975512
# (30, b'rh_THJ4', 0) 0.0 1.2217304764
# (31, b'rh_THJ3', 0) -0.209439510239 0.209439510239
# (32, b'rh_THJ2', 0) -0.698131700798 0.698131700798
# (33, b'rh_THJ1', 0) 0.0 1.57079632679
# (34, b'rh_thtip', 4) 0.0 -1.0
# armdof [0, 1, 2, 3, 4, 6, 7]
# self.fin_actdofs = [9, 10, 11, 14, 15, 16, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33]
# self.fin_zerodofs = [8, 13, 18, 24]

# turn off damping done.
# arm mass inertia /100
# finger mass inertia *10
# there are some 0 mass links
#  TODO: <joint name="rh_LFJ5" type="fixed"> done
#     <parent link="rh_palm"/>
#     <child link="rh_lfmetacarpal"/>
# v2_2 urdf
# does not like IK, so wrist_x bound will not be used for now.

#TODO: replace p with sim?

import pybullet_utils.bullet_client as bc
import pybullet as p
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class InmoovShadowNew:
    def __init__(self,
                 init_noise=True,
                 timestep=1. / 240
                 ):
        self.init_noise = init_noise
        self._timestep = timestep

        self.base_init_pos = np.array([-0.30, 0.348, 0.272])
        # self.base_init_pos = np.array([-0.44, 0.348, 0.272])
        self.base_init_euler = np.array([0,0,0])
        self.arm_dofs = [0, 1, 2, 3, 4, 6, 7]
        self.fin_actdofs = [9, 10, 11, 14, 15, 16, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33]
        self.fin_zerodofs = [8, 13, 18, 24]
        self.fin_tips = [12, 17, 22, 28, 34]
        self.all_findofs = list(np.sort(self.fin_actdofs+self.fin_zerodofs))
        self.init_fin_q = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0])
        self.tar_arm_q = np.zeros(len(self.arm_dofs))       # dummy
        self.tar_fin_q = np.zeros(len(self.fin_actdofs))

        self.ee_id = 7      # link 7 is palm

        self.arm_id = p.loadURDF(os.path.join(currentdir,
                                             "assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf"),
                                 list(self.base_init_pos), p.getQuaternionFromEuler(list(self.base_init_euler)),
                                 flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
                                       | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                                 useFixedBase=1)

        # self.print_all_joints_info()

        # print(self.get_link_pos_quat(0)[0])

        for i in range(-1, p.getNumJoints(self.arm_id)):
            p.changeDynamics(self.arm_id, i, jointDamping=0.0, linearDamping=0.0, angularDamping=0.0)

        self.scale_mass_inertia(-1, self.ee_id, 0.01)
        self.scale_mass_inertia(self.ee_id, p.getNumJoints(self.arm_id), 10.0)

        for i in range(self.ee_id, p.getNumJoints(self.arm_id)):
            p.changeDynamics(self.arm_id, i, lateralFriction=3.0)

        # use np for multi-indexing
        self.ll = np.array([p.getJointInfo(self.arm_id, i)[8] for i in range(p.getNumJoints(self.arm_id))])
        self.ul = np.array([p.getJointInfo(self.arm_id, i)[9] for i in range(p.getNumJoints(self.arm_id))])

        self.maxForce = 200.    # TODO
        self.include_redun_body_pos = False
        self.np_random = None   # seeding inited outside in Env
        # p.stepSimulation()
        # input("press enter")

    def print_all_joints_info(self):
        for i in range(p.getNumJoints(self.arm_id)):
            print(p.getJointInfo(self.arm_id, i)[0:3],
                  p.getJointInfo(self.arm_id, i)[8], p.getJointInfo(self.arm_id, i)[9])

    def scale_mass_inertia(self, start, end, ratio=1.0):
        total_m = 0
        for i in range(start, end):
            dyn = p.getDynamicsInfo(self.arm_id, i)
            mass = dyn[0]
            mass = mass * ratio
            lid = dyn[2]
            lid = (lid[0] * ratio, lid[1] * ratio, lid[2] * ratio,)
            total_m += mass
            p.changeDynamics(self.arm_id, i, mass=mass)
            p.changeDynamics(self.arm_id, i, localInertiaDiagonal=lid)
        # print(total_m)

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(arr + self.np_random.uniform(low=-r, high=r, size=len(arr)))

    def reset(self, wx):
        # reset according to wrist 6D pos
        wx_trans = list(wx[:3])
        wx_euler = list(wx[3:])
        wx_quat = p.getQuaternionFromEuler(wx_euler)
        closeEnough = False
        sp = [-0.44, 0.00, -0.5, -1.8, -0.44, -0.488, -0.8] + [0.0]*len(self.all_findofs)   # dummy init guess IK
        ll = self.ll[self.arm_dofs+self.all_findofs]
        ul = self.ul[self.arm_dofs+self.all_findofs]
        jr = ul - ll
        iter = 0
        while not closeEnough and iter < 20:
            for ind in range(len(self.arm_dofs)):
                p.resetJointState(self.arm_id, self.arm_dofs[ind], sp[ind])

            jointPoses = p.calculateInverseKinematics(self.arm_id, self.ee_id, wx_trans, wx_quat,
                                                      lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
                                                      jointRanges=jr.tolist(),
                                                      restPoses=sp)
            # jointPoses = p.calculateInverseKinematics(self.arm_id, self.ee_id, wx_trans, wx_quat)

            sp = np.array(jointPoses)[range(7)].tolist()
            # print(sp)

            wx_now = p.getLinkState(self.arm_id, self.ee_id)[4]
            dist = np.linalg.norm(np.array(wx_now) - np.array(wx_trans))
            # print("dist=", dist)
            if dist < 2e-3: closeEnough = True
            iter += 1

        good_init = False
        while not good_init:
            if self.init_noise:
                init_arm_q = self.perturb(sp, r=0.002)
                init_fin_q = self.perturb(self.init_fin_q, r=0.02)
            else:
                init_arm_q = np.array(sp)
                init_fin_q = np.array(self.init_fin_q)

            for ind in range(len(self.arm_dofs)):
                p.resetJointState(self.arm_id, self.arm_dofs[ind], init_arm_q[ind], 0.0)
            for ind in range(len(self.fin_actdofs)):
                p.resetJointState(self.arm_id, self.fin_actdofs[ind], init_fin_q[ind], 0.0)
            for ind in range(len(self.fin_zerodofs)):
                p.resetJointState(self.arm_id, self.fin_zerodofs[ind], 0.0, 0.0)

            self.tar_arm_q = init_arm_q
            self.tar_fin_q = init_fin_q

            cps = p.getContactPoints(bodyA=self.arm_id)
            # for cp in cps:
            #     print(cp)
            #     input("penter")
            # print(cps[0][6])
            if len(cps) == 0: good_init = True   # TODO: init hand last and make sure it does not colllide with env
        # input("after reset")

    def get_robot_observation(self):
        obs = []

        links = [self.ee_id] + self.fin_tips
        for link in links:
            pos, orn = self.get_link_pos_quat(link)
            linVel, angVel = self.get_link_v_w(link)
            obs.extend(pos)
            obs.extend(orn)
            obs.extend(linVel)
            obs.extend(angVel)

        a_q, a_dq = self.get_q_dq(self.arm_dofs)
        # print("arm q", a_q)
        obs.extend(list(a_q))
        obs.extend(list(a_dq))
        obs.extend(list(self.get_q_dq(self.fin_actdofs)[0]))    # no finger vel

        # tar pos
        obs.extend(list(self.tar_arm_q))
        # print("tar arm q", self.tar_arm_q)
        obs.extend(list(self.tar_fin_q))

        return obs

    def apply_action(self, a):
        # TODO: a is already scaled, how much to scale? decide in Env.
        self.act = np.array(a)
        self.tar_arm_q += self.act[:len(self.arm_dofs)]     # assume arm controls are the first ones.
        self.tar_arm_q = np.clip(self.tar_arm_q, self.ll[self.arm_dofs], self.ul[self.arm_dofs])
        self.tar_fin_q += self.act[len(self.arm_dofs):]
        self.tar_fin_q = np.clip(self.tar_fin_q, self.ll[self.fin_actdofs], self.ul[self.fin_actdofs])
        p.setJointMotorControlArray(
            bodyIndex=self.arm_id,
            jointIndices=self.arm_dofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(self.tar_arm_q),
            forces=[self.maxForce * 3] * len(self.arm_dofs))  # TODO: wrist force limit?
        p.setJointMotorControlArray(
            bodyIndex=self.arm_id,
            jointIndices=self.fin_actdofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(self.tar_fin_q),
            forces=[self.maxForce] * len(self.tar_fin_q))

    def get_robot_observation_dim(self):
        return len(self.get_robot_observation())

    def get_q_dq(self, dofs):
        joints_state = p.getJointStates(self.arm_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_joints_last_tau(self, dofs):
        joints_state = p.getJointStates(self.arm_id, dofs)
        joints_taus = np.array(joints_state)[:, [3]]
        joints_taus = np.hstack(joints_taus.flatten())
        return joints_taus

    def get_link_pos_quat(self, l_id):
        newPos = p.getLinkState(self.arm_id, l_id)[4]
        newOrn = p.getLinkState(self.arm_id, l_id)[5]
        return newPos, newOrn

    def get_link_v_w(self, l_id):
        newLinVel = p.getLinkState(self.arm_id, l_id, computeForwardKinematics=1, computeLinkVelocity=1)[6]
        newAngVel = p.getLinkState(self.arm_id, l_id, computeForwardKinematics=1, computeLinkVelocity=1)[7]
        return newLinVel, newAngVel


if __name__ == "__main__":
    hz = 240.0
    dt = 1.0 / hz

    sim = bc.BulletClient(connection_mode=p.GUI)

    for i in range(100):
        sim.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=200)

        sim.setGravity(0, 0, 0)
        sim.setTimeStep(dt)
        sim.setRealTimeSimulation(0)

        arm = InmoovShadowNew()
        arm.np_random, seed = gym.utils.seeding.np_random(0)
        # arm.reset([-0.18, 0.105, 0.13, 1.8, -1.57, 0]) # obj [0,0,0]
        # arm.reset([0.02, 0.105, 0.13, 1.8, -1.57, 0]) # obj [0.2, 0, 0]
        # arm.reset([0.02, 0.105-0.2, 0.13, 1.8, -1.57, 0]) # obj [0.2, -0.2, 0]
        arm.reset([-0.18, 0.105 - 0.2, 0.13, 1.8, -1.57, 0])  # obj [0, -0.2, 0]

        print("init", arm.get_robot_observation())
        ls = p.getLinkState(arm.arm_id, arm.ee_id)
        newPos = ls[4]
        print(newPos, p.getEulerFromQuaternion(ls[5]))
        input("press enter to continue")

        for t in range(400):
            arm.apply_action(arm.np_random.uniform(low=-0.003, high=0.003, size=7+17)+np.array([-0.005]*7+[-0.01]*17))
            p.stepSimulation()
            arm.get_robot_observation()
            time.sleep(1. / 240.)
        print("final obz", arm.get_robot_observation())
        ls = p.getLinkState(arm.arm_id, arm.ee_id)
        newPos = ls[4]
        print(newPos, p.getEulerFromQuaternion(ls[5]))

    p.disconnect()


