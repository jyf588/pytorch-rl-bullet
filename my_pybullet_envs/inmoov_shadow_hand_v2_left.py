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
# (29, b'rh_THJ5', 0) -1.0471975512 1.0471975512    # -1 upward
# (30, b'rh_THJ4', 0) 0.0 1.2217304764  -> 2.0        # +1 outward
# (31, b'rh_THJ3', 0) -0.209439510239 0.209439510239    # -1 upward
# (32, b'rh_THJ2', 0) -0.698131700798 0.698131700798 -> 1.22 # same as below +1 flex
# (33, b'rh_THJ1', 0) 0.0 1.57079632679
# (34, b'rh_thtip', 4) 0.0 -1.0
# armdof [0, 1, 2, 3, 4, 6, 7]
# self.fin_actdofs = [9, 10, 11, 14, 15, 16, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33]
# self.fin_zerodofs = [8, 13, 18, 24]

# turn off damping done.
# arm mass inertia /100 done
# finger mass inertia *10 done
# there are some 0 mass links no more
#  <joint name="rh_LFJ5" type="fixed"> done
#     <parent link="rh_palm"/>
#     <child link="rh_lfmetacarpal"/>
# v2_2 urdf
# does not like IK, so wrist_x bound will not be used for now.

# TODO: replace p with sim?
# TODO: test mu
# TODO: compute fk for getLinkPos
# TODO: mass scales of arm.


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
                 timestep=1. / 240,
                 np_random=None,
                 conservative_clip=False,
                 conservative_range=None,
                 sim=p
                 ):
        self.sim = sim
        self.init_noise = init_noise
        self._timestep = timestep
        self.conservative_clip = conservative_clip
        self.conservative_range = conservative_range

        self.base_init_pos = np.array([-0.30, -0.348, 0.272])
        # self.base_init_pos = np.array([-0.44, 0.348, 0.272])
        self.base_init_euler = np.array([0,0,0])
        self.arm_dofs = [0, 1, 2, 3, 4, 6, 7]
        self.fin_actdofs = [9, 10, 11, 14, 15, 16, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33]
        self.fin_zerodofs = [8, 13, 18, 24]
        self.fin_tips = [12, 17, 22, 28, 34]
        self.all_findofs = list(np.sort(self.fin_actdofs+self.fin_zerodofs))
        # fin_init =   [0.21269122473773142,   1.2217305056102568,  0.2094395103670277,  -0.6981316630905725,  1.8797404688696039e-06,]
        #
        # self.init_fin_q = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + fin_init )
        self.init_fin_q = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.1])
        self.tar_arm_q = np.zeros(len(self.arm_dofs))       # dummy
        self.tar_fin_q = np.zeros(len(self.fin_actdofs))

        self.ee_id = 7      # link 7 is palm

        self.maxForce = 200.    # TODO
        self.np_random = np_random

        self.arm_id = self.sim.loadURDF(os.path.join(currentdir,
                                             "assets/inmoov_ros/inmoov_description/robots/left_experimental.urdf"),
                                 list(self.base_init_pos), self.sim.getQuaternionFromEuler(list(self.base_init_euler)),
                                 flags=self.sim.URDF_USE_SELF_COLLISION | self.sim.URDF_USE_INERTIA_FROM_FILE
                                       | self.sim.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                                 useFixedBase=1)

        # self.print_all_joints_info()

        for i in range(-1, self.sim.getNumJoints(self.arm_id)):
            self.sim.changeDynamics(self.arm_id, i, jointDamping=0.0, linearDamping=0.0, angularDamping=0.0)

        self.scale_mass_inertia(-1, self.ee_id, 0.01)
        self.scale_mass_inertia(self.ee_id, self.sim.getNumJoints(self.arm_id), 10.0)

        if self.np_random:
            mu = self.np_random.uniform(0.8, 1.2)
            for i in range(self.ee_id, self.sim.getNumJoints(self.arm_id)):
                self.sim.changeDynamics(self.arm_id, i, lateralFriction=mu)

        # use np for multi-indexing
        self.ll = np.array([self.sim.getJointInfo(self.arm_id, i)[8] for i in range(self.sim.getNumJoints(self.arm_id))])
        self.ul = np.array([self.sim.getJointInfo(self.arm_id, i)[9] for i in range(self.sim.getNumJoints(self.arm_id))])

        self.sim.enableJointForceTorqueSensor(self.arm_id, self.ee_id, 1)

        # self.sim.stepSimulation()
        # input("press enter")

    def change_hand_friction(self, mu):
        for i in range(self.ee_id, self.sim.getNumJoints(self.arm_id)):
            self.sim.changeDynamics(self.arm_id, i, lateralFriction=mu)

    def print_all_joints_info(self):
        for i in range(self.sim.getNumJoints(self.arm_id)):
            print(self.sim.getJointInfo(self.arm_id, i)[0:3],
                  self.sim.getJointInfo(self.arm_id, i)[8], self.sim.getJointInfo(self.arm_id, i)[9])

    def scale_mass_inertia(self, start, end, ratio=1.0):
        total_m = 0
        for i in range(start, end):
            dyn = self.sim.getDynamicsInfo(self.arm_id, i)
            mass = dyn[0]
            mass = mass * ratio
            lid = dyn[2]
            lid = (lid[0] * ratio, lid[1] * ratio, lid[2] * ratio,)
            total_m += mass
            self.sim.changeDynamics(self.arm_id, i, mass=mass)
            self.sim.changeDynamics(self.arm_id, i, localInertiaDiagonal=lid)
        # print(total_m)

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(np.array(arr) + self.np_random.uniform(low=-r, high=r, size=len(arr)))

    def sample_uniform_arm_q(self):
        ll = self.ll[self.arm_dofs]
        ul = self.ul[self.arm_dofs]
        q = [0.0] * (len(ll))
        for ind in range(len(ll)):
            q[ind] = self.np_random.uniform(low=ll[ind], high=ul[ind])
        return q

    # sp = list(self.sample_uniform_arm_q()) + [0.0]*len(self.all_findofs)
    # print(sp)

    def reset_with_certain_arm_q(self, arm_q):
        if self.init_noise:
            init_arm_q = self.perturb(arm_q, r=0.002)
            init_fin_q = self.perturb(self.init_fin_q, r=0.02)
        else:
            init_arm_q = np.copy(arm_q)
            init_fin_q = np.array(self.init_fin_q)

        for ind in range(len(self.arm_dofs)):
            self.sim.resetJointState(self.arm_id, self.arm_dofs[ind], init_arm_q[ind], 0.0)
        for ind in range(len(self.fin_actdofs)):
            self.sim.resetJointState(self.arm_id, self.fin_actdofs[ind], init_fin_q[ind], 0.0)
        for ind in range(len(self.fin_zerodofs)):
            self.sim.resetJointState(self.arm_id, self.fin_zerodofs[ind], 0.0, 0.0)
        self.tar_arm_q = init_arm_q
        self.tar_fin_q = init_fin_q

    def reset_only_certain_finger_states(self, all_fin_q, tar_act_q):
        if self.init_noise:
            all_fin_q = self.perturb(all_fin_q, r=0.01)
            tar_act_q = self.perturb(tar_act_q, r=0.01)
        else:
            all_fin_q = np.array(all_fin_q)
            tar_act_q = np.array(tar_act_q)
        for ind in range(len(self.all_findofs)):
            self.sim.resetJointState(self.arm_id, self.all_findofs[ind], all_fin_q[ind], 0.0)
        self.tar_fin_q = np.array(tar_act_q)

    def reset_with_certain_arm_q_finger_states(self, arm_q, all_fin_q, tar_act_q):
        if self.init_noise:
            arm_q = self.perturb(arm_q, r=0.002)
        else:
            arm_q = np.array(arm_q)
        for ind in range(len(self.arm_dofs)):
            self.sim.resetJointState(self.arm_id, self.arm_dofs[ind], arm_q[ind], 0.0)
        self.tar_arm_q = arm_q
        self.reset_only_certain_finger_states(all_fin_q, tar_act_q)

    def solve_arm_IK(self, w_pos, w_quat, rand_init=False):
        # reset according to wrist 6D pos
        wx_trans = list(w_pos)
        wx_quat = list(w_quat)
        closeEnough = False
        if rand_init:
            sp = list(self.np_random.uniform(low=-0.03, high=0.03, size=7)) + [0.0]*len(self.all_findofs)
            sp[3] -= 1.57
        else:
            sp = [-0.44, 0.00, -0.5, -1.8, -0.44, -0.488, -0.8] + [0.0]*len(self.all_findofs)    # dummy init guess IK
        ll = self.ll[self.arm_dofs+self.all_findofs]
        ul = self.ul[self.arm_dofs+self.all_findofs]
        jr = ul - ll
        iter = 0
        dist = 1e30
        while not closeEnough and iter < 50:
            for ind in range(len(self.arm_dofs)):
                self.sim.resetJointState(self.arm_id, self.arm_dofs[ind], sp[ind])

            jointPoses = self.sim.calculateInverseKinematics(self.arm_id, self.ee_id, wx_trans, wx_quat,
                                                      lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
                                                      jointRanges=jr.tolist(),
                                                      restPoses=sp)
            # jointPoses = self.sim.calculateInverseKinematics(self.arm_id, self.ee_id, wx_trans, wx_quat)

            sp = np.array(jointPoses)[range(7)].tolist()
            # print(sp)

            wx_now = self.sim.getLinkState(self.arm_id, self.ee_id)[4]
            dist = np.linalg.norm(np.array(wx_now) - np.array(wx_trans))
            # print("dist=", dist)
            if dist < 1e-3: closeEnough = True
            iter += 1
        if dist > 1e-3: sp = None     # TODO
        return sp

    def reset(self, w_pos, w_quat, all_fin_q=None, tar_act_q=None):
        sp = self.solve_arm_IK(w_pos, w_quat)
        if all_fin_q is None or tar_act_q is None:       # normal reset for grasping
            good_init = False
            while not good_init:
                self.reset_with_certain_arm_q(sp)
                cps = self.sim.getContactPoints(bodyA=self.arm_id)
                # for cp in cps:
                #     print(cp)
                #     input("penter")
                # print(cps[0][6])
                if len(cps) == 0: good_init = True   # TODO: init hand last and make sure it does not colllide with env
        else:
            # reset for placing, no collision checking
            self.reset_with_certain_arm_q_finger_states(sp, all_fin_q, tar_act_q)
        # input("after reset")

    def get_cur_jac(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs+self.all_findofs)
        n_arm_dofs = len(self.arm_dofs)
        [jac_t, jac_r] = self.sim.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                             list(wq)+list(self.get_q_dq(self.all_findofs)[0]),
                                             [0.] * n_dofs, [0.] * n_dofs)
        jac = np.array([jac_t[0][:n_arm_dofs], jac_t[1][:n_arm_dofs], jac_t[2][:n_arm_dofs],
                        jac_r[0][:n_arm_dofs], jac_r[1][:n_arm_dofs], jac_r[2][:n_arm_dofs]])
        return jac

    def get_norm_diff_tar(self):
        diff_fin = np.array(self.tar_fin_q) - self.get_q_dq(self.fin_actdofs)[0]
        diff_arm = np.array(self.tar_arm_q) - self.get_q_dq(self.arm_dofs)[0]
        return np.linalg.norm(np.concatenate((diff_fin, diff_arm)))

    def get_norm_diff_tar_arm(self):
        diff_arm = np.array(self.tar_arm_q) - self.get_q_dq(self.arm_dofs)[0]
        return np.linalg.norm(diff_arm)

    def get_robot_observation(self, withVel=False, diff_tar=False, flipped=False):
        obs = []

        links = [self.ee_id] + self.fin_tips
        for link in links:
            pos, orn, linVel, angVel = None, None, None, None
            if flipped:
                pos, orn = self.get_link_pos_quat_flipped(link)
                linVel, angVel = self.get_link_v_w_flipped(link)
            else:
                pos, orn = self.get_link_pos_quat(link)
                linVel, angVel = self.get_link_v_w(link)               
            obs.extend(pos)
            obs.extend(orn)
            if withVel:
                obs.extend(linVel)
                obs.extend(angVel)

        a_q, a_dq = self.get_q_dq(self.arm_dofs)
        obs.extend(list(a_q))
        obs.extend(list(a_dq))
        obs.extend(list(self.get_q_dq(self.fin_actdofs)[0]))    # no finger vel
        # print("thumb", self.get_q_dq(self.fin_actdofs)[0][-5:])

        # tar pos
        if diff_tar:
            obs.extend(list(np.array(self.tar_arm_q) - self.get_q_dq(self.arm_dofs)[0]))
            obs.extend(list(np.array(self.tar_fin_q) - self.get_q_dq(self.fin_actdofs)[0]))
        else:
            obs.extend(list(self.tar_arm_q))
            obs.extend(list(self.tar_fin_q))

        return obs


    def apply_action(self, a):
        # TODO: a is already scaled, how much to scale? decide in Env.
        # self.tar_arm_q = self.switchDirections(self.tar_arm_q)
        self.act = np.array(a)
        self.tar_arm_q += self.act[:len(self.arm_dofs)]     # assume arm controls are the first ones.
        if self.conservative_clip:
            cur_q, _ = self.get_q_dq(self.arm_dofs)
            self.tar_arm_q = np.clip(self.tar_arm_q, cur_q - self.conservative_range, cur_q + self.conservative_range)
        else:
            self.tar_arm_q = np.clip(self.tar_arm_q, self.ll[self.arm_dofs], self.ul[self.arm_dofs])
        self.tar_fin_q += self.act[len(self.arm_dofs):]
        self.tar_fin_q = np.clip(self.tar_fin_q, self.ll[self.fin_actdofs], self.ul[self.fin_actdofs])
        self.sim.setJointMotorControlArray(
            bodyIndex=self.arm_id,
            jointIndices=self.arm_dofs,
            controlMode=self.sim.POSITION_CONTROL,
            targetPositions=list(self.tar_arm_q),
            forces=[self.maxForce * 3] * len(self.arm_dofs))  # TODO: wrist force limit?
        self.sim.setJointMotorControlArray(
            bodyIndex=self.arm_id,
            jointIndices=self.fin_actdofs,
            controlMode=self.sim.POSITION_CONTROL,
            targetPositions=list(self.tar_fin_q),
            forces=[self.maxForce] * len(self.tar_fin_q))
        self.sim.setJointMotorControlArray(
            bodyIndex=self.arm_id,
            jointIndices=self.fin_zerodofs,
            controlMode=self.sim.POSITION_CONTROL,
            targetPositions=[0.0]*len(self.fin_zerodofs),
            forces=[self.maxForce / 4.0] * len(self.fin_zerodofs))

    def get_robot_observation_dim(self):
        return len(self.get_robot_observation())

    def get_q_dq(self, dofs):
        joints_state = self.sim.getJointStates(self.arm_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_joints_last_tau(self, dofs):
        joints_state = self.sim.getJointStates(self.arm_id, dofs)
        joints_taus = np.array(joints_state)[:, [3]]
        joints_taus = np.hstack(joints_taus.flatten())
        return joints_taus

    def switch_translation(self, val):
        return (val[0], -val[1], val[2])

    def switch_rotation(self, val, l_id):
        if len(val) == 3:
            return (-val[0], val[1], -val[2])
        elif len(val) == 4:
            if l_id >= 0 and l_id < 5:
                return (-val[0], val[1], -val[2], val[3])
            elif l_id >= 5 and l_id < 35:
                return (-val[0], -val[1], -val[2], -val[3])
        

    def get_link_pos_quat(self, l_id):
        newPos = self.sim.getLinkState(self.arm_id, l_id)[4]
        newOrn = self.sim.getLinkState(self.arm_id, l_id)[5]
        return newPos, newOrn

    def get_link_v_w(self, l_id):
        newLinVel = self.sim.getLinkState(self.arm_id, l_id, computeForwardKinematics=1, computeLinkVelocity=1)[6]
        newAngVel = self.sim.getLinkState(self.arm_id, l_id, computeForwardKinematics=1, computeLinkVelocity=1)[7]
        return newLinVel, newAngVel

    def get_link_pos_quat_flipped(self, l_id):
        newPos = self.switch_translation(self.sim.getLinkState(self.arm_id, l_id)[4])
        newOrn = self.switch_rotation(self.sim.getLinkState(self.arm_id, l_id)[5], l_id)
        return newPos, newOrn

    def get_link_v_w_flipped(self, l_id):
        newLinVel = self.switch_translation(self.sim.getLinkState(self.arm_id, l_id, computeForwardKinematics=1, computeLinkVelocity=1)[6])
        newAngVel = self.switch_rotation(self.sim.getLinkState(self.arm_id, l_id, computeForwardKinematics=1, computeLinkVelocity=1)[7], l_id)
        return newLinVel, newAngVel

    def get_wrist_wrench(self):
        # TODO: do not know if moments include those produced by force
        joint_reaction = list(self.sim.getJointState(self.arm_id, self.ee_id)[2])
        joint_reaction[3] = self.sim.getJointState(self.arm_id, self.ee_id)[3]     # TODO: rot axis 1,0,0
        _, p_quat = self.get_link_pos_quat(self.ee_id)
        # transform to world coord
        f_xyz, _ = self.sim.multiplyTransforms([0, 0, 0], p_quat, joint_reaction[:3], [0, 0, 0, 1])
        m_xyz, _ = self.sim.multiplyTransforms([0, 0, 0], p_quat, joint_reaction[3:], [0, 0, 0, 1])
        return list(f_xyz)+list(m_xyz)

    def get_4_finger_deviation(self):
        #  [9, 10, 11, 14, 15, 16, 19, 20, 21, 25, 26, 27]
        finger_qs = self.get_q_dq(self.fin_actdofs)[0]
        f2 = finger_qs[:3]
        f3 = finger_qs[3:6]
        f4 = finger_qs[6:9]
        f5 = finger_qs[9:12]
        return np.linalg.norm(f5 - f2) + np.linalg.norm(f2 - f3) + np.linalg.norm(f3 - f4) \
                + np.linalg.norm(f4 - f5)


if __name__ == "__main__":
    hz = 240.0
    dt = 1.0 / hz

    sim = bc.BulletClient(connection_mode=p.GUI)

    for i in range(100):
        sim.resetSimulation()
        # sim.setPhysicsEngineParameter(numSolverIterations=200)

        sim.setGravity(0, 0, 0)
        sim.setTimeStep(dt)
        sim.setRealTimeSimulation(0)

        arm = InmoovShadowNew()
        arm.np_random, seed = gym.utils.seeding.np_random(0)
        # arm.reset([-0.18, 0.105, 0.13, 1.8, -1.57, 0]) # obj [0,0,0]
        # arm.reset([0.02, 0.105, 0.13, 1.8, -1.57, 0]) # obj [0.2, 0, 0]
        arm.reset_with_certain_arm_q([0.02, 0.105-0.2, 0.13, 1.8, -1.57, 0, 0]) # obj [0.2, -0.2, 0]
        # arm.reset([-0.18, 0.105 - 0.2, 0.13, 1.8, -1.57, 0])  # obj [0, -0.2, 0]

        print("init", arm.get_robot_observation())
        ls = sim.getLinkState(arm.arm_id, arm.ee_id)
        newPos = ls[4]
        print(newPos, sim.getEulerFromQuaternion(ls[5]))
        input("press enter to continue")

        for t in range(400):
            arm.apply_action(arm.np_random.uniform(low=-0.003, high=0.003, size=7+17)+np.array([-0.005]*7+[-0.01]*17))
            sim.stepSimulation()
            arm.get_robot_observation()
            time.sleep(1. / 240.)
        print("final obz", arm.get_robot_observation())
        ls = sim.getLinkState(arm.arm_id, arm.ee_id)
        newPos = ls[4]
        print(newPos, sim.getEulerFromQuaternion(ls[5]))

    sim.disconnect()


