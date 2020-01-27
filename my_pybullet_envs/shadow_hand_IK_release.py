import pybullet as p
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import pickle

# TODO: render

class ShadowHand:
    def __init__(self):      # default range for reaching

        np.random.seed(300)   # TODO

        self.is_box = False
        self.is_small = True
        self.grasp_pi_name = '0120_cyl_s_1'

        self.fin_actdofs = list(np.array([9, 10, 11, 14, 15, 16, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33]) - 8)
        self.fin_zerodofs = list(np.array([8, 13, 18, 24]) - 8)
        self.fin_tips = list(np.array([12, 17, 22, 28, 34]) - 8)
        self.all_findofs = list(np.sort(self.fin_actdofs+self.fin_zerodofs))
        # self.init_fin_q = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0])
        # self.tar_arm_q = np.zeros(len(self.arm_dofs))       # dummy
        # self.tar_fin_q = np.zeros(len(self.fin_actdofs))

        self.arm_id = p.loadURDF(os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadow_hand_v2_2.urdf"),
                                 [0,0,0], p.getQuaternionFromEuler([0,0,0]),
                                 flags=p.URDF_USE_SELF_COLLISION, useFixedBase=1)
        nDof = p.getNumJoints(self.arm_id)
        for i in range(p.getNumJoints(self.arm_id)):
            print(p.getJointInfo(self.arm_id, i)[2], p.getJointInfo(self.arm_id, i)[8], p.getJointInfo(self.arm_id, i)[9])

        # use np for multi-indexing
        self.ll = np.array([p.getJointInfo(self.arm_id, i)[8] for i in range(p.getNumJoints(self.arm_id))])
        self.ul = np.array([p.getJointInfo(self.arm_id, i)[9] for i in range(p.getNumJoints(self.arm_id))])


        # # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/constraint.py#L11
        self.cid = p.createConstraint(self.arm_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      childFramePosition=list([0,0,0]),
                                      childFrameOrientation=p.getQuaternionFromEuler([0,0,0]))

        input("press enter")

        self.saved_file = None
        with open(os.path.join(currentdir, 'assets/place_init_dist/final_states_' + self.grasp_pi_name + '.pickle'),
                  'rb') as handle:
            self.saved_file = pickle.load(handle)
        assert self.saved_file is not None

        self.o_pos_pf_ave = self.saved_file['ave_obj_pos_in_palm']
        self.o_quat_pf_ave = self.saved_file['ave_obj_quat_in_palm']
        self.o_quat_pf_ave /= np.linalg.norm(self.o_quat_pf_ave)        # in case not normalized
        self.init_states = self.saved_file['init_states']  # a list of dicts

    def reset_with_certain_finger_states(self, all_fin_q=None, tar_act_q=None):
        for ind in range(len(self.all_findofs)):
            p.resetJointState(self.arm_id, self.all_findofs[ind], all_fin_q[ind], 0.0)
        if tar_act_q is not None:
            self.tar_fin_q = np.array(tar_act_q)

    def sample_init_state(self):
        ran_ind = int(np.random.uniform(low=0, high=len(self.init_states) - 0.1))
        return self.init_states[ran_ind]

    def reset_robot_object_from_sample(self, state):
        o_pos = state['obj_pos_in_palm']
        o_quat = state['obj_quat_in_palm']
        all_fin_q_init = state['all_fin_q']
        tar_fin_q_init = state['fin_tar_q']

        self.reset_with_certain_finger_states(all_fin_q_init, tar_fin_q_init)

        # TODO: assume palm at 0,0,0 / 0,0,0,1

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

        return

    def reset(self):
        # TODO: bullet env reload urdfs in reset...
        # TODO: bullet env reset pos with added noise but velocity to zero always.

        init_state = self.sample_init_state()
        self.reset_robot_object_from_sample(init_state)
        input("press enter")

    def move_to_openup_pose(self):
        f_thumb =  np.array([-0.84771132,  0.60768666, -0.13419822,  0.52214954,  0.25141182])

        n_iter = 100
        for i in range(n_iter):
            wq, _ = self.get_q_dq(self.all_findofs)
            step = (f_thumb - wq[-5:]) / n_iter
            wq[-5:] = wq[-5:] + step
            self.reset_with_certain_finger_states(wq)
            print(wq[-5:])
            input("press enter")
        pass

    def solve_openup_IK(self):
        thumb_tip = self.fin_tips[4]

        wq, _ = self.get_q_dq(self.all_findofs)


        dt = 0.001
        it = 0
        while it < 10000:
            n_dofs = len(self.all_findofs)
            [jac_t, _] = p.calculateJacobian(self.arm_id, thumb_tip, [0] * 3,
                                             list(wq),
                                             [0.] * n_dofs, [0.] * n_dofs)

            jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs]])

            outward_xyz = np.array(self.get_link_pos_quat(thumb_tip)[0]) -\
                          p.getBasePositionAndOrientation(self.obj_id)[0]
            _, obj_quat = p.getBasePositionAndOrientation(self.obj_id)
            _, inv_obj_quat = p.invertTransform([0,0,0], obj_quat)
            o_xyz_pf, _ = p.multiplyTransforms([0,0,0], inv_obj_quat, outward_xyz, [0,0,0,1])
            o_xy_pf = np.array(o_xyz_pf[0:2])    # obj height in z direction
            # print(o_xy_pf)
            n_o_xy_pf = o_xy_pf / np.linalg.norm(o_xy_pf)
            n_o_xyz, _ = p.multiplyTransforms([0,0,0], obj_quat, list(n_o_xy_pf)+[0.], [0,0,0,1])

            # print(jac)
            # print(n_o_xyz)

            step, residue, _, _ = np.linalg.lstsq(jac, np.array(n_o_xyz) * dt, 1e-4)

            wq = wq+step
            wq = np.clip(wq, self.ll[self.all_findofs], self.ul[self.all_findofs])  # clip to jl
            self.reset_with_certain_finger_states(wq)
            # print(step)
            print(o_xy_pf)
            print(wq[-5:])
            input("press eneter")
            it += 1
            # input("press enter")
        # _, quat = self.get_link_pos_quat(self.ee_id)

    def get_q_dq(self, dofs):
        joints_state = p.getJointStates(self.arm_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_link_pos_quat(self, l_id):
        newPos = p.getLinkState(self.arm_id, l_id)[4]
        newOrn = p.getLinkState(self.arm_id, l_id)[5]
        return newPos, newOrn



if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)    #or p.DIRECT for non-graphical version

    p.setTimeStep(1./240.)
    p.setGravity(0, 0, -10)

    hand = ShadowHand()
    hand.reset()

    hand.solve_openup_IK()
    # hand.move_to_openup_pose()

    p.disconnect()