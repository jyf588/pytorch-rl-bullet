
import pybullet as p
import time
import math
from datetime import datetime
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# TODO: do a anaylsis if using both xyz as constraint: looks a little bit better.
# TODO: think what to do with the shrinkage of work space during placing.

# (0, b'r_shoulder_out_joint', 0) -1.57079632679 1.57079632679
# (1, b'r_shoulder_lift_joint', 0) -1.57079632679 1.57079632679
# (2, b'r_upper_arm_roll_joint', 0) -1.57079632679 1.57079632679
# (3, b'r_elbow_flex_joint', 0) -3.14159265359 0.0
# (4, b'r_elbow_roll_joint', 0) -1.57079632679 1.57079632679
# (5, b'r_wrist_roll_joint', 4) 0.0 -1.0
# (6, b'rh_WRJ2', 0) -1.0471975512 1.0471975512
# (7, b'rh_WRJ1', 4) 0.0 -1.0
# (8, b'rh_obj_j', 4) 0.0 -1.0

class InmoovArmObj:
    def __init__(self):
        self.base_init_pos = np.array([-0.30, 0.348, 0.272])
        self.base_init_euler = np.array([0,0,0])

        # path = "my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/imaginary_IK_robots/inmoov_arm_v2_2_obj_placing_0114_box_l_4.urdf"
        path = "my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/imaginary_IK_robots/inmoov_arm_v2_2_obj_placing_0114_cyl_s_1.urdf"
        # path = "my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/imaginary_IK_robots/inmoov_arm_v2_2_obj.urdf"

        self.arm_id = p.loadURDF(os.path.join(currentdir,
                                             path),
                                 list(self.base_init_pos), p.getQuaternionFromEuler(list(self.base_init_euler)),
                                 flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
                                       | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, useFixedBase=1)
        self.print_all_joints_info()
        # self.arm_dofs = [0, 1, 2, 3, 4, 6, 7]   # TODO
        self.arm_dofs = [0, 1, 2, 3, 4, 6]

        self.reset(np.array([-0.44, 0.00, -0.5, -1.8, -0.44, -0.488]))

        self.ee_id = 8

        print(self.get_link_6D(self.ee_id))

        self.ll = np.array([p.getJointInfo(self.arm_id, i)[8] for i in range(p.getNumJoints(self.arm_id))])
        self.ul = np.array([p.getJointInfo(self.arm_id, i)[9] for i in range(p.getNumJoints(self.arm_id))])

    def reset(self, init_arm_q=np.array([0.] * 6)):  # TODO
        # init_arm_q += np.random.uniform(low=-0.03, high=0.03, size=6)
        # init_arm_q[3] -= 1.57
        for ind in range(len(self.arm_dofs)):
            p.resetJointState(self.arm_id, self.arm_dofs[ind], init_arm_q[ind], 0.0)

    def print_all_joints_info(self):
        for i in range(p.getNumJoints(self.arm_id)):
            print(p.getJointInfo(self.arm_id, i)[0:3],
                  p.getJointInfo(self.arm_id, i)[8], p.getJointInfo(self.arm_id, i)[9])

    def get_link_6D(self, l_id):
        newPos = p.getLinkState(self.arm_id, l_id)[4]
        newOrn = p.getLinkState(self.arm_id, l_id)[5]
        return list(newPos) + list(p.getEulerFromQuaternion(newOrn))

    def get_link_pos_quat(self, l_id):
        newPos = p.getLinkState(self.arm_id, l_id)[4]
        newOrn = p.getLinkState(self.arm_id, l_id)[5]
        return newPos, newOrn

    def get_cur_jac(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs)
        [jac_t, jac_r] = p.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                             list(wq),
                                             [0.] * n_dofs, [0.] * n_dofs)
        jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
                        jac_r[0][:n_dofs], jac_r[1][:n_dofs]])      # assume jac_r[2][:n_dofs] is dont care
        return jac

    def get_cur_jac_full(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs)
        [jac_t, jac_r] = p.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                             list(wq),
                                             [0.] * n_dofs, [0.] * n_dofs)
        jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
                        jac_r[0][:n_dofs], jac_r[1][:n_dofs], jac_r[2][:n_dofs]])

        return jac

    def get_cur_jac_two_points(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs)
        [jac_t, _] = p.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        [jac_t_z, _] = p.calculateJacobian(self.arm_id, self.ee_id+1, [0, 0, 0],
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
                        jac_t_z[0][:n_dofs], jac_t_z[1][:n_dofs]])
        return jac


    def get_q_dq(self, dofs):
        joints_state = p.getJointStates(self.arm_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    # def calc_IK(self, tar_5d):
    #     for it in range(10000):
    #         print(it)
    #         deviation = np.array(self.get_link_6D(self.ee_id)[:5]) - tar_5d
    #         print("dist", np.linalg.norm(deviation))
    #         print("6D", self.get_link_6D(self.ee_id))
    #
    #         deviation = np.array(list(deviation) + [0.])
    #
    #         wq, _ = self.get_q_dq(self.arm_dofs)
    #
    #         jac = self.get_cur_jac_full()
    #         r, pt, y = self.get_link_6D(self.ee_id)[3:]
    #         T = [[np.cos(pt) * np.cos(y), np.sin(y), 0], [-np.cos(pt) * np.sin(y), np.cos(y), 0], [np.sin(pt), 0, 1]]
    #         T = np.array(T)
    #         Id = np.identity(3)
    #         TT = block_diag(Id, T)
    #
    #         # print(TT.dot(jac))
    #         # print(np.array(list(deviation)+[0.]))
    #
    #         step, residue, _, _ = np.linalg.lstsq(TT.dot(jac), deviation, 1e-4)
    #         # step, residue, _, _ = np.linalg.lstsq(jac, deviation, 1e-4)
    #         print(wq)
    #         print(step)
    #         wq = wq - 0.001 * step
    #
    #         wq = np.clip(wq, self.ll[self.arm_dofs], self.ul[self.arm_dofs])
    #
    #         for ind in range(len(self.arm_dofs)):
    #             p.resetJointState(self.arm_id, self.arm_dofs[ind], wq[ind], 0.0)
    #         print("q", wq)
    #         # TODO: clip to joint limit.

    def calc_IK_two_points(self, tar_5d):
        for it in range(1000):
            # print(it)
            pos, _ = self.get_link_pos_quat(self.ee_id)
            pos_z, _ = self.get_link_pos_quat(self.ee_id + 1)
            deviation = np.array(list(pos)+list(pos_z[:2])) - tar_5d
            # print("dist", np.linalg.norm(deviation))
            # print("6D", self.get_link_6D(self.ee_id))

            wq, _ = self.get_q_dq(self.arm_dofs)

            jac = self.get_cur_jac_two_points()
            step, residue, _, _ = np.linalg.lstsq(jac, deviation, 1e-4)
            # print(step)
            wq = wq - 0.01 * step
            wq = np.clip(wq, self.ll[self.arm_dofs], self.ul[self.arm_dofs])

            for ind in range(len(self.arm_dofs)):
                p.resetJointState(self.arm_id, self.arm_dofs[ind], wq[ind], 0.0)
            # print("q", wq)
            # TODO: clip to joint limit.
        return np.linalg.norm(deviation)

    def get_cur_q_and_jac_two_points_xz(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs)
        [jac_t, _] = p.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        [jac_t_z, _] = p.calculateJacobian(self.arm_id, self.ee_id+1, [0, 0, 0],
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
                        jac_t_z[0][:n_dofs], jac_t_z[2][:n_dofs]])
        return wq, jac

    def calc_IK_two_points_xz(self, tar_5d):
        # tar_5d now is com xyz, (0,0,1) x and (0,0,0.1) z
        # we want (0,0,0.1)z to be at comz+0.1
        deviation = 1e30    # dummy
        wq = None
        it = 0
        # while it < 1000 and np.linalg.norm(deviation) > 1e-3:
        while it < 1000:
            pos, _ = self.get_link_pos_quat(self.ee_id)
            pos_z, _ = self.get_link_pos_quat(self.ee_id + 1)

            deviation = np.array(list(pos)+list([pos_z[0]])+list([pos_z[2]])) - tar_5d    #

            wq, jac = self.get_cur_q_and_jac_two_points_xz()
            step, residue, _, _ = np.linalg.lstsq(jac, deviation, 1e-4)
            wq = wq - 0.01 * step
            wq = np.clip(wq, self.ll[self.arm_dofs], self.ul[self.arm_dofs])    # clip to jl
            self.reset(wq)
            it += 1
        _, quat = self.get_link_pos_quat(self.ee_id)
        return wq, np.linalg.norm(deviation), quat

    def get_cur_q_and_jac_two_points_xyz(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs)
        [jac_t, _] = p.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        [jac_t_z, _] = p.calculateJacobian(self.arm_id, self.ee_id+1, [0, 0, 0],
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
                        jac_t_z[0][:n_dofs], jac_t_z[1][:n_dofs], jac_t_z[2][:n_dofs]])
        return wq, jac

    def calc_IK_two_points_xyz(self, tar_6d):
        # tar_5d now is com xyz, (0,0,1) xyz
        deviation = 1e30    # dummy
        wq = None
        it = 0
        # while it < 1000 and np.linalg.norm(deviation) > 1e-3:
        while it < 1000:
            pos, _ = self.get_link_pos_quat(self.ee_id)
            pos_z, _ = self.get_link_pos_quat(self.ee_id + 1)

            deviation = np.array(list(pos)+list(pos_z)) - tar_6d    #

            wq, jac = self.get_cur_q_and_jac_two_points_xyz()
            step, residue, _, _ = np.linalg.lstsq(jac, deviation, 1e-4)
            wq = wq - 0.01 * step
            wq = np.clip(wq, self.ll[self.arm_dofs], self.ul[self.arm_dofs])    # clip to jl
            self.reset(wq)
            it += 1
        _, quat = self.get_link_pos_quat(self.ee_id)
        return wq, np.linalg.norm(deviation), quat

    def compare_jac_fin_diff(self, test_q):
        perturb = np.random.uniform(low=-1, high=1, size=len(self.arm_dofs))

        for ind in range(len(self.arm_dofs)):
            p.resetJointState(self.arm_id, self.arm_dofs[ind], test_q[ind], 0.0)
        test_x = np.array(self.get_link_6D(self.ee_id)[:5])

        jac = self.get_cur_jac_full()
        r, pt, y = self.get_link_6D(self.ee_id)[3:]
        T = [[np.cos(pt) * np.cos(y), np.sin(y), 0], [-np.cos(pt) * np.sin(y), np.cos(y), 0], [np.sin(pt), 0,1]]
        # T = [[np.cos(pt) * np.cos(r), np.sin(r), 0], [-np.cos(pt) * np.sin(r), np.cos(r), 0], [np.sin(pt), 0,1]]
        # T = [[1, 0, -np.sin(pt)], [0, np.cos(r), np.cos(pt) * np.sin(r)], [0, -np.sin(r), np.cos(pt) * np.cos(r)]]
        # T = [[1, 0, -np.sin(pt)], [0, np.cos(y), np.cos(pt) * np.sin(y)], [0, -np.sin(y), np.cos(pt) * np.cos(y)]]
        T = np.array(T)
        Id = np.identity(3)
        TT = block_diag(Id, np.linalg.inv(T))
        TT = block_diag(Id, T)

        d_test_q = test_q + 0.0001 * perturb
        for ind in range(len(self.arm_dofs)):
            p.resetJointState(self.arm_id, self.arm_dofs[ind], d_test_q[ind], 0.0)
        d_test_x = np.array(self.get_link_6D(self.ee_id)[:5])
        print(d_test_x - test_x)

        # print(TT.jac.dot(d_test_q - test_q))
        print(TT.dot(jac.dot(d_test_q - test_q)))

    def compare_jac_full_fin_diff(self, test_q):
        perturb = np.random.uniform(low=-1, high=1, size=len(self.arm_dofs))

        for ind in range(len(self.arm_dofs)):
            p.resetJointState(self.arm_id, self.arm_dofs[ind], test_q[ind], 0.0)
        test_pos, test_quat = self.get_link_pos_quat(self.ee_id)
        jac = self.get_cur_jac_full()

        d_test_q = test_q + 0.0001 * perturb
        for ind in range(len(self.arm_dofs)):
            p.resetJointState(self.arm_id, self.arm_dofs[ind], d_test_q[ind], 0.0)
        d_test_pos, d_test_quat = self.get_link_pos_quat(self.ee_id)
        print(np.array(d_test_pos) - test_pos)
        eta_test_quat = test_quat[3]
        ep_test_quat = np.array(test_quat[:3])
        eta_d_test_quat = d_test_quat[3]
        ep_d_test_quat = np.array(d_test_quat[:3])
        error = eta_test_quat * ep_d_test_quat - eta_d_test_quat * ep_test_quat \
                - np.cross(ep_d_test_quat, ep_test_quat)
        print(error * 2.0)
        print(jac.dot(d_test_q - test_q))

    def compare_jac_two_point_fin_diff(self, test_q):
        perturb = np.random.uniform(low=-1, high=1, size=len(self.arm_dofs))

        for ind in range(len(self.arm_dofs)):
            p.resetJointState(self.arm_id, self.arm_dofs[ind], test_q[ind], 0.0)
        test_pos, _ = self.get_link_pos_quat(self.ee_id)
        test_pos_z, _ = self.get_link_pos_quat(self.ee_id+1)
        jac = self.get_cur_jac_two_points()

        d_test_q = test_q + 0.0001 * perturb
        for ind in range(len(self.arm_dofs)):
            p.resetJointState(self.arm_id, self.arm_dofs[ind], d_test_q[ind], 0.0)
        d_test_pos, _ = self.get_link_pos_quat(self.ee_id)
        d_test_pos_z, _ = self.get_link_pos_quat(self.ee_id+1)
        print(np.array(d_test_pos) - test_pos)
        print(np.array(d_test_pos_z) - test_pos_z)

        print(jac.dot(d_test_q - test_q))

    def solve_palm_IK(self, w_pos, w_quat):
        # reset according to wrist 6D pos
        wx_trans = list(w_pos)
        wx_quat = list(w_quat)
        closeEnough = False
        sp = [-0.44, 0.00, -0.5, -1.8, -0.44, -0.488] # dummy init guess IK
        ll = self.ll[self.arm_dofs]
        ul = self.ul[self.arm_dofs]
        jr = ul - ll
        iter = 0
        while not closeEnough and iter < 2000:
            for ind in range(len(self.arm_dofs)):
                p.resetJointState(self.arm_id, self.arm_dofs[ind], sp[ind])

            jointPoses = p.calculateInverseKinematics(self.arm_id, self.ee_id-1, wx_trans, wx_quat,
                                                      lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
                                                      jointRanges=jr.tolist(),
                                                      restPoses=sp)
            # jointPoses = p.calculateInverseKinematics(self.arm_id, self.ee_id, wx_trans, wx_quat)

            sp = np.array(jointPoses)[range(6)].tolist()
            # print(sp)

            wx_now = p.getLinkState(self.arm_id, self.ee_id-1)[4]
            dist = np.linalg.norm(np.array(wx_now) - np.array(wx_trans))
            print("dist=", dist)
            if dist < 1e-5: closeEnough = True
            iter += 1
        print(self.get_link_pos_quat(self.ee_id))
        print(self.get_link_pos_quat(self.ee_id-1))


hz = 240.0
dt = 1.0 / hz

p.connect(p.DIRECT)
p.resetSimulation()
# p.setPhysicsEngineParameter(numSolverIterations=200)

p.setGravity(0, 0, 0)
p.setTimeStep(dt)
p.setRealTimeSimulation(0)

floorId = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/plane.urdf'),
                          [0, 0, 0], useFixedBase=1)

a = InmoovArmObj()

# a.solve_palm_IK([-0.18, 0.095, 0.11], p.getQuaternionFromEuler([1.8, -1.57, 0]))
# from IK old reset
# ((-0.17977985739707947, 0.09488461911678314, 0.11017470806837082), (0.5541251301765442, -0.4393734335899353, 0.5536625981330872, 0.43972036242485046))


input("press enter")
# a.compare_jac_two_point_fin_diff([-0.4, -0.7, 0.5, -1.2, -0.5, 0.0])

# a.compare_jac_fin_diff([-0.44, 0.00, -0.5, -1.8, -0.44, -0.488])
# a.compare_jac_fin_diff([-0.44, 1, -0.5, -1.8, -0.44, -0.488])

# init_arm_q = [-0.4, -0.7, 0.5, -1.2, -0.5, 0.0]

fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlim(0.8, -0.3)
ax.set_ylim(-0.3, 0.5)

X = []
Y = []
U = []
V = []
for ind in range(1000):
    tar = list([np.random.uniform(low=-0.3, high=0.5), np.random.uniform(low=-0.3, high=0.8)])

    # residue = a.calc_IK_two_points(tar + [0.] + tar)
    # residue = a.calc_IK_two_points(tar + [0.36] + tar)

    # _, residue, _ = a.calc_IK_two_points_xz(tar + [0.] + [tar[0], 0.1])
    # _, residue, _ = a.calc_IK_two_points_xz(tar + [0.36] + [tar[0], 0.36+0.1])
    _, residue, _ = a.calc_IK_two_points_xyz(tar + [0.36] + tar + [0.36 + 0.1])
    print(residue)
    # input("press enter")

    # a.calc_IK(tar + [0., 0, 0])
    # input("press enter")

    if residue < 1e-3:
        X.append(tar[0])
        Y.append(tar[1])
        _, quat = a.get_link_pos_quat(a.ee_id)
        x_rot, _ = p.multiplyTransforms([0,0,0], quat, [0.1,0,0], [0,0,0,1])
        U.append([x_rot[0]])
        V.append([x_rot[1]])

    a.reset()

q = ax.quiver(Y, X, V, U, angles='xy')
plt.show()

p.disconnect()
