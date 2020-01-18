import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class ImaginaryArmObjSession:
    # (0, b'r_shoulder_out_joint', 0) -1.57079632679 1.57079632679
    # (1, b'r_shoulder_lift_joint', 0) -1.57079632679 1.57079632679
    # (2, b'r_upper_arm_roll_joint', 0) -1.57079632679 1.57079632679
    # (3, b'r_elbow_flex_joint', 0) -3.14159265359 0.0
    # (4, b'r_elbow_roll_joint', 0) -1.57079632679 1.57079632679
    # (5, b'r_wrist_roll_joint', 4) 0.0 -1.0
    # (6, b'rh_WRJ2', 0) -1.0471975512 1.0471975512
    # (7, b'rh_WRJ1', 4) 0.0 -1.0
    # (8, b'rh_obj_j', 4) 0.0 -1.0

    def __init__(self,
                 base_init_pos=np.array([-0.30, 0.348, 0.272]),
                 filename='inmoov_arm_v2_2_obj.urdf'):

        self.sim = bc.BulletClient(connection_mode=p.DIRECT)   # this is always session 1
        # print(self.sim2._client)
        self.sim.resetSimulation()
        self.sim.setGravity(0, 0, 0)
        self.sim.setRealTimeSimulation(0)

        self.base_init_pos = base_init_pos
        self.base_init_euler = np.array([0, 0, 0])

        self.arm_id = self.sim.loadURDF(os.path.join(currentdir,
                    "assets/inmoov_ros/inmoov_description/robots/imaginary_IK_robots/"+filename),
                    list(self.base_init_pos), p.getQuaternionFromEuler(list(self.base_init_euler)),
                    flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=1)

        # self.arm_dofs = [0, 1, 2, 3, 4, 6, 7]
        self.arm_dofs = [0, 1, 2, 3, 4, 6]
        self.ee_id = 8
        self.IK_iters = 1000

        self.reset()

        self.ll = np.array([self.sim.getJointInfo(self.arm_id, i)[8] for i in range(self.sim.getNumJoints(self.arm_id))])
        self.ul = np.array([self.sim.getJointInfo(self.arm_id, i)[9] for i in range(self.sim.getNumJoints(self.arm_id))])

    def __del__(self):        # TODO
        self.sim.disconnect()

    def reset(self, init_arm_q=np.array([0] * 6)):
        for ind in range(len(self.arm_dofs)):
            self.sim.resetJointState(self.arm_id, self.arm_dofs[ind], init_arm_q[ind], 0.0)

    def get_link_pos_quat(self, l_id):
        newPos = self.sim.getLinkState(self.arm_id, l_id)[4]
        newOrn = self.sim.getLinkState(self.arm_id, l_id)[5]
        return newPos, newOrn

    def get_q_dq(self, dofs):
        joints_state = self.sim.getJointStates(self.arm_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_cur_q_and_jac_two_points(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs)
        [jac_t, _] = self.sim.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        [jac_t_z, _] = self.sim.calculateJacobian(self.arm_id, self.ee_id+1, [0, 0, 0],
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
                        jac_t_z[0][:n_dofs], jac_t_z[1][:n_dofs]])
        return wq, jac

    def calc_IK_two_points(self, tar_5d):
        deviation = 1e30    # dummy
        wq = None
        it = 0
        while it < self.IK_iters and np.linalg.norm(deviation) > 1e-3:
            pos, _ = self.get_link_pos_quat(self.ee_id)
            pos_z, _ = self.get_link_pos_quat(self.ee_id + 1)
            deviation = np.array(list(pos)+list(pos_z[:2])) - tar_5d

            wq, jac = self.get_cur_q_and_jac_two_points()
            step, residue, _, _ = np.linalg.lstsq(jac, deviation, 1e-4)
            wq = wq - 0.01 * step
            wq = np.clip(wq, self.ll[self.arm_dofs], self.ul[self.arm_dofs])    # clip to jl
            self.reset(wq)
            it += 1
        _, quat = self.get_link_pos_quat(self.ee_id)
        return wq, np.linalg.norm(deviation), quat

    def get_cur_q_and_jac_two_points_xz(self):
        wq, _ = self.get_q_dq(self.arm_dofs)
        n_dofs = len(self.arm_dofs)
        [jac_t, _] = self.sim.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
                                         list(wq),
                                         [0.] * n_dofs, [0.] * n_dofs)
        [jac_t_z, _] = self.sim.calculateJacobian(self.arm_id, self.ee_id+1, [0, 0, 0],
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
        while it < self.IK_iters and np.linalg.norm(deviation) > 1e-3:
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

    def get_most_comfortable_q_and_refangle(self, tar_x, tar_y, tar_z=0.):
        tar = [tar_x, tar_y, tar_z, tar_x, tar_y]
        q, residue, quat = self.calc_IK_two_points(tar)
        angle = p.getEulerFromQuaternion(quat)
        # self.reset()
        if residue < 1e-3:
            return list(q) + [0.], angle[2]     # the first two will be zero.
        else:
            return None, None

    def get_most_comfortable_q_and_refangle_xz(self, tar_x, tar_y, tar_z=0.):
        tar = [tar_x, tar_y, tar_z, tar_x, tar_z+0.1]
        q, residue, quat = self.calc_IK_two_points_xz(tar)
        angle = p.getEulerFromQuaternion(quat)
        # self.reset()
        if residue < 1e-3:
            return list(q) + [0.], angle[2]     # the first two will be zero.
        else:
            return None, None

# In this class, the skeleton will be the same as the 5D problem, except that the wrist is now open
# it should solve for the solution that, not only the previous 5D constraints are satisfied,
# but also the theta' of object should be theta+delta_th


class ImaginaryArmObjSessionFlexWrist:
    # (0, b'r_shoulder_out_joint', 0) -1.57079632679 1.57079632679
    # (1, b'r_shoulder_lift_joint', 0) -1.57079632679 1.57079632679
    # (2, b'r_upper_arm_roll_joint', 0) -1.57079632679 1.57079632679
    # (3, b'r_elbow_flex_joint', 0) -3.14159265359 0.0
    # (4, b'r_elbow_roll_joint', 0) -1.57079632679 1.57079632679
    # (5, b'r_wrist_roll_joint', 4) 0.0 -1.0
    # (6, b'rh_WRJ2', 0) -1.0471975512 1.0471975512
    # (7, b'rh_WRJ1', 0) -1.57079632679 1.57079632679
    # (8, b'rh_obj_j', 4) 0.0 -1.0

    def __init__(self,
                 base_init_pos=np.array([-0.30, 0.348, 0.272]),
                 filename='inmoov_arm_v2_2_obj_flexwrist.urdf'):    # TODO: hard coded here

        self.sim = bc.BulletClient(connection_mode=p.DIRECT)   # this is always session > 0, init after main session.
        # print(self.sim2._client)
        self.sim.resetSimulation()
        self.sim.setGravity(0, 0, 0)
        self.sim.setRealTimeSimulation(0)

        self.base_init_pos = base_init_pos
        self.base_init_euler = np.array([0, 0, 0])

        self.arm_id = self.sim.loadURDF(os.path.join(currentdir,
                    "assets/inmoov_ros/inmoov_description/robots/imaginary_IK_robots/"+filename),
                    list(self.base_init_pos), p.getQuaternionFromEuler(list(self.base_init_euler)),
                    flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=1)

        # self.print_all_joints_info()

        self.arm_dofs = [0, 1, 2, 3, 4, 6, 7]
        self.ee_id = 8
        self.IK_iters = 25
        self.IK_thres = 1e-3
        # self.vary_angle = 0.6

        self.reset()

        self.ll = np.array([self.sim.getJointInfo(self.arm_id, i)[8] for i in range(self.sim.getNumJoints(self.arm_id))])
        self.ul = np.array([self.sim.getJointInfo(self.arm_id, i)[9] for i in range(self.sim.getNumJoints(self.arm_id))])

    def __del__(self):
        self.sim.disconnect()

    def print_all_joints_info(self):
        for i in range(self.sim.getNumJoints(self.arm_id)):
            print(self.sim.getJointInfo(self.arm_id, i)[0:3],
                  self.sim.getJointInfo(self.arm_id, i)[8], self.sim.getJointInfo(self.arm_id, i)[9])

    def reset(self, init_arm_q=np.array([0] * 7)):
        for ind in range(len(self.arm_dofs)):
            self.sim.resetJointState(self.arm_id, self.arm_dofs[ind], init_arm_q[ind], 0.0)

    def get_link_pos_quat(self, l_id):
        newPos = self.sim.getLinkState(self.arm_id, l_id)[4]
        newOrn = self.sim.getLinkState(self.arm_id, l_id)[5]
        return newPos, newOrn

    def get_q_dq(self, dofs):
        joints_state = self.sim.getJointStates(self.arm_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    # def get_cur_q_and_jac_two_points(self):
    #     wq, _ = self.get_q_dq(self.arm_dofs)
    #     n_dofs = len(self.arm_dofs)
    #     [jac_t, _] = self.sim.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
    #                                      list(wq),
    #                                      [0.] * n_dofs, [0.] * n_dofs)
    #     [jac_t_z, _] = self.sim.calculateJacobian(self.arm_id, self.ee_id+1, [0, 0, 0],
    #                                      list(wq),
    #                                      [0.] * n_dofs, [0.] * n_dofs)
    #     jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
    #                     jac_t_z[0][:n_dofs], jac_t_z[1][:n_dofs]])
    #     return wq, jac
    #
    # def calc_IK_two_points(self, tar_5d):
    #     deviation = 1e30    # dummy
    #     wq = None
    #     it = 0
    #     while it < self.IK_iters and np.linalg.norm(deviation) > 1e-3:
    #         pos, _ = self.get_link_pos_quat(self.ee_id)
    #         pos_z, _ = self.get_link_pos_quat(self.ee_id + 1)
    #         deviation = np.array(list(pos)+list(pos_z[:2])) - tar_5d
    #
    #         wq, jac = self.get_cur_q_and_jac_two_points()
    #         step, residue, _, _ = np.linalg.lstsq(jac, deviation, 1e-4)
    #         wq = wq - 0.01 * step
    #         wq = np.clip(wq, self.ll[self.arm_dofs], self.ul[self.arm_dofs])    # clip to jl
    #         self.reset(wq)
    #         it += 1
    #     return wq, np.linalg.norm(deviation)

    # def calc_IK_full_6D(self, tar_pos, tar_quat):
    #     deviation = 1e30  # dummy
    #     wq = None
    #     it = 0
    #     while it < self.IK_iters and np.linalg.norm(deviation) > 1e-3:
    #         pos, quat = self.get_link_pos_quat(self.ee_id)
    #
    #         eta_quat = quat[3]
    #         ep_quat = np.array(quat[:3])
    #         eta_d_quat = tar_quat[3]
    #         ep_d_quat = np.array(tar_quat[:3])
    #         orn_error = eta_quat * ep_d_quat - eta_d_quat * ep_quat - np.cross(ep_d_quat, ep_quat)  # tar_quat - quat
    #
    #         # print(np.linalg.norm(orn_error))
    #
    #         deviation = np.array(tar_pos) - np.array(pos)
    #         deviation = np.concatenate((deviation, orn_error))
    #
    #         wq, jac = self.get_cur_q_and_jac_full_6D()
    #         step, residue, _, _ = np.linalg.lstsq(jac, deviation, 1e-4)
    #         # print(residue)
    #         wq = wq + 0.001 * step
    #         wq = np.clip(wq, self.ll[self.arm_dofs], self.ul[self.arm_dofs])  # clip to jl
    #
    #         # print(np.linalg.norm(deviation))
    #         # print(wq)
    #         self.reset(wq)
    #         it += 1
    #     return wq, np.linalg.norm(deviation)

    # def get_cur_q_and_jac_full_6D(self):
    #     wq, _ = self.get_q_dq(self.arm_dofs)
    #     n_dofs = len(self.arm_dofs)
    #     [jac_t, jac_r] = self.sim.calculateJacobian(self.arm_id, self.ee_id, [0] * 3,
    #                                          list(wq),
    #                                          [0.] * n_dofs, [0.] * n_dofs)
    #     jac = np.array([jac_t[0][:n_dofs], jac_t[1][:n_dofs], jac_t[2][:n_dofs],
    #                     jac_r[0][:n_dofs], jac_r[1][:n_dofs], jac_r[2][:n_dofs]])
    #
    #     return wq, jac

    # def compare_jac_full_fin_diff(self, test_q):
    #     perturb = np.random.uniform(low=-1, high=1, size=len(self.arm_dofs))
    #
    #     for ind in range(len(self.arm_dofs)):
    #         p.resetJointState(self.arm_id, self.arm_dofs[ind], test_q[ind], 0.0)
    #     test_pos, test_quat = self.get_link_pos_quat(self.ee_id)
    #     _, jac = self.get_cur_q_and_jac_full_6D()
    #
    #     d_test_q = test_q + 0.0001 * perturb
    #     for ind in range(len(self.arm_dofs)):
    #         p.resetJointState(self.arm_id, self.arm_dofs[ind], d_test_q[ind], 0.0)
    #     d_test_pos, d_test_quat = self.get_link_pos_quat(self.ee_id)
    #     print(np.array(d_test_pos) - test_pos)
    #     eta_test_quat = test_quat[3]
    #     ep_test_quat = np.array(test_quat[:3])
    #     eta_d_test_quat = d_test_quat[3]
    #     ep_d_test_quat = np.array(d_test_quat[:3])
    #     error = eta_test_quat * ep_d_test_quat - eta_d_test_quat * ep_test_quat \
    #             - np.cross(ep_d_test_quat, ep_test_quat)
    #     print(error * 2.0)
    #     print(jac.dot(d_test_q - test_q))

    def solve_6D_IK_Bullet(self, w_pos, w_quat, init_q):
        tar_pos = list(w_pos)
        tar_quat = list(w_quat)
        closeEnough = False
        sp = init_q
        ll = self.ll[self.arm_dofs]
        ul = self.ul[self.arm_dofs]
        jr = ul - ll
        iter = 0
        while not closeEnough and iter < self.IK_iters:
            for ind in range(len(self.arm_dofs)):
                self.sim.resetJointState(self.arm_id, self.arm_dofs[ind], sp[ind])

            jointPoses = self.sim.calculateInverseKinematics(self.arm_id, self.ee_id, tar_pos, tar_quat,
                                                             lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
                                                             jointRanges=jr.tolist(),
                                                             restPoses=sp)
            # jointPoses = self.sim.calculateInverseKinematics(self.arm_id, self.ee_id, tar_pos, tar_quat)
            sp = np.array(jointPoses)[range(len(self.arm_dofs))].tolist()

            cur_pos, cur_quat = self.get_link_pos_quat(self.ee_id)
            dist = np.linalg.norm(np.array(cur_pos) - np.array(tar_pos)) \
                   + np.linalg.norm(np.array(cur_quat) - np.array(tar_quat))
            # print("dist=", dist)
            if dist < self.IK_thres: closeEnough = True
            iter += 1
        # print(self.get_link_pos_quat(self.ee_id))
        # print(self.get_link_pos_quat(self.ee_id-1))
        return sp, dist

    def sample_one_comfortable_q(self, tar_x, tar_y, vary_angle):
        tmp = ImaginaryArmObjSession()
        q_c, angle = tmp.get_most_comfortable_q_and_refangle(tar_x, tar_y)
        if q_c is not None:
            angle_t = angle + vary_angle
            tar_quat = p.getQuaternionFromEuler([0, 0, angle_t])
            q, residue = self.solve_6D_IK_Bullet([tar_x, tar_y, 0.0], tar_quat, q_c)

            if residue < self.IK_thres:
                return q
            else:
                return None
        else:
            return None

    def sample_multiple_comfortable_qs_planning(self):
        # -0.5, -0.2, 0, 0.2, 0.5
        pass


obj_urdf = """

<link name="obj">
    <visual>
      <geometry>
<!--          shape does not matter-->
        <box size="0.08 0.08 0.20"/>
      </geometry>
      <origin rpy="0 0 0" xyz="0 0 0.0"/>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
<!--          shape does not matter-->
        <box size="0.08 0.08 0.20"/>
      </geometry>
      <origin rpy="0 0 0" xyz="0 0 0.0"/>
    </collision>
    <inertial>
      <mass value="3.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      <origin rpy="0 0 0" xyz="0 0 0.1"/>
    </inertial>
  </link>

  <joint name="rh_obj_j" type="fixed">
    <parent link="rh_palm"/>
    <child link="obj"/>
<!--  quat  [%f, %f, %f, %f]-->
    <origin rpy="%f %f %f" xyz="%f %f %f"/>
  </joint>

  <link name="obj_z">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>>
  <joint name="rh_obj_j_z" type="fixed">
    <parent link="obj"/>
    <child link="obj_z"/>
    <origin rpy="0 0 0" xyz="0 0 0.1"/>
  </joint>    
    
"""

class URDFWriter():
    def __init__(self,
                 base_path="assets/inmoov_ros/inmoov_description/robots/imaginary_IK_robots"):
        self.base_path = os.path.join(currentdir, base_path)
        return

    def add_obj(self, trans, quat, new_file,
                base_file="inmoov_arm_v2_2_base.urdf",
                ):
        path_name_new = os.path.join(self.base_path, new_file)
        path_name = os.path.join(self.base_path, base_file)
        with open(path_name, 'r') as content_file:
            content = content_file.read()
            last_enter = content.rfind('\n')
            content_body = content[:last_enter]
            content_tail = content[last_enter:]
            euler = p.getEulerFromQuaternion(quat)
            insert = obj_urdf % (quat[0], quat[1], quat[2], quat[3],
                                 euler[0], euler[1], euler[2],
                                 trans[0], trans[1], trans[2])
            content_new = content_body + insert + content_tail

            if os.path.isfile(path_name_new):
                with open(path_name_new, 'r+') as existing_file:
                    exist_content = existing_file.read()
                    if exist_content != content_new:
                        existing_file.seek(0)
                        existing_file.write(content_new)
                        existing_file.truncate()
            else:
                with open(path_name_new, "w") as new_file:
                    new_file.write(content_new)
        return


if __name__ == "__main__":
    hz = 240.0
    dt = 1.0 / hz
    # # test 1
    writer = URDFWriter()
    trans = [0.007118853168487986, -0.06573216456627234, 0.08361430814391192]
    quat = [-0.12210281795880247, 0.5224276497343117, 0.07589733783405209, 0.8404759644090405]

    grasp_pi_name = '0114_box_l_4_new'      # also used to locate fin state pickle
    new_file = 'inmoov_arm_v2_2_obj_placing_'+grasp_pi_name+'.urdf'
    writer.add_obj(trans, quat, new_file)

    # # test 2
    # while True:
    #     grasp_pi_name = '0114_box_l_4'
    #     filename = 'inmoov_arm_v2_2_obj_placing_' + grasp_pi_name + '.urdf'
    #
    #     tar = list([np.random.uniform(low=-0.3, high=0.5), np.random.uniform(low=-0.3, high=0.8)])
    #     # tar = [0.1, 0]
    #
    #     tmp = ImaginaryArmObjSession(filename=filename)
    #     q_c, angle = tmp.get_most_comfortable_q_and_refangle_xz(tar[0], tar[1], 0.36)
    #     print(q_c)
    #     print(angle)
    #
    #
    #     if q_c is not None:
    #         input("press enter")
    #
    #     del tmp

    # # test 3
    # tar = list([np.random.uniform(low=-0.3, high=0.5), np.random.uniform(low=-0.3, high=0.8)])
    # tar = [0.1, 0]
    #
    # tmp = ImaginaryArmObjSession()
    # q_c, angle = tmp.get_most_comfortable_q_and_refangle(tar[0], tar[1])
    # print(q_c)
    # print(angle)
    #
    # a = ImaginaryArmObjSessionFlexWrist()
    #
    # q = a.get_most_comfortable_q(tar[0], tar[1], angle-0.7, q_c)
    # print(q)
    # input("press enter")

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # ax.set_xlim(0.8, -0.3)
    # ax.set_ylim(-0.3, 0.5)
    #
    # X = []
    # Y = []
    # U = []
    # V = []
    # tmp = ImaginaryArmObjSession()
    # a = ImaginaryArmObjSessionFlexWrist()
    # hist = []
    #
    # for ind in range(1000):
    #     tar = list([np.random.uniform(low=-0.3, high=0.5), np.random.uniform(low=-0.3, high=0.8)])
    #
    #     q_c, angle = tmp.get_most_comfortable_q_and_refangle(tar[0], tar[1])
    #     if q_c is not None:
    #         # print(q_c)
    #         # print(angle)
    #
    #         for ind in range(-2, 3):
    #             # angle_t = angle + ind * 0.3
    #             angle_t = angle + np.random.uniform(-0.6, 0.6)
    #
    #             tar_quat = p.getQuaternionFromEuler([0, 0, angle_t])
    #             q, residue = a.solve_6D_IK_Bullet(tar+[0.0], tar_quat, q_c)
    #
    #             # print("residue", residue)
    #             # input("press enter")
    #
    #             if residue < 1e-3:
    #                 hist.append(q[6])
    #
    #                 X.append(tar[0])
    #                 Y.append(tar[1])
    #                 _, quat = a.get_link_pos_quat(a.ee_id)
    #                 x_rot, _ = p.multiplyTransforms([0, 0, 0], quat, [0.1, 0, 0], [0, 0, 0, 1])
    #                 U.append([x_rot[0]])
    #                 V.append([x_rot[1]])
    #
    #     tmp.reset()
    #     a.reset()
    #
    # fig2, axis2 = plt.subplots()
    # n, bins, patches = axis2.hist(hist, 10, facecolor='blue', alpha=0.5)
    #
    # q = ax.quiver(Y, X, V, U, angles='xy', headwidth=0.5)
    # plt.show()
    #
    # p.disconnect()


# how to sample delta_angle during train and test
# given x,y, uniformly get a series of theta around comfortable theta
# If solved q too close to JR, should not train on this theta, nor should pass to planner during testing.
# assume here that all angles not close to JR should be able to grasp.
