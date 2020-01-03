import pybullet as p
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# a new shadow hand.
# connect to world through 6 links, 3 primatic, 3 revolute.

# TODO: render

class ShadowHandVel:
    def __init__(self,
                 base_init_pos=np.array([-0.18, 0.105, 0.13]),      # 0.035 offset from old hand
                 init_fin_pos=np.array([0.4, 0.4, 0.4]*4 + [0.0, 1.0, 0.1, 0.5, 0.2]),    # last was 0.0
                 init_noise=True,
                 act_noise=True,
                 timestep=1./240):

        self.baseInitPos = base_init_pos
        self.init_noise = init_noise
        self.initPos = init_fin_pos
        self.act_noise = act_noise
        self._timestep = timestep

        self.handId = \
            p.loadURDF(os.path.join(currentdir,
                                    "assets/shadow_hand_arm/sr_description/robots/shadowhand_motor_simple_nomass.urdf"),
                       list(self.baseInitPos), p.getQuaternionFromEuler([0, 0, 0]),
                       flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
                             | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                       useFixedBase=1)

        # nDof = p.getNumJoints(self.handId)
        for i in range(p.getNumJoints(self.handId)):
            print(p.getJointInfo(self.handId, i)[0:3], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])

        # input("press enter 0")
        # (0, b'world_x', 1) -0.3 0.3
        # (1, b'world_y', 1) -0.1 0.1
        # (2, b'world_z', 1) -0.1 0.1
        # (3, b'world_rx', 0) -0.7 0.7
        # (4, b'world_ry', 0) -0.7 0.7
        # (5, b'world_rz', 0) -0.7 0.7
        # (6, b'rh_palm_aux_joint', 4) 0.0 -1.0
        self.endEffectorId = 5  # link 5 is palm_aux

        # 4,4,4,5,5
        # 3*4+5=17 finger dofs for now.
        self.activeDofs = []
        start = self.endEffectorId + 2   # add palm aux & 6 root joints
        for i in range(5):
            nDofs = [4,4,4,5,5]
            fig_start = [1,1,1,2,0]     # I fixed some joints
            self.activeDofs += (np.arange(fig_start[i], nDofs[i]) + start).tolist()
            start += nDofs[i]
        assert len(self.activeDofs) == len(init_fin_pos)

        self.ll = np.array([p.getJointInfo(self.handId, i)[8] for i in self.activeDofs])
        self.ul = np.array([p.getJointInfo(self.handId, i)[9] for i in self.activeDofs])

        total_m = 0
        for i in range(self.endEffectorId + 1, p.getNumJoints(self.handId)):       # add palm aux & 6 root joints
            dyn = p.getDynamicsInfo(self.handId, i)
            mass = dyn[0]
            # mass = mass / 100.
            lid = dyn[2]
            # lid = (lid[0] / 10., lid[1] / 10., lid[2] / 10.,)       # TODO
            total_m += mass
            p.changeDynamics(self.handId, i, mass=mass)
            p.changeDynamics(self.handId, i, localInertiaDiagonal=lid)
            p.changeDynamics(self.handId, i, lateralFriction=3.0)
            p.changeDynamics(self.handId, i, jointDamping=0.0)      # we use vel control anyways
            # dyn = p.getDynamicsInfo(self.handId, i)
            # print(dyn[2])
            # print(dyn[0])
            # p.setJointMotorControl2(self.handId, i, p.VELOCITY_CONTROL, force=0.000)  # turn off default control

        print("total hand Mass:", total_m)

        self.maxForce = 1000.

        self.include_redun_body_pos = False

        self.np_random = None   # seeding inited outside in Env

        self.n_dofs = 6 + len(self.activeDofs)      # exclude fixed joints for IK/Jac
        self.act = np.zeros(self.n_dofs)            # dummy, action dim

        p.enableJointForceTorqueSensor(self.handId, self.endEffectorId + 1, True)

    def reset(self):
        # TODO: bullet env reload urdfs in reset...
        # TODO: bullet env reset pos with added noise but velocity to zero always.

        good_init = False
        while not good_init:

            if self.init_noise:
                init_xyz = self.np_random.uniform(low=-0.02, high=0.02, size=3)
                init_rpy = self.np_random.uniform(low=-0.05, high=0.05, size=3)
                init_pos = self.initPos + self.np_random.uniform(low=-0.05, high=0.05, size=len(self.initPos))
            else:
                init_xyz = np.array([0., 0, 0])
                init_rpy = np.array([0., 0, 0])
                init_pos = self.initPos

            for i in range(3):
                p.resetJointState(self.handId, i, init_xyz[i])
                p.resetJointState(self.handId, i+3, init_rpy[i])
            for ind in range(len(self.activeDofs)):
                p.resetJointState(self.handId, self.activeDofs[ind], init_pos[ind], 0.0)

            cps = p.getContactPoints(bodyA=self.handId)
            for cp in cps:
                print(cp)
            #     input("penter")
            # print(cps[0][6])
            if len(cps) == 0: good_init = True   # TODO: init hand last and make sure it does not colllide with env

    # def reset_to_q(self, save_robot_q, needCorrection=False):       # TODO
    #     # assume a certain ordering
    #     initBasePos = save_robot_q[:3]
    #     initEuler = save_robot_q[3:6]
    #     initQuat = p.getQuaternionFromEuler(list(initEuler))
    #     localpos = [0.0, 0.0, 0.035]
    #     localquat = [0.0, 0.0, 0.0, 1.0]
    #     if needCorrection:
    #         initBasePos, initQuat= p.multiplyTransforms(initBasePos, initQuat, localpos, localquat)
    #
    #     # initQuat = p.getQuaternionFromEuler(list(initEuler))
    #     initBaseLinVel = save_robot_q[6:9]
    #     initBaseAugVel = save_robot_q[9:12]
    #
    #     nDof = len(self.activeDofs + self.lockDofs)
    #     assert len(save_robot_q) == (12+nDof)     # TODO: assume finger only q but not dq
    #     initActivePos = save_robot_q[12:12+len(self.activeDofs)]
    #     initLockPos = save_robot_q[12+len(self.activeDofs):12+nDof]
    #
    #     p.removeConstraint(self.cid)
    #     p.resetBasePositionAndOrientation(self.handId, initBasePos, initQuat)
    #     p.resetBaseVelocity(self.handId, initBaseLinVel, initBaseAugVel)
    #
    #     for ind in range(len(self.activeDofs)):
    #         p.resetJointState(self.handId, self.activeDofs[ind], initActivePos[ind], 0.0)
    #     for ind in range(len(self.lockDofs)):
    #         p.resetJointState(self.handId, self.lockDofs[ind], initLockPos[ind], 0.0)
    #
    #     self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
    #                                   childFramePosition=initBasePos,
    #                                   childFrameOrientation=initQuat)
    #     # TODO: no vel here, is this correct? Think
    #
    #     p.stepSimulation()  # TODO
    #
    #     basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
    #     self.baseInitPos = np.array(basePos)
    #     self.baseInitEuler = np.array(p.getEulerFromQuaternion(baseQuat))
    #     self.tarBasePos = np.copy(self.baseInitPos)
    #     self.tarBaseEuler = np.copy(self.baseInitEuler)
    #     self.tarFingerPos = np.copy(initActivePos)

    def get_fingers_q_dq(self):
        dofs = self.activeDofs
        joints_state = p.getJointStates(self.handId, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_wrist_q_dq(self):
        dofs = range(6)
        joints_state = p.getJointStates(self.handId, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_palm_pos_orn(self):
        newPos = p.getLinkState(self.handId, self.endEffectorId, computeForwardKinematics=1)[0]
        newOrn = p.getLinkState(self.handId, self.endEffectorId, computeForwardKinematics=1)[1]
        return newPos, newOrn

    def get_palm_vel(self):
        newLinVel = p.getLinkState(self.handId, self.endEffectorId, computeLinkVelocity=1, computeForwardKinematics=1)[6]
        newAngVel = p.getLinkState(self.handId, self.endEffectorId, computeLinkVelocity=1, computeForwardKinematics=1)[7]
        return newLinVel, newAngVel

    def get_wrist_last_torque(self):
        joints_state = p.getJointStates(self.handId, range(6))
        joints_taus = np.array(joints_state)[:, [3]]
        joints_taus = np.hstack(joints_taus.flatten())
        return joints_taus

    def get_fingers_last_torque(self):
        joints_state = p.getJointStates(self.handId, self.activeDofs)
        joints_taus = np.array(joints_state)[:, [3]]
        joints_taus = np.hstack(joints_taus.flatten())
        return joints_taus

    def get_robot_observation(self):
        obs = []

        # TODO: no palm/finger vel
        basePos, baseQuat = self.get_palm_pos_orn()
        obs.extend(basePos)
        obs.extend(baseQuat)

        fq, fdq = self.get_fingers_q_dq()
        obs.extend(list(fq))

        # baseVels = p.getBaseVelocity(self.handId)
        # obs.extend(baseVels[0])
        # obs.extend(baseVels[1])

        obs.extend(list(self.act * 240))
        # d_quat / dt = 0.5 * w * q, make a bit easier for policy to understand w
        w_tar = self.act[3:6]
        w_tar = list(w_tar) + [0]
        _, d_quat = p.multiplyTransforms([0,0,0], w_tar, [0,0,0], baseQuat)
        obs.extend(list(d_quat))

        if self.include_redun_body_pos:
            for i in range(6, p.getNumJoints(self.handId)):
                pos = p.getLinkState(self.handId, i)[0]  # [0] stores xyz position
                obs.extend(pos)

        return obs

    def get_robot_observation_dim(self):
        return len(self.get_robot_observation())

    def get_tar_dq_from_tar_vel(self, vel):
        wq, _ = self.get_wrist_q_dq()
        [jac_t, jac_r] = p.calculateJacobian(self.handId, self.endEffectorId, [0] * 3,
                                             list(wq)+list(self.get_fingers_q_dq()[0]),
                                             [0.] * self.n_dofs, [0.] * self.n_dofs)
        jac = np.array([jac_t[0][:6], jac_t[1][:6], jac_t[2][:6],
                        jac_r[0][:6], jac_r[1][:6], jac_r[2][:6]])
        # vel = jac * dq, vel is 6D
        tar_dq, residue, _, _ = np.linalg.lstsq(jac, vel, 1e-4)
        if np.abs(residue) > 0.01: print("warning!!Jac inv")
        return tar_dq

    def apply_action(self, a):

        # TODO: a is already scaled, how much to scale? decide in Env.

        self.act = np.array(a)

        root_v = self.act[:6] * 240    # TODO: timestep 240 Hz

        tar_w_dq = self.get_tar_dq_from_tar_vel(root_v)
        if self.act_noise:
            tar_w_dq += self.np_random.uniform(low=-0.05, high=0.05, size=len(tar_w_dq))

        p.setJointMotorControlArray(
            bodyIndex=self.handId,
            jointIndices=range(6),      # 0:6
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=list(tar_w_dq),
            forces=[10000.0]*6)        # TODO: wrist force limit?

        f_v = self.act[6:] * 240
        if self.act_noise:
            f_v += self.np_random.uniform(low=-0.1, high=0.1, size=len(f_v))

        p.setJointMotorControlArray(
            bodyIndex=self.handId,
            jointIndices=self.activeDofs,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=f_v,
            forces=[1000.0]*len(f_v))


if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)    #or p.DIRECT for non-graphical version

    p.setTimeStep(1. / 240.)

    # p.setPhysicsEngineParameter(numSolverIterations=10000000, solverResidualThreshold=1e-8)

    aaa, seed = gym.utils.seeding.np_random(101)
    np.random.seed(101)

    for i in range(100):
        p.resetSimulation()
        p.setGravity(0, 0, -10.)

        floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
        p.changeDynamics(floorId, -1, lateralFriction=2.0, spinningFriction=1.0)
        # cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), [0,0,0.2], useFixedBase=0)
        # p.changeDynamics(cylinderId, -1, lateralFriction=2.0, spinningFriction=1.0)

        a = ShadowHandVel()
        a.np_random = aaa
        a.reset()

        input("press enter to continue")
        # print("init", a.get_robot_observation())
        for t in range(400):
            # a.apply_action(np.random.uniform(low=-0.005, high=0.005, size=6+22)+np.array([0.0025]*6+[0.01]*22))

            act = np.random.uniform(low=-0.001, high=0.001, size=6 + 17) + np.array([0.002] * 3 + [0.003]*3 + [-0.005] * 17)
            # act = np.random.uniform(low=-0.001, high=0.001, size=6+17) + np.array([-0.002] * 3 + [-0.003]*3 + [0.005]*17)
            # act = np.array([0.00]*23)
            a.apply_action(act)

            # a.apply_action(np.array([0.0]*6+[-0.01]*17))
            # a.apply_action(np.array([0.005] * (22+6)))

            p.stepSimulation()

            # n_dofs = 6
            # q, dq = a.get_wrist_q_dq()
            # # print(q)
            # # # q = [0.28414081, 0.27057848, 0.27993475, 0.27265728, 0.29331565, 0.29105865]
            # # # q = [0., 0, 0, 0.27265728, 0.29331565, 0.29105865]
            # [jac_t, jac_r] = p.calculateJacobian(a.handId, a.endEffectorId, [0] * 3, list(q), [0.] * n_dofs,
            #                                      [0.] * n_dofs)
            # print(jac_r)

            print("t", t)
            # print("tar_vel", act[:6]*240)
            # # pos, orn = a.get_palm_pos_orn()
            # # print("palm pos/orn", pos, p.getEulerFromQuaternion(orn))
            # print("palm vel", a.get_palm_vel())
            # print("palm joint vel", a.get_wrist_q_dq()[1])
            # print("tar_fin_vel", act[6:]*240)
            # # print("fin_q", a.get_fingers_q_dq()[0])
            # print("fin_vel", a.get_fingers_q_dq()[1])
            #
            # print("last wrist torque", a.get_wrist_last_torque())
            # print("last finger torque", a.get_fingers_last_torque())

            print(a.get_robot_observation())

            time.sleep(1./240.)
        print("final obz", a.get_robot_observation())

    p.disconnect()