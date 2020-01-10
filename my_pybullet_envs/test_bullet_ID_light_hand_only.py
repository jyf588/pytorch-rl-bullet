import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data

import time
import numpy as np

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class ShadowHand:
    def __init__(self,
                 torque_control=False):
        self.base_init_pos = np.array([0, 0, 0])
        self.base_init_euler = np.array([0,0,0])
        self.init_fin_pos = [0.1, 0.2, 0.3] + [0.35, 0.25, 0.15] + [0.6, 0.2, 0.4] + [0.3, 0.5, 0.4] + \
                            [0.0, 1.0, 0.1, 0.5, 0.2]

        self.handId = p.loadURDF(os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadow_motor_noaux.urdf"),
                                 list(self.base_init_pos), p.getQuaternionFromEuler(list(self.base_init_euler)), useFixedBase=1)
        # TODO: inertia !!

        nDof = p.getNumJoints(self.handId)
        for i in range(p.getNumJoints(self.handId)):
            print(p.getJointInfo(self.handId, i)[2], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])

        #

        for i in range(-1, p.getNumJoints(self.handId)):
            p.changeDynamics(self.handId, i, jointDamping=0.0, linearDamping=0.0, angularDamping=0.0)

        # 4,4,4,5,5
        self.activeDofs = []
        self.lockDofs = []
        start = 0
        for i in range(5):
            nDofs = [4,4,4,5,5]
            fig_start = [1,1,1,2,0]
            self.activeDofs += (np.arange(fig_start[i], nDofs[i]) + start).tolist()
            self.lockDofs += (np.arange(0, fig_start[i]) + start).tolist()
            start += (nDofs[i]+1)
        print(self.activeDofs)
        print(self.lockDofs)
        assert len(self.activeDofs) == len(self.init_fin_pos)

        for ind in range(len(self.activeDofs)):
            p.resetJointState(self.handId, self.activeDofs[ind], self.init_fin_pos[ind], 0.1)
        for ind in range(len(self.lockDofs)):
            p.resetJointState(self.handId, self.lockDofs[ind], 0.1, -0.1)

        for i in range(-1, p.getNumJoints(self.handId)):
            dyn = p.getDynamicsInfo(self.handId, i)
            mass = dyn[0]
            mass = mass * 10.
            lid = dyn[2]
            lid = (lid[0] * 10., lid[1] * 10., lid[2] * 10.,)       # TODO
            # total_m += mass
            p.changeDynamics(self.handId, i, mass=mass)
            p.changeDynamics(self.handId, i, localInertiaDiagonal=lid)
            p.changeDynamics(self.handId, i, lateralFriction=3.0)
            p.changeDynamics(self.handId, i, jointDamping=0.0)      # we use vel control anyways


        # if torque_control:
        #     # activating torque control
        #     sim.setJointMotorControlArray(
        #         bodyIndex=self.arm_id,
        #         jointIndices=self.arm_dofs,
        #         controlMode=p.VELOCITY_CONTROL,
        #         forces=np.zeros(len(self.arm_dofs)))

        # p.stepSimulation()
        input("press enter")

    # def reset(self, q, dq):
    #     for ind, i in enumerate(self.arm_dofs):
    #         p.resetJointState(self.arm_id, i, q[ind], 0.0)  # TODO: dq
    #
    #     # self.init_fin_pos = [0.4, 0.4, 0.4] * 4 + [0.0, 1.0, 0.1, 0.5, 0.2]
    #     # self.init_fin_pos += np.random.uniform(low=-0.05, high=0.05, size=len(self.init_fin_pos))
    #     self.init_fin_pos = [0.1, 0.2, 0.3] + [0.35, 0.25, 0.15] + [0.6, 0.2, 0.4] + [0.3, 0.5, 0.4] + [0.0, 1.0, 0.1, 0.5,
    #                                                                                                0.2]
    #
    #     # self.init_fin_pos = [0.4, 0.4, 0.4] * 3
    #     for ind in range(len(self.fin_actdofs)):
    #         p.resetJointState(self.arm_id, self.fin_actdofs[ind], self.init_fin_pos[ind], 0.0)
    #
    #     cps = p.getContactPoints(bodyA=self.arm_id)
    #     print(cps)

    def get_fingers_q_dq(self):
        dofs = np.sort(self.activeDofs + self.lockDofs).tolist()
        joints_state = p.getJointStates(self.handId, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_fingers_last_torque(self):
        dofs = np.sort(self.activeDofs + self.lockDofs).tolist()
        joints_state = p.getJointStates(self.handId, dofs)
        joints_taus = np.array(joints_state)[:, [3]]
        joints_taus = np.hstack(joints_taus.flatten())
        return joints_taus


hz = 240.0
dt = 1.0 / hz

sim = bc.BulletClient(connection_mode=p.DIRECT)
sim.resetSimulation()
p.setPhysicsEngineParameter(numSolverIterations=200)

sim.setGravity(0, 0, -10)
sim.setTimeStep(dt)
sim.setRealTimeSimulation(0)

hand = ShadowHand()

# arm.reset(np.zeros(7), dq0)

q0, dq0 = hand.get_fingers_q_dq()
print("q0", q0)
print("dq0", dq0)


# sim.applyExternalForce(arm.arm_id, arm.ee_id+1, [-100, 250, -500], [0, 0, 0], p.LINK_FRAME)
# sim.applyExternalTorque(arm.arm_id, arm.ee_id+1, [-20, 10, 30], p.WORLD_FRAME)

# sim.setJointMotorControlArray(
#     bodyIndex=arm.arm_id,
#     jointIndices=[24,25,26,27,28],
#     controlMode=p.TORQUE_CONTROL,
#     forces=[1000, 1000, 1000, 1000, 1000])

sim.setJointMotorControlArray(
            bodyIndex=hand.handId,
            jointIndices=hand.activeDofs,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=np.array(range(17))*0.1,
            forces=[1000.]*len(hand.activeDofs))
# zero dofs are automatically solved to zero,

# taus = [ -9100.24903603, -19530.85978572,  -4457.90000734, -19852.31575715,
#   -2652.5586482 ,  14634.90149399  , 4303.70562124]
#
# sim.setJointMotorControlArray(
#     bodyIndex=arm.arm_id,
#     jointIndices=arm.arm_dofs,
#     controlMode=p.TORQUE_CONTROL,
#     forces=taus)

# sim.setJointMotorControlArray(
#             bodyIndex=arm.arm_id,
#             jointIndices=arm.fin_actdofs,
#             controlMode=p.VELOCITY_CONTROL,
#             targetVelocities=np.array(range(17))*0.1,
#             forces=[1000000.]*len(arm.fin_actdofs))

# sim.setJointMotorControlArray(
#             bodyIndex=arm.arm_id,
#             jointIndices=arm.fin_actdofs,
#             controlMode=p.POSITION_CONTROL,
#             targetPositions=list(np.array(arm.init_fin_pos) + np.array(range(17))*0.1/240),
#             targetVelocities=list(np.zeros(17)),
#             positionGains=list(np.ones(17)*0.1),
#             velocityGains=list(np.zeros(17)),
#             forces=[1000000.]*len(arm.fin_actdofs))

# TODO: what does it mean to acc/de-acc (match next vel) a zero-mass link?

sim.stepSimulation()

q1, dq1 = hand.get_fingers_q_dq()
print("q1", q1)
print("dq1", dq1)
print("tau_velc", hand.get_fingers_last_torque())

acc = (np.array([0,0,1,2, 0,3,4,5, 0,6,7,8, 0,0,9,10,11, 12,13,14,15,16])*0.1 - dq0)*240
print("tardq1", dq0+acc/240)
tau_c = sim.calculateInverseDynamics(hand.handId,
                                     list(q0),
                                     list(dq0),
                                     list(acc))

# tau_c = sim.calculateInverseDynamics(hand.handId,
#                                      list([0.0]*22),
#                                      list([0.0]*22),
#                                      list([0.0]*22))
print("tau_c", tau_c)

# # a = np.random.uniform(-10.0, 10.0, 24)
# # a = np.ones(24)
# # M = p.calculateMassMatrix(arm.arm_id, list(q0)+list(arm.init_fin_pos))
# # M = np.array(M).reshape((24,24))
# # print(M.dot(a.T))
#
#
# # TODO: Note, the solved tau is what I want, tau_id = M(ddq) - C - tau - J^Tf
#
# print(arm.get_arm_last_torque())
# print(arm.get_fingers_last_torque())
# print(p.getJointState(arm.arm_id, 29))
#
# input("press enter")





