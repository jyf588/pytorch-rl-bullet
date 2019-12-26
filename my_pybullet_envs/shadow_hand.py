import pybullet as p
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# TODO: render

class ShadowHand:
    def __init__(self,
                 base_init_pos=np.array([-0.18, 0.105, 0.13]),      # 0.035 offset from old hand
                 base_init_euler=np.array([1.8, -1.57, 0]),
                 init_fin_pos = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0]),
                 base_ll = np.array([-0.1, -0.1, -0.1]),   # default range for reaching
                 base_ul = np.array([0.3, 0.1, 0.1])):      # default range for reaching

        self.baseInitPos = base_init_pos
        self.baseInitEuler = base_init_euler
        self.initPos = init_fin_pos
        self.base_ll = self.baseInitPos + base_ll
        self.base_ul = self.baseInitPos + base_ul
        self.ori_ll = self.baseInitEuler - 0.7        # TODO: is this right?
        self.ori_ul = self.baseInitEuler + 0.7

        self.handId = p.loadURDF(os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadowhand_motor.urdf"),
                                 list(self.baseInitPos), p.getQuaternionFromEuler(list(self.baseInitEuler)),
                                 flags=p.URDF_USE_SELF_COLLISION)
        # self.handId = p.loadURDF(os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadowhand_motor.urdf"),
        #                          list(self.baseInitPos), p.getQuaternionFromEuler(list(self.baseInitEuler)))
        # nDof = p.getNumJoints(self.handId)
        # for i in range(p.getNumJoints(self.handId)):
        #     print(p.getJointInfo(self.handId, i)[0:3], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])
        # print(p.getLinkState(self.handId, 0)[0])
        # input("press enter 0")

        # 4,4,4,5,5
        self.activeDofs = []
        self.lockDofs = []
        start = 1   # add palm aux
        for i in range(5):
            nDofs = [4,4,4,5,5]
            fig_start = [1,1,1,2,0] # default 0,0,0,0,0 to include all finger dofs
            self.activeDofs += (np.arange(fig_start[i], nDofs[i]) + start).tolist()
            self.lockDofs += (np.arange(0, fig_start[i]) + start).tolist()
            start += (nDofs[i]+1)
        # print(self.activeDofs)
        # print(self.lockDofs)
        assert len(self.activeDofs) == len(init_fin_pos)

        self.ll = np.array([p.getJointInfo(self.handId, i)[8] for i in self.activeDofs])
        self.ul = np.array([p.getJointInfo(self.handId, i)[9] for i in self.activeDofs])

        for ind in range(len(self.activeDofs)):
            p.resetJointState(self.handId, self.activeDofs[ind], self.initPos[ind], 0.0)
        for ind in range(len(self.lockDofs)):
            p.resetJointState(self.handId, self.lockDofs[ind], 0.0, 0.0)

        total_m = 0
        for i in range(0, p.getNumJoints(self.handId)):       # add palm aux
            p.changeDynamics(self.handId, i, lateralFriction=3.0)
            # TODO: increase mass for now
            mass = p.getDynamicsInfo(self.handId, i)[0]
            mass = mass * 10.
            total_m += mass
            p.changeDynamics(self.handId, i, mass=mass)
            # p.setJointMotorControl2(self.handId, i, p.VELOCITY_CONTROL, force=0.000)

        # print("total hand Mass:", total_m)
        # # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/constraint.py#L11
        self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      childFramePosition=list(self.baseInitPos),
                                      childFrameOrientation=p.getQuaternionFromEuler(list(self.baseInitEuler)))

        self.tarBasePos = np.copy(self.baseInitPos)
        self.tarBaseEuler = np.copy(self.baseInitEuler)     # euler angles
        self.tarFingerPos = np.copy(self.initPos)    # used for position control and as part of state

        self.maxForce = 200.
        # self.maxForce = 0.

        self.include_redun_body_pos = False

        self.np_random = None   # seeding inited outside in Env

        # p.enableJointForceTorqueSensor(self.handId, 0, True)

        # print(self.tarFingerPos)
        # print(self.ll)
        # print(self.ul)
        assert len(self.tarFingerPos) == len(self.ll)

    def reset(self):
        # TODO: bullet env reload urdfs in reset...
        # TODO: bullet env reset pos with added noise but velocity to zero always.

        goodInit = False
        while not goodInit:
            # initBasePos = self.baseInitPos
            # initEuler = self.baseInitEuler
            initBasePos = np.array(self.baseInitPos)
            initBasePos[0] += self.np_random.uniform(low=-0.02, high=0.02)
            initBasePos[1] += self.np_random.uniform(low=-0.02, high=0.02)
            initBasePos[2] += self.np_random.uniform(low=-0.02, high=0.02)
            initEuler = np.array(self.baseInitEuler) + self.np_random.uniform(low=-0.05, high=0.05, size=3)
            initQuat = p.getQuaternionFromEuler(list(initEuler))
            #
            # print(p.getEulerFromQuaternion(initQuat))

            # init self.np_random outside, in Env
            # initPos = self.initPos
            initPos = self.initPos + self.np_random.uniform(low=-0.05, high=0.05, size=len(self.initPos))

            p.removeConstraint(self.cid)
            p.resetBasePositionAndOrientation(self.handId, initBasePos, initQuat)

            for ind in range(len(self.activeDofs)):
                p.resetJointState(self.handId, self.activeDofs[ind], initPos[ind], 0.0)
            for ind in range(len(self.lockDofs)):
                p.resetJointState(self.handId, self.lockDofs[ind], 0.0, 0.0)

            self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                          childFramePosition=initBasePos,
                                          childFrameOrientation=initQuat)

            p.stepSimulation()  # TODO

            cps = p.getContactPoints(bodyA=self.handId)
            # for cp in cps:
            #     print(cp)
            #     input("penter")
            # print(cps[0][6])
            if len(cps) == 0: goodInit = True   # TODO: init hand last and make sure it does not colllide with env

            self.tarBasePos = np.copy(initBasePos)
            self.tarBaseEuler = np.copy(initEuler)
            self.tarFingerPos = np.copy(initPos)

    def reset_to_q(self, save_robot_q, needCorrection=False):
        # assume a certain ordering
        initBasePos = save_robot_q[:3]
        initEuler = save_robot_q[3:6]
        initQuat = p.getQuaternionFromEuler(list(initEuler))
        localpos = [0.0, 0.0, 0.035]
        localquat = [0.0, 0.0, 0.0, 1.0]
        if needCorrection:
            initBasePos, initQuat= p.multiplyTransforms(initBasePos, initQuat, localpos, localquat)

        # initQuat = p.getQuaternionFromEuler(list(initEuler))
        initBaseLinVel = save_robot_q[6:9]
        initBaseAugVel = save_robot_q[9:12]

        nDof = len(self.activeDofs + self.lockDofs)
        assert len(save_robot_q) == (12+nDof)     # TODO: assume finger only q but not dq
        initActivePos = save_robot_q[12:12+len(self.activeDofs)]
        initLockPos = save_robot_q[12+len(self.activeDofs):12+nDof]

        p.removeConstraint(self.cid)
        p.resetBasePositionAndOrientation(self.handId, initBasePos, initQuat)
        p.resetBaseVelocity(self.handId, initBaseLinVel, initBaseAugVel)

        for ind in range(len(self.activeDofs)):
            p.resetJointState(self.handId, self.activeDofs[ind], initActivePos[ind], 0.0)
        for ind in range(len(self.lockDofs)):
            p.resetJointState(self.handId, self.lockDofs[ind], initLockPos[ind], 0.0)

        self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      childFramePosition=initBasePos,
                                      childFrameOrientation=initQuat)
        # TODO: no vel here, is this correct? Think

        p.stepSimulation()  # TODO

        basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
        self.baseInitPos = np.array(basePos)
        self.baseInitEuler = np.array(p.getEulerFromQuaternion(baseQuat))
        self.tarBasePos = np.copy(self.baseInitPos)
        self.tarBaseEuler = np.copy(self.baseInitEuler)
        self.tarFingerPos = np.copy(initActivePos)

    def get_raw_state_fingers(self, includeVel=True):
        dofs = self.activeDofs + self.lockDofs
        joints_state = p.getJointStates(self.handId, dofs)
        if includeVel:
            joints_state = np.array(joints_state)[:,[0,1]]
        else:
            joints_state = np.array(joints_state)[:, [0]]
        return np.hstack(joints_state.flatten())

    def get_robot_observation(self):
        obs = []

        basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
        obs.extend(basePos)
        obs.extend(baseQuat)
        #
        # print("pos", basePos)
        # print("euler", p.getEulerFromQuaternion(baseQuat))

        obs.extend(list(self.get_raw_state_fingers(includeVel=False)))
        # print(self.get_raw_state_fingers())

        # TODO: no finger vel

        baseVels = p.getBaseVelocity(self.handId)
        obs.extend(baseVels[0])
        obs.extend(baseVels[1])
        #
        # print("linvel", baseVels[0])
        # print("angvel", baseVels[1])

        obs.extend(list(self.tarFingerPos))
        obs.extend(list(self.tarBasePos))
        tarQuat = p.getQuaternionFromEuler(list(self.tarBaseEuler))
        obs.extend(tarQuat)

        if self.include_redun_body_pos:
            for i in range(p.getNumJoints(self.handId)):
                pos = p.getLinkState(self.handId, i)[0]  # [0] stores xyz position
                obs.extend(pos)

        return obs

    def get_robot_observation_dim(self):
        return len(self.get_robot_observation())

    def get_finger_dist_from_init(self):
        return np.linalg.norm(self.get_raw_state_fingers(includeVel=False) - self.initPos)

    # def get_three_finger_deviation(self):
    #     fingers_q = self.get_raw_state_fingers(includeVel=False)
    #     assert len(fingers_q) == 16     # TODO
    #     f1 = fingers_q[:4]
    #     f2 = fingers_q[4:8]
    #     f3 = fingers_q[8:12]
    #     # TODO: is this different from dist to mean
    #     return np.linalg.norm(f1-f2) + np.linalg.norm(f2-f3) + np.linalg.norm(f1-f3)

    def apply_action(self, a):

        # print("action", a)
        # TODO: should encourage same q for first 3 fingers for now

        # TODO: a is already scaled, how much to scale? decide in Env.
        # should be delta control (policy outputs delta position), but better add to old tar pos instead of cur pos
        # TODO: but tar pos should be part of state vector (? how accurate is pos_con?)

        a = np.array(a)

        dxyz = a[0:3]
        dOri = a[3:6]
        self.tarBasePos += dxyz
        self.tarBasePos = np.clip(self.tarBasePos, self.base_ll, self.base_ul)

        self.tarBaseEuler += dOri
        self.tarBaseEuler = np.clip(self.tarBaseEuler, self.ori_ll, self.ori_ul)

        # print(self.tarBasePos)
        # print(p.getBasePositionAndOrientation(self.handId))
        tarQuat = p.getQuaternionFromEuler(list(self.tarBaseEuler))

        p.changeConstraint(self.cid, list(self.tarBasePos), tarQuat, maxForce=self.maxForce * 10.0)   # TODO: wrist force larger

        # # tmp, apply ext forces here
        # p.removeConstraint(self.cid)
        #
        # # force = [1305.2429943942445, 814.8305070256084, -88.75239210661124]
        # # tor = [111.09001369756704, -5.01105164325144, -8.617024915673836]
        # # base_pos, base_quat = p.getBasePositionAndOrientation(self.handId)
        # # inv_base_pos, inv_base_quat = p.invertTransform(base_pos, base_quat)
        # # l_force = p.multiplyTransforms([0,0,0], inv_base_quat, force, [0,0,0,1])
        # # print("l_force", l_force)
        #
        # # tor = [111.09001369756704, -5.01105164325144, -8.617024915673836]
        # # root_pos, _ = p.getBasePositionAndOrientation(self.handId)
        # # p.applyExternalForce(self.handId, -1, force, root_pos, flags=p.WORLD_FRAME)
        # # p.applyExternalTorque(self.handId, -1, tor, flags=p.WORLD_FRAME)
        #
        # force = [-87.69514789251878, -1456.2983454130156, -496.959669106391]
        # # force = [10.106244179329314, -939.3308505536016, -297.4481669327156]
        # # tor = [60.119571606556605, -1.9417214666191618, -8.61702491336549]
        # tor = [111.09001369756704, -5.01105164325144, -8.617024915673836]
        # # tor = [40.38932609461888, -2.2311416025337847, -4.656959654796246]
        # p.applyExternalForce(self.handId, -1, force, [0,0,0], flags=p.LINK_FRAME)
        # p.applyExternalTorque(self.handId, -1, tor, flags=p.WORLD_FRAME)
        # print("aaaaaaaaa")

        self.tarFingerPos += a[6:]      # length should match
        self.tarFingerPos = np.clip(self.tarFingerPos, self.ll, self.ul)

        for i in range(len(self.activeDofs)):
            p.setJointMotorControl2(self.handId,
                                    jointIndex=self.activeDofs[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.tarFingerPos[i],
                                    force=self.maxForce)
        # for i in range(len(self.lockDofs)):
        #     p.setJointMotorControl2(self.handId,
        #                             jointIndex=self.activeDofs[i],
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPosition=0.0,
        #                             force=self.maxForce / 4.0)

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)    #or p.DIRECT for non-graphical version

    p.setTimeStep(1./240.)
    p.setGravity(0, 0, -10)

    floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
    p.changeDynamics(floorId, -1, lateralFriction=3.0)

    a = ShadowHand()
    a.np_random, seed = gym.utils.seeding.np_random(0)

    for i in range(100):
        np.random.seed(0)
        a.reset()

        input("press enter to continue")
        print("init", a.get_robot_observation())
        for t in range(800):
            # a.apply_action(np.random.uniform(low=-0.005, high=0.005, size=6+22)+np.array([0.0025]*6+[0.01]*22))
            a.apply_action(
                np.random.uniform(low=-0.001, high=0.001, size=6 + 17) + np.array([0.002] * 3 + [0.002]*3 + [-0.01] * 17))
            # a.apply_action(np.array([0.0]*6+[0.01]*22))
            # a.apply_action(np.array([0.005] * (22+6)))

            p.stepSimulation()
            # print(p.getConstraintState(a.cid)[:3])
            act_palm_com, _ = p.getBasePositionAndOrientation(a.handId)
            print("after ts", np.array(a.tarBasePos) - np.array(act_palm_com))
            # a.apply_action(np.array([0.0] * 23))  # equivalent
            # p.stepSimulation()
            # print(p.getConstraintState(a.cid)[:3])
            time.sleep(1./240.)
        print("final obz", a.get_robot_observation())

    p.disconnect()

#
# import pybullet as p
# import time
# import gym, gym.utils.seeding
# import numpy as np
# import math
#
# import os
# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#
# # TODO: mass good or not?
# # TODO: render
#
# class ShadowHand:
#     def __init__(self,
#                  base_init_pos=np.array([-0.17, 0.07, 0.1]),
#                  base_init_euler=np.array([1.8, -1.57, 0]),
#                  init_fin_pos = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0]),
#                  base_ll = np.array([-0.1, -0.1, -0.1]),   # default range for reaching
#                  base_ul = np.array([0.3, 0.1, 0.1])):      # default range for reaching
#
#         self.baseInitPos = base_init_pos
#         self.baseInitEuler = base_init_euler
#         self.initPos = init_fin_pos
#         self.base_ll = self.baseInitPos + base_ll
#         self.base_ul = self.baseInitPos + base_ul
#         self.ori_ll = self.baseInitEuler - 1.57        # TODO: is this right?
#         self.ori_ul = self.baseInitEuler + 1.57
#
#         self.handId = p.loadURDF(os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadowhand_motor.urdf"),
#                                  list(self.baseInitPos), p.getQuaternionFromEuler(list(self.baseInitEuler)),
#                                  flags=p.URDF_USE_SELF_COLLISION)
#         # nDof = p.getNumJoints(self.handId)
#         # for i in range(p.getNumJoints(self.handId)):
#         #     print(p.getJointInfo(self.handId, i)[0:3], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])
#
#         # 4,4,4,5,5
#         self.activeDofs = []
#         self.lockDofs = []
#         start = 0
#         for i in range(5):
#             nDofs = [4,4,4,5,5]
#             fig_start = [1,1,1,2,0] # default 0,0,0,0,0 to include all finger dofs
#             self.activeDofs += (np.arange(fig_start[i], nDofs[i]) + start).tolist()
#             self.lockDofs += (np.arange(0, fig_start[i]) + start).tolist()
#             start += (nDofs[i]+1)
#         # print(self.activeDofs)
#         # print(self.lockDofs)
#         assert len(self.activeDofs) == len(init_fin_pos)
#
#         self.ll = np.array([p.getJointInfo(self.handId, i)[8] for i in self.activeDofs])
#         self.ul = np.array([p.getJointInfo(self.handId, i)[9] for i in self.activeDofs])
#
#         for ind in range(len(self.activeDofs)):
#             p.resetJointState(self.handId, self.activeDofs[ind], self.initPos[ind], 0.0)
#         for ind in range(len(self.lockDofs)):
#             p.resetJointState(self.handId, self.lockDofs[ind], 0.0, 0.0)
#
#         total_m = 0
#         for i in range(-1, p.getNumJoints(self.handId)):
#             p.changeDynamics(self.handId, i, lateralFriction=3.0)
#             # TODO: increase mass for now
#             mass = p.getDynamicsInfo(self.handId, i)[0]
#             # inertia = p.getDynamicsInfo(self.handId, i)[2]
#             mass = mass * 10.
#             total_m += mass
#             # inertia = [ele * 100. for ele in inertia]
#             p.changeDynamics(self.handId, i, mass=mass)
#             # p.changeDynamics(self.handId, i, localInertiaDiagnoal=inertia)    # TODO: default inertia from bullet
#
#         print("total hand Mass:", total_m)
#         # # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/constraint.py#L11
#         self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
#                                       childFramePosition=list(self.baseInitPos),
#                                       childFrameOrientation=p.getQuaternionFromEuler(list(self.baseInitEuler)))
#
#         self.tarBasePos = np.copy(self.baseInitPos)
#         self.tarBaseEuler = np.copy(self.baseInitEuler)     # euler angles
#         self.tarFingerPos = np.copy(self.initPos)    # used for position control and as part of state
#
#         self.maxForce = 200.
#         # self.maxForce = 0.
#
#         self.include_redun_body_pos = False
#
#         self.np_random = None   # seeding inited outside in Env
#
#         # print(self.tarFingerPos)
#         # print(self.ll)
#         # print(self.ul)
#         assert len(self.tarFingerPos) == len(self.ll)
#
#     def reset(self):
#         # TODO: bullet env reload urdfs in reset...
#         # TODO: bullet env reset pos with added noise but velocity to zero always.
#
#         goodInit = False
#         while not goodInit:
#             initBasePos = self.baseInitPos
#             initEuler = self.baseInitEuler
#             # initBasePos = np.array(self.baseInitPos)
#             # initBasePos[0] += self.np_random.uniform(low=-0.015, high=0.015)
#             # initBasePos[1] += self.np_random.uniform(low=-0.015, high=0.015)
#             # initBasePos[2] += self.np_random.uniform(low=-0.015, high=0.015)  # enlarge here
#             # initEuler = np.array(self.baseInitEuler) + self.np_random.uniform(low=-0.05, high=0.05, size=3)
#             initQuat = p.getQuaternionFromEuler(list(initEuler))
#             #
#             # print(p.getEulerFromQuaternion(initQuat))
#
#             # init self.np_random outside, in Env
#             initPos = self.initPos
#             # initPos = self.initPos + self.np_random.uniform(low=-0.05, high=0.05, size=len(self.initPos))
#
#             p.removeConstraint(self.cid)
#             p.resetBasePositionAndOrientation(self.handId, initBasePos, initQuat)
#
#             for ind in range(len(self.activeDofs)):
#                 p.resetJointState(self.handId, self.activeDofs[ind], initPos[ind], 0.0)
#             for ind in range(len(self.lockDofs)):
#                 p.resetJointState(self.handId, self.lockDofs[ind], 0.0, 0.0)
#
#             self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
#                                           childFramePosition=initBasePos,
#                                           childFrameOrientation=initQuat)
#
#             # print(p.getNumJoints(self.handId))
#             # for i in range(p.getNumJoints(self.handId)):
#             #     print(p.getJointState(self.handId, i)[0])
#
#             p.stepSimulation()  # TODO
#
#             cps = p.getContactPoints(bodyA=self.handId)
#             # for cp in cps:
#             #     print(cp)
#             #     input("penter")
#             # print(cps[0][6])
#             if len(cps) == 0: goodInit = True   # TODO: init hand last and make sure it does not colllide with env
#
#             # print("aaa")
#             # basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
#             # print(basePos)
#             # print(p.getEulerFromQuaternion(baseQuat))
#             # print(p.getEulerFromQuaternion(p.getQuaternionFromEuler([1.8, -1.57, 0])))
#             self.tarBasePos = np.copy(initBasePos)
#             self.tarBaseEuler = np.copy(initEuler)
#             self.tarFingerPos = np.copy(initPos)
#
#     def reset_to_q(self, save_robot_q, needCorrection=True):
#         # assume a certain ordering
#         initBasePos = save_robot_q[:3]
#         initEuler = save_robot_q[3:6]
#         initQuat = p.getQuaternionFromEuler(list(initEuler))
#         localpos = [0.0, 0.0, 0.035]
#         localquat = [0.0, 0.0, 0.0, 1.0]
#         if needCorrection:
#             initBasePos, initQuat= p.multiplyTransforms(initBasePos, initQuat, localpos, localquat)
#
#         # initQuat = p.getQuaternionFromEuler(list(initEuler))
#         initBaseLinVel = save_robot_q[6:9]
#         initBaseAugVel = save_robot_q[9:12]
#
#         nDof = len(self.activeDofs + self.lockDofs)
#         assert len(save_robot_q) == (12+nDof)     # TODO: assume finger only q but not dq
#         initActivePos = save_robot_q[12:12+len(self.activeDofs)]
#         initLockPos = save_robot_q[12+len(self.activeDofs):12+nDof]
#
#         p.removeConstraint(self.cid)
#         p.resetBasePositionAndOrientation(self.handId, initBasePos, initQuat)
#         p.resetBaseVelocity(self.handId, initBaseLinVel, initBaseAugVel)
#
#         for ind in range(len(self.activeDofs)):
#             p.resetJointState(self.handId, self.activeDofs[ind], initActivePos[ind], 0.0)
#         for ind in range(len(self.lockDofs)):
#             p.resetJointState(self.handId, self.lockDofs[ind], initLockPos[ind], 0.0)
#
#         self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
#                                       childFramePosition=initBasePos,
#                                       childFrameOrientation=initQuat)
#         # TODO: no vel here, is this correct? Think
#
#         p.stepSimulation()  # TODO
#
#         basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
#         self.baseInitPos = np.array(basePos)
#         self.baseInitEuler = np.array(p.getEulerFromQuaternion(baseQuat))
#         self.tarBasePos = np.copy(self.baseInitPos)
#         self.tarBaseEuler = np.copy(self.baseInitEuler)
#         self.tarFingerPos = np.copy(initActivePos)
#
#     def get_raw_state_fingers(self, includeVel=True):
#         dofs = self.activeDofs + self.lockDofs
#         joints_state = p.getJointStates(self.handId, dofs)
#         if includeVel:
#             joints_state = np.array(joints_state)[:,[0,1]]
#         else:
#             joints_state = np.array(joints_state)[:, [0]]
#         return np.hstack(joints_state.flatten())
#
#     def get_robot_observation(self):
#         obs = []
#
#         basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
#         obs.extend(basePos)
#         obs.extend(baseQuat)
#
#         print("pos", basePos)
#         print("euler", p.getEulerFromQuaternion(baseQuat))
#
#         obs.extend(list(self.get_raw_state_fingers(includeVel=False)))
#         # print(self.get_raw_state_fingers())
#
#         # TODO: no finger vel
#
#         baseVels = p.getBaseVelocity(self.handId)
#         obs.extend(baseVels[0])
#         obs.extend(baseVels[1])
#
#         print("linvel", baseVels[0])
#         print("angvel", baseVels[1])
#
#         obs.extend(list(self.tarFingerPos))
#         # print(self.tarFingerPos)
#         obs.extend(list(self.tarBasePos))
#         tarQuat = p.getQuaternionFromEuler(list(self.tarBaseEuler))
#         obs.extend(tarQuat)
#
#         if self.include_redun_body_pos:
#             for i in range(p.getNumJoints(self.handId)):
#                 pos = p.getLinkState(self.handId, i)[0]  # [0] stores xyz position
#                 obs.extend(pos)
#
#         return obs
#
#     def get_robot_observation_dim(self):
#         return len(self.get_robot_observation())
#
#     def get_finger_dist_from_init(self):
#         return np.linalg.norm(self.get_raw_state_fingers(includeVel=False) - self.initPos)
#
#     # def get_three_finger_deviation(self):
#     #     fingers_q = self.get_raw_state_fingers(includeVel=False)
#     #     assert len(fingers_q) == 16     # TODO
#     #     f1 = fingers_q[:4]
#     #     f2 = fingers_q[4:8]
#     #     f3 = fingers_q[8:12]
#     #     # TODO: is this different from dist to mean
#     #     return np.linalg.norm(f1-f2) + np.linalg.norm(f2-f3) + np.linalg.norm(f1-f3)
#
#     def apply_action(self, a):
#
#         # print("action", a)
#         # TODO: should encourage same q for first 3 fingers for now
#
#         # TODO: a is already scaled, how much to scale? decide in Env.
#         # should be delta control (policy outputs delta position), but better add to old tar pos instead of cur pos
#         # TODO: but tar pos should be part of state vector (? how accurate is pos_con?)
#
#         a = np.array(a)
#
#         dxyz = a[0:3]
#         dOri = a[3:6]
#         self.tarBasePos += dxyz
#         self.tarBasePos = np.clip(self.tarBasePos, self.base_ll, self.base_ul)
#
#         self.tarBaseEuler += dOri
#         self.tarBaseEuler = np.clip(self.tarBaseEuler, self.ori_ll, self.ori_ul)
#
#         # print(self.tarBasePos)
#         # print(p.getBasePositionAndOrientation(self.handId))
#         tarQuat = p.getQuaternionFromEuler(list(self.tarBaseEuler))
#         p.changeConstraint(self.cid, list(self.tarBasePos), tarQuat, maxForce=self.maxForce * 10.0)   # TODO: wrist force larger
#
#         self.tarFingerPos += a[6:]      # length should match
#         self.tarFingerPos = np.clip(self.tarFingerPos, self.ll, self.ul)
#
#         # p.setJointMotorControlArray(self.handId,
#         #                             self.activeDofs,
#         #                             p.POSITION_CONTROL,
#         #                             targetPositions=list(self.tarFingerPos))
#         # p.setJointMotorControlArray(self.handId,
#         #                             self.activeDofs,
#         #                             p.POSITION_CONTROL,
#         #                             targetPositions=list(self.tarFingerPos),
#         #                             forces=[self.maxForce]*len(self.tarFingerPos))
#
#         for i in range(len(self.activeDofs)):
#             p.setJointMotorControl2(self.handId,
#                                     jointIndex=self.activeDofs[i],
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=self.tarFingerPos[i],
#                                     force=self.maxForce)
#         # for i in range(len(self.lockDofs)):
#         #     p.setJointMotorControl2(self.handId,
#         #                             jointIndex=self.activeDofs[i],
#         #                             controlMode=p.POSITION_CONTROL,
#         #                             targetPosition=0.0,
#         #                             force=self.maxForce / 4.0)
#
# if __name__ == "__main__":
#     physicsClient = p.connect(p.GUI)    #or p.DIRECT for non-graphical version
#     # p.setAdditionalSearchPath(pybullet_data.getDataPath())  #optionally
#     # p.setGravity(0,0,-10)
#     # planeId = p.loadURDF("plane.urdf")
#     # cubeStartPos = [0,0,1]
#     # cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
#     # # boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
#     #
#     # boxId = p.loadURDF("/home/yifengj/Downloads/allegro_hand_description/allegro_hand_description_right.urdf", cubeStartPos,
#     #                    cubeStartOrientation)
#     #
#     # for i in range (1000000):
#     #     p.stepSimulation()
#     #     time.sleep(1./240.)
#     # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#     # print(cubePos,cubeOrn)
#
#     p.setTimeStep(1./240.)
#     p.setGravity(0, 0, -10)
#
#     floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
#     p.changeDynamics(floorId, -1, lateralFriction=3.0)
#
#     a = ShadowHand()
#     a.np_random, seed = gym.utils.seeding.np_random(0)
#
#     for i in range(100):
#         np.random.seed(0)
#         a.reset()
#
#         # a = InmoovHand()
#         #
#         # p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
#
#         # a = AllegroHand()
#
#         input("press enter to continue")
#         print("init", a.get_robot_observation())
#         for t in range(800):
#             # a.apply_action(np.random.uniform(low=-0.005, high=0.005, size=6+22)+np.array([0.0025]*6+[0.01]*22))
#             a.apply_action(
#                 np.random.uniform(low=-0.001, high=0.001, size=6 + 17) + np.array([0.002] * 3 + [0.0005]*3 + [-0.01] * 17))
#             # a.apply_action(np.array([0.0]*6+[0.01]*22))
#             # a.apply_action(np.array([0.005] * (22+6)))
#
#             p.stepSimulation()
#             # print(p.getConstraintState(a.cid)[:3])
#             act_palm_com, _ = p.getBasePositionAndOrientation(a.handId)
#             print("after ts", np.array(a.tarBasePos) - np.array(act_palm_com))
#             # a.apply_action(np.array([0.0] * 23))  # equivalent
#             # p.stepSimulation()
#             # print(p.getConstraintState(a.cid)[:3])
#             time.sleep(1./240.)
#         print("final obz", a.get_robot_observation())
#
#     p.disconnect()
#
#
#     # def seed(self, seed=None):
#     #     self.np_random, seed = gym.utils.seeding.np_random(seed)
#     #     self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
#     #     return [seed]
