import pybullet as p
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# TODO: mass good or not?
# TODO: render


class InmoovShadowHand:
    def __init__(self,
                 palm_init_pos=np.array([-0.17, 0.07, 0.1]),
                 palm_init_euler=np.array([1.8, -1.57, 0]),
                 init_fin_pos = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0])):  # TODO

        self.baseToPalmOffset = np.array([-0.27157567, 0.37579589,  -1.17620252])
        self.palmInitPos = palm_init_pos
        self.palmInitOri = palm_init_euler
        self.initFinPos = init_fin_pos

        self.robotId = p.loadURDF(os.path.join(currentdir, "assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand.urdf"),
                                  (self.palmInitPos + self.baseToPalmOffset).tolist(), flags=p.URDF_USE_SELF_COLLISION)
        # nDof = p.getNumJoints(self.handId)
        # for i in range(p.getNumJoints(self.handId)):
        #     print(p.getJointInfo(self.handId, i)[2], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])

        self.armDofs = [27, 28, 29, 30, 31, 33, 34]
        self.armIKDofs  = np.array([9, 10, 11, 12, 13, 14, 15])-2   # exclude fixed TODO: hard coded here: may change after we clean URDF
        self.endEffectorId = self.armDofs[-1]
        self.arm_ll = np.array([p.getJointInfo(self.robotId, i)[8] for i in self.armDofs])
        self.arm_ul = np.array([p.getJointInfo(self.robotId, i)[9] for i in self.armDofs])
        self.arm_jr = self.arm_ul - self.arm_ll
        # # TODO: the corresponding IK q of the above offset [-0.37157567, 0.47579589, -1.17620252] is: (with fixed waist)
        # self.arm_rp = [-0.4600493, -0.5079458, -0.550884, -0.460955, -0.166899, 0.446743, -0.787904]
        # TODO: the corresponding IK q of the above offset [-0.27157567, 0.37579589,  -1.17620252]  is: (with fixed waist)
        self.arm_rp =[-0.1766934, 0.0010606, -0.347004, -1.296158, -0.169959, 0.2112, -0.9534262]
        assert len(self.arm_rp) == len(self.armDofs)

        # 4,4,4,5,5
        self.activeFinDofs = []
        self.zeroFinDofs = []
        tmp = 36
        start = tmp      # TODO: hard coded here: may change after we clean URDF
        for i in range(5):
            nDofs = [4,4,4,5,5]
            fig_start = [1,1,1,2,0] # default 0,0,0,0,0 to include all finger dofs
            self.activeFinDofs += (np.arange(fig_start[i], nDofs[i]) + start).tolist()
            self.zeroFinDofs += (np.arange(0, fig_start[i]) + start).tolist()
            start += (nDofs[i]+1)
        print(self.activeFinDofs)
        print(self.zeroFinDofs)
        assert len(self.activeFinDofs) == len(init_fin_pos)

        self.fll = np.array([p.getJointInfo(self.robotId, i)[8] for i in self.activeFinDofs])
        self.ful = np.array([p.getJointInfo(self.robotId, i)[9] for i in self.activeFinDofs])

        for ind in range(len(self.armDofs)):
            p.resetJointState(self.robotId, self.armDofs[ind], self.arm_rp[ind], 0.0)
        for ind in range(len(self.activeFinDofs)):
            p.resetJointState(self.robotId, self.activeFinDofs[ind], self.initFinPos[ind], 0.0)
        for ind in range(len(self.zeroFinDofs)):
            p.resetJointState(self.robotId, self.zeroFinDofs[ind], 0.0, 0.0)
        # TODO: left arm and head does not matter for now since we fix waist

        for i in range(tmp, start):
            p.changeDynamics(self.robotId, i, lateralFriction=3.0)
            # TODO: increase finger/palm mass for now to make them more stable
            mass = p.getDynamicsInfo(self.robotId, i)[0]
            mass = mass * 100.
            p.changeDynamics(self.robotId, i, mass=mass)
            # TODO: default inertia from bullet

        newPos, newOrn = self.get_palm_pos_orn()
        self.palmInitPos = np.copy(newPos)
        self.palmInitOri = np.copy(p.getEulerFromQuaternion(newOrn))
        self.tarPalmPos = np.copy(self.palmInitPos)
        self.tarPalmOri = np.copy(self.palmInitOri)     # euler angles
        self.tarFingerPos = np.copy(self.initFinPos)    # used for position control and as part of state

        self.maxForce = 100.
        # self.maxForce = 0.

        self.include_redun_body_pos = False

        self.np_random = None   # seeding inited outside in Env

        # print(self.tarFingerPos)
        # print(self.ll)
        # print(self.ul)
        assert len(self.tarFingerPos) == len(self.fll)

    def reset(self):
        # TODO: bullet env reload urdfs in reset...
        # TODO: bullet env reset pos with added noise but velocity to zero always.

        goodInit = False
        while not goodInit:
            armRp = self.arm_rp
            armRp += self.np_random.uniform(low=-0.01, high=0.01, size=len(self.armDofs))

            # # initBasePos = self.baseInitPos
            # # initOri = self.baseInitOri
            # initBasePos = np.array(self.baseInitPos)
            # initBasePos[0] += self.np_random.uniform(low=-0.03, high=0.03)
            # initBasePos[1] += self.np_random.uniform(low=-0.03, high=0.03)
            # initBasePos[2] += self.np_random.uniform(low=-0.03, high=0.03)  # enlarge here
            # initOri = np.array(self.baseInitOri) + self.np_random.uniform(low=-0.05, high=0.05, size=3)
            # initQuat = p.getQuaternionFromEuler(list(initOri))
            # #
            # # print(p.getEulerFromQuaternion(initQuat))

            # init self.np_random outside, in Env
            # initPos = self.initPos
            initPos = self.initFinPos + self.np_random.uniform(low=-0.05, high=0.05, size=len(self.initFinPos))

            # p.removeConstraint(self.cid)
            # p.resetBasePositionAndOrientation(self.robotId, initBasePos, initQuat)

            for ind in range(len(self.armDofs)):
                p.resetJointState(self.robotId, self.armDofs[ind], armRp[ind], 0.0)
            for ind in range(len(self.activeFinDofs)):
                p.resetJointState(self.robotId, self.activeFinDofs[ind], initPos[ind], 0.0)
            for ind in range(len(self.zeroFinDofs)):
                p.resetJointState(self.robotId, self.zeroFinDofs[ind], 0.0, 0.0)

            # self.cid = p.createConstraint(self.robotId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
            #                               childFramePosition=initBasePos,
            #                               childFrameOrientation=initQuat)

            p.stepSimulation()  # TODO

            cps = p.getContactPoints(bodyA=self.robotId)
            # for cp in cps:
            #     print(cp)
            #     input("penter")
            # print(cps[0][6])
            if len(cps) == 0: goodInit = True   # TODO: init hand last and make sure it does not colllide with env

            newPos, newOrn = self.get_palm_pos_orn()
            self.palmInitPos = np.copy(newPos)
            self.palmInitOri = np.copy(p.getEulerFromQuaternion(newOrn))
            self.tarPalmPos = np.copy(self.palmInitPos)
            self.tarPalmOri = np.copy(self.palmInitOri)  # euler angles
            self.tarFingerPos = np.copy(self.initFinPos)  # used for position control and as part of state

    def get_palm_pos_orn(self):
        newPos = p.getLinkState(self.robotId, self.endEffectorId)[4]
        newOrn = p.getLinkState(self.robotId, self.endEffectorId)[5]
        return newPos, newOrn

    def get_raw_state_arm(self, includeVel=True):
        joints_state = p.getJointStates(self.robotId, self.armDofs)
        if includeVel:
            joints_state = np.array(joints_state)[:, [0, 1]]
        else:
            joints_state = np.array(joints_state)[:, [0]]
        return np.hstack(joints_state.flatten())

    def get_raw_state_fingers(self, includeVel=True):
        dofs = self.activeFinDofs + self.zeroFinDofs
        joints_state = p.getJointStates(self.robotId, dofs)
        if includeVel:
            joints_state = np.array(joints_state)[:,[0,1]]
        else:
            joints_state = np.array(joints_state)[:, [0]]
        return np.hstack(joints_state.flatten())

    def get_robot_observation(self):
        obs = []

        obs.extend(list(self.get_raw_state_arm(includeVel=True)))

        # TODO: palm pos and vel

        obs.extend(list(self.get_raw_state_fingers(includeVel=False)))
        # print(self.get_raw_state_fingers())
        # TODO: no finger vel

        obs.extend(list(self.tarFingerPos))
        # print(self.tarFingerPos)
        obs.extend(list(self.tarPalmPos))
        tarQuat = p.getQuaternionFromEuler(list(self.tarPalmOri))
        obs.extend(tarQuat)

        if self.include_redun_body_pos:
            for i in range(p.getNumJoints(self.robotId)):
                pos = p.getLinkState(self.robotId, i)[0]  # [0] stores xyz position
                obs.extend(pos)

        return obs

    def get_robot_observation_dim(self):
        return len(self.get_robot_observation())

    def get_finger_dist_from_init(self):
        return np.linalg.norm(self.get_raw_state_fingers(includeVel=False) - self.initFinPos)

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

        # bounds
        xyz_ll = self.palmInitPos + np.array([-0.1, -0.1, -0.1])    # TODO: how to set these / palmInitPos->FK
        xyz_ul = self.palmInitPos + np.array([0.2, 0.1, 0.1])
        ori_lb = self.palmInitOri - 1.57        # TODO: is this right?
        ori_ub = self.palmInitOri + 1.57

        a = np.array(a)

        dxyz = a[0:3]
        dOri = a[3:6]
        self.tarPalmPos += dxyz
        self.tarPalmPos = np.clip(self.tarPalmPos, xyz_ll, xyz_ul)
        self.tarPalmOri += dOri
        self.tarPalmOri = np.clip(self.tarPalmOri, ori_lb, ori_ub)
        tarQuat = p.getQuaternionFromEuler(list(self.tarPalmOri))

        cur_arm_q = self.get_raw_state_arm(includeVel=False)

        # run IK
        armQIK = p.calculateInverseKinematics(self.robotId, self.endEffectorId, list(self.tarPalmPos), tarQuat,
                                              lowerLimits=self.arm_ll.tolist(), upperLimits=self.arm_ul.tolist(),
                                              jointRanges=self.arm_jr.tolist(), restPoses=cur_arm_q)
        armQIK = np.array(armQIK)[self.armIKDofs]
        armQIK = armQIK.tolist()

        for ji,i in enumerate(self.armDofs):
            p.setJointMotorControl2(bodyIndex=self.robotId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=armQIK[ji],
                                    targetVelocity=0,
                                    force=self.maxForce * 30.,        # TODO wrist force larger
                                    positionGain=0.1,
                                    velocityGain=1)

        self.tarFingerPos += a[6:]      # length should match
        self.tarFingerPos = np.clip(self.tarFingerPos, self.fll, self.ful)

        # print("tar", self.tarFingerPos)
        # print(self.get_raw_state_fingers(includeVel=False))

        for i in range(len(self.activeFinDofs)):
            p.setJointMotorControl2(self.robotId,
                                    jointIndex=self.activeFinDofs[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.tarFingerPos[i],
                                    force=self.maxForce)
        # for i in range(len(self.zeroFinDofs)):
        #     p.setJointMotorControl2(self.robotId,
        #                             jointIndex=self.activeFinDofs[i],
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPosition=0.0,
        #                             force=self.maxForce / 4.0)

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)    #or p.DIRECT for non-graphical version
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())  #optionally
    # p.setGravity(0,0,-10)
    # planeId = p.loadURDF("plane.urdf")
    # cubeStartPos = [0,0,1]
    # cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    # # boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
    #
    # boxId = p.loadURDF("/home/yifengj/Downloads/allegro_hand_description/allegro_hand_description_right.urdf", cubeStartPos,
    #                    cubeStartOrientation)
    #
    # for i in range (1000000):
    #     p.stepSimulation()
    #     time.sleep(1./240.)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)

    p.setTimeStep(1./480.)
    p.setGravity(0, 0, -5)

    cylinderId = p.loadURDF(os.path.join(currentdir, 'assets/cylinder.urdf'), [0.,0.,0.1], useFixedBase=0)  # 0.2/2
    p.changeDynamics(cylinderId, -1, lateralFriction=3.0)
    floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
    p.changeDynamics(floorId, -1, lateralFriction=3.0)

    a = InmoovShadowHand()
    a.np_random, seed = gym.utils.seeding.np_random(0)

    np.random.seed(0)

    for i in range(100):
        a.reset()

        p.resetBasePositionAndOrientation(cylinderId,
                                          [0.,0.,0.1],
                                          p.getQuaternionFromEuler([0, 0, 0]))

        print("init", a.get_robot_observation())
        ls = p.getLinkState(a.robotId, a.endEffectorId)
        newPos = ls[4]
        print(newPos, p.getEulerFromQuaternion(ls[5]))
        input("press enter to continue")
        action = np.random.uniform(low=-1.0, high=1.0, size=6+17)

        action[:3] *= 0.005
        action[3:6] *= 0.005
        action[6:] *= 0.01

        for t in range(800):
            # a.apply_action(np.random.uniform(low=-0.005, high=0.005, size=6+17)+np.array([0.005]*1+[0.0]*5+[0.01]*17))
            # a.apply_action(
            #     np.random.uniform(low=-0.005, high=0.005, size=6 + 17) + action)
            # action = [0.0]*23
            a.apply_action(action)
            # a.apply_action(np.array([0.0]*6+[0.01]*22))
            # a.apply_action(np.array([0.005] * (22+6)))

            p.stepSimulation()
            p.stepSimulation()
            time.sleep(1./600.)
        print("final obz", a.get_robot_observation())
        ls = p.getLinkState(a.robotId, a.endEffectorId)
        newPos = ls[4]
        print(newPos, p.getEulerFromQuaternion(ls[5]))

    p.disconnect()


    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
    #     return [seed]
