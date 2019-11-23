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

class ShadowHand:
    def __init__(self,
                 base_init_pos=np.array([-0.13, 0.07, 0.1]),
                 base_init_euler=np.array([1.8, -1.57, 0]),
                 # init_fin_pos=np.array([0.0, 0.4, 0.4, 0.4]*3+[0.0,0.0,0.4,0.4,0.4]+[0.0,0.6,0.1,0.4,0.5])):  # TODO
                 # init_fin_pos = np.array([0.0, 0.4, 0.4, 0.4] * 3 + [0.0, 0.0, 0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0])):  # TODO
                 init_fin_pos = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0])):  # TODO

        self.baseInitPos = base_init_pos
        self.baseInitOri = base_init_euler
        self.initPos = init_fin_pos

        self.handId = p.loadURDF(os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadowhand_motor.urdf"),
                                 list(self.baseInitPos), p.getQuaternionFromEuler(list(self.baseInitOri)),
                                 flags=p.URDF_USE_SELF_COLLISION)
        # nDof = p.getNumJoints(self.handId)
        # for i in range(p.getNumJoints(self.handId)):
        #     print(p.getJointInfo(self.handId, i)[2], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])

        # 4,4,4,5,5
        self.activeDofs = []
        self.lockDofs = []
        start = 0
        for i in range(5):
            nDofs = [4,4,4,5,5]
            fig_start = [1,1,1,2,0] # default 0,0,0,0,0 to include all finger dofs
            self.activeDofs += (np.arange(fig_start[i], nDofs[i]) + start).tolist()
            self.lockDofs += (np.arange(0, fig_start[i]) + start).tolist()
            start += (nDofs[i]+1)
        print(self.activeDofs)
        print(self.lockDofs)
        assert len(self.activeDofs) == len(init_fin_pos)

        self.ll = np.array([p.getJointInfo(self.handId, i)[8] for i in self.activeDofs])
        self.ul = np.array([p.getJointInfo(self.handId, i)[9] for i in self.activeDofs])

        for ind in range(len(self.activeDofs)):
            p.resetJointState(self.handId, self.activeDofs[ind], self.initPos[ind], 0.0)
        for ind in range(len(self.lockDofs)):
            p.resetJointState(self.handId, self.lockDofs[ind], 0.0, 0.0)

        for i in range(-1, p.getNumJoints(self.handId)):
            p.changeDynamics(self.handId, i, lateralFriction=3.0)
            # TODO: increase mass for now
            mass = p.getDynamicsInfo(self.handId, i)[0]
            # inertia = p.getDynamicsInfo(self.handId, i)[2]
            mass = mass * 100.
            # inertia = [ele * 100. for ele in inertia]
            p.changeDynamics(self.handId, i, mass=mass)
            # p.changeDynamics(self.handId, i, localInertiaDiagnoal=inertia)    # TODO: default inertia from bullet

        # # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/constraint.py#L11
        self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      childFramePosition=list(self.baseInitPos),
                                      childFrameOrientation=p.getQuaternionFromEuler(list(self.baseInitOri)))

        self.tarBasePos = np.copy(self.baseInitPos)
        self.tarBaseOri = np.copy(self.baseInitOri)     # euler angles
        self.tarFingerPos = np.copy(self.initPos)    # used for position control and as part of state

        self.maxForce = 100.
        # self.maxForce = 0.

        self.include_redun_body_pos = False

        self.np_random = None   # seeding inited outside in Env

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
            # initOri = self.baseInitOri
            initBasePos = np.array(self.baseInitPos)
            initBasePos[0] += self.np_random.uniform(low=-0.03, high=0.03)
            initBasePos[1] += self.np_random.uniform(low=-0.03, high=0.03)
            initBasePos[2] += self.np_random.uniform(low=-0.03, high=0.03)  # enlarge here
            initOri = np.array(self.baseInitOri) + self.np_random.uniform(low=-0.05, high=0.05, size=3)
            initQuat = p.getQuaternionFromEuler(list(initOri))
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
            self.tarBaseOri = np.copy(initOri)
            self.tarFingerPos = np.copy(initPos)

    def get_raw_state_fingers(self, includeVel=True):
        joints_state = p.getJointStates(self.handId, self.activeDofs)
        if includeVel:
            joints_state = np.array(joints_state)[:,[0,1]]
        else:
            joints_state = np.array(joints_state)[:, [0]]
        # print(joints_state.flatten())
        return np.hstack(joints_state.flatten())

    def get_robot_observation(self):
        obs = []

        basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
        obs.extend(basePos)
        obs.extend(baseQuat)

        obs.extend(list(self.get_raw_state_fingers(includeVel=False)))
        # print(self.get_raw_state_fingers())

        # TODO: no finger vel

        baseVels = p.getBaseVelocity(self.handId)
        obs.extend(baseVels[0])
        obs.extend(baseVels[1])

        obs.extend(list(self.tarFingerPos))
        # print(self.tarFingerPos)
        obs.extend(list(self.tarBasePos))
        tarQuat = p.getQuaternionFromEuler(list(self.tarBaseOri))
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
        xyz_ll = self.baseInitPos + np.array([-0.1, -0.1, -0.1])
        xyz_ul = self.baseInitPos + np.array([0.3, 0.1, 0.1])
        self.tarBasePos = np.clip(self.tarBasePos, xyz_ll, xyz_ul)

        ori_lb = self.baseInitOri - 1.57        # TODO: is this right?
        ori_ub = self.baseInitOri + 1.57
        self.tarBaseOri += dOri
        self.tarBaseOri = np.clip(self.tarBaseOri, ori_lb, ori_ub)

        # print(self.tarBasePos)
        # print(p.getBasePositionAndOrientation(self.handId))
        tarQuat = p.getQuaternionFromEuler(list(self.tarBaseOri))
        p.changeConstraint(self.cid, list(self.tarBasePos), tarQuat, maxForce=self.maxForce * 5.0)   # TODO: wrist force larger

        self.tarFingerPos += a[6:]      # length should match
        self.tarFingerPos = np.clip(self.tarFingerPos, self.ll, self.ul)

        # p.setJointMotorControlArray(self.handId,
        #                             self.activeDofs,
        #                             p.POSITION_CONTROL,
        #                             targetPositions=list(self.tarFingerPos))
        # p.setJointMotorControlArray(self.handId,
        #                             self.activeDofs,
        #                             p.POSITION_CONTROL,
        #                             targetPositions=list(self.tarFingerPos),
        #                             forces=[self.maxForce]*len(self.tarFingerPos))

        for i in range(len(self.activeDofs)):
            p.setJointMotorControl2(self.handId,
                                    jointIndex=self.activeDofs[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.tarFingerPos[i],
                                    force=self.maxForce)
        for i in range(len(self.lockDofs)):
            p.setJointMotorControl2(self.handId,
                                    jointIndex=self.activeDofs[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.0,
                                    force=self.maxForce / 4.0)

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

    floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
    p.changeDynamics(floorId, -1, lateralFriction=3.0)

    a = ShadowHand()
    a.np_random, seed = gym.utils.seeding.np_random(0)

    for i in range(100):
        np.random.seed(0)
        a.reset()

        # a = InmoovHand()
        #
        # p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)

        # a = AllegroHand()

        input("press enter to continue")
        print("init", a.get_robot_observation())
        for t in range(800):
            # a.apply_action(np.random.uniform(low=-0.005, high=0.005, size=6+22)+np.array([0.0025]*6+[0.01]*22))
            a.apply_action(
                np.random.uniform(low=-0.005, high=0.005, size=6 + 17) + np.array([0.0025] * 6 + [-0.01] * 17))
            # a.apply_action(np.array([0.0]*6+[0.01]*22))
            # a.apply_action(np.array([0.005] * (22+6)))

            p.stepSimulation()
            p.stepSimulation()
            time.sleep(1./480.)
        print("final obz", a.get_robot_observation())

    p.disconnect()


    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
    #     return [seed]
