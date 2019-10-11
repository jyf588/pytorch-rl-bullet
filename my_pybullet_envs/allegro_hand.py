import pybullet as p
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# TODO: no mass
# TODO: table contact violated
# TODO: render

class AllegroHand:
    def __init__(self):

        self.baseInitPos = np.array([0, 0, 0.5])
        self.baseInitOri = np.array([1.57, 0, 0])
        # TODO: always init to certain "holding" pose
        self.initPos = np.array([0.0, 0.9, 0.8, 0.0] * 3 + [1.2, 0.5, 0.0, 0.3])

        # TODO: note, no self-collision flag
        self.handId = p.loadURDF(os.path.join(currentdir, "assets/allegro_hand_description/allegro_hand_description_right.urdf"),
                                 list(self.baseInitPos), flags=p.URDF_USE_SELF_COLLISION)
        nDof = p.getNumJoints(self.handId)
        # for i in range(p.getNumJoints(self.handId)):
        #     print(p.getJointInfo(self.handId, i)[2], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])

        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/constraint.py#L11
        self.cid = p.createConstraint(self.handId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], list(self.baseInitPos))

        for i in range(-1, p.getNumJoints(self.handId)):
            p.changeDynamics(self.handId, i, lateralFriction=3.0)

        # exclude fixed joints, actual DoFs are [0:4, 5:9, 10:14, 15:19]
        self.activeDofs = []
        for i in range(4):
            self.activeDofs += (np.arange(4) + 5 * i).tolist()

        self.ll = np.array([p.getJointInfo(self.handId, i)[8] for i in range(nDof)])
        self.ul = np.array([p.getJointInfo(self.handId, i)[9] for i in range(nDof)])    # use np for multi indexing
        self.ll = self.ll[self.activeDofs]
        self.ul = self.ul[self.activeDofs]

        # self.tarFingerPos = np.clip(np.zeros(len(self.activeDofs)), self.ll, self.ul)    # initial tar pos
        # self.tarFingerPos = [0.0, 0.9, 0.8] * 3 + [1.2, 0.5, 0.0, 0.3]

        self.tarBasePos = np.copy(self.baseInitPos)
        self.tarBaseOri = np.copy(self.baseInitOri)     # euler angles
        self.tarFingerPos = np.copy(self.initPos)    # used for position control and as part of state

        self.maxForce = 1000.

        self.include_redun_body_pos = False

        self.np_random = None   # seeding inited outside in Env

        # self.dir = 0.01 * np.ones(len(self.activeDofs))

        # print(self.tarFingerPos)
        # print(self.ll)
        # print(self.ul)
        assert len(self.tarFingerPos) == len(self.ll)

        # self.reset()

    def reset(self):
        # TODO: bullet env reload urdfs in reset...
        # TODO: bullet env reset pos with added noise but velocity to zero always.

        initBasePos = np.array(self.baseInitPos) + self.np_random.uniform(low=-0.1, high=0.1, size=3)
        initOri = np.array(self.baseInitOri) + self.np_random.uniform(low=-0.2, high=0.2, size=3)

        # initBasePos = np.array(self.baseInitPos)
        # initOri = np.array(self.baseInitOri)
        initQuat = p.getQuaternionFromEuler(list(initOri))

        p.resetBasePositionAndOrientation(self.handId, initBasePos, initQuat)
        p.changeConstraint(self.cid, list(initBasePos), jointChildFrameOrientation=initQuat)

        # init self.np_random outside, in Env
        initPos = self.initPos + self.np_random.uniform(low=-0.05, high=0.05, size=len(self.initPos))

        for ind in range(len(self.activeDofs)):
            p.resetJointState(self.handId, self.activeDofs[ind], initPos[ind])

        # p.resetJointStatesMultiDof(self.handId, self.activeDofs, targetValues=list(initPos), targetVelocities=list(initPos*0.0))
        p.setJointMotorControlArray(self.handId, self.activeDofs, p.POSITION_CONTROL, list(initPos))

        self.tarBasePos = np.copy(initBasePos)
        self.tarBaseOri = np.copy(initOri)
        self.tarFingerPos = np.copy(initPos)

        # print(initBasePos)
        # print(initOri)
        # input("press enter to continue")

    def get_raw_state_fingers(self):
        joints_state = p.getJointStates(self.handId, self.activeDofs)
        joints_state = np.array(joints_state)[:,[0]]    # only position, no vel.
        # print(joints_state.flatten())
        return np.hstack(joints_state.flatten())

    def get_robot_observation(self):
        obs = []

        obs.extend(list(self.get_raw_state_fingers()))
        basePos, baseQuat = p.getBasePositionAndOrientation(self.handId)
        obs.extend(basePos)
        obs.extend(baseQuat)

        obs.extend(list(self.tarFingerPos))
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

    def apply_action(self, a):
        # TODO: should encourage same q for first 3 fingers for now

        # TODO: a is already scaled, how much to scale? decide in Env.
        # should be delta control (policy outputs delta position), but better add to old tar pos instead of cur pos
        # TODO: but tar pos should be part of state vector (? how accurate is pos_con?)

        a = np.array(a)

        dxyz = a[0:3]
        dOri = a[3:6]
        self.tarBasePos += dxyz
        self.tarBasePos[:2] = np.clip(self.tarBasePos[:2], -1, 1)
        self.tarBasePos[2] = np.clip(self.tarBasePos[2], 0, 1)

        self.tarBaseOri += dOri

        # so that state/obs is bounded
        for i in range(len(self.tarBaseOri)):
            if self.tarBaseOri[i] > math.pi: self.tarBaseOri[i] -= 2 * math.pi
            if self.tarBaseOri[i] < -math.pi: self.tarBaseOri[i] += 2 * math.pi

        # print(self.tarBasePos, self.tarBaseOri)

        tarQuat = p.getQuaternionFromEuler(list(self.tarBaseOri))
        p.changeConstraint(self.cid, list(self.tarBasePos), jointChildFrameOrientation=tarQuat, maxForce=self.maxForce)
        # p.changeConstraint(self.cid, list(self.tarBasePos), jointChildFrameOrientation=tarQuat)

        # print(self.get_raw_state_fingers())

        # for ind, (pos,ll,ul) in enumerate(zip(self.tarFingerPos, self.ll, self.ul)):
        #     if pos < ll: self.dir[ind] = 0.01
        #     if pos > ul: self.dir[ind] = -0.01

        self.tarFingerPos += a[6:]      # length should match
        self.tarFingerPos = np.clip(self.tarFingerPos, self.ll, self.ul)

        # p.setJointMotorControlArray(self.handId,
        #                             self.activeDofs,
        #                             p.POSITION_CONTROL,
        #                             targetPositions=list(self.tarFingerPos))
        p.setJointMotorControlArray(self.handId,
                                    self.activeDofs,
                                    p.POSITION_CONTROL,
                                    targetPositions=list(self.tarFingerPos),
                                    forces=[self.maxForce]*len(self.tarFingerPos))
        #
        # cubePos, cubeOrn = p.getBasePositionAndOrientation(self.handId)
        # cubePos = list(cubePos)
        # cubeOrn = list(cubeOrn)
        # cubeEuler = list(p.getEulerFromQuaternion(cubeOrn))
        # cubePos[0] += 0.005
        # cubeEuler[0] += 0.005
        # # cubeEuler[1] += 0.005
        #
        #
        #
        # cubeOrn = p.getQuaternionFromEuler(cubeEuler)
        # p.resetBasePositionAndOrientation(self.handId, cubePos, cubeOrn)

        # p.setJointMotorControl2(bodyUniqueId=objUid,
        #                         jointIndex=0,
        #                         controlMode=p.VELOCITY_CONTROL,
        #                         targetVelocity=targetVel,
        #                         force=maxForce)


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

    a = AllegroHand()
    a.np_random, seed = gym.utils.seeding.np_random(0)

    for i in range(100):
        a.reset()
        input("press enter to continue")
        print("init", a.get_robot_observation())
        act = a.np_random.uniform(low=-0.01, high=0.01, size=6+16)
        act[3:] = 0.0
        for t in range(400):
            a.apply_action(a.np_random.uniform(low=-0.01, high=0.01, size=6+16))
            p.stepSimulation()
            time.sleep(1./240.)
        print("final z", a.get_robot_observation())


    p.disconnect()


    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
    #     return [seed]
