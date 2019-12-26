import pybullet as p
import time
import math
from datetime import datetime
import numpy as np

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class InmoovShadowHand:
    def __init__(self,
                 base_init_pos=np.array([0., 0, 0.0]),
                 base_init_euler=np.array([0.0, 0, 0]),
                 init_fin_pos=np.array([0.0, 0.9, 0.8, 0.0] * 3 + [1.2, 0.5, 0.0, 0.3])):

        self.baseInitPos = base_init_pos
        self.baseInitOri = base_init_euler
        # self.initPos = init_fin_pos

        self.endEffectorId = 32
        start = 60  # TODO

        self.robotId = p.loadURDF(os.path.join(currentdir, "assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2.urdf"),
                                  list(self.baseInitPos), p.getQuaternionFromEuler(list(self.baseInitOri)),
                                  flags=p.URDF_USE_SELF_COLLISION)
        self.cid = p.createConstraint(self.robotId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                      childFramePosition=list(self.baseInitPos),
                                      childFrameOrientation=p.getQuaternionFromEuler(list(self.baseInitOri)))

        # for i in range(p.getNumJoints(self.robotId)):
        #     print(p.getJointInfo(self.robotId, i)[0:2], p.getJointInfo(self.robotId, i)[8], p.getJointInfo(self.robotId, i)[9])

        # mass surgery (inconsistency between inmoov and shadow), default inertia from bullet
        for i in range(-1, p.getNumJoints(self.robotId)):
            if i < (self.endEffectorId-1) or i >= start:
                # mass = p.getDynamicsInfo(self.robotId, i)[0]
                # mass = mass / 100.
                p.changeDynamics(self.robotId, i, mass=1.0)
        total_m = 0
        for i in range(self.endEffectorId-1, start):    # starting from wristlink
            # increase wrist/finger/palm mass for now to make them more stable
            mass = p.getDynamicsInfo(self.robotId, i)[0]
            mass = mass * 10.
            p.changeDynamics(self.robotId, i, mass=mass)
            total_m += mass
            p.changeDynamics(self.robotId, i, lateralFriction=3.0)


        # # exclude fixed joints, actual DoFs are [0:4, 5:9, 10:14, 15:19]
        # self.activeDofs = []
        # for i in range(4):
        #     self.activeDofs += (np.arange(4) + 5 * i).tolist()

        # for ind in range(len(self.activeDofs)):
        #     p.resetJointState(self.handId, self.activeDofs[ind], self.initPos[ind], 0.0)

        self.tarBasePos = np.copy(self.baseInitPos)
        self.tarBaseOri = np.copy(self.baseInitOri)     # euler angles

        self.savedState = p.saveState()

def getNumpyIso3dMatrix(pos, quat):
    matlist = p.getMatrixFromQuaternion(quat)
    # print(list(matlist[:3]))
    # print(pos[0])
    r1 = list(matlist[:3]) + [pos[0]]
    r2 = list(matlist[3:6]) + [pos[1]]
    r3 = list(matlist[6:]) + [pos[2]]
    mat = np.array([r1, r2, r3, [0,0,0,1]])
    return repr(mat)



if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)    #or p.DIRECT for non-graphical version

    p.setGravity(0, 0, -10.)

    ts = 1./240
    p.setTimeStep(ts)

    tarOrn = p.getQuaternionFromEuler([1.8, -1.57, 0])    # TODO: these two are equivalent, init orin important
    print(tarOrn)
    # print(p.getQuaternionFromEuler([1.8, -1.57, 0.0]))
    # tarPos = [0.27157567, -0.37579589,  1.17620252] # this is almost workspace size: 0.45*0.45
    tarPos = [0.03955104947090149, 0.2158297449350357, 0.11094027012586594]
    # tarPos = [0.03955104947090149, 0.4158297449350357, 0.11094027012586594]
    baseToPalmOffset = np.array([-0.27157567, 0.37579589, -1.17620252])
    # palmInitPos = np.array([-0.19, 0.10, 0.1])
    palmInitPos = np.array([-0.12, 0.10, 0.1])
    tarPos = list(np.array(tarPos) - (baseToPalmOffset+palmInitPos))
    # tarOrn = [0.49626946449279785, -0.4890630841255188, 0.5149370431900024, 0.4993733763694763] this is wrong. why??

    # tarPos =( np.array([0.3419269919395447, -0.30924251675605774, -0.10815609246492386])
    #             - np.array([-0.24, 0.0, -1.25]) ).tolist()
    # tarPos =( np.array([0.014969042502343655, -0.10807766765356064, -0.24303656816482544])
    #             - np.array([-0.24, 0.0, -1.25]) ).tolist()
    # tarOrn = [0.889319658279419, -0.3515283763408661, 0.19866934418678284, 0.21463678777217865]
    # tarOrn = [0.9999897480010986, -0.0009517080034129322, 0.0041458094492554665, -0.0015737927751615644]

    localpos = [0.0, 0.0, 0.035]
    localquat = [0.0, 0.0, 0.0, 1.0]
    com_offset, _ = p.multiplyTransforms([0, 0, 0], tarOrn, localpos, localquat)

    tarPos = (np.array([0.2, -0.4, 0.0]) + np.array([-0.17, 0.07, 0.13])
              - np.array([-0.24, 0.0, -1.25])).tolist()
    tarOrn = p.getQuaternionFromEuler([1.8, -1.57, 0])

    print(getNumpyIso3dMatrix(tarPos, tarOrn))

    # input("press enter")

    hh = p.loadURDF(
        os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadowhand_motor.urdf"),
        [0, 0,  2], tarOrn,
        flags=p.URDF_USE_SELF_COLLISION)
    # hhls = p.getLinkState(hh, 0)
    # print(p.getEulerFromQuaternion(hhls[5]))
    # print("com", hhls[0])
    # print("link", hhls[4])
    # print(np.array(hhls[4])-np.array(hhls[0]))

    a = InmoovShadowHand()

    # relatedDofs = [3,5, 27,28,29,30,31, 33,34]  # the first two are waist
    # relatedDofs = [ 27, 28, 29, 30, 31, 33, 34]  # the first two are waist
    # relatedDofs = [29,30,31,32,33,35,36]
    relatedDofs = [25, 26, 27, 28, 29, 31, 32]  # TODO: can infer this from name
    # p.resetJointState(a.robotId, 28, -3.0, 0.0)
    # input("press enter")

    endEffectorId = 32
    ll = np.array([p.getJointInfo(a.robotId, i)[8] for i in relatedDofs] + [0.0] * (22 + 5))
    ul = np.array([p.getJointInfo(a.robotId, i)[9] for i in relatedDofs] + [0.0] * (22 + 5))
    jr = ul - ll
    # rp = [-0.4] * 7 + [0.0] * (22+5)

    # ll = np.array([p.getJointInfo(a.handId, i)[8] for i in relatedDofs])
    # ul = np.array([p.getJointInfo(a.handId, i)[9] for i in relatedDofs])
    # jr = ul - ll
    # # rp = [0.0]*2 + [-0.4]*7
    # rp = [-0.4] * 7
    # rp = [-0.16919578296797122, -0.0009699891314406999, -0.3581625283670476, -1.2989875377550304, -0.16123744170090526, 0.20768650872184122, -0.9450596386560098]
    # rp = [-0.14776994, -0.9777694,  -0.41016512, -0.05685512,  0.25548058,  0.54198472, -1.45115343]


    # new arm
    # rp = [-0.1945118169896666, -0.0007405090283826236, -0.34967920743357844, -1.2902111011768136, -0.18613783524299568, 0.20819152723285422, -0.9467171542759178]
    # rp = [-0.1465392499677634, -0.9546348297151802, -0.3480594089712433, -0.10860042509897395, 0.1898162556686379, 0.5134336458563873, -1.43955327415545]
    # rp = [-0.15940447864050794, -0.9709721832731079, -0.41924073906644443, -0.07158307340794688, 0.24907596109523372, 0.5333995582886921, -1.4374453331791757]
    # rp = [0.019202363834398186, -0.6653182233354258, -0.31505911105328116, -0.7078318963228405, 0.24373731512731367, 0.2360276386197912, -1.3505682977395823]
    rp = [0.006061704398361391, -0.6617032749359424, -0.29566629255483945, -0.7092797222263957, 0.1906545791906203, 0.21795662674634791, -1.1473480929124535] \
            + [0.0] * (22+5)
    rp = [0.4542808031756164, -0.20451686277306846, -0.37993986046606604, -2.0767408118051467, 0.5556177943690238, -0.44765794066788944, -0.6540148070338858] \
         + [0.0] * (22 + 5)

    closeEnough = False
    maxIter = 1300
    iter = 0
    dist2 = 1e30

    print(ll)
    print(ul)
    print(jr)
    print(rp)

    # rp = [-0.5] * 7
    for ind in range(len(relatedDofs)):
        p.resetJointState(a.robotId, relatedDofs[ind], rp[ind], 0.0)

    p.stepSimulation()
    time.sleep(ts)

    input("press enter")
    # ls = p.getLinkState(a.handId, endEffectorId)
    # newPos = ls[4]
    # print(newPos)
    # print(p.getEulerFromQuaternion(ls[5]))

    while (not closeEnough and iter < maxIter):
        # jointPoses = p.calculateInverseKinematics(a.handId, endEffectorId, tarPos, tarOrn,
        #                                               lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
        #                                               jointRanges=jr.tolist(),
        #                                               restPoses=rp)     # TODO: this seems worse....
        jointPoses = p.calculateInverseKinematics(a.robotId, endEffectorId, tarPos, tarOrn)
        # print(jointPoses)
        # jointPoses = np.array(jointPoses)[[0,1,9,10,11,12,13,14,15]]
        # jointPoses = np.array(jointPoses)[np.array([9, 10, 11, 12, 13, 14, 15])-2]
        # print(jointPoses)
        jointPoses = np.array(jointPoses)[np.array(range(7))]
        jointPoses = jointPoses.tolist()

        curQ = p.getJointStates(a.robotId, relatedDofs)
        curQ = np.array(curQ)[:, [0]].flatten()
        tarVel = (jointPoses - curQ) / ts


        # is the printlink order wrong or the actual orn from IK is actually wrong?

        for ji,i in enumerate(relatedDofs):
            # p.resetJointState(a.handId, i, jointPoses[ji])
            # p.setJointMotorControl2(bodyIndex=a.handId,
            #                         jointIndex=i,
            #                         controlMode=p.POSITION_CONTROL,
            #                         targetPosition=jointPoses[i],
            #                         targetVelocity=0,
            #                         force=500,
            #                         positionGain=0.03,
            #                         velocityGain=1)     # TODO
            p.setJointMotorControl2(bodyIndex=a.robotId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[ji],
                                    targetVelocity=tarVel[ji],
                                    force=5000,
                                    positionGain=0.8,  # 1.0 can be unstable
                                    velocityGain=0.8)
            # TODO: maxiter can also be useful

        # for i in range(p.getNumJoints(a.handId)):
        #     if i not in relatedDofs:
        #         p.setJointMotorControl2(bodyIndex=a.handId,
        #                                 jointIndex=i,
        #                                 controlMode=p.POSITION_CONTROL,
        #                                 targetPosition=0.0,
        #                                 targetVelocity=0,
        #                                 force=5000000,
        #                                 positionGain = 1,
        #                                 velocityGain = 1)

        p.stepSimulation()
        time.sleep(ts)

        joints_state = p.getJointStates(a.robotId, relatedDofs)
        rp = np.array(joints_state)[:, [0]]
        # vel = np.array(joints_state)[:, [1]]
        # print("tar q", jointPoses)
        # print("act q", rp)

        # for i in range(numJoints):
        #     jointInfo = p.getJointInfo(bodyId, i)
        #     qIndex = jointInfo[3]
        #     if qIndex > -1:
        #         p.resetJointState(bodyId, i, jointPoses[qIndex - 7])
        ls = p.getLinkState(a.robotId, endEffectorId)
        newPos = ls[4]
        diff = [tarPos[0] - newPos[0], tarPos[1] - newPos[1], tarPos[2] - newPos[2]]
        dist2 = np.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
        print("dist2=", dist2)
        # print(p.getEulerFromQuaternion(ls[5]))

        if iter <= 200:
            tarPos[2] += 0.001
        elif iter <= 500:
            tarPos[1] += 0.001
        elif iter <= 700:
            tarPos[2] -= 0.001
        elif iter >= 900:
            input("press enter to continue")

        closeEnough = False
        iter = iter + 1
    print("iter=", iter)

    ls = p.getLinkState(a.robotId, endEffectorId)
    newPos = ls[4]
    print(newPos)
    print(p.getEulerFromQuaternion(ls[5]))
    print(tarPos)
    print(p.getEulerFromQuaternion(tarOrn))


    input("press enter to continue")

    p.disconnect()
