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

        self.handId = p.loadURDF(os.path.join(currentdir, "assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand.urdf"),
                                 list(self.baseInitPos), p.getQuaternionFromEuler(list(self.baseInitOri)),
                                 flags=p.URDF_USE_SELF_COLLISION)

        # for i in range(p.getNumJoints(self.handId)):
        #     print(p.getJointInfo(self.handId, i)[0:2], p.getJointInfo(self.handId, i)[8], p.getJointInfo(self.handId, i)[9])

        # # exclude fixed joints, actual DoFs are [0:4, 5:9, 10:14, 15:19]
        # self.activeDofs = []
        # for i in range(4):
        #     self.activeDofs += (np.arange(4) + 5 * i).tolist()

        # for ind in range(len(self.activeDofs)):
        #     p.resetJointState(self.handId, self.activeDofs[ind], self.initPos[ind], 0.0)

        self.tarBasePos = np.copy(self.baseInitPos)
        self.tarBaseOri = np.copy(self.baseInitOri)     # euler angles

        self.savedState = p.saveState()

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
    # p.setGravity(0, 0, -5)

    # floorId = p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0], useFixedBase=1)
    # p.changeDynamics(floorId, -1, lateralFriction=3.0)

    tarOrn = p.getQuaternionFromEuler([1.8, -1.57, 0.0])    # TODO: these two are equivalent
    print(tarOrn)
    # print(p.getQuaternionFromEuler([0.0, -1.57, 1.8]))
    tarPos = [0.27157567, -0.37579589,  1.17620252] # this is almost workspace size: 0.45*0.45

    print(p.getMatrixFromQuaternion(tarOrn))

    input("press enter")

    hh = p.loadURDF(
        os.path.join(currentdir, "assets/shadow_hand_arm/sr_description/robots/shadowhand_motor.urdf"),
        [0.37157567, -0.97579589,  1.17620252], tarOrn,
        flags=p.URDF_USE_SELF_COLLISION)
    hhls = p.getLinkState(hh, 0)
    print(p.getEulerFromQuaternion(hhls[5]))

    a = InmoovShadowHand()

    # relatedDofs = [3,5, 27,28,29,30,31, 33,34]  # the first two are waist
    relatedDofs = [ 27, 28, 29, 30, 31, 33, 34]  # the first two are waist
    endEffectorId = 34
    ll = np.array([p.getJointInfo(a.handId, i)[8] for i in relatedDofs])
    ul = np.array([p.getJointInfo(a.handId, i)[9] for i in relatedDofs])
    jr = ul - ll
    # rp = [0.0]*2 + [-0.4]*7
    rp = [-0.4] * 7

    closeEnough = False
    maxIter = 500
    iter = 0
    dist2 = 1e30

    print(ll)
    print(ul)
    print(jr)
    print(rp)

    # rp = [-0.5] * 7
    for ind in range(len(relatedDofs)):
        p.resetJointState(a.handId, relatedDofs[ind], rp[ind], 0.0)
    # ls = p.getLinkState(a.handId, endEffectorId)
    # newPos = ls[4]
    # print(newPos)
    # print(p.getEulerFromQuaternion(ls[5]))
    # input("press enter 1")

    while (not closeEnough and iter < maxIter):
        jointPoses = p.calculateInverseKinematics(a.handId, endEffectorId, tarPos, tarOrn,
                                                      lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
                                                      jointRanges=jr.tolist(),
                                                      restPoses=rp)
        print(jointPoses)
        # jointPoses = np.array(jointPoses)[[0,1,9,10,11,12,13,14,15]]
        jointPoses = np.array(jointPoses)[np.array([9, 10, 11, 12, 13, 14, 15])-2]
        jointPoses = jointPoses.tolist()


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
            p.setJointMotorControl2(bodyIndex=a.handId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[ji],
                                    targetVelocity=0,
                                    force=50000000,
                                    positionGain=1,
                                    velocityGain=1)

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
        p.stepSimulation()

        joints_state = p.getJointStates(a.handId, relatedDofs)
        rp = np.array(joints_state)[:, [0]]
        print("tar q", jointPoses)
        print("act q", rp)

        # for i in range(numJoints):
        #     jointInfo = p.getJointInfo(bodyId, i)
        #     qIndex = jointInfo[3]
        #     if qIndex > -1:
        #         p.resetJointState(bodyId, i, jointPoses[qIndex - 7])
        ls = p.getLinkState(a.handId, endEffectorId)
        newPos = ls[4]
        diff = [tarPos[0] - newPos[0], tarPos[1] - newPos[1], tarPos[2] - newPos[2]]
        dist2 = np.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
        print("dist2=", dist2)
        print(p.getEulerFromQuaternion(ls[5]))
        closeEnough = False
        iter = iter + 1
    print("iter=", iter)

    input("press enter to continue")

    p.disconnect()

# p.connect(p.GUI)
#
# # p.loadURDF("plane.urdf", [0, 0, -0.3])
# kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
# p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
# kukaEndEffectorIndex = 6
# numJoints = p.getNumJoints(kukaId)
# if (numJoints != 7):
#   exit()
#
# #lower limits for null space
# ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# #upper limits for null space
# ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# #joint ranges for null space
# jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# #restposes for null space
# rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# #joint damping coefficents
# jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#
# for i in range(numJoints):
#   p.resetJointState(kukaId, i, rp[i])
#
# p.setGravity(0, 0, 0)
# t = 0.
# prevPose = [0, 0, 0]
# prevPose1 = [0, 0, 0]
# hasPrevPose = 0
# useNullSpace = 1
#
# useOrientation = 1
# #If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
# #This can be used to test the IK result accuracy.
# useSimulation = 1
# useRealTimeSimulation = 0
# ikSolver = 0
# p.setRealTimeSimulation(useRealTimeSimulation)
# #trailDuration is duration (in seconds) after debug lines will be removed automatically
# #use 0 for no-removal
# trailDuration = 15
#
# i=0
# while 1:
#   i+=1
#   #p.getCameraImage(320,
#   #                 200,
#   #                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
#   #                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
#   if (useRealTimeSimulation):
#     dt = datetime.now()
#     t = (dt.second / 60.) * 2. * math.pi
#   else:
#     t = t + 0.01
#
#   if (useSimulation and useRealTimeSimulation == 0):
#     p.stepSimulation()
#
#   for i in range(1):
#     pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
#     #end effector points down, not up (in case useOrientation==1)
#     orn = p.getQuaternionFromEuler([0, -math.pi, 0])
#
#     if (useNullSpace == 1):
#       if (useOrientation == 1):
#         jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul,
#                                                   jr, rp)
#       else:
#         jointPoses = p.calculateInverseKinematics(kukaId,
#                                                   kukaEndEffectorIndex,
#                                                   pos,
#                                                   lowerLimits=ll,
#                                                   upperLimits=ul,
#                                                   jointRanges=jr,
#                                                   restPoses=rp)
#     else:
#       if (useOrientation == 1):
#         jointPoses = p.calculateInverseKinematics(kukaId,
#                                                   kukaEndEffectorIndex,
#                                                   pos,
#                                                   orn,
#                                                   jointDamping=jd,
#                                                   solver=ikSolver,
#                                                   maxNumIterations=100,
#                                                   residualThreshold=.01)
#       else:
#         jointPoses = p.calculateInverseKinematics(kukaId,
#                                                   kukaEndEffectorIndex,
#                                                   pos,
#                                                   solver=ikSolver)
#
#     if (useSimulation):
#       for i in range(numJoints):
#         p.setJointMotorControl2(bodyIndex=kukaId,
#                                 jointIndex=i,
#                                 controlMode=p.POSITION_CONTROL,
#                                 targetPosition=jointPoses[i],
#                                 targetVelocity=0,
#                                 force=500,
#                                 positionGain=0.03,
#                                 velocityGain=1)
#     else:
#       #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
#       for i in range(numJoints):
#         p.resetJointState(kukaId, i, jointPoses[i])
#
#   ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
#   if (hasPrevPose):
#     p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
#     p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
#   prevPose = pos
#   prevPose1 = ls[4]
#   hasPrevPose = 1
# p.disconnect()