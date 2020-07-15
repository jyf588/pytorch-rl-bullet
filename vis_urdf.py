import pybullet as p
import pybullet_data
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

l_arm = p.loadURDF(
    "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2_left_mod.urdf", [0.5, 0, 1], useFixedBase=1)


r_arm = p.loadURDF(
    "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2_original.urdf", [0, 0, 1], useFixedBase=1)

PI = 3.141

p.setJointMotorControlArray(bodyIndex=l_arm, jointIndices=range(
    35), controlMode=p.POSITION_CONTROL, targetPositions=[0.5] * 35)

p.setJointMotorControlArray(bodyIndex=r_arm, jointIndices=range(
    35), controlMode=p.POSITION_CONTROL, targetPositions=[0.5] * 35)

# p.setJointMotorControlArray(bodyIndex=l_arm, jointIndices=range(8, 29), controlMode=p.POSITION_CONTROL, targetPositions=[0.5] * 21)
# p.setJointMotorControlArray(bodyIndex=r_arm, jointIndices=range(8, 29), controlMode=p.POSITION_CONTROL, targetPositions=[0.5] * 21)
# p.setJointMotorControlArray(bodyIndex=l_arm, jointIndices=range(29, 35), controlMode=p.POSITION_CONTROL, targetPositions=[0.5] * 6)
# p.setJointMotorControlArray(bodyIndex=r_arm, jointIndices=range(29, 35), controlMode=p.POSITION_CONTROL, targetPositions=[0.5] * 6)

id1 = p.addUserDebugParameter("rh_THJ5", -1.4047, 1.4047, 0)
id2 = p.addUserDebugParameter("lh_THJ5", -1.4047, 1.4047, 0)

id3 = p.addUserDebugParameter("rh_THJ4'", 0, 1.22, 0)
id4 = p.addUserDebugParameter("lh_THJ4'", 0, 1.22, 0)

id5 = p.addUserDebugParameter("rh_THJ3'", -0.2094, 0.2094, 0)
id6 = p.addUserDebugParameter("lh_THJ3'", -0.2094, 0.2094, 0)

id7 = p.addUserDebugParameter("rh_THJ2", -0.6917, 0.6917, 0)
id8 = p.addUserDebugParameter("lh_THJ2", -0.6917, 0.6917, 0)

id9 = p.addUserDebugParameter("rh_THJ1", 0, 1.57079632679, 0)
id10 = p.addUserDebugParameter("lh_THJ1", 0, 1.57079632679, 0)

while True:
    p.stepSimulation()

    output1 = p.readUserDebugParameter(id1)
    output2 = p.readUserDebugParameter(id2)
    output3 = p.readUserDebugParameter(id3)
    output4 = p.readUserDebugParameter(id4)
    output5 = p.readUserDebugParameter(id5)
    output6 = p.readUserDebugParameter(id6)
    output7 = p.readUserDebugParameter(id7)
    output8 = p.readUserDebugParameter(id8)
    output9 = p.readUserDebugParameter(id9)
    output10 = p.readUserDebugParameter(id10)

    # p.setJointMotorControl2(bodyUniqueId=r_arm, jointIndex=29,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output1)
    # p.setJointMotorControl2(bodyUniqueId=r_arm, jointIndex=30,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output3)
    # p.setJointMotorControl2(bodyUniqueId=r_arm, jointIndex=31,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output5)
    # p.setJointMotorControl2(bodyUniqueId=r_arm, jointIndex=32,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output7)                                               
    # p.setJointMotorControl2(bodyUniqueId=r_arm, jointIndex=33,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output9)

    # p.setJointMotorControl2(bodyUniqueId=l_arm, jointIndex=29,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output2)
    # p.setJointMotorControl2(bodyUniqueId=l_arm, jointIndex=30,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output4)
    # p.setJointMotorControl2(bodyUniqueId=l_arm, jointIndex=31,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output6)
    # p.setJointMotorControl2(bodyUniqueId=l_arm, jointIndex=32,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output8)
    # p.setJointMotorControl2(bodyUniqueId=l_arm, jointIndex=33,
    #                         controlMode=p.POSITION_CONTROL, targetPosition=output10)

# p.resetJointState(arm, 0, 0)

# num = p.getNumJoints(l_arm)
# for i in range(0, num):
#     print(p.getJointInfo(l_arm,i))

# p.setJointMotorControl2(bodyUniqueId=arm, jointIndex=0,
#                         controlMode=p.VELOCITY_CONTROL, targetVelocity=0.5)

# p.resetJointState(arm, 0, 0)

# p.setJointMotorControl2(bodyUniqueId=arm, jointIndex=1,
#                         controlMode=p.POSITION_CONTROL, targetPosition=0.5)


# while True:
#     p.stepSimulation()

# for _ in range(500):
#     p.stepSimulation()

# # pybullet.setJointMotorControlArray(
# #     robot, range(0,6), pybullet.VELOCITY_CONTROL,
# #     targetVelocities=[10] * 6)

# p.resetJointState(arm, 0, 0)

# p.setTimeStep(0.001)

# p.setRealTimeSimulation(1)

# for _ in range(1000):
#     p.stepSimulation()


# while True:
#     p.stepSimulation()
