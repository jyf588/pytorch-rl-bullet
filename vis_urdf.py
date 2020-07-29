import pybullet as p
import pybullet_data
import time
import pybullet_data
import sys

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

l_arm = p.loadURDF(
    "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2_left_mod.urdf", [0, 0.5, 1], useFixedBase=1)

l_arm_2 = p.loadURDF(
    "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2_left_mod.urdf", [0, -0.5, 1], useFixedBase=1)

r_arm = p.loadURDF(
    "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf", [0, -0.5, 1], useFixedBase=1)

pos = [0]*35
pos[30] = 2
p.setJointMotorControlArray(bodyUniqueId=l_arm, jointIndices=range(35), controlMode=p.POSITION_CONTROL, targetPositions=pos)
p.setJointMotorControlArray(bodyUniqueId=r_arm, jointIndices=range(35), controlMode=p.POSITION_CONTROL, targetPositions=[-0.3]*35)
p.resetJointState(2, 30, 1.2)
p.resetJointState(1, 30, 2)
while True:
    p.stepSimulation()


for i in range(35):
    print(p.getJointInfo(1, i))