import pybullet as p
import pybullet_data
import time
import pybullet_data
import sys

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

l_arm = p.loadURDF(
    "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/left_experimental.urdf", [0, 0.5, 1], useFixedBase=1)

r_arm = p.loadURDF(
    "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf", [0, -0.5, 1], useFixedBase=1)

# joint_difs = {}

def check_big_difference(val1, val2, l_id=100, j_id=100):
    # if abs(val1-val2) > 1E-10:
    if val1 - val2 == 0:
        return False
    # elif val1 == -val2:
    #     if l_id not in joint_difs:
    #         joint_difs[l_id] = []
    #     if j_id not in joint_difs[l_id]:
    #         joint_difs[l_id].append(j_id)
    #     return False
    return True

def check_if_same(l, r):
    if check_big_difference(l[0], r[0]):
        return False
    if check_big_difference(l[1], -r[1]):
        return False
    if check_big_difference(l[2], r[2]):
        return False
    return True

def fixup_orientation(l, l_id, j_id):

    if l_id >= 0 and l_id < 5:
        return (-l[0], l[1], -l[2], l[3])
    elif l_id >= 5 and l_id < 35:
        return (l[0], -l[1], l[2], -l[3])



def check_if_same_r(l, r, l_id, j_id):

    l = fixup_orientation(l, l_id, j_id)
    if check_big_difference(l[0], r[0], l_id, j_id):
        return False
    if check_big_difference(l[1], r[1], l_id, j_id):
        return False
    if check_big_difference(l[2], r[2], l_id, j_id):
        return False
    if check_big_difference(l[3], r[3], l_id, j_id):
        return False
    return True


trans = open("/Users/michaelhayashi/Desktop/test/trans.txt", "w+")
rot = open("/Users/michaelhayashi/Desktop/test/rot.txt", "w+")

setting = "frame" #center of mass
if setting == "com":
    pos_val = 0
    or_val = 1
elif setting == "frame":
    pos_val = 4
    or_val = 5

pos_val = None
or_val = None

# angles = [-0.3, 0.3]
angles = [0.1, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, -0.1, -0.3, -0.4, -0.5, -1.0, -1.5, -2.0]
trans_differences = []
rot_differences = []

# broken = {
#     (0.0, 0.0, 0.0): [],
#     (1.0, 0.0, 0.0): [],
#     (-1.0, 0.0, 0.0): [],
#     (0.0, -1.0, 0.0): [],
#     (0.0, 1.0, 0.0): [],
#     (0.0, 0.0, -1.0): [],
#     (0.0, 0.0, 1.0): [],
# }

for angle in angles:
    trans.write("-------\n")
    trans.write("Angle:" + str(angle) + "\n")
    trans.write("-------\n\n")

    rot.write("-------\n")
    rot.write("Angle:" + str(angle) + "\n")
    rot.write("-------\n\n")
    for i in range(35):
        if i != 0:
            p.resetJointState(l_arm, i-1, 0)
            p.resetJointState(r_arm, i-1, 0)
        if angle > p.getJointInfo(l_arm, i)[9] or angle < p.getJointInfo(l_arm, i)[8]:
            continue
        p.resetJointState(l_arm, i, angle)
        p.resetJointState(r_arm, i, angle)
        for j in range(35):
            l_arm_link = p.getLinkState(l_arm, j)
            r_arm_link = p.getLinkState(r_arm, j)
            for k in range(1):
                if k == 0:
                    # com
                    pos_val = 0
                    or_val = 1
                elif k == 1:
                    # frame
                    pos_val = 4
                    or_val = 5
                is_same = check_if_same(l_arm_link[pos_val], r_arm_link[pos_val])
                is_same_r = check_if_same_r(l_arm_link[or_val], r_arm_link[or_val], j, i)
                if not is_same:
                    if i not in trans_differences:
                        trans_differences.append(i)
                    print("Broken here: ", i, j, " Pos: ", pos_val)
                    print("Left: ", l_arm_link[pos_val])
                    print("Righ: ", r_arm_link[pos_val])
                    trans.write("Broken here: Joint:" + str(i) + ", Link: " + str(j) + "\n")
                    trans.write("Left: " + str(l_arm_link[pos_val]) + "\n")
                    trans.write("Righ: " + str(r_arm_link[pos_val]) + "\n\n")
                if not is_same_r:
                    if i not in rot_differences:
                        rot_differences.append(i)
                    print("r_Broken here: ", i, j, " Pos: ", or_val)
                    print("Left: ", str(l_arm_link[or_val]))
                    print("Righ: ", str(r_arm_link[or_val]))
                    rot.write("r_Broken here: Joint:" + str(i) + ", Link: " + str(j) + "\n")
                    rot.write("Left: " + str(l_arm_link[or_val]) + "\n")
                    rot.write("Righ: " + str(r_arm_link[or_val]) + "\n\n")
                    # print("r_Broken here: ", i, j, " Pos: ", or_val)
                    # print("Left: ", p.getEulerFromQuaternion(list(l_arm_link[or_val])))
                    # print("Righ: ", p.getEulerFromQuaternion(list(r_arm_link[or_val])))
                    # rot.write("r_Broken here: Joint:" + str(i) + ", Link: " + str(j) + "\n")
                    # rot.write("Left: " + str(p.getEulerFromQuaternion(list(l_arm_link[or_val]))) + "\n")
                    # rot.write("Righ: " + str(p.getEulerFromQuaternion(list(r_arm_link[or_val]))) + "\n\n")
        # broken[p.getJointInfo(l_arm, i)[13]].extend(rot_differences)
        # rot_differences = []

print("Trans Differences: ", trans_differences)
print("Rot Differences: ", rot_differences)
# print("Joints: ", joint_difs)

# for key, value in broken.items():
#     broken[key] = list(dict.fromkeys(value))

# import pdb; pdb.set_trace()
p.disconnect()
