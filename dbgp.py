import pybullet as p

def switch_orientation(link):
    link[0] = (link[0][0], -link[0][1], link[0][2])
    link[1] = (link)

def get_links():
    for i in range(35):
        print(p.getLinkState(1, i), "\n")

def get_joints():
    for i in range(35):
        print(p.getJointInfo(1, i))