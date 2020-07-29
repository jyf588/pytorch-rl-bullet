from openravepy.misc import InitOpenRAVELogging
InitOpenRAVELogging()
import time
from openravepy import *
import numpy as np
from pdb import set_trace as bp
import os.path
import time
from scipy.interpolate import interp1d
import sys
MODE = int(sys.argv[1])  # 0 reach, 1 move, 2 retract
IS_LARGE_OBS = (len(sys.argv) > 2) and bool(sys.argv[2] == "l")
MAX_OBJECTS = 6
print("mode", MODE)
print("is large obstacles", IS_LARGE_OBS)
env = Environment()
# env.SetViewer('qtcoin')
urdf_module = RaveCreateModule(env, "urdf")
if MODE == 0:
    urdf_path = "/data/or_planning_scripts/inmoov_arm_v2_2_reaching_BB.urdf"
elif MODE == 1:
    urdf_path = "/data/or_planning_scripts/inmoov_arm_v2_2_moving_BB.urdf"
elif MODE == 2:
    urdf_path = "/data/or_planning_scripts/inmoov_arm_v2_2_retract_BB.urdf"
srdf_path = "/data/or_planning_scripts/inmoov_shadow_hand_v2.srdf"
np.set_printoptions(formatter={"int_kind": "{:,}".format})
inmoov_name = urdf_module.SendCommand(
    "LoadURI {} {}".format(urdf_path, srdf_path)
)
robot = env.GetRobots()[0]
baseT_translation = np.array([-0.30, 0.348, 0.272])
BaseT = np.array(
    [
        [1, 0, 0, baseT_translation[0]],
        [0, 1, 0, baseT_translation[1]],
        [0, 0, 1, baseT_translation[2]],
        [0, 0, 0, 1],
    ]
)
robot.SetTransform(BaseT)
if MODE == 1:
    table = env.ReadKinBodyXMLFile('tabletop_move.kinbody.xml')
    env.Add(table)
else:
    table = env.ReadKinBodyXMLFile('tabletop_reach.kinbody.xml')
    env.Add(table)      # move towards -x 4cm as clearance
if MODE == 0:
    ceil = env.ReadKinBodyXMLFile('ceiling_reach.kinbody.xml')
    env.Add(ceil)
torso = env.ReadKinBodyXMLFile('torso.kinbody.xml')
env.Add(torso)
manip = robot.SetActiveManipulator("right_arm")
# print(manip.GetArmIndices()) # [ 3  2  4  0  1 6 5] out of 7
RaveSetDebugLevel(DebugLevel.Debug)  # set output level to debug
#########################################################################################################################################
# listen for file and open it when it appears
if MODE == 0:
    file_path = "/data/PB_REACH.npz"
elif MODE == 1:
    file_path = "/data/PB_MOVE.npz"
else:
    file_path = "/data/PB_RETRACT.npz"
Qinit = [0.0] * 7
Qdestin = [0.0] * 7
while not os.path.exists(file_path):
    time.sleep(0.02)
if os.path.isfile(file_path):
    time.sleep(0.3)         # TODO: wait for file write
    try:
        loaded_data = np.load(file_path)
        OBJECTS = loaded_data["arr_0"]
        Qinit = loaded_data["arr_1"]
        Qdestin = loaded_data["arr_2"]
        os.remove(file_path)
    except Exception:
        os.remove(file_path)
else:
    raise ValueError("%s isn't a file!" % file_path)
# Calculate transformation matrices for obstacles and target object
def get_transf_mat_from_pos_orient(xyz, theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, xyz[0]],
            [np.sin(theta), np.cos(theta), 0, xyz[1]],
            [0, 0, 1, xyz[2]],
            [0, 0, 0, 1],
        ]
    )
Tobj = []
start = 1 if MODE == 1 else 0
for i in range(start, OBJECTS.shape[0]):
    xyz = OBJECTS[i, 0:3]
    theta = OBJECTS[i, -1]
    Tobj.append(get_transf_mat_from_pos_orient(xyz, theta))
print(Tobj)
assert len(Tobj) <= MAX_OBJECTS
for i in range(0, len(Tobj)):  # TODO: <=6 objs
    if IS_LARGE_OBS:
        kinbody = "obs%s_l.kinbody.xml" % i
    else:
        kinbody = "obs%s.kinbody.xml" % i
    exec "obs%s=env.ReadKinBodyXMLFile(kinbody)" % i
    exec "env.Add(obs%s)" % i
    exec "obs%s.SetTransform(Tobj[i])" % i
    print("aaa")
# raw_input('press any key 3')
start_time = time.time()
try:
    with robot:
        robot.SetDOFValues(Qinit[0:7], [3, 2, 4, 0, 1, 6, 5])
        manipprob = interfaces.BaseManipulation(
            robot, maxvelmult=1.0
        ) # create the interface for basic manipulation programs
        if MODE == 2 or MODE == 0:   # start may be in collision, need jitter
            res = manipprob.MoveManipulator(
                goal=Qdestin, outputtrajobj=True, jitter=0.25, execute=True
            ) # call motion planner
        elif MODE == 1:   # start may be in collision, need jitter, transport Bounding Box larger
            res = manipprob.MoveManipulator(
                goal=Qdestin, outputtrajobj=True, jitter=0.3, execute=True
            ) # call motion planner
    traj = res.GetAllWaypoints2D()[:,0:-1]
    spec = res.GetConfigurationSpecification()
    # raw_input('press any key 4')
except PlanningError:
    # print(e)
    traj = []
end_time = time.time()
print("Duration: %.2f sec" % (end_time - start_time))
if len(traj) == 0:
    Traj_I = np.array([])
    Traj_S = np.array([])
else:
    #interpolated trajectory resolution
    if MODE == 0:
        n = 400
    elif MODE == 1:
        n = 500
    elif MODE == 2:
        n = 300
    t = np.cumsum(traj[:, -1])
    T = np.linspace(t[0], t[-1], n)
    Traj_I = np.zeros((n, traj.shape[1] - 8))  # TODO
    Traj_S = np.zeros((n, traj.shape[1] - 8))  # TODO
    for i in range(traj.shape[1] - 8):
        f = interp1d(t, traj[:, i], kind="linear")
        Traj_I[:, i] = f(T)
    idx = 0
    for ts in T:
        with env:
            trajdata = res.Sample(ts)
            Traj_S[idx, :] = spec.ExtractJointValues(
                trajdata, robot, [3, 2, 4, 0, 1, 6, 5], 0
            )
            idx += 1
# for test in [0,1,2,3,4,10]:
#     print(Traj_I[test, :])
#     print(Traj_S[test, :])
#     print(T[test])
if MODE == 0:
    save_path = "/data/OR_REACH.npz"
elif MODE == 1:
    save_path = "/data/OR_MOVE.npz"
else:
    save_path = "/data/OR_RETRACT.npz"
np.savez(save_path, Traj_I, Traj_S)
# bp()
# raw_input('press any key 4')
# keep rerunning this script until it's terminated from the terminal
os.execv(sys.executable, ["python"] + sys.argv)