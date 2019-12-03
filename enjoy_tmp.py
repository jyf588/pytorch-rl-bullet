import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

import gym
import my_pybullet_envs

import pybullet as p

import time

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

# TODO: seems some jittering motion during tarZ<0

# may need to refactor this into robot class
def planning(robot):
    l = 200
    n_wp = None
    for ind in range(len(Traj) - 1):
        c_wp = Traj[ind]
        n_wp = Traj[ind + 1]

        assert len(c_wp) == 16      # TODO: why 16?
        c_armq = np.array(c_wp[:7])
        n_armq = np.array(n_wp[:7])
        # armdq = wp[7:14]
        for t in range(l):
            armq = c_armq * (1. - t / l) + n_armq * (t / l)
            for ji, i in enumerate(robot.armDofs):
                # p.resetJointState(a.robotId, i, armq[ji])
                p.setJointMotorControl2(bodyIndex=robot.robotId,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=armq[ji],
                                        targetVelocity=0.,
                                        force=3000,  # TODO
                                        positionGain=0.2,   # TODO
                                        velocityGain=1)
            for i in range(len(robot.activeFinDofs)):
                p.setJointMotorControl2(robot.robotId,
                                        jointIndex=robot.activeFinDofs[i],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=robot.tarFingerPos[i],
                                        force=robot.maxForce)
            for ind in range(len(robot.zeroFinDofs)):
                p.setJointMotorControl2(robot.robotId,
                                        jointIndex=robot.zeroFinDofs[ind],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=0.0,
                                        force=robot.maxForce)
            p.stepSimulation()
            p.setTimeStep(1. / 480.)

    cps = p.getContactPoints(bodyA=robot.robotId)
    print(len(cps) == 0)
    for _ in range(1000):
        armq = np.array(n_wp[:7])
        for ji, i in enumerate(robot.armDofs):
            # p.resetJointState(a.robotId, i, armq[ji])
            p.setJointMotorControl2(bodyIndex=robot.robotId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=armq[ji],
                                    targetVelocity=0.,
                                    force=3000,  # TODO wrist force larger
                                    positionGain=1.0,
                                    velocityGain=1)
        for i in range(len(robot.activeFinDofs)):
            p.setJointMotorControl2(robot.robotId,
                                    jointIndex=robot.activeFinDofs[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=robot.tarFingerPos[i],
                                    force=robot.maxForce)
        for ind in range(len(robot.zeroFinDofs)):
            p.setJointMotorControl2(robot.robotId,
                                    jointIndex=robot.zeroFinDofs[ind],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.0,
                                    force=robot.maxForce)
        p.stepSimulation()
        p.setTimeStep(1. / 480.)

    newPos, newOrn = robot.get_palm_pos_orn()
    robot.palmInitPos = np.copy(newPos)
    robot.palmInitOri = np.copy(p.getEulerFromQuaternion(newOrn))
    robot.tarPalmPos = np.copy(robot.palmInitPos)
    robot.tarPalmOri = np.copy(robot.palmInitOri)  # euler angles
    robot.tarFingerPos = np.copy(robot.initFinPos)  # used for position control and as part of state
    robot.xyz_ll = robot.palmInitPos + np.array([-0.1, -0.1, -0.1])  # TODO: how to set these
    robot.xyz_ul = robot.palmInitPos + np.array([0.3, 0.1, 0.1])
    print(robot.xyz_ll)
    print(robot.xyz_ul)

# TODO: change this to read it OpenRave file
Traj = [[ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
[-0.01109587, 0.00853604, -0.00930493, -0.00154651, -0.00402582, 0.00332282, 0.00984393, -1.0533695,  0.81035606, -0.88334944, -0.1468154, -0.3821846, 0.31544643, 0.93451781, 0.02106739, 1.],
[-0.28142909, 0.13287289, -0.22060106, -0.15989818, -0.12374688, 0.0644147,
 0.13134642, -5.30498903, 2.11410234, -4.0864303, -3.57768696, -2.43370846,
 1.12146199, 1.92327538, 0.08503239, 1.],
[-0.31711221, 0.14699525, -0.24806974, -0.18410401, -0.14014211, 0.07193479,
 0.14301815, -5.63127171, 2.21415594, -4.33224482, -3.84098286, -2.59114887
, 1.18331816, 1.65390905, 0.00652565, 1.],
[-0.37071074, 0.16980463, -0.29307915, -0.22435413, -0.16714174, 0.08418859
, 0.15744151, -5.1333584,  2.36683955, -4.70736225, -4.24277718, -2.83140579
, 1.27771181, 1.24285113, 0.00995827, 1.],
[-0.43487499, 0.19870244, -0.35938357, -0.28468647, -0.20715413, 0.10211886
, 0.17036908, -4.46485649, 1.95593151, -5.21099755, -4.78222908, -3.15397643
, 1.40444539, 0.69096188, 0.01337004, 1.],
[ -4.35550062e-01,  1.98998070e-01, -3.60171573e-01, -2.85410606e-01
, -2.07631685e-01,  1.02331499e-01,  1.70473171e-01, -4.45729020e+00
,  1.95128074e+00, -5.20365874e+00, -4.78833474e+00, -3.15762737e+00
,  1.40587979e+00,  6.84715447e-01,  1.51325688e-04,  1.00000000e+00],
[ -4.46913905e-01,  2.03943254e-01, -3.73471328e-01, -2.97933211e-01
, -2.15881311e-01,  1.05927922e-01,  1.72106422e-01, -4.32793850e+00
,  1.87177213e+00, -5.07819584e+00, -4.89271593e+00, -3.22004314e+00
,  1.37446468e+00,  5.77927694e-01,  2.58703415e-03,  1.00000000e+00],
[ -4.60777782e-01,  2.09890633e-01, -3.89792745e-01, -3.14122580e-01
, -2.26243122e-01,  1.10350716e-01,  1.73773303e-01, -4.16469255e+00
,  1.77142975e+00, -4.91985769e+00, -5.02444829e+00, -3.12731887e+00
,  1.33481781e+00,  4.43158170e-01,  3.26491897e-03,  1.00000000e+00],
[-0.62627117, 0.27098661, -0.59941907, -0.61403706, -0.35743617, 0.17530051
, 0.11215021, -1.41724091, 0.28927909, -2.15063734, -5.09137957, -1.29769674
, 0.85587652, -2.52164592, 0.05929608, 1.],
[-0.53693192, 0.17535432, -0.58145177, -1.02816041, -0.33326575, 0.22049331,
 -0.38858924, 3.20741999, -2.20555798, 2.51066583, -3.2068202,  1.78202332
, 0.04969635, -7.51216634, 0.09981041, 1.],
[ -5.17295721e-01,  1.61970048e-01, -5.65901269e-01, -1.04666838e+00
, -3.22267824e-01,  2.20645879e-01, -4.31845748e-01,  3.47954136e+00
, -2.35235761e+00,  2.78494328e+00, -3.09593015e+00,  1.96323826e+00
,  2.25960595e-03, -7.21851729e+00,  5.87298094e-03,  1.00000000e+00],
[-0.49052675, 0.145178,  -0.54421731, -1.06886917, -0.30703737, 0.22044517,
 -0.48344868, 3.81940571, -2.2262356,  3.12750045, -2.95743469, 2.18956556,
 -0.05698623, -6.85176609, 0.00733502, 1.],
[ -4.86984579e-01,  1.43132161e-01, -5.41313101e-01, -1.07158864e+00
, -3.05025442e-01,  2.20389177e-01, -4.89746484e-01,  3.86213783e+00
, -2.21037792e+00,  3.17057115e+00, -2.94002126e+00,  2.17349359e+00
, -6.44353783e-02, -6.80565340e+00,  9.22253715e-04,  1.00000000e+00],
[-0.43053628, 0.11482047, -0.50079981, -1.10961054, -0.27723364, 0.2187799,
 -0.57718568, 4.48855335, -1.97791816, 2.82277159, -2.68475557, 1.93789246,
 -0.17363339, -6.12968179, 0.01351943, 1.],
[ -4.09637435e-01,  1.05608237e-01, -4.87666058e-01, -1.12216530e+00
, -2.68214392e-01,  2.17862781e-01, -6.05772082e-01,  4.30013240e+00
, -1.89614370e+00,  2.70042302e+00, -2.59495848e+00,  1.85501295e+00
, -2.12046952e-01, -5.89188921e+00,  4.75585166e-03,  1.00000000e+00],
[ -1.76335583e-01,  1.05845221e-03, -3.46301291e-01, -1.29353319e+00
, -1.69614821e-01,  2.10772305e-01, -9.51495442e-01, -6.98412061e-02
,  4.19220996e-04, -1.37159497e-01, -5.12329482e-01, -6.71793148e-02
,  8.34805529e-02, -3.76858647e-01,  1.10300611e-01,  1.00000000e+00],
[ -1.76693400e-01,  1.06060000e-03, -3.47004000e-01, -1.29615800e+00
, -1.69959000e-01,  2.11200000e-01, -9.53426200e-01,  0.00000000e+00
,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00
,  0.00000000e+00,  0.00000000e+00,  1.02465896e-02,  1.00000000e+00]]

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=10, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det


# from my_pybullet_envs.inmoov_shadow_hand_grasp_env_tmp import InmoovShadowHandGraspEnvTmp
# # env = gym.make(args.env_name)
# env = InmoovShadowHandGraspEnvTmp()
# env.seed(args.seed)
# # if str(env.__class__.__name__).find('TimeLimit') >= 0:
# #     env = TimeLimitMask(env)
# # if log_dir is not None:
# #     env = bench.Monitor(
# #         env,
# #         os.path.join(log_dir, str(rank)),
# #         allow_early_resets=allow_early_resets)
# env = VecNormalize(env, ret=False)


# must use vector version of make_env as to use vec_normalize
env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cuda',
    allow_early_resets=False)

# dont know why there are so many wrappers in make_vec_envs...
env_core = env.venv.venv.envs[0]
robot = env_core.robot
# print(robot.get_robot_observation())

# # Get a render function
# render_func = get_render_func(env)
#
# print(render_func)

# We need to use the same statistics for normalization as used in training
ori_env_name = 'InmoovShadowHandGraspBulletEnv-v0'  # TODO
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, ori_env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)


# if render_func is not None:
#     render_func('human')

obs = env.reset()

#
# if args.env_name.find('Bullet') > -1:
#     import pybullet as p
#
#     torsoId = -1
#     for i in range(p.getNumBodies()):
#         if (p.getBodyInfo(i)[0].decode() == "r_forearm_link"):
#             torsoId = i

reward_total = 0

timer = 0

resetted = True

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # where is the reset() called?
    # modify each reset so that reset to rest arm pose
    # after each reset(done), depends on whether planning is finished, run planning_traj or step
    if resetted:
        print("Done reset! Start planning")
        input("press enter")
        planning(robot)
        print(robot.get_raw_state_arm())
        resetted = False
        # input("press enter")

    # maybe in demo, only roll out for 300 steps.
    # then start to first go up in z, then +y, then go down in z, then drop.
    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    timer += 1

    reward_total += reward

    if timer >= 300:    # done grasping
        print(robot.tarPalmPos)
        print(robot.get_palm_pos_orn())
        print(p.getBasePositionAndOrientation(env_core.cylinderId))
        robot.xyz_ll = robot.palmInitPos + np.array([-1.0, -1.0, -0.2])    # TODO: enlarge this for transporting
        robot.xyz_ul = robot.palmInitPos + np.array([1.0, 1.0, 0.5])
        # input("press enter")

        print("tr:", reward_total)
        reward_total = 0.
        resetted = True
        # run traj here

        action = [0,0,0.001] + [0.0] * 20
        for t in range(200):
            robot.apply_action(action)
            for _ in range(2):
                p.stepSimulation()
            # p.stepSimulation()
            time.sleep(1./480.)

        print(robot.tarPalmPos)
        print(robot.get_palm_pos_orn())
        print(p.getBasePositionAndOrientation(env_core.cylinderId))
        # input("press enter")

        action = [0, 0.001, 0.0] + [0.0] * 20
        for t in range(250):
            robot.apply_action(action)
            for _ in range(2):
                p.stepSimulation()
            time.sleep(1./480.)
        print(robot.tarPalmPos)
        print(robot.get_palm_pos_orn())
        print(p.getBasePositionAndOrientation(env_core.cylinderId))
        # input("press enter")

        action = [0, 0, -0.001] + [0.0] * 20
        for t in range(200):
            robot.apply_action(action)
            for _ in range(2):
                p.stepSimulation()
            time.sleep(1./480.)
        print(robot.tarPalmPos)
        print(robot.get_palm_pos_orn())
        print(p.getBasePositionAndOrientation(env_core.cylinderId))
        # input("press enter")

        action = [-0.000] + [-0.00]*5 + [-0.02] * 17
        for t in range(200):
            robot.apply_action(action)
            for _ in range(2):
                p.stepSimulation()
            time.sleep(1./480.)
        print(robot.tarPalmPos)
        print(robot.get_palm_pos_orn())
        print(p.getBasePositionAndOrientation(env_core.cylinderId))
        # input("press enter")

        for _ in range(1000):
            p.stepSimulation()
            time.sleep(1. / 480.)

        obs = env.reset()
        timer = 0

    masks.fill_(0.0 if done else 1.0)

    # if args.env_name.find('Bullet') > -1:
    #     if robot is not None:
    #         distance = 0.8
    #         yaw = 0
    #         humanPos, humanOrn = robot.get_palm_pos_orn()
    #         p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    # if render_func is not None:
    #     render_func('human')
    # p.getCameraImage()