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

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

ts = 1/240

# may need to refactor this into robot class
def planning(robot):
    l = 20
    n_armq = None
    for ind in range(len(Traj) - 1):
        c_wp = Traj[ind]
        n_wp = Traj[ind + 1]

        assert len(c_wp) == 15      # TODO: q 7 + dq 7 + dt 1
        c_armq = np.array(c_wp[:7])
        n_armq = np.array(n_wp[:7])
        for t in range(l):
            tar_armq = c_armq * (1. - t / l) + n_armq * (t / l)

            # for ji, i in enumerate(robot.arm_dofs):
            #     p.resetJointState(robot.arm_id, i, tar_armq[ji])
            # for ind in range(len(robot.fin_actdofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_actdofs[ind], robot.init_fin_q[ind], 0.0)
            # for ind in range(len(robot.fin_zerodofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_zerodofs[ind], 0.0, 0.0)

            p.setJointMotorControlArray(
                bodyIndex=robot.arm_id,
                jointIndices=robot.arm_dofs,
                controlMode=p.POSITION_CONTROL,
                targetPositions=list(tar_armq),
                forces=[robot.maxForce * 3] * len(robot.arm_dofs))
            p.setJointMotorControlArray(
                bodyIndex=robot.arm_id,
                jointIndices=robot.fin_actdofs,
                controlMode=p.POSITION_CONTROL,
                targetPositions=list(robot.tar_fin_q),
                forces=[robot.maxForce] * len(robot.tar_fin_q))
            p.setJointMotorControlArray(
                bodyIndex=robot.arm_id,
                jointIndices=robot.fin_zerodofs,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[0.0] * len(robot.fin_zerodofs),
                forces=[robot.maxForce / 4.0] * len(robot.fin_zerodofs))
            p.stepSimulation()
            print(robot.tar_fin_q)
            time.sleep(ts)

    cps = p.getContactPoints(bodyA=robot.arm_id)   # TODO
    print(len(cps) == 0)
    for _ in range(100):
        robot.tar_arm_q = n_armq
        p.stepSimulation()
        time.sleep(1. / 240.)    # TODO: stay still for a while

# TODO: change this to read it OpenRave file
###
Traj = np.array(
    [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [ -6.60049762e-03,   1.41110095e-04,  -4.99070289e-03,
         -1.23043404e-02,  -5.85865581e-03,  -1.97250077e-03,
          1.45828301e-04,  -5.95041847e-01,   1.27212244e-02,
         -4.49917149e-01,  -1.10924932e+00,  -5.28164023e-01,
         -1.77823033e-01,   1.31465757e-02,   2.21849863e-02],
       [ -2.81502883e-01,   7.23930158e-02,  -2.36019150e-01,
         -2.75513085e-01,  -2.04477881e-01,  -9.38069720e-03,
          1.43452157e-01,  -5.27677111e+00,   1.53055267e+00,
         -4.48476478e+00,  -4.51279182e+00,  -3.71426894e+00,
          1.95867058e-02,   3.04782344e+00,   9.36345852e-02],
       [ -4.02723852e-01,   1.07926737e-01,  -3.39175920e-01,
         -3.77782796e-01,  -2.89549113e-01,  -1.02094500e-02,
          2.14239674e-01,  -6.32189926e+00,   1.86938653e+00,
         -5.38548598e+00,  -5.27258332e+00,  -4.42552064e+00,
         -9.88834674e-02,   3.72527104e+00,   2.09025630e-02],
       [ -4.52033774e-01,   1.21460513e-01,  -3.81196218e-01,
         -4.18754737e-01,  -3.24039605e-01,  -1.11208266e-02,
          2.43380757e-01,  -6.70055240e+00,   1.70480150e+00,
         -5.71182005e+00,  -5.54785811e+00,  -4.68320932e+00,
         -1.41805575e-01,   3.97071238e+00,   7.57306283e-03],
       [ -4.69039750e-01,   1.25678378e-01,  -3.95694279e-01,
         -4.32819218e-01,  -3.35922666e-01,  -1.14953000e-02,
          2.53257307e-01,  -6.82627278e+00,   1.65015599e+00,
         -5.82016945e+00,  -5.63925483e+00,  -4.76876710e+00,
         -1.56056570e-01,   3.88525284e+00,   2.51440763e-03],
       [ -4.82877630e-01,   1.28955017e-01,  -4.07321976e-01,
         -4.44240799e-01,  -3.45587846e-01,  -1.18208113e-02,
          2.61006862e-01,  -6.92688877e+00,   1.60642234e+00,
         -5.73634134e+00,  -5.71240105e+00,  -4.83724032e+00,
         -1.67461863e-01,   3.81685822e+00,   2.01231976e-03],
       [ -4.96832731e-01,   1.32170482e-01,  -4.18878050e-01,
         -4.55908935e-01,  -3.55475066e-01,  -1.21723462e-02,
          2.68683153e-01,  -6.82541412e+00,   1.56231546e+00,
         -5.65179785e+00,  -5.78617150e+00,  -4.90629789e+00,
         -1.78964490e-01,   3.74787993e+00,   2.02949292e-03],
       [ -5.27357579e-01,   1.39051121e-01,  -4.44151577e-01,
         -4.82600347e-01,  -3.77490434e-01,  -1.30448924e-02,
          2.85376988e-01,  -6.59801433e+00,   1.46347407e+00,
         -5.46233994e+00,  -5.95148752e+00,  -4.77505103e+00,
         -2.04741323e-01,   3.59330290e+00,   4.54799592e-03],
       [ -9.07503333e-01,   1.84945993e-01,  -7.57913457e-01,
         -8.92313344e-01,  -6.79059824e-01,  -5.09049236e-02,
          4.68008143e-01,  -2.34930153e+00,  -3.83267580e-01,
         -1.92252919e+00,  -3.69173893e+00,  -2.32284907e+00,
         -6.86352827e-01,   7.05202647e-01,   8.49742560e-02],
       [ -9.62546781e-01,   1.30764948e-01,  -7.94098144e-01,
         -1.11728340e+00,  -7.63782318e-01,  -1.09258926e-01,
          4.24661086e-01,   4.53466145e-01,  -1.48286450e+00,
          6.76237122e-01,  -4.05679826e+00,  -5.95207669e-01,
         -1.32350643e+00,  -2.19818483e+00,   5.80677496e-02],
       [ -9.36010491e-01,   8.76389366e-02,  -7.63081022e-01,
         -1.20962692e+00,  -7.69354500e-01,  -1.45887210e-01,
          3.53840370e-01,   1.66351925e+00,  -1.95759905e+00,
          1.79821568e+00,  -3.31008989e+00,   1.50675716e-01,
         -1.59858798e+00,  -3.45167903e+00,   2.50698840e-02],
       [ -9.32210714e-01,   8.32601543e-02,  -7.58991749e-01,
         -1.21687961e+00,  -7.68948175e-01,  -1.49385169e-01,
          3.46078931e-01,   1.77034050e+00,  -1.99950774e+00,
          1.89726188e+00,  -3.24417185e+00,   2.16520926e-01,
         -1.56251939e+00,  -3.56233519e+00,   2.21312313e-03],
       [ -8.88157383e-01,   4.89282355e-02,  -7.13123784e-01,
         -1.27482600e+00,  -7.58964982e-01,  -1.76918297e-01,
          2.66513558e-01,   2.71786081e+00,  -1.49826551e+00,
          2.77581627e+00,  -2.65946913e+00,   8.00577649e-01,
         -1.24258563e+00,  -4.54387162e+00,   1.96307287e-02],
       [ -8.86805695e-01,   4.81894871e-02,  -7.11752460e-01,
         -1.27613921e+00,  -7.58564922e-01,  -1.77531576e-01,
          2.64257493e-01,   2.74176071e+00,  -1.48562236e+00,
          2.76311494e+00,  -2.64472080e+00,   8.15309682e-01,
         -1.23451574e+00,  -4.56862954e+00,   4.95158301e-04],
       [ -8.75600471e-01,   4.22125614e-02,  -7.00445819e-01,
         -1.28691582e+00,  -7.54903758e-01,  -1.82541090e-01,
          2.44758143e-01,   2.62882310e+00,  -1.37907560e+00,
          2.65607786e+00,  -2.52043323e+00,   9.39459938e-01,
         -1.16650889e+00,  -4.77727023e+00,   4.17281385e-03],
       [ -8.44067679e-01,   2.66055375e-02,  -6.68446317e-01,
         -1.31683206e+00,  -7.40382887e-01,  -1.96179701e-01,
          1.87521949e-01,   2.28119150e+00,  -1.05111560e+00,
          2.32660862e+00,  -2.13786549e+00,   1.32160500e+00,
         -9.57178019e-01,  -4.13505640e+00,   1.28442765e-02],
       [ -7.54673345e-01,   1.04938743e-02,  -5.73522833e-01,
         -1.39212294e+00,  -6.73105802e-01,  -2.24202545e-01,
          2.72834979e-02,   6.04071100e-01,   5.31100774e-01,
          7.37111053e-01,  -2.92198533e-01,   8.49808421e-01,
          5.27217768e-02,  -1.03674798e+00,   6.19661684e-02],
       [ -7.48410650e-01,   1.60000509e-02,  -5.65880849e-01,
         -1.39515230e+00,  -6.64295430e-01,  -2.23655953e-01,
          1.65350342e-02,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   2.07349596e-02],
       [ -7.54705123e-01,   2.32905107e-02,  -5.73996687e-01,
         -1.40185247e+00,  -6.56835166e-01,  -2.33033309e-01,
          8.26751709e-03,  -6.50008762e-01,   7.52860881e-01,
         -8.38094940e-01,  -6.91903790e-01,   7.70395970e-01,
         -9.68367470e-01,  -8.53758242e-01,   1.93673494e-02],
       [ -7.54705123e-01,   2.32905107e-02,  -5.73996687e-01,
         -1.40185247e+00,  -6.56835166e-01,  -2.33033309e-01,
          8.26751709e-03,  -6.50008762e-01,   7.52860881e-01,
         -8.38094940e-01,  -6.91903790e-01,   7.70395970e-01,
         -9.68367470e-01,  -8.53758242e-01,   6.93889390e-18],
       [ -7.60999597e-01,   3.05809706e-02,  -5.82112526e-01,
         -1.40855264e+00,  -6.49374902e-01,  -2.42410664e-01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   1.93673494e-02]]
    )
###

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='ShadowHandDemoBulletEnv-v1',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models_0114_cyl_l_2/ppo/',    # TODO
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    type=int,
    default=0,
    help='whether to use a non-deterministic policy, 1 true 0 false')
parser.add_argument(
    '--iter',
    type=int,
    default=-1,
    help='which iter pi to test')
args = parser.parse_args()

# TODO
is_cuda = False
device = 'cuda' if is_cuda else 'cpu'

args.det = not args.non_det

p.connect(p.GUI)

p.resetSimulation()
# p.setPhysicsEngineParameter(numSolverIterations=200)
p.setTimeStep(ts)
p.setGravity(0, 0, -10)

# ignore, not working
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
    device=device,
    allow_early_resets=False)

# dont know why there are so many wrappers in make_vec_envs...
env_core = env.venv.venv.envs[0]
robot = env_core.robot

table_id = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/tabletop.urdf'), [0.27, 0.1, 0.0], useFixedBase=1)  # TODO
tx = 0.1
ty = 0.0
oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box.urdf'), [tx, ty, 0.1], useFixedBase=0)   # tar obj
oid2 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box.urdf'), [0.1, 0.2, 0.1], useFixedBase=0)
oid3 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder.urdf'), [0.2, -0.15, 0.1], useFixedBase=0)
env_core.assign_estimated_obj_pos(tx, ty)

# print(robot.get_robot_observation())

# # Get a render function
# render_func = get_render_func(env)
#
# print(render_func)

# We need to use the same statistics for normalization as used in training
ori_env_name = 'InmoovHandGraspBulletEnv-v1'
if args.iter >= 0:
    path = os.path.join(args.load_dir, ori_env_name + "_" + str(args.iter) + ".pt")
else:
    path = os.path.join(args.load_dir, ori_env_name + ".pt")

if is_cuda:
    actor_critic, ob_rms = torch.load(path)
else:
    actor_critic, ob_rms = torch.load(path, map_location='cpu')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)


# if render_func is not None:
#     render_func('human')

env.reset()

#
# if args.env_name.find('Bullet') > -1:
#     import pybullet as p
#
#     torsoId = -1
#     for i in range(p.getNumBodies()):
#         if (p.getBodyInfo(i)[0].decode() == "r_forearm_link"):
#             torsoId = i

control_steps = 0

print("Done reset! Start planning")
input("press enter")
planning(robot)
print(robot.get_q_dq(robot.arm_dofs))
print(robot.tar_arm_q)
print(robot.tar_fin_q)
input("press enter")

# robot.reset_with_certain_arm_q
robot.reset_with_certain_arm_q([-7.60999597e-01,   3.05809706e-02,  -5.82112526e-01,
         -1.40855264e+00,  -6.49374902e-01,  -2.42410664e-01,
          0.00000000e+00])

sample = torch.from_numpy(env.action_space.sample() * 0.0)
obs, _, _, _ = env.step(sample)
print(obs)

print(oid1)

for i in range(100):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    obs, reward, done, _ = env.step(action)
    control_steps += 1

# if control_steps >= 100:  # done grasping
for _ in range(1000):
    p.stepSimulation()
    time.sleep(ts)

masks.fill_(0.0 if done else 1.0)

input("press enter")

# while True:
#
#     # where is the reset() called?
#     # modify each reset so that reset to rest arm pose
#     # after each reset(done), depends on whether planning is finished, run planning_traj or step
#     if resetted:
#
#
#     obs = env_core.getExtendedObservation()
#     with torch.no_grad():
#         value, action, _, recurrent_hidden_states = actor_critic.act(
#             obs, recurrent_hidden_states, masks, deterministic=args.det)
#
#     obs, reward, done, _ = env.step(action)
#     control_steps += 1
#
#     if control_steps >= 100:    # done grasping
#         for _ in range(1000):
#             p.stepSimulation()
#             time.sleep(ts)
#
#         obs = env.reset()
#         timer = 0
#
#     masks.fill_(0.0 if done else 1.0)
#
#     # if args.env_name.find('Bullet') > -1:
#     #     if robot is not None:
#     #         distance = 0.8
#     #         yaw = 0
#     #         humanPos, humanOrn = robot.get_palm_pos_orn()
#     #         p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
#
#     # if render_func is not None:
#     #     render_func('human')
#     # p.getCameraImage()