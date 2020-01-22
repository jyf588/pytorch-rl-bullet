import argparse
import os
# workaround to unpickle olf model files
import sys
from pdb import set_trace as bp
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
homedir = os.path.expanduser("~")

ts = 1/240



# may need to refactor this into robot class
def planning(robot):
    for ind in range(len(Traj) - 1):
            tar_armq = Traj[ind,0:7]
            # for ji, i in enumerate(robot.arm_dofs):
            #     p.resetJointState(robot.arm_id, i, tar_armq[ji])
            # for ind in range(len(robot.fin_actdofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_actdofs[ind], robot.init_fin_q[ind], 0.0)
            # for ind in range(len(robot.fin_zerodofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_zerodofs[ind], 0.0, 0.0)
            #print(tar_armq)
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
            # print(robot.tar_fin_q)
            time.sleep(ts)

    cps = p.getContactPoints(bodyA=robot.arm_id)
    print(len(cps) == 0)
    for _ in range(50):
        robot.tar_arm_q = tar_armq
        p.stepSimulation()
        #time.sleep(1. / 240.)    # TODO: stay still for a while
    

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
    default='./trained_models_0114_box_l_4/ppo/',    # TODO
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
is_cuda = True
device = 'cuda' if is_cuda else 'cpu'
args.det = not args.non_det
#np.random.seed(123)

p.connect(p.GUI)
p.resetSimulation()
p.setTimeStep(ts)
p.setGravity(0, 0, -10)

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

table_id = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/tabletop.urdf'), [0.27, 0.1, 0.0], useFixedBase=1)  # TODO

############################# HERE IS THE INPUT FROM VISION AND LANGUAGE MODULE
tx = np.random.uniform(low=0, high=0.25) # target object location
ty = np.random.uniform(low=-0.1, high=0.5)

destin_x = np.random.uniform(low=0, high=0.25) # destination location for target object
destin_y = np.random.uniform(low=-0.1, high=0.5)
destin_z = 0

#tx = 0.1
#ty = 0.0
#est_tx = tx
#est_ty = ty
est_tx = tx + np.random.uniform(low=-0.01, high=0.01)
est_ty = ty + np.random.uniform(low=-0.01, high=0.01)
OBJECTS = np.array([[est_tx,est_ty,0,0],[0.8, 0.8, 0, 0]])
# oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder.urdf'), [a_tx, a_ty, 0.1], useFixedBase=0)   # tar obj
oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box.urdf'), [tx, ty, 0.1], useFixedBase=0)   # tar obj
# oid2 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box.urdf'), [0.1, 0.2, 0.1], useFixedBase=0)
# oid3 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder.urdf'), [0.2, -0.15, 0.1], useFixedBase=0)
env_core.assign_estimated_obj_pos(est_tx, est_ty)
p.changeDynamics(oid1, -1, lateralFriction=1.0)
p.changeDynamics(table_id, -1, lateralFriction=1.0)

# print(oid1)

# # Get a render function
# render_func = get_render_func(env)
#
# print(render_func)

from my_pybullet_envs.inmoov_shadow_grasp_env_v2 import ImaginaryArmObjSession
sess = ImaginaryArmObjSession()
Qreach = np.array(sess.get_most_comfortable_q(OBJECTS[0,0],OBJECTS[0,1])) 
Qdestin = np.array(sess.get_most_comfortable_q(destin_x,destin_y)) #################################################### NEEDS TO HAVE Z and object orientation!!!

# send command to Openrave
file_path = homedir+'/container_data/PB_REACH.npz'
np.savez(file_path,OBJECTS,Qreach)

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

# obs_old = env.reset()
# print(obs_old)

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

# TODO: change this to read it OpenRave file
###

#Get planned trajectory from Openrave:

file_path = homedir+'/container_data/OR_REACH.npy'
while not os.path.exists(file_path):
    time.sleep(1)
if os.path.isfile(file_path):
    Traj = np.load(file_path)
    os.remove(file_path)
else:
    raise ValueError("%s isn't a file!" % file_path)        
        
###
    

print("Trajectory obtained from OpenRave!")
input("press enter")
planning(env_core.robot)
# print(robot.get_q_dq(robot.arm_dofs))
# print(robot.tar_arm_q)
# print(robot.tar_fin_q)
# input("press enter")


# env_core.robot.reset_with_certain_arm_q([-7.60999597e-01,   3.05809706e-02,  -5.82112526e-01,
#          -1.40855264e+00,  -6.49374902e-01,  -2.42410664e-01,
#           0.00000000e+00])
# env_core.robot.tar_arm_q = [-7.60999597e-01,   3.05809706e-02,  -5.82112526e-01,
#          -1.40855264e+00,  -6.49374902e-01,  -2.42410664e-01,
#           0.00000000e+00]
# p.stepSimulation()
# print("tar arm q after reset", robot.tar_arm_q)
# time.sleep(3)


obs = torch.Tensor([env_core.getExtendedObservation()])
if is_cuda:
    obs = obs.cuda()

# print("tar arm q after getting obs using env_core", env_core.robot.tar_arm_q)
# print("tar arm q after getting obs", robot.tar_arm_q)
#print("init obs", obs)
# input("press enter")
# # print(obs)
#
# print("diff", obs - obs_old)


for i in range(200):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # print(action)

    obs, reward, done, _ = env.step(action)
    control_steps += 1

# if control_steps >= 100:  # done grasping
# for _ in range(1000):
#     p.stepSimulation()
#     time.sleep(ts)

masks.fill_(0.0 if done else 1.0)

Qmove_init = np.concatenate((env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0],env_core.robot.get_q_dq(env_core.robot.arm_dofs)[1]))
file_path = homedir+'/container_data/PB_MOVE.npz'
np.savez(file_path,OBJECTS,Qmove_init,Qdestin)

file_path = homedir+'/container_data/OR_MOVE.npy'
while not os.path.exists(file_path):
    time.sleep(1)
if os.path.isfile(file_path):
    Traj = np.load(file_path)
    os.remove(file_path)
else:
    raise ValueError("%s isn't a file!" % file_path)

print("Trajectory obtained from OpenRave!")
input("press enter")
planning(env_core.robot)
#print(env_core.getExtendedObservation())
bp()
#input("press enter")