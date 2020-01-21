import argparse
import pickle
import os
import sys
from pdb import set_trace as bp
import numpy as np
import torch
#import gym
import my_pybullet_envs
import pybullet as p
import time
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
import inspect
from NLP_module import NLPmod
from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSession ,URDFWriter

from my_pybullet_envs.inmoov_shadow_place_env_v3 import InmoovShadowHandPlaceEnvV3

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
homedir = os.path.expanduser("~")

# TODO: main module depends on the following code/model:
# demo env: especially observation
# the settings of inmoov hand v2
# obj sizes & frame representation

# constants
DEMO_ENV_NAME = 'ShadowHandDemoBulletEnv-v1'
GRASP_PI = '0114_cyl_s_1'
GRASP_DIR = "./trained_models_%s/ppo/" % GRASP_PI
GRASP_PI_ENV_NAME = 'InmoovHandGraspBulletEnv-v1'
PLACE_PI = '0114_cyl_s_1_place_v3_2'
PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI
PLACE_PI_ENV_NAME = 'InmoovHandPlaceBulletEnv-v3'

GRASP_END_STEP = 23

STATE_NORM = True
OBJ_MU = 1.0
FLOOR_MU = 1.0
# HAND_MU = 3.0

sys.path.append('a2c_ppo_acktr')
parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--non-det', type=int, default=0)
args = parser.parse_args()
args.det = not args.non_det

IS_CUDA = True
DEVICE = 'cuda' if IS_CUDA else 'cpu'
TS = 1./240
TABLE_OFFSET = [0.25, 0.1, 0.0]     # TODO: chaged to 0.2 for vision, 0.25 may collide, need to change OR reaching.
HALF_OBJ_HEIGHT_L = 0.09
HALF_OBJ_HEIGHT_S = 0.07    # todo: 0.065 now
PLACE_CLEARANCE = 0.16

# test only one obj
g_tx = 0.2
g_ty = -0.2
p_tx = 0.15
p_ty = 0.4
p_tz = 0.0  # TODO: depending on placing on floor or not


def planning(robot, Traj, i_g_obs, recurrent_hidden_states, masks):
    g_obs = i_g_obs
    for ind in range(0, len(Traj), 3):
            tar_armq = Traj[ind, 0:7]
            # for ji, i in enumerate(robot.arm_dofs):
            #     p.resetJointState(robot.arm_id, i, tar_armq[ji])
            # for ind in range(len(robot.fin_actdofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_actdofs[ind], robot.init_fin_q[ind], 0.0)
            # for ind in range(len(robot.fin_zerodofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_zerodofs[ind], 0.0, 0.0)

            with torch.no_grad():
                value, action, _, recurrent_hidden_states = g_actor_critic.act(
                    g_obs, recurrent_hidden_states, masks, deterministic=args.det)
                # print(g_obs)
                # print(action)
                # input("press enter")
            action[:7] = 0
            env_core.robot.tar_arm_q = tar_armq
            g_obs, _, _ , _ = env.step(action)
            masks.fill_(0.0 if done else 1.0)

            # #print(tar_armq)
            # p.setJointMotorControlArray(
            #     bodyIndex=robot.arm_id,
            #     jointIndices=robot.arm_dofs,
            #     controlMode=p.POSITION_CONTROL,
            #     targetPositions=list(tar_armq),
            #     forces=[robot.maxForce * 3] * len(robot.arm_dofs))
            # p.setJointMotorControlArray(
            #     bodyIndex=robot.arm_id,
            #     jointIndices=robot.fin_actdofs,
            #     controlMode=p.VELOCITY_CONTROL)
            # p.setJointMotorControlArray(
            #     bodyIndex=robot.arm_id,
            #     jointIndices=robot.fin_zerodofs,
            #     controlMode=p.VELOCITY_CONTROL)
            # p.stepSimulation()
            # # print(robot.tar_fin_q)
            # time.sleep(TS)

    # for _ in range(50):
    #     robot.tar_arm_q = tar_armq
    #     p.stepSimulation()
    #     #time.sleep(1. / 240.)    # TODO: stay still for a while


def get_relative_state_for_reset(oid):
    obj_pos, obj_quat = p.getBasePositionAndOrientation(oid)  # w2o
    hand_pos, hand_quat = env_core.robot.get_link_pos_quat(env_core.robot.ee_id)  # w2p
    inv_h_p, inv_h_q = p.invertTransform(hand_pos, hand_quat)  # p2w
    o_p_hf, o_q_hf = p.multiplyTransforms(inv_h_p, inv_h_q, obj_pos, obj_quat)  # p2w*w2o

    fin_q, _ = env_core.robot.get_q_dq(env_core.robot.all_findofs)

    state = {'obj_pos_in_palm': o_p_hf, 'obj_quat_in_palm': o_q_hf,
             'all_fin_q': fin_q, 'fin_tar_q': env_core.robot.tar_fin_q}
    return state


def load_policy_params(dir, env_name, iter=None):
    if iter is not None:
        path = os.path.join(dir, env_name + "_" + str(iter) + ".pt")
    else:
        path = os.path.join(dir, env_name + ".pt")
    if IS_CUDA:
        actor_critic, ob_rms = torch.load(path)
    else:
        actor_critic, ob_rms = torch.load(path, map_location='cpu')
    vec_norm = get_vec_normalize(env)
    if not STATE_NORM: assert ob_rms is None
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    return actor_critic, ob_rms, recurrent_hidden_states, masks     # probably only first one is used


# set up world
p.connect(p.GUI)
p.resetSimulation()

desired_obj_pos = [p_tx, p_ty, PLACE_CLEARANCE + p_tz]
# TODO: write an imagivary session?
a = InmoovShadowHandPlaceEnvV3(renders=False, is_box=False, is_small=True, place_floor=True, grasp_pi_name='0114_cyl_s_1')
Qdestin =a.get_optimal_init_arm_q(desired_obj_pos)
print("place arm q", Qdestin)
p.resetSimulation()
# arm_q = [-0.8095155039980575, -0.42793360197051106, -0.7269143588514168, -1.2848182939515076, -0.7268314292697703, -0.7463880732365392, -0.7885470027289124]



p.setTimeStep(TS)
p.setGravity(0, 0, -10)


env = make_vec_envs(
    DEMO_ENV_NAME,
    args.seed,
    1,
    None,
    None,
    device=DEVICE,
    allow_early_resets=False)
env_core = env.venv.venv.envs[0].env.env

# env.reset()  # TODO: do we need to call reset After loading the objects?
# env_core.robot.change_hand_friction(HAND_MU)

oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder_small.urdf'), [g_tx, g_ty, HALF_OBJ_HEIGHT_S+0.001],
                  useFixedBase=0)
p.changeDynamics(oid1, -1, lateralFriction=OBJ_MU)
table_id = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/tabletop.urdf'), TABLE_OFFSET,
                      useFixedBase=1)
p.changeDynamics(table_id, -1, lateralFriction=FLOOR_MU)


# ready to grasp
sess = ImaginaryArmObjSession()
Qreach = np.array(sess.get_most_comfortable_q_and_refangle(g_tx, g_ty)[0])
env_core.robot.reset_with_certain_arm_q(Qreach)
env_core.assign_estimated_obj_pos(g_tx, g_ty)   # env_core.tx = will not work
g_actor_critic, g_ob_rms, recurrent_hidden_states, masks = load_policy_params(GRASP_DIR, GRASP_PI_ENV_NAME) # latter 2 returns dummy
# g_obs = torch.Tensor([env_core.getExtendedObservation()])
# print(g_obs)
g_obs = env.reset()
print("gobs", g_obs)
if IS_CUDA:
    g_obs = g_obs.cuda()
input("ready to grasp")

# grasp!
control_steps = 0
for i in range(GRASP_END_STEP):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = g_actor_critic.act(
            g_obs, recurrent_hidden_states, masks, deterministic=args.det)
        # print(g_obs)
        # print(action)
        # input("press enter")

    g_obs, reward, done, _ = env.step(action)

    print(g_obs)
    print(action)
    print(control_steps)
    control_steps += 1
    # input("press enter g_obs")

    masks.fill_(0.0 if done else 1.0)

import copy
final_g_obs = copy.copy(g_obs)

#   if control_steps >= 100:  # done grasping
#      for _ in range(1000):
#       p.stepSimulation()
#       time.sleep(ts)
# masks.fill_(0.0 if done else 1.0)

state = get_relative_state_for_reset(oid1)
print("after grasping", state)
input("after grasping")

##########################----- SEND MOVE COMMAND TO OPENRAVE
Qmove_init = np.concatenate((env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0],env_core.robot.get_q_dq(env_core.robot.arm_dofs)[1])) # OpenRave initial condition
file_path = homedir+'/container_data/PB_MOVE.npz'
np.savez(file_path, [], Qmove_init, Qdestin)

# Wait for command from OpenRave
file_path = homedir+'/container_data/OR_MOVE.npy'
while not os.path.exists(file_path):
    time.sleep(1)
if os.path.isfile(file_path):
    Traj2 = np.load(file_path)
    os.remove(file_path)
else:
    raise ValueError("%s isn't a file!" % file_path)
print("Trajectory obtained from OpenRave!")
input("press enter")


##################################---- EXECUTE PLANNED MOVING TRAJECTORY
planning(env_core.robot, Traj2, final_g_obs, recurrent_hidden_states, masks)
state = get_relative_state_for_reset(oid1)
print("after moving", state)
input("after moving")


# p.disconnect()
#
#
# p.connect(p.GUI)
#
# env = make_vec_envs(
#     DEMO_ENV_NAME,
#     args.seed,
#     1,
#     None,
#     None,
#     device=DEVICE,
#     allow_early_resets=False)
# env_core = env.venv.venv.envs[0].env.env

# o_pos_pf = state['obj_pos_in_palm']
# o_quat_pf = state['obj_quat_in_palm']
# all_fin_q_init = state['all_fin_q']
# tar_fin_q_init = state['fin_tar_q']
# env_core.robot.reset_with_certain_arm_q_finger_states(Qdestin, all_fin_q_init, tar_fin_q_init)
# p_pos, p_quat = env_core.robot.get_link_pos_quat(env_core.robot.ee_id)
# o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)
#
# # table_id = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/tabletop.urdf'), TABLE_OFFSET,
# #                       useFixedBase=1)
# # p.changeDynamics(table_id, -1, lateralFriction=FLOOR_MU)
# # oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder_small.urdf'), o_pos, o_quat,
# #                   useFixedBase=0)
# # p.changeDynamics(oid1, -1, lateralFriction=OBJ_MU)
# p.resetBasePositionAndOrientation(oid1, o_pos, o_quat)
#
# state = get_relative_state_for_reset(oid1)
#
# print("before releasing", state)
# # print(env_core.withVel)
# input("before realsing")

p_actor_critic, p_ob_rms, recurrent_hidden_states, masks = load_policy_params(PLACE_DIR, PLACE_PI_ENV_NAME) # latter 2 returns dummy
p_obs = env.reset()
print("pobs", p_obs)
if IS_CUDA:
    p_obs = p_obs.cuda()
input("ready to place")

# place!
control_steps = 0
for i in range(95):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = p_actor_critic.act(
            p_obs, recurrent_hidden_states, masks, deterministic=args.det)
        # print(g_obs)

    p_obs, reward, done, _ = env.step(action)

    # print(action)
    # print(g_obs)
    # input("press enter g_obs")

    masks.fill_(0.0 if done else 1.0)
    # g_obs = torch.Tensor([env_core.getExtendedObservation(withVel=True)])
    control_steps += 1
#   if control_steps >= 100:  # done grasping
#      for _ in range(1000):
#       p.stepSimulation()
#       time.sleep(ts)
# masks.fill_(0.0 if done else 1.0)

for test_t in range(400):
    open_up_q = np.array([0.2, 0.2, 0.2] * 4 + [-0.4, 1.9, -0.0, 0.5, 0.0])
    devi = open_up_q - env_core.robot.get_q_dq(env_core.robot.fin_actdofs)[0]
    env_core.robot.apply_action(np.array([0.0] * 7 + list(devi / 150.)))
    p.stepSimulation()
    time.sleep(TS)





