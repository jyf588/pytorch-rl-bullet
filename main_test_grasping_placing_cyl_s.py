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
g_tx = 0.1
g_ty = 0.35
p_tx = 0.1
p_ty = 0.3
p_tz = 0.0  # TODO: depending on placing on floor or not


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
for i in range(95):
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

#   if control_steps >= 100:  # done grasping
#      for _ in range(1000):
#       p.stepSimulation()
#       time.sleep(ts)
# masks.fill_(0.0 if done else 1.0)

state = get_relative_state_for_reset(oid1)

print("after grasping", state)

p.disconnect()


p.connect(p.GUI)

env = make_vec_envs(
    DEMO_ENV_NAME,
    args.seed,
    1,
    None,
    None,
    device=DEVICE,
    allow_early_resets=False)
env_core = env.venv.venv.envs[0].env.env

desired_obj_pos = [p_tx, p_ty, PLACE_CLEARANCE + p_tz]

a = InmoovShadowHandPlaceEnvV3(renders=False, is_box=False, is_small=True, place_floor=True, grasp_pi_name='0114_cyl_s_1')
arm_q =a.get_optimal_init_arm_q(desired_obj_pos)
print("place arm q", arm_q)


o_pos_pf = state['obj_pos_in_palm']
o_quat_pf = state['obj_quat_in_palm']
all_fin_q_init = state['all_fin_q']
tar_fin_q_init = state['fin_tar_q']
env_core.robot.reset_with_certain_arm_q_finger_states(arm_q, all_fin_q_init, tar_fin_q_init)
p_pos, p_quat = env_core.robot.get_link_pos_quat(env_core.robot.ee_id)
o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)

table_id = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/tabletop.urdf'), TABLE_OFFSET,
                      useFixedBase=1)
p.changeDynamics(table_id, -1, lateralFriction=FLOOR_MU)
oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder_small.urdf'), o_pos, o_quat,
                  useFixedBase=0)
p.changeDynamics(oid1, -1, lateralFriction=OBJ_MU)


state = get_relative_state_for_reset(oid1)

print("before releasing", state)
print(env_core.withVel)

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





