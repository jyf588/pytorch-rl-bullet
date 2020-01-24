import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

import gym
import my_pybullet_envs

import pickle

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

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
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    type=int,
    default=1,
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


# # Get a render function
# render_func = get_render_func(env)
#
# print(render_func)

# We need to use the same statistics for normalization as used in training
if args.iter >= 0:
    path = os.path.join(args.load_dir, args.env_name + "_" + str(args.iter) + ".pt")
else:
    path = os.path.join(args.load_dir, args.env_name + ".pt")

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

obs = env.reset()


if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

reward_total = 0

timer = 0

finish = False

while not finish:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    timer += 1

    reward_total += reward

    # print(reward_total)
    # if 95 <= timer <= 101:
    #     input("press enter")

    if not done and 85 <= timer <= 95:   # TODO: timer/r, need to change if Pi different
        # input("press enter")
        env_core.append_final_state()
        print(len(env_core.final_states))

        if len(env_core.final_states) > 100000:      # TODO: length
            finish = True

    if done:
        print("tr:", reward_total)
        if reward_total < 4000:     # TODO: timer/r, need to change if Pi different
            for i in range(0, 11):
                env_core.final_states.pop()
        reward_total = 0.
        timer = 0

    masks.fill_(0.0 if done else 1.0)

    # if args.env_name.find('Bullet') > -1:
    #     if torsoId > -1:
    #         distance = 5
    #         yaw = 0
    #         humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
    #         p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    # if render_func is not None:
    #     render_func('human')
    # p.getCameraImage()

with open('my_pybullet_envs/assets/place_init_dist/final_states_0120_box_s_1.pickle', 'wb') as handle:      # TODO: change name
    o_pos_pf_ave, o_quat_pf_ave_ri = env_core.calc_average_obj_in_palm_rot_invariant()
    _, o_quat_pf_ave = env_core.calc_average_obj_in_palm()
    print(o_pos_pf_ave, o_quat_pf_ave_ri)
    stored_info = {'init_states': env_core.final_states, 'ave_obj_pos_in_palm': o_pos_pf_ave,
                   'ave_obj_quat_in_palm_rot_ivr': o_quat_pf_ave_ri,
                   'ave_obj_quat_in_palm': o_quat_pf_ave}
    pickle.dump(stored_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
