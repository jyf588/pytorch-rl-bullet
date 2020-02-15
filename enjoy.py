import argparse
import os
# workaround to unpickle olf model files
import sys
import time

import numpy as np
import torch

import gym
import my_pybullet_envs

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
parser.add_argument(
    '--success_reward_thresh',
    type=int,
    default=4000,
    help='The threshold reward value above which it is considered a success.')
parser.add_argument(
    '--n_trials',
    type=int,
    default=2000,
    help='The number of trials to run.')

args, unknown = parser.parse_known_args()  # this is an 'internal' method
# which returns 'parsed', the same as what parse_args() would return
# and 'unknown', the remainder of that
# the difference to parse_args() is that it does not exit when it finds redundant arguments


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


for arg, value in pairwise(unknown):  # note: assume always --arg value (no --arg)
    assert arg.startswith(("-", "--"))
    parser.add_argument(arg, type=float)  # assume always float (pass bool as 0 or 1)

args_w_extra = parser.parse_args()
args_dict = vars(args)
args_w_extra_dict = vars(args_w_extra)
extra_dict = {k: args_w_extra_dict[k] for k in set(args_w_extra_dict) - set(args_dict)}

# TODO
is_cuda = True
device = 'cuda' if is_cuda else 'cpu'

args.det = not args.non_det

if 'renders' not in extra_dict:
    extra_dict['renders'] = True

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device=device,
    allow_early_resets=False,
    **extra_dict)

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

if ob_rms:
    print(ob_rms.mean)
    print(ob_rms.var)
    print(ob_rms.count)
    input("ob_rms")


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
print("obs", obs)
# input("reset, press enter")
done = False

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

reward_total = 0
control_step = 0
n_success, n_trials = 0, 0
start_time = time.time()

while n_trials < args.n_trials:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # if done:
    #     input("reset, press enter")

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    # print(obs)
    # print(action)
    # print(control_step)
    control_step += 1
    # input("press enter obs")

    reward_total += reward

    if done:
        if reward_total > args.success_reward_thresh:
            n_success += 1
        n_trials += 1
        print(f"{args.load_dir}\t"
            f"tr: {reward_total.numpy()[0][0]:.1f}\t"
            f"Avg Success: {n_success / n_trials * 100: .2f} ({n_success}/{n_trials})"
            f"(Avg. time/trial: {(time.time() - start_time)/n_trials:.2f})")
        reward_total = 0.

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    # if render_func is not None:
    #     render_func('human')
    # p.getCameraImage()
