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
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

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


# # Get a render function
# render_func = get_render_func(env)
#
# print(render_func)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

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
save_qs = []
while not finish:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    timer += 1

    reward_total += reward

    if not done and timer >= 380 and reward_total > 12000:   # TODO
        save_q = []

        # pos = p.getLinkState(robot.robotId, robot.endEffectorId)[0]
        # orn = p.getLinkState(robot.robotId, robot.endEffectorId)[1]
        # print(pos, orn)
        #
        # localpos = p.getLinkState(robot.robotId, robot.endEffectorId)[2]
        # localorn = p.getLinkState(robot.robotId, robot.endEffectorId)[3]
        # print(localpos, localorn)
        #
        pos, orn = p.getBasePositionAndOrientation(robot.handId)
        # x,r = p.multiplyTransforms(pos, orn, localpos, localorn)
        # print(x,r)

        linVel, angVel = p.getBaseVelocity(robot.handId)
        save_q.extend(pos)
        save_q.extend(p.getEulerFromQuaternion(orn))
        save_q.extend(linVel)
        save_q.extend(angVel)
        # print(save_q)
        save_q.extend(list(robot.get_raw_state_fingers(includeVel=False)))
        # print(save_q)

        clPos, clOrn = p.getBasePositionAndOrientation(env_core.cylinderId)
        save_q.extend(clPos)
        save_q.extend(p.getEulerFromQuaternion(clOrn))
        clVels = p.getBaseVelocity(env_core.cylinderId)
        save_q.extend(clVels[0])
        save_q.extend(clVels[1])
        # print(save_q)

        save_qs.append(save_q)

        print(len(save_qs))

        if len(save_qs) > 60000:
            finish = True

    if done:
        print("tr:", reward_total)
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

with open('final_states_1209.pickle', 'wb') as handle:
    pickle.dump(save_qs, handle, protocol=pickle.HIGHEST_PROTOCOL)
