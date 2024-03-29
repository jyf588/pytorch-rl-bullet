import argparse
import os

# workaround to unpickle olf model files
import sys
import time

import numpy as np
import torch

import gym
import my_pybullet_envs

import pickle

import json

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append("a2c_ppo_acktr")

parser = argparse.ArgumentParser(description="RL")
parser.add_argument(
    "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    help="log interval, one log per n updates (default: 10)",
)
parser.add_argument(
    "--env-name",
    default="PongNoFrameskip-v4",
    help="environment to train on (default: PongNoFrameskip-v4)",
)
parser.add_argument(
    "--load-dir",
    default="./trained_models/",
    help="directory to save agent logs (default: ./trained_models/)",
)
parser.add_argument(
    "--non-det",
    type=int,
    default=1,
    help="whether to use a non-deterministic policy, 1 true 0 false",
)
parser.add_argument(
    "--iter", type=int, default=-1, help="which iter pi to test"
)
parser.add_argument(
    "--r_thres",
    type=int,
    default=4000,
    help="The threshold reward value above which it is considered a success.",
)
parser.add_argument(
    "--n_trials", type=int, default=10000, help="The number of trials to run."
)  # TODO
parser.add_argument("--save_final_states", type=int, default=0)
parser.add_argument("--save_final_s", type=int, default=20)
parser.add_argument("--save_final_e", type=int, default=50)


args, unknown = parser.parse_known_args()  # this is an 'internal' method
# which returns 'parsed', the same as what parse_args() would return
# and 'unknown', the remainder of that
# the difference to parse_args() is that it does not exit when it finds redundant arguments


def try_numerical(string):
    # convert all extra arguments to numerical type (float) if possible
    # assume always float (pass bool as 0 or 1)
    # else, keep the argument as string type
    try:
        num = float(string)
        return num
    except ValueError:
        return string


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


for arg, value in pairwise(
    unknown
):  # note: assume always --arg value (no --arg)
    assert arg.startswith(("-", "--"))
    parser.add_argument(
        arg, type=try_numerical
    )  # assume always float (pass bool as 0 or 1)

args_w_extra = parser.parse_args()
args_dict = vars(args)
args_w_extra_dict = vars(args_w_extra)
extra_dict = {
    k: args_w_extra_dict[k] for k in set(args_w_extra_dict) - set(args_dict)
}

save_final_state_pkl = bool(args.save_final_states)
is_cuda = True
device = "cuda" if is_cuda else "cpu"

args.det = not args.non_det

# If renders is provided, turn it on. Otherwise, turn it off.
if "renders" not in extra_dict:
    extra_dict["renders"] = False


env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device=device,
    allow_early_resets=False,
    **extra_dict,
)
# dont know why there are so many wrappers in make_vec_envs...
env_core = env.venv.venv.envs[0].env.env

# # Get a render function
# render_func = get_render_func(env)
#
# print(render_func)

# We need to use the same statistics for normalization as used in training
# args.env_name = 'InmoovHandPlaceBulletEnv-v4'
if args.iter >= 0:
    path = os.path.join(
        args.load_dir, args.env_name + "_" + str(args.iter) + ".pt"
    )
else:
    path = os.path.join(args.load_dir, args.env_name + ".pt")

if is_cuda:
    actor_critic, ob_rms = torch.load(path)
else:
    actor_critic, ob_rms = torch.load(path, map_location="cpu")

if ob_rms:
    print(ob_rms.mean)
    print(ob_rms.var)
    print(ob_rms.count)
    input("ob_rms")


vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(
    1, actor_critic.recurrent_hidden_state_size
)
masks = torch.zeros(1, 1)

collect_start = args.save_final_s
collect_end = args.save_final_e
save_path = None
if args.save_final_states:
    grasp_pi_name = args.load_dir[15 : args.load_dir.find("/")]
    save_path = (
        "my_pybullet_envs/assets/place_init_dist/final_states_"
        + grasp_pi_name
        + "_"
        + str(collect_start)
        + "_"
        + str(collect_end)
        + ".pickle"
    )
    print("SAVE: ", save_path)

# if render_func is not None:
#     render_func('human')

# heights = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]
heights = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.22, 0.26, 0.30, 0.34, 0.40]
N1 = len(heights)
# radiuses = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# radiuses = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]) / 0.8
radiuses = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
N2 = len(radiuses)
good_mat = np.zeros((N1, N2))

list_of_info = []

for N1_idx, height in enumerate(heights):
    for N2_idx, radius in enumerate(list(radiuses)):
        env_core.overwrite_size = True
        env_core.overwrite_height = height
        env_core.overwrite_radius = radius

        obs = env.reset()
        print("obs", obs)
        # input("reset, press enter")
        done = False

        if args.env_name.find("Bullet") > -1:
            import pybullet as p

            torsoId = -1
            for i in range(p.getNumBodies()):
                if p.getBodyInfo(i)[0].decode() == "torso":
                    torsoId = i

        reward_total = 0
        control_step = 0
        n_success, n_trials = 0, 0
        start_time = time.time()

        list_length = 0

        while n_trials < args.n_trials:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det
                )

            # if done:
            #     input("reset, press enter")

            # Obser reward and next obs
            obs, reward, done, info = env.step(action)

            # print(obs)
            # print(action)
            # print(control_step)
            control_step += 1
            # input("press enter obs")

            reward_total += reward

            if (
                save_final_state_pkl
                and not done
                and collect_start <= control_step < collect_end
            ):  # TODO: timer/r, need to change if Pi different
                # input("press enter")
                env_core.append_final_state()
                print(len(env_core.final_states))
            if done:
                if reward_total > args.r_thres:
                    n_success += 1
                else:
                    if save_final_state_pkl:
                        pop_length = len(env_core.final_states) - list_length
                        for i in range(0, pop_length):
                            env_core.final_states.pop()


                list_of_info.append(info[0])        # somehow this dict is warpped by env to a list
                if info[0]["success"]:
                    good_mat[N1_idx, N2_idx] += 1


                n_trials += 1
                print(
                    f"{args.load_dir}\t"
                    f"tr: {reward_total.numpy()[0][0]:.1f}\t"
                    f"Avg Success: {n_success / n_trials * 100: .2f} ({n_success}/{n_trials})"
                    f"(Avg. time/trial: {(time.time() - start_time)/n_trials:.2f})"
                )
                reward_total = 0.0
                control_step = 0
                if save_final_state_pkl:
                    list_length = len(env_core.final_states)

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


with open("grasp_stats.json", "w") as f:
    json.dump(list_of_info, f, sort_keys=True, indent=2, separators=(",", ": "))

print(good_mat)
import scipy.io as sio
sio.savemat('good_mat.mat', {'good_mat': good_mat})

if save_final_state_pkl:
    with open(save_path, "wb") as handle:  # TODO: change name
        o_pos_pf_ave, o_quat_pf_ave_ri = (
            env_core.calc_average_obj_in_palm_rot_invariant()
        )
        _, o_quat_pf_ave = env_core.calc_average_obj_in_palm()
        print(o_pos_pf_ave, o_quat_pf_ave_ri)
        stored_info = {
            "init_states": env_core.final_states,
            "ave_obj_pos_in_palm": o_pos_pf_ave,
            "ave_obj_quat_in_palm_rot_ivr": o_quat_pf_ave_ri,
            "ave_obj_quat_in_palm": o_quat_pf_ave,
        }
        pickle.dump(stored_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
