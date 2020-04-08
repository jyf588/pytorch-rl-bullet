import argparse
import copy
import json
import pickle
import pprint
import os
import sys
from tqdm import tqdm
from typing import *
from my_pybullet_envs import utils

import numpy as np
import torch

import math

import my_pybullet_envs
from demo import policy, openrave
import pybullet as p
import time

import inspect
from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import (
    ImaginaryArmObjSession,
)

from my_pybullet_envs.inmoov_shadow_demo_env_v4 import (
    InmoovShadowHandDemoEnvV4,
)

from my_pybullet_envs.inmoov_shadow_hand_v2 import (
    InmoovShadowNew,
)

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
homedir = os.path.expanduser("~")


# TODO: main module depends on the following code/model:
# demo env: especially observation  # change obs vec (note diffTar)
# the settings of inmoov hand v2        # TODO: init thumb 0.0 vs 0.1
# obj sizes & frame representation & friction & obj xy range
# frame skip
# vision delay

"""Parse arguments"""
sys.path.append("a2c_ppo_acktr")
parser = argparse.ArgumentParser(description="RL")
parser.add_argument("--seed", type=int, default=101)    # only keep np.random
parser.add_argument("--non-det", type=int, default=0)
args = parser.parse_args()
np.random.seed(args.seed)
args.det = not args.non_det

"""Configurations."""

USE_GV5 = True  # TODO: is false, use gv6

NUM_TRIALS = 20

GRASP_END_STEP = 40
PLACE_END_STEP = 90

INIT_NOISE = True
DET_CONTACT = 0  # 0 false, 1 true

OBJ_MU = 1.0
FLOOR_MU = 1.0
HAND_MU = 1.0
OBJ_MASS = 3.5

IS_CUDA = True
DEVICE = "cuda" if IS_CUDA else "cpu"

GRASP_PI = "0313_2_n_25_45"
GRASP_DIR = "./trained_models_%s/ppo/" % "0313_2_n"
PLACE_PI = "0313_2_placeco_0316_1"  # 50ms
PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI

GRASP_PI_ENV_NAME = "InmoovHandGraspBulletEnv-v5"
PLACE_PI_ENV_NAME = "InmoovHandPlaceBulletEnv-v9"

USE_VISION_DELAY = True
VISION_DELAY = 2
PLACING_CONTROL_SKIP = 6
GRASPING_CONTROL_SKIP = 6

INIT_FIN_Q = np.array([0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0])


def planning(trajectory):
    for idx in range(len(trajectory) + 50):
        if idx > len(trajectory) - 1:
            tar_arm_q = trajectory[-1]
        else:
            tar_arm_q = trajectory[idx]
        env_core.robot.tar_arm_q = tar_arm_q
        env_core.robot.apply_action([0.0] * 24)
        p.stepSimulation()
        time.sleep(utils.TS / 2.0)

    # print("end of traj", traj[-1, 0:7])
    # for time_step in range(0, len(traj)):
    #     tar_armq = traj[time_step, 0:7]
    #     env_core.robot.tar_arm_q = tar_armq
    #     env_core.robot.apply_action([0.0] * 24)
    #     p.stepSimulation()
    #     time.sleep(utils.TS / 2.0)
    #
    # tar_armq = traj[-1, 0:7]
    # for _ in range(50):
    #     # print(env_core.robot.tar_arm_q)
    #     env_core.robot.tar_arm_q = tar_armq
    #     env_core.robot.apply_action([0.0] * 24)  # stay still for a while
    #     p.stepSimulation()
    #     # pose_saver.get_poses()
    #     # print("act", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    # #     #time.sleep(1. / 240.)


def get_relative_state_for_reset(oid):
    obj_pos, obj_quat = p.getBasePositionAndOrientation(oid)  # w2o
    hand_pos, hand_quat = env_core.robot.get_link_pos_quat(
        env_core.robot.ee_id
    )  # w2p
    inv_h_p, inv_h_q = p.invertTransform(hand_pos, hand_quat)  # p2w
    o_p_hf, o_q_hf = p.multiplyTransforms(
        inv_h_p, inv_h_q, obj_pos, obj_quat
    )  # p2w*w2o

    fin_q, _ = env_core.robot.get_q_dq(env_core.robot.all_findofs)

    relative_state = {
        "obj_pos_in_palm": o_p_hf,
        "obj_quat_in_palm": o_q_hf,
        "all_fin_q": fin_q,
        "fin_tar_q": env_core.robot.tar_fin_q,
    }
    return relative_state


# def load_policy_params(path_name, env_name, iter_num=None):
#     if iter_num is not None:
#         path = os.path.join(path_name, env_name + "_" + str(iter_num) + ".pt")
#     else:
#         path = os.path.join(path_name, env_name + ".pt")
#     if IS_CUDA:
#         actor_critic, ob_rms = torch.load(path)
#     else:
#         actor_critic, ob_rms = torch.load(path, map_location="cpu")
#
#     # dummy
#     recurrent_hidden_states = torch.zeros(
#         1, actor_critic.recurrent_hidden_state_size
#     )
#     masks = torch.zeros(1, 1)
#     return (
#         actor_critic,
#         ob_rms,
#         recurrent_hidden_states,
#         masks,
#     )  # probably only first one is used
#     # assume no ob_rms


# def wrap_over_grasp_obs(obs):
#     obs = torch.Tensor([obs])
#     if IS_CUDA:
#         obs = obs.cuda()
#     return obs
#
#
# def unwrap_action(act_tensor):
#     act_numpy = act_tensor.squeeze()
#     act_numpy = act_numpy.cpu() if IS_CUDA else act_numpy
#     return act_numpy.numpy()
#
#
# def get_traj_from_openrave_container(objects, q_start, q_end, save_file_path, read_file_path):
#
#     if q_start is not None:
#         np.savez(save_file_path, objects, q_start, q_end)      # move
#     else:
#         np.savez(save_file_path, objects, q_end)       # reach has q_start 0
#
#     # Wait for command from OpenRave
#
#     assert not os.path.exists(read_file_path)
#     while not os.path.exists(read_file_path):
#         time.sleep(0.2)
#     if os.path.isfile(read_file_path):
#         traj = np.load(read_file_path)
#         print("loaded")
#         try:
#             os.remove(read_file_path)
#             print("deleted")
#             # input("press enter")
#         except OSError as e:  # name the Exception `e`
#             print("Failed with:", e.strerror)  # look what it says
#             # input("press enter")
#     else:
#         raise ValueError("%s isn't a file!" % read_file_path)
#     print("Trajectory obtained from OpenRave!")
#     # input("press enter")
#     return traj


def sample_obj_dict(is_thicker=False):
    # a dict containing obj info
    # "shape", "radius", "height", "position", "orientation"

    min_r = utils.HALF_W_MIN_BTM if is_thicker else utils.HALF_W_MIN

    obj_dict = {
        "shape": utils.SHAPE_IND_TO_NAME_MAP[np.random.randint(2)],
        "radius": np.random.uniform(min_r, utils.HALF_W_MAX),
        "height": np.random.uniform(utils.H_MIN, utils.H_MAX),
        "position": [
            np.random.uniform(utils.TX_MIN, utils.TX_MAX),
            np.random.uniform(utils.TY_MIN, utils.TY_MAX),
            0.0
        ],
        "orientation": p.getQuaternionFromEuler(
            [0., 0., np.random.uniform(low=0, high=2.0 * math.pi)]
        ),
        "mass": OBJ_MASS,
        "mu": OBJ_MU,

    }

    if obj_dict["shape"] == "box":
        obj_dict["radius"] *= 0.8
    obj_dict["position"][2] = obj_dict["height"] / 2.0 + 0.001  # TODO

    return obj_dict


def get_stacking_obs(
    top_oid: int,
    btm_oid: int,
):
    """Retrieves stacking observations.

    Args:
        top_oid: The object ID of the top object.
        btm_oid: The object ID of the bottom object.

    Returns:
        t_pos: The xyz position of the top object.
        t_up: The up vector of the top object.
        b_pos: The xyz position of the bottom object.
        b_up: The up vector of the bottom object.
        t_half_height: Half of the height of the top object.
    """

    t_pos, t_quat = p.getBasePositionAndOrientation(top_oid)
    b_pos, b_quat = p.getBasePositionAndOrientation(btm_oid)

    t_up = utils.quat_to_upv(t_quat)
    b_up = utils.quat_to_upv(b_quat)

    t_half_height = objs[top_id]["height"] / 2      # TODO: add noise to GT

    return t_pos, t_up, b_pos, b_up, t_half_height


success_count = 0

"""Pre-calculation & Loading"""
# latter 2 returns dummy
g_actor_critic, _, _, _ = policy.load(
    GRASP_DIR, GRASP_PI_ENV_NAME, IS_CUDA
)
p_actor_critic, _, recurrent_hidden_states, masks = policy.load(
    PLACE_DIR, PLACE_PI_ENV_NAME, IS_CUDA
)

o_pos_pf_ave, o_quat_pf_ave, _ = \
    utils.read_grasp_final_states_from_pickle(GRASP_PI)

p_pos_of_ave, p_quat_of_ave = p.invertTransform(
    o_pos_pf_ave, o_quat_pf_ave
)

"""Start Bullet session."""

p.connect(p.GUI)

for trial in range(NUM_TRIALS):
    """Sample two objects"""

    while True:
        top_dict = sample_obj_dict()
        btm_dict = sample_obj_dict(is_thicker=True)

        # TODO: add noise to GT
        g_tx, g_ty = top_dict["position"][0], top_dict["position"][1]
        p_tx, p_ty, p_tz = btm_dict["position"][0], btm_dict["position"][1], btm_dict["height"]
        t_half_height = top_dict["height"]/2

        g_tx += np.random.uniform(low=-0.01, high=0.01)
        g_ty += np.random.uniform(low=-0.01, high=0.01)
        t_half_height += np.random.uniform(low=-0.01, high=0.01)
        p_tx += np.random.uniform(low=-0.01, high=0.01)
        p_ty += np.random.uniform(low=-0.01, high=0.01)
        p_tz += np.random.uniform(low=-0.01, high=0.01)

        # required by OpenRave
        OBJECTS = np.array([top_dict["position"][:2]+[0., 0.], btm_dict["position"][:2]+[0., 0.]])
        is_box = (top_dict["shape"] == "box")

        if (g_tx - p_tx)**2 + (g_ty - p_ty)**2 > 0.2**2:
            break

    """Imaginary arm session to get q_reach"""
    sess = ImaginaryArmObjSession()

    Qreach = np.array(sess.get_most_comfortable_q_and_refangle(g_tx, g_ty)[0])

    del sess

    desired_obj_pos = [p_tx, p_ty, utils.PLACE_START_CLEARANCE + p_tz]

    table_id = utils.create_table(FLOOR_MU)

    robot = InmoovShadowNew(
        init_noise=False,
        timestep=utils.TS,
        np_random=np.random,
    )

    # TODO: [1] is the 2nd candidate
    Qdestin = utils.get_n_optimal_init_arm_qs(
        robot, p_pos_of_ave, p_quat_of_ave, desired_obj_pos, table_id
    )[0]
    p.resetSimulation()

    """Clean up the simulation, since this is only imaginary."""

    """Setup Bullet world."""
    """ Create table, robot, bottom obj, top obj"""
    p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
    p.setPhysicsEngineParameter(deterministicOverlappingPairs=DET_CONTACT)
    p.setTimeStep(utils.TS)
    p.setGravity(0, 0, -utils.GRAVITY)

    # top_id = utils.create_sym_prim_shape_helper_new(top_dict)

    table_id = utils.create_table(FLOOR_MU)

    env_core = InmoovShadowHandDemoEnvV4(
        np_random=np.random,
        init_noise=INIT_NOISE,
        timestep=utils.TS,
        withVel=False,
        diffTar=True,
        robot_mu=HAND_MU,
        control_skip=GRASPING_CONTROL_SKIP,
    )
    env_core.change_init_fin_q(INIT_FIN_Q)

    btm_id = utils.create_sym_prim_shape_helper_new(btm_dict)
    top_id = utils.create_sym_prim_shape_helper_new(top_dict)
    objs = {
        btm_id: btm_dict,
        top_id: top_dict,
    }

    # env_core.robot.reset_with_certain_arm_q([0.0]*7)

    """Prepare for grasping. Reach for the object."""

    print(f"Qreach: {Qreach}")
    # reach_save_path = homedir + "/container_data/PB_REACH.npz"
    # reach_read_path = homedir + "/container_data/OR_REACH.npy"
    # Traj_reach = get_traj_from_openrave_container(OBJECTS, None, Qreach, reach_save_path, reach_read_path)
    #
    # planning(Traj_reach)
    # input("press enter")
    env_core.robot.reset_with_certain_arm_q(Qreach)
    # input("press enter 2")

    # p.stepSimulation()

    # g_obs = env_core.get_robot_contact_txty_halfh_obs_nodup(g_tx, g_ty, t_half_height)
    g_obs = env_core.get_robot_contact_txty_halfh_obs_nodup(g_tx, g_ty, t_half_height)
    g_obs = policy.wrap_obs(g_obs, IS_CUDA)

    """Grasp"""
    control_steps = 0
    for i in range(GRASP_END_STEP):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = g_actor_critic.act(
                g_obs, recurrent_hidden_states, masks, deterministic=args.det
            )

        env_core.step(policy.unwrap_action(action, IS_CUDA))
        g_obs = env_core.get_robot_contact_txty_halfh_obs_nodup(
            g_tx, g_ty, t_half_height
        )
        g_obs = policy.wrap_obs(g_obs, IS_CUDA)

        # print(g_obs)
        # print(action)
        # print(control_steps)
        # control_steps += 1
        # input("press enter g_obs")
        masks.fill_(1.0)
        # pose_saver.get_poses()

    final_g_obs = copy.copy(g_obs)
    del g_obs, g_tx, g_ty, t_half_height

    state = get_relative_state_for_reset(top_id)
    print("after grasping", state)
    print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    # input("after grasping")

    """Send move command to OpenRAVE"""
    Qmove_init = env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0]
    print(f"Qmove_init: {Qmove_init}")
    print(f"Qdestin: {Qdestin}")
    move_save_path = homedir + "/container_data/PB_MOVE.npz"
    move_read_path = homedir + "/container_data/OR_MOVE.npy"
    Traj_move = openrave.get_traj_from_openrave_container(OBJECTS, Qmove_init, Qdestin, move_save_path, move_read_path)

    """Execute planned moving trajectory"""

    if Traj_move is None or len(Traj_move) == 0:
        continue        # TODO: transporting failed
    else:
        planning(Traj_move)
    print("after moving", get_relative_state_for_reset(top_id))
    print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    # input("after moving")

    print("palm", env_core.robot.get_link_pos_quat(env_core.robot.ee_id))

    # pose_saver.get_poses()
    # print(f"Pose before placing")
    # pprint.pprint(pose_saver.poses[-1])

    """Prepare for placing"""
    env_core.change_control_skip_scaling(c_skip=PLACING_CONTROL_SKIP)

    t_pos, t_up, b_pos, b_up, t_half_height = get_stacking_obs(
        top_oid=top_id,
        btm_oid=btm_id,
    )

    l_t_pos, l_t_up, l_b_pos, l_b_up, l_t_half_height = (
        t_pos,
        t_up,
        b_pos,
        b_up,
        t_half_height,
    )

    # TODO: an unly hack to force Bullet compute forward kinematics
    p_obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
        p_tx, p_ty, p_tz, t_half_height, is_box, t_pos, t_up, b_pos, b_up
    )
    p_obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
        p_tx, p_ty, p_tz, t_half_height, is_box, t_pos, t_up, b_pos, b_up
    )

    p_obs = policy.wrap_obs(p_obs, IS_CUDA)
    print("pobs", p_obs)
    # input("ready to place")

    """Execute placing"""
    print(f"Executing placing...")
    for i in tqdm(range(PLACE_END_STEP)):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = p_actor_critic.act(
                p_obs, recurrent_hidden_states, masks, deterministic=args.det
            )

        env_core.step(policy.unwrap_action(action, IS_CUDA))

        if (i + 1) % VISION_DELAY == 0:
            l_t_pos, l_t_up, l_b_pos, l_b_up, l_t_half_height = (
                t_pos,
                t_up,
                b_pos,
                b_up,
                t_half_height,
            )
            t_pos, t_up, b_pos, b_up, t_half_height = get_stacking_obs(
                top_oid=top_id,
                btm_oid=btm_id,
            )

        p_obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
            p_tx, p_ty, p_tz, l_t_half_height, is_box, l_t_pos, l_t_up, l_b_pos, l_b_up
        )

        p_obs = policy.wrap_obs(p_obs, IS_CUDA)

        # print(action)
        # print(p_obs)
        # input("press enter g_obs")

        masks.fill_(1.0)
        # pose_saver.get_poses()

    # print(f"Pose after placing")
    # pprint.pprint(pose_saver.poses[-1])

    print(f"Starting release trajectory")
    # execute_release_traj()
    for ind in range(0, 100):
        p.stepSimulation()
        time.sleep(utils.TS)
        # pose_saver.get_poses()

    # if SAVE_POSES:
    #     pose_saver.save()

    t_pos, t_quat = p.getBasePositionAndOrientation(top_id)
    if t_pos[2] > 0.15 and (t_pos[0] - p_tx)**2 + (t_pos[1] - p_ty)**2 < 0.1**2:
        # a very rough check
        success_count += 1

    p.resetSimulation()

    print("*******", success_count * 1.0 / (trial+1))

p.disconnect()
print("*******total", success_count * 1.0 / NUM_TRIALS)