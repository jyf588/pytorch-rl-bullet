import argparse
import copy
import json
import pickle
import pprint
import os
import sys
from tqdm import tqdm
from typing import *
from my_pybullet_envs import utils_left as utils
from state_saver import StateSaver

import numpy as np
import torch

import math

import my_pybullet_envs
from system import policy, openrave
import pybullet as p
import time

import inspect
from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSession

from my_pybullet_envs.inmoov_shadow_demo_env_v4_left_no_orientation import InmoovShadowHandDemoEnvV4

from my_pybullet_envs.inmoov_shadow_hand_v2_no_orientation import InmoovShadowNew

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
homedir = os.path.expanduser("~")


# TODO: main module depends on the following code/model:
# demo env: especially observation  # change obs vec (note diffTar)
# the settings of inmoov hand v2        # init thumb 0.0 vs 0.1
# obj sizes & frame representation & friction & obj xy range
# frame skip
# vision delay

# state_saver = StateSaver(out_dir="/home/yifengj/try")

"""Parse arguments"""
sys.path.append("a2c_ppo_acktr")
parser = argparse.ArgumentParser(description="RL")
parser.add_argument("--seed", type=int, default=101)  # only keep np.random
parser.add_argument("--use_height", type=int, default=0)
parser.add_argument("--test_placing", type=int, default=0)
parser.add_argument("--long_move", type=int, default=0)
parser.add_argument("--non-det", type=int, default=0)
parser.add_argument("--render", type=int, default=0)
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--add_place_stack_bit", type=int, default=0)
args = parser.parse_args()
np.random.seed(args.seed)
args.det = not args.non_det

"""Configurations."""

USE_GV5 = False  # is false, use gv6
DUMMY_SLEEP = bool(args.sleep)
WITH_REACHING = True
WITH_RETRACT = True
USE_HEIGHT_INFO = bool(args.use_height)
ADD_PLACE_STACK_BIT = bool(args.add_place_stack_bit)
TEST_PLACING = bool(args.test_placing)    # if false, test stacking
ADD_SURROUNDING_OBJS = True
LONG_MOVE = bool(args.long_move)
SURROUNDING_OBJS_MAX_NUM = 4

ADD_WHITE_NOISE = False
RENDER = bool(args.render)

CLOSE_THRES = 0.25

NUM_TRIALS = 300

GRASP_END_STEP = 35
PLACE_END_STEP = 55

INIT_NOISE = True
DET_CONTACT = 0  # 0 false, 1 true

OBJ_MU = 1.0
FLOOR_MU = 1.0
HAND_MU = 1.0
OBJ_MASS = 3.5

IS_CUDA = False
DEVICE = "cuda" if IS_CUDA else "cpu"

ITER = None

if USE_GV5:
    GRASP_PI = "0313_2_n_25_45"
    GRASP_DIR = "./trained_models_%s/ppo/" % "0313_2_n"
    PLACE_PI = "0313_2_placeco_0316_1"  # 50ms
    PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI

    GRASP_PI_ENV_NAME = "InmoovHandGraspBulletEnv-v5"
    PLACE_PI_ENV_NAME = "InmoovHandPlaceBulletEnv-v9"

    INIT_FIN_Q = np.array(
        [0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.0]
    )
else:
    # use gv6
    if USE_HEIGHT_INFO:
        GRASP_PI = "0404_0_n_20_40"
        GRASP_DIR = "./trained_models_%s/ppo/" % "0404_0_n"

        PLACE_PI = "0404_0_n_place_0404_0"
        PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI

        if ADD_PLACE_STACK_BIT:
            GRASP_PI = "0729_12_n_25_45"
            GRASP_DIR = "./trained_models_%s/ppo/" % "0729_12_n"
            PLACE_PI = "0729_12_n_place_0729_12"          # 68/83
            PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI
    else:
        GRASP_PI = "0411_0_n_25_45"
        GRASP_DIR = "./trained_models_%s/ppo/" % "0411_0_n"

        PLACE_PI = "0411_0_n_place_0411_0"
        PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI

        if ADD_PLACE_STACK_BIT:
            GRASP_PI = "0426_0_n_25_45"
            GRASP_DIR = "./trained_models_%s/ppo/" % "0426_0_n"

            PLACE_PI = "0426_0_n_place_0426_0"
            PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI

    GRASP_PI_ENV_NAME = "InmoovHandGraspBulletEnv-v6"
    PLACE_PI_ENV_NAME = "InmoovHandPlaceBulletEnv-v9"

    INIT_FIN_Q = np.array(
        [0.4, 0.4, 0.4] * 3 + [0.4, 0.4, 0.4] + [0.0, 1.0, 0.1, 0.5, 0.1]
    )

USE_VISION_DELAY = True
VISION_DELAY = 2
PLACING_CONTROL_SKIP = 6
GRASPING_CONTROL_SKIP = 6


def switchDirections(target_list):
    for i in range(len(target_list)):
        # 5th joint ignored in urdf
        if i not in (1, 3, 5, 6):
            target_list[i] *= -1

    return target_list


def planning(trajectory, retract_stage=False):
    # TODO: total traj length 300+5 now

    max_force = env_core.robot.maxForce

    init_tar_fin_q = env_core.robot.tar_fin_q
    init_fin_q = env_core.robot.get_q_dq(env_core.robot.fin_actdofs)[0]

    init_arm_dq = env_core.robot.get_q_dq(env_core.robot.arm_dofs)[1]
    init_arm_q = env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0]
    last_tar_arm_q = init_arm_q

    env_core.robot.tar_arm_q = trajectory[-1]  # TODO: important!

    print("init_tar_fin_q")
    print(["{0:0.3f}".format(n) for n in init_tar_fin_q])
    print("init_fin_q")
    print(["{0:0.3f}".format(n) for n in init_fin_q])

    for idx in range(len(trajectory) + 5):
        if idx > len(trajectory) - 1:
            tar_arm_q = trajectory[-1]
        else:
            tar_arm_q = trajectory[idx]

        #tar_arm_q = switchDirections(tar_arm_q)

        if retract_stage:
            proj_arm_q = init_arm_q + (idx+1) * init_arm_dq * utils.TS
            blending = np.clip(
                idx / (len(trajectory) * 0.6), 0.0, 1.0
            )
            tar_arm_q = tar_arm_q * blending + proj_arm_q * (1 - blending)

        tar_arm_vel = (tar_arm_q - last_tar_arm_q) / utils.TS

        p.setJointMotorControlArray(
            bodyIndex=env_core.robot.arm_id,
            jointIndices=env_core.robot.arm_dofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(tar_arm_q),
            targetVelocities=list(tar_arm_vel),
            forces=[max_force * 5] * len(env_core.robot.arm_dofs),
        )

        if retract_stage and idx >= len(trajectory) * 0.1:  # TODO: hardcoded
            blending = np.clip(
                (idx - len(trajectory) * 0.1) /
                (len(trajectory) * 0.8), 0.0, 1.0
            )
            # cur_fin_q = env_core.robot.get_q_dq(env_core.robot.fin_actdofs)[0]
            tar_fin_q = env_core.robot.init_fin_q * blending + init_fin_q * (
                1 - blending
            )
        else:
            # try to keep fin q close to init_fin_q (keep finger pose)
            # add at most offset 0.05 in init_tar_fin_q direction so that grasp is tight
            tar_fin_q = np.clip(
                init_tar_fin_q, init_fin_q - 0.1, init_fin_q + 0.1)

        # clip to joint limit
        tar_fin_q = np.clip(
            tar_fin_q,
            env_core.robot.ll[env_core.robot.fin_actdofs],
            env_core.robot.ul[env_core.robot.fin_actdofs],
        )

        p.setJointMotorControlArray(
            bodyIndex=env_core.robot.arm_id,
            jointIndices=env_core.robot.fin_actdofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(tar_fin_q),
            forces=[max_force] * len(env_core.robot.fin_actdofs),
        )
        p.setJointMotorControlArray(
            bodyIndex=env_core.robot.arm_id,
            jointIndices=env_core.robot.fin_zerodofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.0] * len(env_core.robot.fin_zerodofs),
            forces=[max_force / 4.0] * len(env_core.robot.fin_zerodofs),
        )

        diff = np.linalg.norm(
            env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0] - tar_arm_q
        )

        if idx == len(trajectory) + 4:
            print("diff final", diff)
            print(
                "vel final",
                np.linalg.norm(env_core.robot.get_q_dq(
                    env_core.robot.arm_dofs)[1]),
            )
            print("fin dofs")
            print(
                [
                    "{0:0.3f}".format(n)
                    for n in env_core.robot.get_q_dq(env_core.robot.fin_actdofs)[0]
                ]
            )
            print("cur_fin_tar_q")
            print(["{0:0.3f}".format(n) for n in env_core.robot.tar_fin_q])

        for _ in range(1):
            p.stepSimulation()
            # state_saver.save_state()
        if DUMMY_SLEEP:
            time.sleep(utils.TS * 0.6)

        last_tar_arm_q = tar_arm_q


def get_relative_state_for_reset(oid):
    obj_pos, obj_quat = p.getBasePositionAndOrientation(oid)  # w2o
    hand_pos, hand_quat = env_core.robot.get_link_pos_quat(
        env_core.robot.ee_id)  # w2p
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


def sample_obj_dict(is_thicker=False, whole_table_top=False):
    # a dict containing obj info
    # "shape", "radius", "height", "position", "orientation", "mass", "mu"

    min_r = utils.HALF_W_MIN_BTM if is_thicker else utils.HALF_W_MIN
    if whole_table_top:
        x_min = utils.X_MIN
        x_max = utils.X_MAX
        y_min = utils.Y_MIN
        y_max = utils.Y_MAX
    else:
        x_min = utils.TX_MIN
        x_max = utils.TX_MAX
        y_min = utils.TY_MIN
        y_max = utils.TY_MAX

    obj_dict = {
        "shape": utils.SHAPE_IND_TO_NAME_MAP[np.random.randint(2)],
        "radius": np.random.uniform(min_r, utils.HALF_W_MAX),
        "height": np.random.uniform(utils.H_MIN, utils.H_MAX),
        "position": [
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            0.0,
        ],
        "orientation": p.getQuaternionFromEuler(
            [0.0, 0.0, np.random.uniform(low=0, high=2.0 * math.pi)]
        ),
        "mass": OBJ_MASS,
        "mu": OBJ_MU,
    }

    # color_dict = {0:"red", 1:"yellow", 2:"blue", 3:"green"}
    # obj_dict["color"] = color_dict[np.random.randint(3)]
    if obj_dict["shape"] == "box":
        obj_dict["radius"] *= 0.8
    obj_dict["position"][2] = obj_dict["height"] / 2.0

    return obj_dict


def load_obj_and_construct_state(obj_dicts_list):

    state = {}
    # load surrounding first
    for idx in range(2, len(obj_dicts_list)):
        bullet_id = utils.create_sym_prim_shape_helper_new(obj_dicts_list[idx])
        state[bullet_id] = obj_dicts_list[idx]

    bottom_id = None
    # ignore btm if placing on tabletop
    if not TEST_PLACING:
        obj_dicts_list[1]["color"] = "green"
        bottom_id = utils.create_sym_prim_shape_helper_new(obj_dicts_list[1])
        state[bottom_id] = obj_dicts_list[1]

    # TODO:tmp load grasp obj last
    obj_dicts_list[0]["color"] = "red"
    topobj_id = utils.create_sym_prim_shape_helper_new(obj_dicts_list[0])
    state[topobj_id] = obj_dicts_list[0]
    # # irregular objects
    # obj_dicts_list[0]['position'][2] = 0.0
    # obj_dicts_list[0]['shape'] = 'prism'
    # obj_pos = obj_dicts_list[0]['position']
    # # obj_pos[2] = 0.0
    # topobj_id = p.loadURDF("my_pybullet_envs/assets/cone.urdf", basePosition=obj_pos,
    #                        baseOrientation=obj_dicts_list[0]['orientation'], useFixedBase=0)
    # p.changeDynamics(topobj_id, -1, lateralFriction=OBJ_MU)
    # state[topobj_id] = obj_dicts_list[0]

    return state, topobj_id, bottom_id


def construct_obj_array_for_openrave(obj_dicts_list):
    arr = []
    for idx, obj_dict in enumerate(obj_dicts_list):
        if idx == 1 and TEST_PLACING:
            # ignore btm if placing on tabletop
            continue
        # grasp obj should be at first
        arr.append(obj_dict["position"][:2] + [0.0, 0.0])
    return np.array(arr)


def get_grasp_policy_obs_tensor(tx, ty, half_height, is_box):
    if USE_GV5:
        assert USE_HEIGHT_INFO
        obs = env_core.get_robot_contact_txty_halfh_obs_nodup(
            tx, ty, half_height)
    else:
        if USE_HEIGHT_INFO:
            obs = env_core.get_robot_contact_txtytz_halfh_shape_obs_no_dup(
                tx, ty, 0.0, half_height, is_box
            )
        else:
            obs = env_core.get_robot_contact_txty_shape_obs_no_dup(
                tx, ty, is_box)
    obs = policy.wrap_obs(obs, IS_CUDA)
    return obs


def get_stack_policy_obs_tensor(
    tx, ty, tz, t_half_height, is_box, t_pos, t_up, b_pos, b_up
):
    if USE_HEIGHT_INFO:
        obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
            tx, ty, tz, t_half_height, is_box, t_pos, t_up, b_pos, b_up
        )
    else:
        obs = env_core.get_robot_contact_txty_shape_2obj6dUp_obs_nodup_from_up(
            tx, ty, is_box, t_pos, t_up, b_pos, b_up
        )

    if ADD_PLACE_STACK_BIT:
        if TEST_PLACING:
            obs.extend([1.0])
        else:
            obs.extend([-1.0])
    obs = policy.wrap_obs(obs, IS_CUDA)
    return obs


def is_close(obj_dict_a, obj_dict_b, dist=CLOSE_THRES):
    xa, ya = obj_dict_a["position"][0], obj_dict_a["position"][1]
    xb, yb = obj_dict_b["position"][0], obj_dict_b["position"][1]
    return (xa - xb) ** 2 + (ya - yb) ** 2 < dist ** 2


def get_stacking_obs(
    obj_state: dict, top_oid: int, btm_oid: int,
):
    """Retrieves stacking observations.

    Args:
        obj_state: world obj state dict of dicts
        top_oid: The object ID of the top object.
        btm_oid: The object ID of the bottom object.

    Returns:
        top_pos: The xyz position of the top object.
        top_up: The up vector of the top object.
        btm_pos: The xyz position of the bottom object.
        btm_up: The up vector of the bottom object.
        top_half_height: Half of the height of the top object.
    """

    top_pos, top_quat = p.getBasePositionAndOrientation(top_oid)
    # # # irregular objects
    # top_pos = list(top_pos)
    # top_pos[2] += 0.05
    if btm_oid is None:
        btm_pos, btm_quat = [0.0, 0, 0], [0.0, 0, 0, 1]
    else:
        btm_pos, btm_quat = p.getBasePositionAndOrientation(btm_oid)

    top_up = utils.quat_to_upv(top_quat)
    btm_up = utils.quat_to_upv(btm_quat)

    top_pos = (top_pos[0], -top_pos[1], top_pos[2])
    btm_pos = (btm_pos[0], -btm_pos[1], btm_pos[2])
    top_up = (top_up[0], -top_up[1], top_up[2])
    btm_up = (btm_up[0], -btm_up[1], btm_up[2])

    top_half_height = obj_state[top_oid]["height"] / 2

    if ADD_WHITE_NOISE:
        top_pos = utils.perturb(np.random, top_pos, r=0.02)
        btm_pos = utils.perturb(np.random, btm_pos, r=0.02)
        top_up = utils.perturb(np.random, top_up, r=0.03)
        btm_up = utils.perturb(np.random, btm_up, r=0.03)
        top_half_height = utils.perturb_scalar(
            np.random, top_half_height, r=0.01)


    return top_pos, top_up, btm_pos, btm_up, top_half_height


def gen_surrounding_objs(obj_dicts_list):
    # gen objs and modifies obj_dicts_list accordingly
    if ADD_SURROUNDING_OBJS:
        num_obj = np.random.randint(SURROUNDING_OBJS_MAX_NUM) + 1  # 1,2,3,4
        retries = 0
        while len(obj_dicts_list) - 2 < num_obj and retries < 50:
            new_obj_dict = sample_obj_dict(whole_table_top=True)
            is_close_arr = [
                is_close(new_obj_dict, obj_dict) for obj_dict in obj_dicts_list
            ]
            if not any(is_close_arr):
                obj_dicts_list.append(new_obj_dict)
            retries += 1
    return obj_dicts_list


success_count = 0
openrave_success_count = 0

"""Pre-calculation & Loading"""
g_actor_critic, _, _, _ = policy.load(GRASP_DIR, GRASP_PI_ENV_NAME, IS_CUDA)
p_actor_critic, _, recurrent_hidden_states, masks = policy.load(
    PLACE_DIR, PLACE_PI_ENV_NAME, IS_CUDA, ITER
)

o_pos_pf_ave, o_quat_pf_ave, _ = utils.read_grasp_final_states_from_pickle(
    GRASP_PI)

p_pos_of_ave, p_quat_of_ave = p.invertTransform(o_pos_pf_ave, o_quat_pf_ave)

"""Start Bullet session."""
if RENDER:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

for trial in range(NUM_TRIALS):
    """Sample two/N objects"""

    all_dicts = []

    while True:
        top_dict = sample_obj_dict()
        btm_dict = sample_obj_dict(is_thicker=True)

        g_tx, g_ty = top_dict["position"][0], top_dict["position"][1]
        p_tx, p_ty, p_tz = (
            btm_dict["position"][0],
            btm_dict["position"][1],
            btm_dict["height"],
        )
        t_half_height = top_dict["height"] / 2

        if ADD_WHITE_NOISE:
            g_tx += np.random.uniform(low=-0.015, high=0.015)
            g_ty += np.random.uniform(low=-0.015, high=0.015)
            t_half_height += np.random.uniform(low=-0.01, high=0.01)
            p_tx += np.random.uniform(low=-0.015, high=0.015)
            p_ty += np.random.uniform(low=-0.015, high=0.015)
            p_tz += np.random.uniform(low=-0.015, high=0.015)

        if TEST_PLACING:
            # overwrite ptz
            p_tz = 0.0

        is_box = int(top_dict["shape"] == "box")

        dist = CLOSE_THRES * 2.0 if LONG_MOVE else CLOSE_THRES
        if is_close(top_dict, btm_dict, dist=dist):
            continue  # discard & re-sample
        else:
            all_dicts = [top_dict, btm_dict]
            gen_surrounding_objs(all_dicts)
            del top_dict, btm_dict
            break

    """Imaginary arm session to get q_reach"""

    if USE_GV5:
        sess = ImaginaryArmObjSession()
        Qreach = np.array(
            sess.get_most_comfortable_q_and_refangle(g_tx, g_ty)[0])
        del sess
    else:
        # maybe not necessary to create table and robot twice. Decide later
        desired_obj_pos = [g_tx, g_ty, 0.0]

        table_id = utils.create_table(FLOOR_MU)

        robot = InmoovShadowNew(
            init_noise=False, timestep=utils.TS, np_random=np.random,
        )

        Qreach = utils.get_n_optimal_init_arm_qs(
            robot,
            utils.PALM_POS_OF_INIT,
            p.getQuaternionFromEuler(utils.PALM_EULER_OF_INIT),
            desired_obj_pos,
            table_id,
            wrist_gain=3.0,
        )[0]

        p.resetSimulation()

    if USE_HEIGHT_INFO:
        desired_obj_pos = [p_tx, p_ty, utils.PLACE_START_CLEARANCE + p_tz]
    else:
        if TEST_PLACING:
            desired_obj_pos = [p_tx, p_ty, utils.PLACE_START_CLEARANCE + 0.0]
        else:
            desired_obj_pos = [p_tx, p_ty,
                               utils.PLACE_START_CLEARANCE + utils.H_MAX]

    table_id = utils.create_table(FLOOR_MU)

    robot = InmoovShadowNew(
        init_noise=False, timestep=utils.TS, np_random=np.random,)

    Qdestin = utils.get_n_optimal_init_arm_qs(
        robot, p_pos_of_ave, p_quat_of_ave, desired_obj_pos, table_id
    )[0]
    del table_id, robot, desired_obj_pos
    p.resetSimulation()

    """Clean up the simulation, since this is only imaginary."""

    """Setup Bullet world."""
    """ Create table, robot, bottom obj, top obj"""
    p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
    p.setPhysicsEngineParameter(deterministicOverlappingPairs=DET_CONTACT)
    p.setTimeStep(utils.TS)
    p.setGravity(0, 0, -utils.GRAVITY)

    table_id = utils.create_table(FLOOR_MU)

    env_core = InmoovShadowHandDemoEnvV4(
        np_random=np.random,
        init_noise=INIT_NOISE,
        timestep=utils.TS,
        withVel=False,
        diffTar=True,
        robot_mu=HAND_MU,
        control_skip=GRASPING_CONTROL_SKIP,
        sleep=DUMMY_SLEEP,
    )
    env_core.change_init_fin_q(INIT_FIN_Q)

    # Qreach = [-0.23763184568016021, 0.3488536398002119, -0.7548165106788512,
    #     -1.9150681944827617, -0.7430404958659877, 0.1319044308197112, -1.02087799385173826]
    # Qreach = [-0.23763184568016021, 0.3488536398002119, -0.4548165106788512,
    #     -2.1150681944827617, -0.7430404958659877, 0.1319044308197112, -1.22087799385173826]
    # Qreach = [-0.23763184568016021, 0.5088536398002119, -0.4548165106788512,
    #     -2.1150681944827617, -0.7430404958659877, 0.1319044308197112, -1.22087799385173826]
    # Qreach = [-0.23763184568016021, 0.5088536398002119, -0.5548165106788512,
    #     -2.1150681944827617, -0.7430404958659877, 0.1319044308197112, -1.22087799385173826]
    # env_core.robot.reset_with_certain_arm_q(Qreach)
    # input("press enter")


    objs, top_id, btm_id = load_obj_and_construct_state(all_dicts)
    OBJECTS = construct_obj_array_for_openrave(all_dicts)

    # p p.getBasePositionAndOrientation(btm_id)
    """ Flip y positions """
    for i in range(len(OBJECTS)):
        OBJECTS[i][1] *= -1
    """
    {2: {'shape': 'box', 'radius': 0.035941487572190636, 'height': 0.1737531760717065, 'position': [0.1985981609573155, -0.229288899069151, 0.08687658803585326], 'orientation': (0.0, 0.0, 0.2847587652805973, 0.9585992100955799), 'mass': 3.5, 'mu': 1.0},
    """
    for key, value in objs.items():
        objs[key]['position'][1] *= -1
        objs[key]['orientation'] = (-objs[key]['orientation'][0], objs[key]['orientation'][1], -objs[key]['orientation'][2], objs[key]['orientation'][3])

    # state_saver.track(
    #     trial=trial,
    #     task="stack",
    #     tx_act=0.0,
    #     ty_act=0.0,
    #     tx=0.0,
    #     ty=0.0,
    #     odicts=objs,
    #     robot_id=env_core.robot.arm_id,
    # )

    """Prepare for grasping. Reach for the object."""

    print(f"Qreach: {Qreach}")

    if WITH_REACHING:
        env_core.robot.reset_with_certain_arm_q([0.0] * 7)
        reach_save_path = homedir + "/container_data_left/PB_REACH.npz"
        reach_read_path = homedir + "/container_data_left/OR_REACH.npz"
        Traj_reach = openrave.get_traj_from_openrave_container(
            OBJECTS, np.array(
                [0.0] * 7), Qreach, reach_save_path, reach_read_path
        )

        if Traj_reach is None or len(Traj_reach) == 0:
            p.resetSimulation()
            print("*******", success_count * 1.0 / (trial + 1))
            continue  # reaching failed
        else:
            planning(Traj_reach)

        print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    else:
        env_core.robot.reset_with_certain_arm_q(Qreach)
        # input("press enter")

    g_obs = get_grasp_policy_obs_tensor(g_tx, g_ty, t_half_height, is_box)

    """Grasp"""
    env_core.change_control_skip_scaling(           # demo uses 12
        c_skip=GRASPING_CONTROL_SKIP
    )
    control_steps = 0
    for i in range(GRASP_END_STEP):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = g_actor_critic.act(
                g_obs, recurrent_hidden_states, masks, deterministic=args.det
            )


        for i in range(GRASPING_CONTROL_SKIP):
            env_core.step_sim(policy.unwrap_action(action, IS_CUDA))
            # state_saver.save_state()

        env_core.step(policy.unwrap_action(action, IS_CUDA))

        g_obs = get_grasp_policy_obs_tensor(g_tx, g_ty, t_half_height, is_box)

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

    # state = get_relative_state_for_reset(top_id)
    # print("after grasping", state)
    # print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    # # input("after grasping")

    """Send move command to OpenRAVE"""
    Qmove_init = env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0]
    print(f"Qmove_init: {Qmove_init}")
    print(f"Qdestin: {Qdestin}")
    move_save_path = homedir + "/container_data_left/PB_MOVE.npz"
    move_read_path = homedir + "/container_data_left/OR_MOVE.npz"
    Traj_move = openrave.get_traj_from_openrave_container(
        OBJECTS, Qmove_init, Qdestin, move_save_path, move_read_path
    )

    """Execute planned moving trajectory"""

    if Traj_move is None or len(Traj_move) == 0:
        p.resetSimulation()
        print("*******", success_count * 1.0 / (trial + 1))
        continue  # transporting failed
    else:
        planning(Traj_move)

    print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    # input("after moving")

    print("palm", env_core.robot.get_link_pos_quat(env_core.robot.ee_id))

    # pose_saver.get_poses()
    # print(f"Pose before placing")
    # pprint.pprint(pose_saver.poses[-1])
    #
    # input("ready to place")
    # ##### fake: reset###
    # # reset only arm but not obj/finger
    # # reset obj/finger but not arm
    # # reset finger vel/obj vel only
    # # reset obj but not arm/finger -- good
    # # reset obj vel but not pos -- somewhat good
    # # reset obj but not arm/finger
    #
    # # # TODO:tmp
    # # state = get_relative_state_for_reset(top_id)
    # # print("after grasping", state)
    #
    # o_pos_pf = state['obj_pos_in_palm']
    # o_quat_pf = state['obj_quat_in_palm']
    # all_fin_q_init = state['all_fin_q']
    # tar_fin_q_init = state['fin_tar_q']
    # # env_core.robot.reset_with_certain_arm_q_finger_states(Qdestin, all_fin_q_init, tar_fin_q_init)
    # # env_core.robot.reset_only_certain_finger_states(all_fin_q_init, tar_fin_q_init)
    #
    # p_pos, p_quat = env_core.robot.get_link_pos_quat(env_core.robot.ee_id)
    # o_pos, o_quat = p.multiplyTransforms(p_pos, p_quat, o_pos_pf, o_quat_pf)
    # p.resetBasePositionAndOrientation(top_id, o_pos, o_quat)
    # p.stepSimulation()
    # # env_core.robot.reset_with_certain_arm_q_finger_states(Qdestin, all_fin_q_init, tar_fin_q_init)
    # # env_core.robot.reset_only_certain_finger_states(all_fin_q_init, tar_fin_q_init)
    # p.resetBasePositionAndOrientation(top_id, o_pos, o_quat)
    # p.stepSimulation()
    # #####
    # # input("reset")

    """Prepare for placing"""
    env_core.change_control_skip_scaling(c_skip=PLACING_CONTROL_SKIP)

    t_pos, t_up, b_pos, b_up, t_half_height = get_stacking_obs(
        obj_state=objs, top_oid=top_id, btm_oid=btm_id,
    )

    l_t_pos, l_t_up, l_b_pos, l_b_up, l_t_half_height = (
        t_pos,
        t_up,
        b_pos,
        b_up,
        t_half_height,
    )

    # an ugly hack to force Bullet compute forward kinematics
    _ = get_stack_policy_obs_tensor(
        p_tx, p_ty, p_tz, t_half_height, is_box, t_pos, t_up, b_pos, b_up
    )
    p_obs = get_stack_policy_obs_tensor(
        p_tx, p_ty, p_tz, t_half_height, is_box, t_pos, t_up, b_pos, b_up
    )
    # print("pobs", p_obs)
    # input("ready to place")

    """Execute placing"""
    env_core.change_control_skip_scaling(
        c_skip=PLACING_CONTROL_SKIP
    )
    print(f"Executing placing...")
    for i in tqdm(range(PLACE_END_STEP)):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = p_actor_critic.act(
                p_obs, recurrent_hidden_states, masks, deterministic=args.det
            )

        for i in range(PLACING_CONTROL_SKIP):
            env_core.step_sim(policy.unwrap_action(action, IS_CUDA))
            # state_saver.save_state()
        # env_core.step(policy.unwrap_action(action, IS_CUDA))

        if (i + 1) % VISION_DELAY == 0:
            l_t_pos, l_t_up, l_b_pos, l_b_up, l_t_half_height = (
                t_pos,
                t_up,
                b_pos,
                b_up,
                t_half_height,
            )
            t_pos, t_up, b_pos, b_up, t_half_height = get_stacking_obs(
                obj_state=objs, top_oid=top_id, btm_oid=btm_id,
            )

        p_obs = get_stack_policy_obs_tensor(
            p_tx, p_ty, p_tz, l_t_half_height, is_box, l_t_pos, l_t_up, l_b_pos, l_b_up
        )

        # print(action)
        # print(p_obs)
        # input("press enter g_obs")

        masks.fill_(1.0)
        # pose_saver.get_poses()

    # print(f"Pose after placing")
    # pprint.pprint(pose_saver.poses[-1])

    if WITH_RETRACT:
        print(f"Starting release trajectory")
        Qretract_init = env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0]
        # Qretract_end = [-0.238, 0.349, -0.755,
        #                 -1.915, -0.743, 0.132,
        #                 -1.021]
        Qretract_end = [0.0] * 7

        retract_save_path = homedir + "/container_data_left/PB_RETRACT.npz"
        retract_read_path = homedir + "/container_data_left/OR_RETRACT.npz"

        # note: p_tz is 0 for placing
        OBJECTS[0, :] = np.array([p_tx, p_ty, p_tz, 0.0])

        Traj_reach = openrave.get_traj_from_openrave_container(
            OBJECTS, Qretract_init,  Qretract_end, retract_save_path, retract_read_path
        )

        if Traj_reach is None or len(Traj_reach) == 0:
            p.resetSimulation()
            print("*******", success_count * 1.0 / (trial + 1))
            continue  # retracting failed
        else:
            planning(Traj_reach, retract_stage=True)

    t_pos, t_quat = p.getBasePositionAndOrientation(top_id)
    if (
        t_pos[2] - p_tz > 0.05
        and (t_pos[0] - p_tx) ** 2 + (t_pos[1] - p_ty) ** 2 < 0.1 ** 2
    ):
        # TODO: ptz noisy a very rough check
        success_count += 1

    openrave_success_count += 1
    p.resetSimulation()
    print("*******", success_count * 1.0 / (trial + 1), trial + 1)
    print(
        "******* w/o OR",
        success_count * 1.0 / openrave_success_count,
        openrave_success_count,
    )

p.disconnect()
print("*******total", success_count * 1.0 / NUM_TRIALS)
print("*******total w/o OR", success_count * 1.0 / openrave_success_count)

f = open("final_stats.txt", "a")
f.write(f"*******total: {success_count * 1.0 / NUM_TRIALS:.3f})")
f.write(
    f"*******total w/o OR: {success_count * 1.0 / openrave_success_count:.3f})")
f.write("\n")
f.close()
