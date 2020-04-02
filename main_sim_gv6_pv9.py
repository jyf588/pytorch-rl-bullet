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
import math

import numpy as np
import torch

import my_pybullet_envs
import pybullet as p
import time

import inspect
from NLP_module import NLPmod
# from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import (
#     ImaginaryArmObjSession,
# )

from my_pybullet_envs.inmoov_shadow_place_env_v9 import (
    InmoovShadowHandPlaceEnvV9,
)
from my_pybullet_envs.inmoov_shadow_demo_env_v4 import (
    InmoovShadowHandDemoEnvV4,
)

no_vision = False
try:
    from ns_vqa_dart.bullet.state_saver import StateSaver
    from ns_vqa_dart.bullet.vision_inference import VisionInference
    import ns_vqa_dart.bullet.util as util
except ImportError:
    no_vision = True
# from pose_saver import PoseSaver

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
homedir = os.path.expanduser("~")


# TODO: main module depends on the following code/model:
# demo env: especially observation  # change obs vec (note diffTar)
# the settings of inmoov hand v2
# obj sizes & frame representation & friction & obj xy range
# frame skip
# vision delay

# what is different from cyl env?
# 1. policy load names
# 2. obj load
# 3. tmp add obj obs, some policy does not use GT
# 5. 4 grid / 6 grid

"""Parse arguments"""
sys.path.append("a2c_ppo_acktr")
parser = argparse.ArgumentParser(description="RL")
parser.add_argument("--seed", type=int, default=101)  # numpy and env seeds
parser.add_argument("--non-det", type=int, default=0)
parser.add_argument("--use_vision", action="store_true")
parser.add_argument(
    "--pose_path",
    type=str,
    default="main_sim_stack_new.json",
    help="The path to the json file where poses are saved.",
)
parser.add_argument("--scene", type=int, help="The scene to use.")
parser.add_argument("--shape", type=str, help="Shape of top shape.")
parser.add_argument("--size", type=str, help="Shape of top size.")
args = parser.parse_args()
args.det = not args.non_det

# np.random.seed(11)     # turn this off so each scene will have different random shapes

"""Configurations."""

import demo_scenes_rand_size

gt_odicts = demo_scenes_rand_size.SCENES[args.scene]

# TODO: merge some config with utils

PLACE_FLOOR = False  # TODO: language
g_tz = 0.0
# TODO:tmp
if args.scene == 4:
    g_tz = 0.16

MIX_SHAPE_PI = True

SAVE_POSES = True  # Whether to save object and robot poses to a JSON file.
USE_VISION_MODULE = args.use_vision and (not no_vision)
RENDER = True  # If true, uses OpenGL. Else, uses TinyRenderer.

GRASP_END_STEP = 30
PLACE_END_STEP = 50

STATE_NORM = False

INIT_NOISE = True
DET_CONTACT = 0  # 0 false, 1 true

OBJ_MU = 1.0
FLOOR_MU = 1.0
HAND_MU = 1.0

IS_CUDA = True  # TODO:tmp odd. seems no need to use cuda
DEVICE = "cuda" if IS_CUDA else "cpu"

TS = 1.0 / 240


# HALF_OBJ_HEIGHT_L = 0.09
# HALF_OBJ_HEIGHT_S = 0.065
# SIZE2HALF_H = {"small": HALF_OBJ_HEIGHT_S, "large": HALF_OBJ_HEIGHT_L}
# SHAPE2SIZE2RADIUS = {
#     "box": {"small": 0.025, "large": 0.04},
#     "cylinder": {"small": 0.04, "large": 0.05},
# }

# Ground-truth scene:
HIDE_SURROUNDING_OBJECTS = False  # If true, hides the surrounding objects.


top_obj_idx = 1  # TODO: in fact, moved obj, infer from language
btm_obj_idx = 2
# TODO: in fact, reference obj (place between not considered), infer from language

# Override the shape and size of the top object if provided as arguments. TODO: Language
if args.shape is not None:
    gt_odicts[top_obj_idx]["shape"] = args.shape

# # should not have size attr
# if args.size is not None:
#     gt_odicts[top_obj_idx]["size"] = args.size

if HIDE_SURROUNDING_OBJECTS:
    gt_odicts = [gt_odicts[top_obj_idx], gt_odicts[btm_obj_idx]]
    top_obj_idx = 0
    btm_obj_idx = 1

# top_size = gt_odicts[top_obj_idx]["size"]
# btm_size = gt_odicts[btm_obj_idx]["size"]
# P_TZ = SIZE2HALF_H[btm_size] * 2
# T_HALF_HEIGHT = SIZE2HALF_H[top_size]


IS_BOX = gt_odicts[top_obj_idx]["shape"] == "box"  # TODO: infer from language

if MIX_SHAPE_PI:
    GRASP_PI = "0325_graspco_16_n_w0_25_45"
    GRASP_DIR = "./trained_models_%s/ppo/" % "0325_graspco_16_n_w0"  # TODO
    # PLACE_PI = "0313_2_placeco_0316_1"  # 50ms
    PLACE_PI = "0325_graspco_16_n_w0_placeco_0331_2"  # 50ms
    PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI
else:
    if IS_BOX:
        GRASP_PI = "0311_box_2_n_20_50"
        GRASP_DIR = "./trained_models_%s/ppo/" % "0311_box_2_n"  # TODO
        PLACE_PI = "0311_box_2_placeco_0316_0"  # 50ms
        PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI
    else:
        GRASP_PI = "0311_cyl_2_n_20_50"
        GRASP_DIR = "./trained_models_%s/ppo/" % "0311_cyl_2_n"  # TODO
        PLACE_PI = "0311_cyl_2_placeco_0316_0"  # 50ms
        PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI

if IS_BOX:
    if PLACE_FLOOR:
        sentence = "Put the green box in front of the blue cylinder"
    else:
        sentence = "Put the green box on top of the blue cylinder"
else:
    if PLACE_FLOOR:
        sentence = "Put the green cylinder in front of the blue cylinder"
    else:
        sentence = "Put the green cylinder on top of the blue cylinder"

GRASP_PI_ENV_NAME = "InmoovHandGraspBulletEnv-v6"
PLACE_PI_ENV_NAME = "InmoovHandPlaceBulletEnv-v9"

USE_VISION_DELAY = True
VISION_DELAY = 2
PLACING_CONTROL_SKIP = 6
GRASPING_CONTROL_SKIP = 6


def planning(Traj, recurrent_hidden_states, masks, pose_saver=None):
    print("end of traj", Traj[-1, 0:7])
    for ind in range(0, len(Traj)):
        tar_armq = Traj[ind, 0:7]
        env_core.robot.tar_arm_q = tar_armq
        env_core.robot.apply_action([0.0] * 24)
        p.stepSimulation()
        time.sleep(TS)
        # pose_saver.get_poses()

    for _ in range(50):
        # print(env_core.robot.tar_arm_q)
        env_core.robot.tar_arm_q = tar_armq
        env_core.robot.apply_action([0.0] * 24)  # stay still for a while
        p.stepSimulation()
        # pose_saver.get_poses()
        # print("act", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    #     #time.sleep(1. / 240.)


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

    state = {
        "obj_pos_in_palm": o_p_hf,
        "obj_quat_in_palm": o_q_hf,
        "all_fin_q": fin_q,
        "fin_tar_q": env_core.robot.tar_fin_q,
    }
    return state


def load_policy_params(dir, env_name, iter=None):
    if iter is not None:
        path = os.path.join(dir, env_name + "_" + str(iter) + ".pt")
    else:
        path = os.path.join(dir, env_name + ".pt")
    if IS_CUDA:
        actor_critic, ob_rms = torch.load(path)
    else:
        actor_critic, ob_rms = torch.load(path, map_location="cpu")
    # vec_norm = get_vec_normalize(env) # TODO: assume no state normalize
    # if not STATE_NORM: assert ob_rms is None
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.ob_rms = ob_rms
    recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size
    )
    masks = torch.zeros(1, 1)
    return (
        actor_critic,
        ob_rms,
        recurrent_hidden_states,
        masks,
    )  # probably only first one is used


def wrap_over_grasp_obs(obs):
    obs = torch.Tensor([obs])
    if IS_CUDA:
        obs = obs.cuda()
    return obs


def unwrap_action(act_tensor):
    action = act_tensor.squeeze()
    action = action.cpu() if IS_CUDA else action
    return action.numpy()


def get_traj_from_openrave_container(
    objs, q_start, q_end, save_file_path, read_file_path
):

    if q_start is not None:
        np.savez(save_file_path, objs, q_start, q_end)  # move
    else:
        np.savez(save_file_path, objs, q_end)  # reach has q_start 0

    # Wait for command from OpenRave

    assert not os.path.exists(read_file_path)
    while not os.path.exists(read_file_path):
        time.sleep(0.2)
    if os.path.isfile(read_file_path):
        traj = np.load(read_file_path)
        print("loaded")
        try:
            os.remove(read_file_path)
            print("deleted")
            # input("press enter")
        except OSError as e:  # name the Exception `e`
            print("Failed with:", e.strerror)  # look what it says
            # input("press enter")
    else:
        raise ValueError("%s isn't a file!" % read_file_path)
    print("Trajectory obtained from OpenRave!")
    # input("press enter")
    return traj


def construct_obj_dict_bullet(odicts):
    odicts_internal = []
    for odict in odicts:
        odict_new = {}

        odict_new["mass"] = np.random.uniform(utils.MASS_MIN, utils.MASS_MAX)
        odict_new["mu"] = OBJ_MU
        odict_new["height"] = np.random.uniform(utils.H_MIN, utils.H_MAX)

        # TODO:tmp
        if args.scene == 4 and odict["color"] == "red":
            odict_new["height"] = 0.16

        odict_new["half_width"] = np.random.uniform(
            utils.HALF_W_MIN, utils.HALF_W_MAX
        )
        odict_new["shape"] = utils.SHAPE_NAME_MAP[odict["shape"]]
        if odict_new["shape"] == p.GEOM_BOX:
            odict_new["half_width"] *= 0.8
        elif odict_new["shape"] == p.GEOM_SPHERE:
            odict_new["height"] *= 0.75
        odict_new["color"] = utils.COLOR2RGBA[odict["color"]]

        odicts_internal.append(odict_new)

    return odicts_internal


def construct_bullet_scene(odicts, odicts_internal):
    # p.resetSimulation()
    table_id = p.loadURDF(
        os.path.join(currentdir, "my_pybullet_envs/assets/tabletop.urdf"),
        utils.TABLE_OFFSET,
        useFixedBase=1,
    )
    p.changeVisualShape(table_id, -1, rgbaColor=utils.COLOR2RGBA["grey"])
    p.changeDynamics(table_id, -1, lateralFriction=FLOOR_MU)

    for o_ind in range(len(odicts)):

        odict = odicts[o_ind]
        odict_internal = odicts_internal[o_ind]

        assert len(odict["position"]) == 4
        # x y and z height, TODO figure out this, z and height seems unnecessary

        real_loc = np.array(odict["position"][0:3])
        real_loc += [0.0, 0, odict_internal["height"] / 2.0]
        quat = p.getQuaternionFromEuler(
            [0.0, 0.0, np.random.uniform(low=0, high=2.0 * math.pi)]
        )
        odict_internal["id"] = utils.create_sym_prim_shape_helper(
            odict_internal, real_loc, quat
        )


def get_stacking_obs(
    top_oid: int,
    btm_oid: int,
    use_vision: bool,
    vision_module=None,
    verbose: Optional[bool] = False,
):
    """Retrieves stacking observations.

    Args:
        top_oid: The object ID of the top object.
        btm_oid: The object ID of the bottom object.
        use_vision: Whether to use vision or GT.
        vision_module: The vision module to use to generate predictions.

    Returns:
        t_pos: The xyz position of the top object.
        t_up: The up vector of the top object.
        b_pos: The xyz position of the bottom object.
        b_up: The up vector of the bottom object.
        t_half_height: Half of the height of the top object.
    """
    if use_vision:
        top_odict, btm_odict = stacking_vision_module.predict(
            client_oids=[top_oid, btm_oid]
        )
        t_pos = top_odict["position"]
        b_pos = btm_odict["position"]
        t_up = top_odict["up_vector"]
        b_up = top_odict["up_vector"]
        t_half_height = top_odict["height"] / 2

        if verbose:
            print(f"Stacking vision module predictions:")
            pprint.pprint(top_odict)
            pprint.pprint(btm_odict)
    else:
        t_pos, t_quat = p.getBasePositionAndOrientation(top_oid)
        b_pos, b_quat = p.getBasePositionAndOrientation(btm_oid)

        rot = np.array(p.getMatrixFromQuaternion(t_quat))
        t_up = [rot[2], rot[5], rot[8]]
        rot = np.array(p.getMatrixFromQuaternion(b_quat))
        b_up = [rot[2], rot[5], rot[8]]

        t_half_height = gt_odicts_internal[top_obj_idx]["height"] / 2.0     # TODO: duplicate
    return t_pos, t_up, b_pos, b_up, t_half_height


"""Pre-calculation & Loading"""
# latter 2 returns dummy
g_actor_critic, g_ob_rms, _, _ = load_policy_params(
    GRASP_DIR, GRASP_PI_ENV_NAME
)
p_actor_critic, p_ob_rms, recurrent_hidden_states, masks = load_policy_params(
    PLACE_DIR, PLACE_PI_ENV_NAME
)

gt_odicts_internal = construct_obj_dict_bullet(gt_odicts)

"""Vision and language"""
if USE_VISION_MODULE:
    # Construct the bullet scene using DIRECT rendering, because that's what
    # the vision module was trained on.
    vision_p = util.create_bullet_client(mode="direct")
    # odicts_new = construct_bullet_scene(odicts=gt_odicts)

    # Initialize the vision module for initial planning. We apply camera offset
    # because the default camera position is for y=0, but the table is offset
    # in this case.
    state_saver = StateSaver(p=vision_p)
    for obj_i in range(len(gt_odicts_internal)):
        odict = gt_odicts[obj_i]
        shape = odict["shape"]
        # size = odict["size"]
        state_saver.track_object(
            oid=gt_odicts_internal[obj_i]["id"],
            shape=shape,
            color=odict["color"],
            radius=gt_odicts_internal[obj_i][
                "half_width"
            ],  # TODO: streamlize naming
            height=gt_odicts_internal[obj_i]["height"],
        )
    initial_vision_module = VisionInference(
        state_saver=state_saver,
        checkpoint_path="/home/michelle/outputs/ego_v009/checkpoint_best.pt",
        camera_position=[
            -0.20450591046900168,
            0.03197646764976494,
            0.4330631992464512,
        ],
        camera_offset=[-0.05, utils.TABLE_OFFSET[1], 0.0],
        camera_directed_offset=[0.02, 0.0, 0.0],
        apply_offset_to_preds=True,
        html_dir="/home/michelle/html/vision_inference_initial",
    )

    # initial_vision_module = VisionInference(
    #     state_saver=state_saver,
    #     checkpoint_path="/home/michelle/outputs/stacking_v003/checkpoint_best.pt",
    #     camera_position=[-0.2237938867122504, 0.0, 0.5425],
    #     camera_offset=[0.0, TABLE_OFFSET[1], 0.0],
    #     apply_offset_to_preds=False,
    #     html_dir="/home/michelle/html/demo_delay_vision_v003_{top_shape}",
    # )
    pred_odicts = initial_vision_module.predict(
        client_oids=[odict_new["id"] for odict_new in gt_odicts_internal]
    )

    # Artificially pad with a fourth dimension because language module
    # expects it.
    for i in range(len(pred_odicts)):
        pred_odicts[i]["position"] = pred_odicts[i]["position"] + [0.0]

    print(f"Vision module predictions:")
    pprint.pprint(pred_odicts)
    vision_p.disconnect()
    language_input_objs = pred_odicts
else:
    language_input_objs = gt_odicts
    initial_vision_module = None
    # stacking_vision_module = None

[OBJECTS, placing_xyz] = NLPmod(sentence, language_input_objs)
print("placing xyz from language", placing_xyz)

# Define the grasp position.
if USE_VISION_MODULE:
    top_pos = pred_odicts[top_obj_idx]["position"]
    g_half_h = pred_odicts[top_obj_idx]["height"] / 2
else:
    top_pos = gt_odicts[top_obj_idx]["position"]
    # TODO: naming, init position actually
    g_half_h = gt_odicts_internal[top_obj_idx]["height"] / 2.0
g_tx, g_ty = top_pos[0], top_pos[1]
print(f"Grasp position: ({g_tx}, {g_ty})\theight: {g_half_h}")

if PLACE_FLOOR:
    p_tx, p_ty, p_tz = placing_xyz[0], placing_xyz[1], placing_xyz[2]
else:
    # Define the target xyz position to perform placing.
    p_tx, p_ty = placing_xyz[0], placing_xyz[1]
    if USE_VISION_MODULE:
        # Temp: replace with GT
        # p_tx = gt_odicts[btm_obj_idx]["position"][0]
        # p_ty = gt_odicts[btm_obj_idx]["position"][1]
        # p_tz = P_TZ
        p_tz = pred_odicts[btm_obj_idx]["height"]
    else:
        p_tz = gt_odicts_internal[btm_obj_idx]["height"]
print(f"Placing position: ({p_tx}, {p_ty}, {p_tz})")

"""Start Bullet session."""
if RENDER:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

"""Imaginary arm session to get q_reach"""
# sess = ImaginaryArmObjSession()
#
# Qreach = np.array(sess.get_most_comfortable_q_and_refangle(g_tx, g_ty)[0])



a = InmoovShadowHandPlaceEnvV9(renders=False, grasp_pi_name=GRASP_PI)
a.seed(args.seed)
# TODO:tmp, get_n_optimal_init_arm_qs need to do collision checking
table_id = p.loadURDF(
    os.path.join(currentdir, "my_pybullet_envs/assets/tabletop.urdf"),
    utils.TABLE_OFFSET,
    useFixedBase=1,
)

desired_obj_pos = [g_tx, g_ty, g_tz]
Qreach = utils.get_n_optimal_init_arm_qs(a.robot, utils.PALM_POS_OF_INIT,
                                         p.getQuaternionFromEuler(utils.PALM_EULER_OF_INIT),
                                         desired_obj_pos, table_id, wrist_gain=3.0)[0]  # TODO

desired_obj_pos = [p_tx, p_ty, utils.PLACE_START_CLEARANCE + p_tz]
p_pos_of_ave, p_quat_of_ave = p.invertTransform(
    a.o_pos_pf_ave, a.o_quat_pf_ave
)
# TODO: [1] is the 2nd candidate
Qdestin = utils.get_n_optimal_init_arm_qs(
    a.robot, p_pos_of_ave, p_quat_of_ave, desired_obj_pos, table_id
)[0]

print("place arm q", Qdestin)
p.resetSimulation()  # Clean up the simulation, since this is only imaginary.

"""Setup Bullet world."""
p.setPhysicsEngineParameter(numSolverIterations=utils.BULLET_CONTACT_ITER)
p.setPhysicsEngineParameter(deterministicOverlappingPairs=DET_CONTACT)
p.setTimeStep(TS)
p.setGravity(0, 0, -10)

# Load bullet objects again, since they were cleared out by the imaginary
# arm session.
print(f"Loading objects:")
pprint.pprint(gt_odicts)

env_core = InmoovShadowHandDemoEnvV4(
    seed=args.seed,
    init_noise=INIT_NOISE,
    timestep=TS,
    withVel=False,
    diffTar=True,
    robot_mu=HAND_MU,
    control_skip=GRASPING_CONTROL_SKIP,
)  # TODO: does obj/robot order matter

env_core.robot.reset_with_certain_arm_q([0.0] * 7)

construct_bullet_scene(gt_odicts, gt_odicts_internal)

top_oid = gt_odicts_internal[top_obj_idx]["id"]
btm_oid = gt_odicts_internal[btm_obj_idx]["id"]

# # Initialize a PoseSaver to save poses throughout robot execution.
# pose_saver = PoseSaver(
#     path=args.pose_path,
#     odicts=gt_odicts_internal,
#     # oids=[gt_odict_new["id"] for gt_odict_new in gt_odicts_internal],
#     robot_id=env_core.robot.arm_id,
# )

"""Prepare for grasping. Reach for the object."""

print(f"Qreach: {Qreach}")
reach_save_path = homedir + "/container_data/PB_REACH.npz"
reach_read_path = homedir + "/container_data/OR_REACH.npy"
Traj_reach = get_traj_from_openrave_container(
    OBJECTS, None, Qreach, reach_save_path, reach_read_path
)

planning(Traj_reach, recurrent_hidden_states, masks)
# input("press enter")
# env_core.robot.reset_with_certain_arm_q(Qreach)
# input("press enter 2")

# pose_saver.get_poses()
# print(f"Pose after reset")
# pprint.pprint(pose_saver.poses[-1])

g_obs = env_core.get_robot_contact_txtytz_halfh_shape_obs_no_dup(g_tx, g_ty, g_tz, g_half_h, IS_BOX)
g_obs = wrap_over_grasp_obs(g_obs)

"""Grasp"""
control_steps = 0
for i in range(GRASP_END_STEP):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = g_actor_critic.act(
            g_obs, recurrent_hidden_states, masks, deterministic=args.det
        )

    env_core.step(unwrap_action(action))
    g_obs = env_core.get_robot_contact_txtytz_halfh_shape_obs_no_dup(g_tx, g_ty, g_tz, g_half_h, IS_BOX)
    g_obs = wrap_over_grasp_obs(g_obs)

    # print(g_obs)
    # print(action)
    # print(control_steps)
    # control_steps += 1
    # input("press enter g_obs")
    masks.fill_(1.0)
    # pose_saver.get_poses()

# print(f"Pose after grasping")
# pprint.pprint(pose_saver.poses[-1])

final_g_obs = copy.copy(g_obs)
del g_obs, g_tx, g_ty, g_tz, g_actor_critic, g_ob_rms, g_half_h

state = get_relative_state_for_reset(top_oid)
print("after grasping", state)
print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
# input("after grasping")

"""Send move command to OpenRAVE"""
Qmove_init = env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0]
print(f"Qmove_init: {Qmove_init}")
print(f"Qdestin: {Qdestin}")
move_save_path = homedir + "/container_data/PB_MOVE.npz"
move_read_path = homedir + "/container_data/OR_MOVE.npy"
Traj_move = get_traj_from_openrave_container(
    OBJECTS, Qmove_init, Qdestin, move_save_path, move_read_path
)

"""Execute planned moving trajectory"""
planning(Traj_move, recurrent_hidden_states, masks)
print("after moving", get_relative_state_for_reset(top_oid))
print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
# input("after moving")

print("palm", env_core.robot.get_link_pos_quat(env_core.robot.ee_id))

# pose_saver.get_poses()
# print(f"Pose before placing")
# pprint.pprint(pose_saver.poses[-1])

"""Prepare for placing"""
env_core.change_control_skip_scaling(c_skip=PLACING_CONTROL_SKIP)

if USE_VISION_MODULE:
    # Initialize the vision module for stacking.
    top_shape = gt_odicts[top_obj_idx]["shape"]
    state_saver = StateSaver(p=p)
    state_saver.set_robot_id(env_core.robot.arm_id)
    for obj_i in range(len(gt_odicts_internal)):
        odict = gt_odicts[obj_i]
        shape = odict["shape"]
        # size = odict["size"]
        state_saver.track_object(
            oid=gt_odicts_internal[obj_i]["id"],
            shape=shape,
            color=odict["color"],
            radius=gt_odicts_internal[obj_i]["half_width"],
            height=gt_odicts_internal[obj_i]["height"],
        )
    stacking_vision_module = VisionInference(
        state_saver=state_saver,
        checkpoint_path="/home/michelle/outputs/stacking_v003/checkpoint_best.pt",
        camera_position=[-0.2237938867122504, 0.0, 0.5425],
        camera_offset=[0.0, utils.TABLE_OFFSET[1], 0.0],
        apply_offset_to_preds=False,
        html_dir="/home/michelle/html/demo_delay_vision_v003_{top_shape}",
    )
else:
    stacking_vision_module = None

t_pos, t_up, b_pos, b_up, t_half_height = get_stacking_obs(
    top_oid=top_oid,
    btm_oid=btm_oid,
    use_vision=USE_VISION_MODULE,
    vision_module=stacking_vision_module,
    verbose=True,
)

l_t_pos, l_t_up, l_b_pos, l_b_up, l_t_half_height = (
    t_pos,
    t_up,
    b_pos,
    b_up,
    t_half_height,
)

# TODO: an unly hack to force Bullet compute forward kinematics

# TODO: deprecate IS_BOX.
if MIX_SHAPE_PI:
    p_obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
        p_tx, p_ty, p_tz, t_half_height, IS_BOX, t_pos, t_up, b_pos, b_up
    )
    p_obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
        p_tx, p_ty, p_tz, t_half_height, IS_BOX, t_pos, t_up, b_pos, b_up
    )
else:
    p_obs = env_core.get_robot_contact_txtytz_halfh_2obj6dUp_obs_nodup_from_up(
        p_tx, p_ty, p_tz, t_half_height, t_pos, t_up, b_pos, b_up
    )
    p_obs = env_core.get_robot_contact_txtytz_halfh_2obj6dUp_obs_nodup_from_up(
        p_tx, p_ty, p_tz, t_half_height, t_pos, t_up, b_pos, b_up
    )

p_obs = wrap_over_grasp_obs(p_obs)
print("pobs", p_obs)
# input("ready to place")

"""Execute placing"""
print(f"Executing placing...")
for i in tqdm(range(PLACE_END_STEP)):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = p_actor_critic.act(
            p_obs, recurrent_hidden_states, masks, deterministic=args.det
        )

    env_core.step(unwrap_action(action))

    if USE_VISION_DELAY:
        if (i + 1) % VISION_DELAY == 0:
            l_t_pos, l_t_up, l_b_pos, l_b_up, l_t_half_height = (
                t_pos,
                t_up,
                b_pos,
                b_up,
                t_half_height,
            )
            t_pos, t_up, b_pos, b_up, t_half_height = get_stacking_obs(
                top_oid=top_oid,
                btm_oid=btm_oid,
                use_vision=USE_VISION_MODULE,
                vision_module=stacking_vision_module,
            )
        if MIX_SHAPE_PI:
            p_obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
                p_tx,
                p_ty,
                p_tz,
                l_t_half_height,
                IS_BOX,
                l_t_pos,
                l_t_up,
                l_b_pos,
                l_b_up,
            )
        else:
            p_obs = env_core.get_robot_contact_txtytz_halfh_2obj6dUp_obs_nodup_from_up(
                p_tx,
                p_ty,
                p_tz,
                l_t_half_height,
                l_t_pos,
                l_t_up,
                l_b_pos,
                l_b_up,
            )
    else:
        t_pos, t_quat, b_pos, b_quat, t_half_height = get_stacking_obs(
            top_oid=top_oid,
            btm_oid=btm_oid,
            use_vision=USE_VISION_MODULE,
            vision_module=stacking_vision_module,
        )

        if MIX_SHAPE_PI:
            p_obs = env_core.get_robot_contact_txtytz_halfh_shape_2obj6dUp_obs_nodup_from_up(
                p_tx,
                p_ty,
                p_tz,
                t_half_height,
                IS_BOX,
                t_pos,
                t_up,
                b_pos,
                b_up,
            )
        else:
            p_obs = env_core.get_robot_contact_txtytz_halfh_2obj6dUp_obs_nodup_from_up(
                p_tx, p_ty, p_tz, t_half_height, t_pos, t_up, b_pos, b_up
            )

    p_obs = wrap_over_grasp_obs(p_obs)

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
    time.sleep(TS)
    # pose_saver.get_poses()

# if SAVE_POSES:
#     pose_saver.save()

if USE_VISION_MODULE:
    initial_vision_module.close()
    stacking_vision_module.close()
