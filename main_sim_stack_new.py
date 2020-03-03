import argparse
import pickle
import os
import sys

import numpy as np
import torch

import my_pybullet_envs
import pybullet as p
import time

import inspect
from NLP_module import NLPmod
from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import (
    ImaginaryArmObjSession,
)

# from my_pybullet_envs.inmoov_shadow_place_env_v3 import InmoovShadowHandPlaceEnvV3
from my_pybullet_envs.inmoov_shadow_place_env_v6 import (
    InmoovShadowHandPlaceEnvV6,
)  # TODO
from my_pybullet_envs.inmoov_shadow_demo_env_v3 import (
    InmoovShadowHandDemoEnvV3,
)

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
homedir = os.path.expanduser("~")

# change obs vec (note diffTar)

# TODOs
# A modified bullet/const classes
#   - vary joints the same as send joints it seems, ignore
#   - mods the send_joints to new urdf: 1. no palm aux 2. metacarpal is fixed (just always send 0?)
#   - represent everything in shoulder frame.
#   - fps is problematic, probably want to simulate several steps and send one pose
#   - simplify the current bullet class: only needs to transform pose. (warp around pose)
# Where in the code does it handle variable-size objects
#   - In the C# code
#   - now it is hard-coded in python & c# that there are 3 in the order of box-box-cyl
#   - ideally GT bullet can dump a json file that C# can read and call setObj


# TODO: main module depends on the following code/model:
# demo env: especially observation
# the settings of inmoov hand v2
# obj sizes & frame representation

# what is different from cyl env?
# 1. policy load names
# 2. obj load
# 3. tmp add obj obs, some policy does not use GT
# 4. different release traj
# 5. 4 grid / 6 grid

# TODO:tmp add a flag to always load the same transporting traj
FIX_MOVE = True
FIX_MOVE_PATH = os.path.join(homedir, "container_data/OR_MOVE.npy")

# constants
# DEMO_ENV_NAME = 'ShadowHandDemoBulletEnv-v1'        # TODO: no longer used
# GRASP_PI = '0219_cyl_2'
GRASP_PI = "0219_box_2"
GRASP_DIR = "./trained_models_%s/ppo/" % GRASP_PI
GRASP_PI_ENV_NAME = "InmoovHandGraspBulletEnv-v4"
# PLACE_PI = '0219_cyl_2_place_0220_1'
PLACE_PI = "0219_box_2_place_0220_3"
PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI
PLACE_PI_ENV_NAME = "InmoovHandPlaceBulletEnv-v6"

# # TODO: infer from language
# pi_is_box = True
# pi_is_small = False

GRASP_END_STEP = 90
PLACE_END_STEP = 95

STATE_NORM = False
NOISY_OBS = True
# Noisy tx ty
# Noisy obj 6d if applicable

OBJ_MU = 1.0
FLOOR_MU = 1.0
# HAND_MU = 3.0

sys.path.append("a2c_ppo_acktr")
parser = argparse.ArgumentParser(description="RL")
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--non-det", type=int, default=0)
args = parser.parse_args()
args.det = not args.non_det

IS_CUDA = True  # TODO:tmp odd. seems no need to use cuda
DEVICE = "cuda" if IS_CUDA else "cpu"

TS = 1.0 / 240
TABLE_OFFSET = [
    0.25,
    0.2,
    0.0,
]  # TODO: chaged to 0.2 for vision, 0.25 may collide, need to change OR reaching.
HALF_OBJ_HEIGHT_L = 0.09
HALF_OBJ_HEIGHT_S = 0.065
PLACE_CLEARANCE = 0.14  # TODO: different for diff envs

COLORS = {
    "red": [0.8, 0.0, 0.0, 1.0],
    "grey": [0.4, 0.4, 0.4, 1.0],
    "yellow": [0.8, 0.8, 0.0, 1.0],
    "blue": [0.0, 0.0, 0.8, 1.0],
    "green": [0.0, 0.8, 0.0, 1.0],
}

g_tx = 0.2
g_ty = 0.4
p_tz = 0.18  # TODO: placing on cyl
# Ground-truth scene:
obj1 = {
    "shape": "box",
    "color": "yellow",
    "position": [0.15, 0.7, 0, 0],
    "size": "large",
}  # ref 1
# obj2 = {'shape':'box','color':'red','position': [g_tx,g_ty,0,0],'size': 'small'} # target
# obj2 = {'shape':'cylinder','color':'green','position': [g_tx,g_ty,0,0],'size': 'small'} # target
obj2 = {
    "shape": "box",
    "color": "green",
    "position": [g_tx, g_ty, 0, 0],
    "size": "small",
}  # target
obj3 = {
    "shape": "cylinder",
    "color": "blue",
    "position": [0.2, -0.05, 0, 0],
    "size": "large",
}  # ref 2
obj4 = {
    "shape": "box",
    "color": "yellow",
    "position": [0.0, 0.1, 0, 0],
    "size": "large",
}  # irrelevant
objs = [obj1, obj2, obj3, obj4]
# objs = [obj1, obj2, obj3]
Target_ind = 1  # TODO:tmp
# command
# sentence = "Put the small red box between the blue cylinder and yellow box"
# sentence = "Put the small green cylinder on top of the blue cylinder"
sentence = "Put the small green box on top of the blue cylinder"

BULLET_SOLVER_ITER = 200


def planning(Traj, i_g_obs, recurrent_hidden_states, masks):
    print("end of traj", Traj[-1, 0:7])
    for ind in range(0, len(Traj)):
        tar_armq = Traj[ind, 0:7]
        env_core.robot.tar_arm_q = tar_armq
        env_core.robot.apply_action([0.0] * 24)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    for _ in range(50):
        # print(env_core.robot.tar_arm_q)
        env_core.robot.tar_arm_q = tar_armq
        env_core.robot.apply_action([0.0] * 24)  # stay still for a while
        p.stepSimulation()
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


def construct_bullet_scene(objs):  # TODO: copied from inference code
    # p.resetSimulation()
    obj_ids = []
    for obj in objs:
        ob_shape = obj["shape"]
        assert len(obj["position"]) == 4  # x y and z height

        real_loc = np.array(obj["position"][0:3])

        if obj["size"] == "small":
            ob_shape += "_small"
            real_loc += [0, 0, HALF_OBJ_HEIGHT_S + 0.001]
        else:
            real_loc += [0, 0, HALF_OBJ_HEIGHT_L + 0.001]
        urdf_file = (
            "my_pybullet_envs/assets/" + ob_shape + ".urdf"
        )  # TODO: hardcoded path

        obj_id = p.loadURDF(
            os.path.join(currentdir, urdf_file), real_loc, useFixedBase=0
        )
        p.changeVisualShape(obj_id, -1, rgbaColor=COLORS[obj["color"]])

        p.changeDynamics(obj_id, -1, lateralFriction=OBJ_MU)
        obj_ids.append(obj_id)

    table_id = p.loadURDF(
        os.path.join(currentdir, "my_pybullet_envs/assets/tabletop.urdf"),
        TABLE_OFFSET,
        useFixedBase=1,
    )  # main sim uses 0.27, 0.1/ constuct table at last
    p.changeVisualShape(table_id, -1, rgbaColor=COLORS["grey"])
    p.changeDynamics(table_id, -1, lateralFriction=FLOOR_MU)
    return obj_ids


################# pre-calculation & loading
[OBJECTS, target_xyz] = NLPmod(sentence, objs)
print("tar xyz from language", target_xyz)
p_tx = target_xyz[0]
p_ty = target_xyz[1]

# latter 2 returns dummy
g_actor_critic, g_ob_rms, _, _ = load_policy_params(
    GRASP_DIR, GRASP_PI_ENV_NAME
)
p_actor_critic, p_ob_rms, recurrent_hidden_states, masks = load_policy_params(
    PLACE_DIR, PLACE_PI_ENV_NAME
)

p.connect(p.GUI)
p.resetSimulation()

sess = ImaginaryArmObjSession()
Qreach = np.array(sess.get_most_comfortable_q_and_refangle(g_tx, g_ty)[0])

desired_obj_pos = [p_tx, p_ty, PLACE_CLEARANCE + p_tz]
a = InmoovShadowHandPlaceEnvV6(renders=False, grasp_pi_name=GRASP_PI)
a.seed(args.seed)
Qdestin = a.get_optimal_init_arm_q(desired_obj_pos)
print("place arm q", Qdestin)

################## set up world
p.resetSimulation()
p.setPhysicsEngineParameter(numSolverIterations=BULLET_SOLVER_ITER)
p.setTimeStep(TS)
p.setGravity(0, 0, -10)

env_core = InmoovShadowHandDemoEnvV3(noisy_obs=NOISY_OBS, seed=args.seed)

obj_ids = construct_bullet_scene(objs)
oid1 = obj_ids[Target_ind]  # TODO:tmp

#################### ready to grasp

env_core.robot.reset_with_certain_arm_q(Qreach)  # TODO

g_obs = env_core.get_robot_contact_txty_halfh_obs(
    g_tx, g_ty, 0.065
)  # TODO: hardcoded
g_obs = wrap_over_grasp_obs(g_obs)

# grasp!
control_steps = 0
for i in range(GRASP_END_STEP):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = g_actor_critic.act(
            g_obs, recurrent_hidden_states, masks, deterministic=args.det
        )

    env_core.step(unwrap_action(action))
    # g_obs = env_core.get_robot_contact_txty_obs(g_tx, g_ty)
    g_obs = env_core.get_robot_contact_txty_halfh_obs(
        g_tx, g_ty, 0.065
    )  # TODO: hardcoded
    g_obs = wrap_over_grasp_obs(g_obs)

    # print(g_obs)
    # print(action)
    # print(control_steps)
    # control_steps += 1
    # input("press enter g_obs")
    masks.fill_(1.0)

import copy

final_g_obs = copy.copy(g_obs)
del g_obs, g_tx, g_ty, g_actor_critic, g_ob_rms


state = get_relative_state_for_reset(oid1)
print("after grasping", state)
print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
# input("after grasping")

##########################----- SEND MOVE COMMAND TO OPENRAVE
if FIX_MOVE:
    Traj2 = np.load(FIX_MOVE_PATH)
else:
    Qmove_init = np.concatenate(
        (
            env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0],
            env_core.robot.get_q_dq(env_core.robot.arm_dofs)[1],
        )
    )  # OpenRave initial condition
    file_path = homedir + "/container_data/PB_MOVE.npz"
    np.savez(file_path, OBJECTS, Qmove_init, Qdestin)

    # Wait for command from OpenRave
    file_path = homedir + "/container_data/OR_MOVE.npy"
    assert not os.path.exists(file_path)
    while not os.path.exists(file_path):
        time.sleep(0.01)
    if os.path.isfile(file_path):
        Traj2 = np.load(file_path)
        print("loaded")
        try:
            os.remove(file_path)
            print("deleted")
            # input("press enter")
        except OSError as e:  # name the Exception `e`
            print("Failed with:", e.strerror)  # look what it says
            # input("press enter")
    else:
        raise ValueError("%s isn't a file!" % file_path)
    print("Trajectory obtained from OpenRave!")
    # input("press enter")


##################################---- EXECUTE PLANNED MOVING TRAJECTORY
planning(Traj2, final_g_obs, recurrent_hidden_states, masks)
print("after moving", get_relative_state_for_reset(oid1))
print("arm q", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
# input("after moving")

print("palm", env_core.robot.get_link_pos_quat(env_core.robot.ee_id))

#################### ready to place

env_core.diffTar = True  # TODO:tmp!!!

t_pos, t_quat = p.getBasePositionAndOrientation(oid1)  # TODO!!!
b_pos, b_quat = p.getBasePositionAndOrientation(obj_ids[2])  #  TODO!!!
# TODO: an unly hack to force Bullet compute forward kinematics
p_obs = env_core.get_robot_2obj6dUp_contact_txty_halfh_obs(
    p_tx, p_ty, t_pos, t_quat, b_pos, b_quat, 0.065
)  # TODO:hardcoded
p_obs = env_core.get_robot_2obj6dUp_contact_txty_halfh_obs(
    p_tx, p_ty, t_pos, t_quat, b_pos, b_quat, 0.065
)
p_obs = wrap_over_grasp_obs(p_obs)
print("pobs", p_obs)
# input("ready to place")
# place!
for i in range(PLACE_END_STEP):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = p_actor_critic.act(
            p_obs, recurrent_hidden_states, masks, deterministic=args.det
        )

    env_core.step(unwrap_action(action))
    t_pos, t_quat = p.getBasePositionAndOrientation(
        oid1
    )  # real time update TODO!!!
    b_pos, b_quat = p.getBasePositionAndOrientation(
        obj_ids[2]
    )  # real time update TODO!!!
    p_obs = env_core.get_robot_2obj6dUp_contact_txty_halfh_obs(
        p_tx, p_ty, t_pos, t_quat, b_pos, b_quat, 0.065
    )
    p_obs = wrap_over_grasp_obs(p_obs)

    # print(action)
    # print(p_obs)
    # input("press enter g_obs")

    masks.fill_(1.0)

# execute_release_traj()
for ind in range(0, 100):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

