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
from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import ImaginaryArmObjSession
from my_pybullet_envs.inmoov_shadow_place_env_v3 import InmoovShadowHandPlaceEnvV3
from my_pybullet_envs.inmoov_shadow_demo_env_v3 import InmoovShadowHandDemoEnvV3

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
homedir = os.path.expanduser("~")


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

# constants
DEMO_ENV_NAME = 'ShadowHandDemoBulletEnv-v1'
# GRASP_PI = '0120_cyl_l_0'
GRASP_PI = '0120_box_s_1'
GRASP_DIR = "./trained_models_%s/ppo/" % GRASP_PI
GRASP_PI_ENV_NAME = 'InmoovHandGraspBulletEnv-v1'
# PLACE_PI = '0120_cyl_l_0_place_f_nogt_0'
PLACE_PI = '0120_box_s_1_place_f_nogt_1'
PLACE_DIR = "./trained_models_%s/ppo/" % PLACE_PI
PLACE_PI_ENV_NAME = 'InmoovHandPlaceBulletEnv-v3'

# TODO: infer from language
pi_is_box = True
pi_is_small = False

GRASP_END_STEP = 90
PLACE_END_STEP = 95

STATE_NORM = False
NOISY_OBS = True
# Noisy tx ty
# Noisy obj 6d if applicable

OBJ_MU = 1.0
FLOOR_MU = 1.0
# HAND_MU = 3.0

sys.path.append('a2c_ppo_acktr')
parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=101)
parser.add_argument('--non-det', type=int, default=0)
args = parser.parse_args()
args.det = not args.non_det

IS_CUDA = True     # TODO:tmp odd. seems no need to use cuda
DEVICE = 'cuda' if IS_CUDA else 'cpu'

TS = 1./240
TABLE_OFFSET = [0.25, 0.2, 0.0]     # TODO: chaged to 0.2 for vision, 0.25 may collide, need to change OR reaching.
HALF_OBJ_HEIGHT_L = 0.09
HALF_OBJ_HEIGHT_S = 0.065
PLACE_CLEARANCE = 0.16

# test only one obj
g_tx = 0.1
g_ty = 0.43
p_tx = 0.2
p_ty = -0.0
# g_tx = 0.1
# g_ty = 0.43
# p_tx = 0.2
# p_ty = 0.2
# g_tx = 0.2
# g_ty = -0.1
# p_tx = 0.15
# p_ty = 0.4
p_tz = 0.0  # TODO: depending on placing on floor or not


def planning(Traj, i_g_obs, recurrent_hidden_states, masks):
    print("end of traj", Traj[-1, 0:7])
    for ind in range(0, len(Traj)):
        tar_armq = Traj[ind, 0:7]
        env_core.robot.tar_arm_q = tar_armq
        env_core.robot.apply_action([0.0]*24)
        p.stepSimulation()
        time.sleep(1. / 240.)

    # for _ in range(50):
    #     # print(env_core.robot.tar_arm_q)
    #     env_core.robot.tar_arm_q = tar_armq
    #     env_core.robot.apply_action([0.0] * 24)         # stay still for a while
    #     p.stepSimulation()
    #     # print("act", env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0])
    # #     #time.sleep(1. / 240.)


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
    # vec_norm = get_vec_normalize(env) # TODO: assume no state normalize
    # if not STATE_NORM: assert ob_rms is None
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.ob_rms = ob_rms
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    return actor_critic, ob_rms, recurrent_hidden_states, masks     # probably only first one is used


def execute_release_traj():
    for i in range(-1, p.getNumJoints(env_core.robot.arm_id)):
        p.setCollisionFilterPair(oid1, env_core.robot.arm_id, -1, i, enableCollision=0)      # TODO:tmp
    env_core.robot.tar_arm_q = env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0]
    env_core.robot.tar_fin_q = env_core.robot.get_q_dq(env_core.robot.fin_actdofs)[0]
    tar_wrist_xyz = list(env_core.robot.get_link_pos_quat(env_core.robot.ee_id)[0])
    ik_q = None
    for test_t in range(300):
        if test_t < 200:
            tar_wrist_xyz[0] -= 0.001
            ik_q = p.calculateInverseKinematics(env_core.robot.arm_id, env_core.robot.ee_id, tar_wrist_xyz)
        env_core.robot.tar_arm_q = np.array(ik_q[:len(env_core.robot.arm_dofs)])
        env_core.robot.apply_action(np.array([0.0] * len(env_core.action_scale)))
        p.stepSimulation()
        time.sleep(TS)


def wrap_over_grasp_obs(obs):
    obs = torch.Tensor([obs])
    if IS_CUDA:
        obs = obs.cuda()
    return obs


def unwrap_action(act_tensor):
    action = act_tensor.squeeze()
    action = action.cpu() if IS_CUDA else action
    return action.numpy()


# set up world
p.connect(p.GUI)
p.resetSimulation()

sess = ImaginaryArmObjSession()
Qreach = np.array(sess.get_most_comfortable_q_and_refangle(g_tx, g_ty)[0])

desired_obj_pos = [p_tx, p_ty, PLACE_CLEARANCE + p_tz]
a = InmoovShadowHandPlaceEnvV3(renders=False, is_box=pi_is_box, is_small=pi_is_small, place_floor=True, grasp_pi_name=GRASP_PI)
a.seed(args.seed)
Qdestin =a.get_optimal_init_arm_q(desired_obj_pos)
print("place arm q", Qdestin)
# arm_q = [-0.8095155039980575, -0.42793360197051106, -0.7269143588514168, -1.2848182939515076, -0.7268314292697703, -0.7463880732365392, -0.7885470027289124]

# latter 2 returns dummy
g_actor_critic, g_ob_rms, _, _ = load_policy_params(GRASP_DIR, GRASP_PI_ENV_NAME)
p_actor_critic, p_ob_rms, recurrent_hidden_states, masks = load_policy_params(PLACE_DIR, PLACE_PI_ENV_NAME)

p.resetSimulation()
p.setTimeStep(TS)
p.setGravity(0, 0, -10)


env_core = InmoovShadowHandDemoEnvV3(noisy_obs=NOISY_OBS, seed=args.seed)


# oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder_small.urdf'), [g_tx, g_ty, HALF_OBJ_HEIGHT_S+0.001],
#                   useFixedBase=0)
# oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box.urdf'), [g_tx, g_ty, HALF_OBJ_HEIGHT_L+0.001],
#                   useFixedBase=0)
# oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder.urdf'), [g_tx, g_ty, HALF_OBJ_HEIGHT_L+0.001],
#                   useFixedBase=0)
oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box_small.urdf'), [g_tx, g_ty, HALF_OBJ_HEIGHT_S+0.001],
                  useFixedBase=0)
p.changeDynamics(oid1, -1, lateralFriction=OBJ_MU)
table_id = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/tabletop.urdf'), TABLE_OFFSET,
                      useFixedBase=1)
p.changeDynamics(table_id, -1, lateralFriction=FLOOR_MU)


# ready to grasp

env_core.robot.reset_with_certain_arm_q(Qreach)     # TODO
env_core.reset()
g_obs = env_core.get_robot_contact_txty_obs(g_tx, g_ty)
g_obs = wrap_over_grasp_obs(g_obs)

# grasp!
control_steps = 0
for i in range(GRASP_END_STEP):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = g_actor_critic.act(
            g_obs, recurrent_hidden_states, masks, deterministic=args.det)

    env_core.step(unwrap_action(action))
    g_obs = env_core.get_robot_contact_txty_obs(g_tx, g_ty)
    g_obs = wrap_over_grasp_obs(g_obs)

    print(g_obs)
    print(action)
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
Qmove_init = np.concatenate((env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0],env_core.robot.get_q_dq(env_core.robot.arm_dofs)[1])) # OpenRave initial condition
file_path = homedir+'/container_data/PB_MOVE.npz'
np.savez(file_path, [], Qmove_init, Qdestin)


# Wait for command from OpenRave
file_path = homedir+'/container_data/OR_MOVE.npy'
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

env_core.reset()
# t_pos, t_quat = p.getBasePositionAndOrientation(oid1)
print("palm", env_core.robot.get_link_pos_quat(env_core.robot.ee_id))
# p_obs = env_core.get_robot_obj6d_contact_txty_obs(p_tx, p_ty, t_pos, t_quat)
# p_obs = env_core.get_robot_obj6d_contact_txty_obs(p_tx, p_ty, t_pos, t_quat)    # TODO:tmp why?? computefk=1

p_obs = env_core.get_robot_contact_txty_obs(p_tx, p_ty)
p_obs = env_core.get_robot_contact_txty_obs(p_tx, p_ty)    # TODO:tmp why?? computefk=1
print("palm", env_core.robot.get_link_pos_quat(env_core.robot.ee_id))


p_obs = wrap_over_grasp_obs(p_obs)
print("pobs", p_obs)
# input("ready to place")
# place!
for i in range(PLACE_END_STEP):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = p_actor_critic.act(
            p_obs, recurrent_hidden_states, masks, deterministic=args.det)

    env_core.step(unwrap_action(action))
    # t_pos, t_quat = p.getBasePositionAndOrientation(oid1)   # real time update
    # p_obs = env_core.get_robot_obj6d_contact_txty_obs(p_tx, p_ty, t_pos, t_quat)
    p_obs = env_core.get_robot_contact_txty_obs(p_tx, p_ty)
    p_obs = wrap_over_grasp_obs(p_obs)

    # print(action)
    # print(p_obs)
    # input("press enter g_obs")

    masks.fill_(1.0)

execute_release_traj()





