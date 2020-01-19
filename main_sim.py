#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:19:50 2020

@author: yannis
"""


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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
homedir = os.path.expanduser("~")

ts = 1/240

grasp_pi_name = '0114_box_l_4'
saved_file = None
with open(os.path.join(currentdir, 'my_pybullet_envs/assets/place_init_dist/final_states_'+grasp_pi_name+'.pickle'), 'rb') as handle:
    saved_file = pickle.load(handle)
assert saved_file is not None
o_pos_pf_ave = saved_file['ave_obj_pos_in_palm']
o_quat_pf_ave_ri = saved_file['ave_obj_quat_in_palm_rot_ivr']
writer = URDFWriter()
new_file = 'inmoov_arm_v2_2_obj_placing_' + grasp_pi_name + '.urdf'
writer.add_obj(o_pos_pf_ave, o_quat_pf_ave_ri, new_file)

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
    default='ShadowHandDemoBulletEnv-v1',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models_0114_box_l_4/ppo/',    # TODO
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    type=int,
    default=0,
    help='whether to use a non-deterministic policy, 1 true 0 false')
parser.add_argument(
    '--iter',
    type=int,
    default=-1,
    help='which iter pi to test')
args = parser.parse_args()
is_cuda = True
device = 'cuda' if is_cuda else 'cpu'
args.det = not args.non_det
p.connect(p.GUI)
p.resetSimulation()
p.setTimeStep(ts)
p.setGravity(0, 0, -10)
env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device=device,
    allow_early_resets=False)
env_core = env.venv.venv.envs[0]
table_id = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/tabletop.urdf'), [0.27, 0.1, 0.0], useFixedBase=1)
p.changeDynamics(table_id, -1, lateralFriction=1.0) 

# Prepare grasping policy: We need to use the same statistics for normalization as used in training
ori_env_name = 'InmoovHandGraspBulletEnv-v1'
if args.iter >= 0:
    path = os.path.join(args.load_dir, ori_env_name + "_" + str(args.iter) + ".pt")
else:
    path = os.path.join(args.load_dir, ori_env_name + ".pt")
if is_cuda:
    actor_critic, ob_rms = torch.load(path)
else:
    actor_critic, ob_rms = torch.load(path, map_location='cpu')
vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms
recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)
env.reset()

################################----- HERE IS THE INPUT FROM VISION AND LANGUAGE MODULE
#np.random.seed(123)
tx = np.random.uniform(low=0, high=0.25) # target object location
ty = np.random.uniform(low=-0.2, high=0.5)
e = np.random.uniform(low=-0.01,high=0.01,size=(6,))
est_x = tx + np.random.uniform(low=-0.01,high=0.01)
est_y = ty + np.random.uniform(low=-0.01,high=0.01)

# Ground-truth scene:
obj1 = {'shape':'box','color':'yellow','position':np.array([0.45,0.1,0,0])} #ref 1
obj2 = {'shape':'box','color':'red','position':np.array([tx,ty,0,0])} # target
obj3 = {'shape':'cylinder','color':'blue','position':np.array([0.1,0.5,0,0])} # ref 2
obj4 = {'shape':'cylinder','color':'yellow','position':np.array([0.5,-0.4,0,0])} #irrelevant
Scene = [obj1, obj2, obj3, obj4]

# Vision module output:
obj1 = {'shape':'box','color':'yellow','position':np.array([0.45+e[0],0.1+e[1],0,0])} #ref 1
obj2 = {'shape':'box','color':'red','position':np.array([est_x,est_y,0,0])} # target
obj3 = {'shape':'cylinder','color':'blue','position':np.array([0.1+e[2],0.5+e[3],0,0])} # ref 2
obj4 = {'shape':'cylinder','color':'yellow','position':np.array([0.5+e[4],-0.4+e[5],0,0])} #irrelevant
Vision_output = [obj1, obj2, obj3, obj4]

sentence = "Put the smaller red box between the blue cylinder and yellow box"

[OBJECTS, target_xyz] = NLPmod(sentence, Vision_output)
destin_x = target_xyz[0] #np.random.uniform(low=0, high=0.25) # destination location for target object
destin_y = target_xyz[1] #np.random.uniform(low=-0.1, high=0.5)
destin_z = target_xyz[2]

################################# ---- OR TRY SINGLE OBJECT

# tx = np.random.uniform(low=0, high=0.25) # target object location
# ty = np.random.uniform(low=-0.1, high=0.5)
# destin_x = np.random.uniform(low=0, high=0.25) # destination location for target object
# destin_y = np.random.uniform(low=-0.1, high=0.5)
# destin_z = 0
# est_x = tx + np.random.uniform(low=-0.01, high=0.01)
# est_y = ty + np.random.uniform(low=-0.01, high=0.01)

# obj1 = {'shape':'box','color':'red','position':np.array([tx,ty,0,0])} # target
# obj2 = {'shape':'cylinder','color':'yellow','position':np.array([0.8,0.5,0,0])} #irrelevant
# Scene = [obj1, obj2]

# obj1 = {'shape':'box','color':'red','position':np.array([est_x,est_y,0,0])} # target
# Vision_output = [obj1, obj2]

# OBJECTS = np.array([[est_x,est_y,0,0],[0.8, 0.8, 0, 0]])


for i in range(len(Scene)):
    ob_shape = Scene[i]['shape']
    real_loc = Scene[i]['position'][0:3]+np.array([0,0,0.1])
    urdf_file = 'my_pybullet_envs/assets/'+ob_shape+'.urdf'
    exec("oid%s = p.loadURDF(os.path.join(currentdir,urdf_file), real_loc, useFixedBase=0)"  % i)
    est_loc = Vision_output[i]['position'][0:3]+np.array([0,0,0.1])
    env_core.assign_estimated_obj_pos(est_loc[0], est_loc[1])
# oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder.urdf'), [a_tx, a_ty, 0.1], useFixedBase=0)   # tar obj
#oid1 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box.urdf'), [tx, ty, 0.1], useFixedBase=0)   # tar obj
# oid2 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/box.urdf'), [0.1, 0.2, 0.1], useFixedBase=0)
# oid3 = p.loadURDF(os.path.join(currentdir, 'my_pybullet_envs/assets/cylinder.urdf'), [0.2, -0.15, 0.1], useFixedBase=0)
    exec('p.changeDynamics(oid%s, -1, lateralFriction=1.0)' % i)



##################################---- SEND REACH COMMAND TO OPENRAVE

# Get target reaching pose from PyBullet to pass it to OpenRave:

sess = ImaginaryArmObjSession()
Qreach = np.array(sess.get_most_comfortable_q_and_refangle(OBJECTS[0,0],OBJECTS[0,1])[0]) 

file_path = homedir+'/container_data/PB_REACH.npz'
np.savez(file_path,OBJECTS,Qreach)

# Wait for command from OpenRave
file_path = homedir+'/container_data/OR_REACH.npy'
while not os.path.exists(file_path):
    time.sleep(1)
if os.path.isfile(file_path):
    Traj = np.load(file_path)
    os.remove(file_path)
else:
    raise ValueError("%s isn't a file!" % file_path)
print("Trajectory obtained from OpenRave!")
input("press enter")

##################################---- EXECUTE PLANNED REACHING TRAJECTORY
#This executes and OpenRave-planned trajectory on the robot
def planning(robot,Traj):
    for ind in range(len(Traj)):
            tar_armq = Traj[ind,0:7]
            # for ji, i in enumerate(robot.arm_dofs):
            #     p.resetJointState(robot.arm_id, i, tar_armq[ji])
            # for ind in range(len(robot.fin_actdofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_actdofs[ind], robot.init_fin_q[ind], 0.0)
            # for ind in range(len(robot.fin_zerodofs)):
            #     p.resetJointState(robot.arm_id, robot.fin_zerodofs[ind], 0.0, 0.0)
            #print(tar_armq)
            p.setJointMotorControlArray(
                bodyIndex=robot.arm_id,
                jointIndices=robot.arm_dofs,
                controlMode=p.POSITION_CONTROL,
                targetPositions=list(tar_armq),
                forces=[robot.maxForce * 3] * len(robot.arm_dofs))
            p.setJointMotorControlArray(
                bodyIndex=robot.arm_id,
                jointIndices=robot.fin_actdofs,
                controlMode=p.POSITION_CONTROL,
                targetPositions=list(robot.tar_fin_q),
                forces=[robot.maxForce] * len(robot.tar_fin_q))
            p.setJointMotorControlArray(
                bodyIndex=robot.arm_id,
                jointIndices=robot.fin_zerodofs,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[0.0] * len(robot.fin_zerodofs),
                forces=[robot.maxForce / 4.0] * len(robot.fin_zerodofs))
            p.stepSimulation()
            # print(robot.tar_fin_q)
            time.sleep(ts)

    cps = p.getContactPoints(bodyA=robot.arm_id)
    print(len(cps) == 0)
    for _ in range(50):
        robot.tar_arm_q = tar_armq
        p.stepSimulation()
        #time.sleep(1. / 240.)    # TODO: stay still for a while

planning(env_core.robot,Traj)


##########################---- APPLY GRASPING POLICY 
obs = torch.Tensor([env_core.getExtendedObservation()])
if is_cuda:
    obs = obs.cuda()
control_steps = 0    
for i in range(200):
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    obs, reward, done, _ = env.step(action)
    control_steps += 1
    #potentially crop grasping trajectory
#   if control_steps >= 100:  # done grasping
#      for _ in range(1000):
#       p.stepSimulation()
#       time.sleep(ts)    
masks.fill_(0.0 if done else 1.0)



##########################----- SEND MOVE COMMAND TO OPENRAVE
sess = ImaginaryArmObjSession(filename=new_file)
Qmove_init = np.concatenate((env_core.robot.get_q_dq(env_core.robot.arm_dofs)[0],env_core.robot.get_q_dq(env_core.robot.arm_dofs)[1])) # OpenRave initial condition
Qdestin = np.array(sess.get_most_comfortable_q_and_refangle_xz(destin_x,destin_y, destin_z+0.16)[0]) # Get target configuration from PyBullet 

file_path = homedir+'/container_data/PB_MOVE.npz'
np.savez(file_path,OBJECTS,Qmove_init,Qdestin)        

# Wait for command from OpenRave
file_path = homedir+'/container_data/OR_MOVE.npy'
while not os.path.exists(file_path):
    time.sleep(1)
if os.path.isfile(file_path):
    Traj2 = np.load(file_path)
    os.remove(file_path)
else:
    raise ValueError("%s isn't a file!" % file_path)
print("Trajectory obtained from OpenRave!")
input("press enter")


##################################---- EXECUTE PLANNED MOVING TRAJECTORY
planning(env_core.robot,Traj2)



bp()