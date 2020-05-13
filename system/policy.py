"""Policy-related functions."""

import numpy as np
import os
import torch
from typing import *
import pybullet as pb

import my_pybullet_envs.utils as utils
from system.options import SHAPE2POLICY_NAMES, POLICY_OPTIONS


def get_shape2policy_dict(opt):
    shape2policy_dict = {}
    for shape, policy_names in SHAPE2POLICY_NAMES.items():
        grasp_policy, _, _, _ = load(
            policy_dir=policy_names.grasp_dir,
            env_name=POLICY_OPTIONS.grasp_env_name,
            is_cuda=opt.is_cuda,
        )
        place_policy, _, hidden_states, masks = load(
            policy_dir=policy_names.place_dir,
            env_name=POLICY_OPTIONS.place_env_name,
            is_cuda=opt.is_cuda,
        )
        (o_pos_pf_ave, o_quat_pf_ave, _,) = utils.read_grasp_final_states_from_pickle(
            policy_names.grasp_pi
        )
        p_pos_of_ave, p_quat_of_ave = pb.invertTransform(o_pos_pf_ave, o_quat_pf_ave)

        shape2policy_dict[shape] = {
            "grasp_policy": grasp_policy,
            "place_policy": place_policy,
            "hidden_states": hidden_states,
            "masks": masks,
            "o_quat_pf_ave": o_quat_pf_ave,
            "p_pos_of_ave": p_pos_of_ave,
            "p_quat_of_ave": p_quat_of_ave,
        }
    return shape2policy_dict


def load(policy_dir: str, env_name: str, is_cuda: bool, iter: Optional[int] = None):
    """Loads parameters for a specified policy.

    Args:
        policy_dir: The directory to load the policy from.
        env_name: The environment name of the policy.
        is_cuda: Whether to use gpu.
        iter: The iteration of the policy model to load.
    
    Returns:
        actor_critic: The actor critic model.
        ob_rms: ?
        recurrent_hidden_states: The recurrent hidden states of the model.
        masks: ? 
    """
    if iter is not None:
        path = os.path.join(policy_dir, env_name + "_" + str(iter) + ".pt")
    else:
        path = os.path.join(policy_dir, env_name + ".pt")
    if is_cuda:
        actor_critic, ob_rms = torch.load(path)
    else:
        actor_critic, ob_rms = torch.load(path, map_location="cpu")
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    return (
        actor_critic,
        ob_rms,
        recurrent_hidden_states,
        masks,
    )


def wrap_obs(obs: np.ndarray, is_cuda: bool) -> torch.Tensor:
    obs = torch.Tensor([obs])
    if is_cuda:
        obs = obs.cuda()
    return obs


def unwrap_action(action: torch.Tensor, is_cuda: bool) -> np.ndarray:
    action = action.squeeze()
    action = action.cpu() if is_cuda else action
    return action.numpy()
