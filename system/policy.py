"""Policy-related functions."""

import numpy as np
import os
import torch
from typing import *


def load(
    policy_dir: str, env_name: str, is_cuda: bool, iter: Optional[int] = None
):
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
    recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size
    )
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
