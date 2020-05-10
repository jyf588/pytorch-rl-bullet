"""Runs the bullet-unity system."""
import os
import sys
import time
import copy
import pprint
import argparse
from typing import *
import warnings

warnings.filterwarnings("ignore")

import exp.loader
import system.base_scenes
import bullet2unity.states
import scene.loader as scene_loader
from system.env import DemoEnvironment
import my_pybullet_envs.utils as utils
import ns_vqa_dart.bullet.util as util
import bullet2unity.interface as interface
from exp.options import EXPERIMENT_OPTIONS
from system.options import (
    SYSTEM_OPTIONS,
    BULLET_OPTIONS,
    POLICY_OPTIONS,
    NAME2POLICY_MODELS,
    VISION_OPTIONS,
)

global args


PLAN_TARGET_POSITION = [-0.06, 0.3, 0.0]
STAGE2ANIMATION_Z_OFFSET = {
    "plan": 0.3,
    "reach": 0.1,
    "grasp": 0.1,
    "retract": 0.3,
}
TASK2ANIMATION_Z_OFFSET = {"place": 0.1, "stack": 0.2}


async def send_to_client(websocket, path):
    """Sends and receives data to and from Unity.
    
    Args:
        websocket: The websocket protocol instance.
        path: The URI path.
    """
    start_time = time.time()
    n_trials = 0

    # Run all sets in experiment.
    opt = SYSTEM_OPTIONS[args.exp]
    sets = list(EXPERIMENT_OPTIONS[args.exp].keys())
    for set_name in sets:
        set_opt = EXPERIMENT_OPTIONS[args.exp][set_name]
        scenes = scene_loader.load_scenes(exp=args.exp, set_name=set_name)
        task = set_opt["task"]
        states_path = os.path.join(
            exp.loader.get_exp_set_dir(exp=args.exp, set_name=set_name), "states.p"
        )

        for scene_idx in range(len(scenes)):
            scene = scenes[scene_idx]
            avg_time = 0 if n_trials == 0 else (time.time() - start_time) / n_trials
            print(
                f"Exp: {args.exp}\t"
                f"Set: {set_name}\t"
                f"Task: {task}\t"
                f"Scene ID: {scene_idx}\t"
                f"Trial: {n_trials}\t"
                f"Avg Time: {avg_time:.2f}"
            )

            # Modify the scene for placing, and determine placing destination.
            place_dst_xy, place_dest_object = None, None
            if task == "place":
                scene, place_dst_xy, place_dest_object = convert_scene_for_placing(
                    opt, scene
                )

            # Initialize the environment.
            env = DemoEnvironment(
                opt=opt,
                bullet_opt=BULLET_OPTIONS,
                policy_opt=POLICY_OPTIONS,
                name2policy_models=NAME2POLICY_MODELS,
                vision_opt=VISION_OPTIONS,
                trial=scene_idx,
                scene=scene,
                task=task,
                place_dst_xy=place_dst_xy,
                states_path=states_path,
            )

            while 1:
                stage, _ = env.get_current_stage()
                state = env.get_state()

                is_render_step = False
                if opt.render_unity:
                    render_frequency = opt.render_frequency
                    if opt.obs_mode == "vision" and stage in ["plan", "place"]:
                        render_frequency = POLICY_OPTIONS.vision_delay

                    # Renders unity and steps.
                    if env.timestep % render_frequency == 0:
                        is_render_step = True
                        u_opt = get_unity_options(env, opt.render_obs)
                        for (rend_obs, rend_place, send_image, should_step,) in u_opt:
                            state_id = f"{scene_idx:06}_{env.timestep:06}"
                            render_state = compute_render_state(
                                state, env.obs, place_dest_object, rend_obs, rend_place
                            )

                            # Compute camera targets.
                            if send_image:
                                new_targets = compute_bullet_camera_targets(env=env)
                                if new_targets is not None:
                                    bullet_cam_targets = copy.deepcopy(new_targets)

                            # Compute the animation target.
                            b_ani_tar = (
                                compute_b_ani_target(env=env)
                                if opt.animate_head
                                else None
                            )

                            # Encode, send, receive, and decode.
                            message = interface.encode(
                                state_id=state_id,
                                bullet_state=render_state,
                                bullet_animation_target=b_ani_tar,
                                bullet_cam_targets=bullet_cam_targets,
                            )
                            await websocket.send(message)
                            reply = await websocket.recv()
                            data = interface.decode(
                                state_id, reply, bullet_cam_targets=bullet_cam_targets,
                            )

                            # Hand the data to the env for processing.
                            if send_image:
                                env.set_unity_data(data)
                            if should_step:
                                is_done, success = env.step()
                if not is_render_step:
                    is_done, success = env.step()

                # Break out if we're done with the sequence, or it failed.
                if is_done or not success:
                    env.cleanup()
                    n_trials += 1
                    break
    sys.exit(0)


def convert_scene_for_placing(opt, scene: List) -> Tuple:
    """Converts the scene into a modified scene for placing, with the following steps:
        1. Denote the (x, y) location of object index 0 as the placing destination (x, y).
        2. Remove object index 0 from the scene list.
    
    Args:
        scene: The original scene.
    
    Returns:
        new_scene: The scene modified for placing.
        place_dst_xy: The (x, y) placing destination for the placing task.
    """
    # Make sure that removing the destination object from the scene won't modify the
    # source object index.
    assert opt.scene_place_src_idx < opt.scene_place_dst_idx

    # Remove the destination object from the scene.
    new_scene = copy.deepcopy(
        scene[: opt.scene_place_dst_idx] + scene[opt.scene_place_dst_idx + 1 :]
    )

    # Use the location of the destination object as the (x, y) placing destination.
    place_dst_xy = scene[opt.scene_place_dst_idx]["position"][:2]

    # Construct an imaginary object with the same shape attribute as the source object
    # to visualize the placing destination.
    place_dest_object = copy.deepcopy(scene[opt.scene_place_src_idx])
    place_dest_object["position"] = place_dst_xy + [0.0]
    place_dest_object["height"] = 0.005
    return new_scene, place_dst_xy, place_dest_object


def get_unity_options(env, render_obs):
    render_place = render_obs and env.task == "place"
    if env.obs_mode == "vision" and env.stage in ["plan", "place"]:
        unity_options = [(False, False, True, True)]
        if args.render_obs:
            unity_options += [(True, render_place, False, False)]
    else:
        if env.obs_mode == "vision":
            unity_options = [(args.render_obs, render_place, False, True)]
        elif env.obs_mode == "gt":
            unity_options = [(False, render_place, False, True)]
    return unity_options


def compute_bullet_camera_targets(env):
    stage, stage_ts = env.get_current_stage()
    task = env.task
    if stage == "plan":
        cam_target = PLAN_TARGET_POSITION
    elif stage == "place" and stage_ts == 0:
        if task == "place":
            cam_target = env.place_dst_xy + [env.initial_obs[env.src_idx]["height"]]
        elif task == "stack":
            pprint.pprint(env.initial_obs)
            cam_target = bullet2unity.states.get_object_camera_target(
                bullet_odicts=env.initial_obs, oidx=env.dst_idx
            )
    else:
        return None

    # Set the camera target.
    bullet_camera_targets = bullet2unity.states.create_bullet_camera_targets(
        camera_control="position",
        bullet_odicts=env.initial_obs,
        use_oids=False,
        should_save=False,
        should_send=True,
        position=cam_target,
    )
    return bullet_camera_targets


def compute_b_ani_target(env):
    stage, _ = env.get_current_stage()
    task = env.task
    if stage in ["plan", "retract"]:
        b_ani_tar = None
    else:
        if stage in ["reach", "grasp"]:
            b_ani_tar = env.initial_obs[env.src_idx]["position"]
        elif stage in ["transport", "place", "release"]:
            if task == "place":
                b_ani_tar = env.place_dst_xy + [env.initial_obs[env.src_idx]["height"]]
            elif task == "stack":
                b_ani_tar = env.initial_obs[env.dst_idx]["position"]
            else:
                raise ValueError(f"Unsupported task: {task}")
        else:
            raise ValueError(f"Unsupported stage: {stage}.")
        b_ani_tar = copy.deepcopy(b_ani_tar)
        if stage in STAGE2ANIMATION_Z_OFFSET:
            z_offset = STAGE2ANIMATION_Z_OFFSET[stage]
        elif task in TASK2ANIMATION_Z_OFFSET:
            z_offset = TASK2ANIMATION_Z_OFFSET[task]
        b_ani_tar[2] += z_offset
    return b_ani_tar


def compute_render_state(state, obs, place_dest_object, render_obs, render_place):
    # If we are rendering observations, add them to the
    # render state.
    render_state = copy.deepcopy(state)
    if render_obs:
        render_state = add_hallucinations_to_state(
            state=render_state, h_odicts=obs, color=None,
        )
    if render_place:
        render_state = add_hallucinations_to_state(
            state=render_state, h_odicts=[place_dest_object], color="clear",
        )
    return render_state


def add_hallucinations_to_state(state: Dict, h_odicts: Dict, color: str):
    state = copy.deepcopy(state)
    h_odicts = copy.deepcopy(h_odicts)
    n_existing_objects = len(state["objects"])
    for oi, odict in enumerate(h_odicts):
        # Set the color to be the clear version of the object color.
        if color is None:
            ocolor = odict["color"]
            hallu_color = f"clear_{ocolor}"
        else:
            hallu_color = color
        odict["color"] = hallu_color
        state["objects"][f"h_{n_existing_objects + oi}"] = odict
    return state


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str, help="The name of the experiment to run.")
    parser.add_argument(
        "--hostname",
        type=str,
        # default="172.27.76.64",
        default="127.0.0.1",
        help="The hostname of the server.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port of the server."
    )
    args = parser.parse_args()

    # Start the python server.
    interface.run_server(hostname=args.hostname, port=args.port, handler=send_to_client)
