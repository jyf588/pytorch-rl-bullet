"""Runs the bullet-unity system."""
import os
import sys
import time
import copy
import pprint
import argparse
import pybullet as p
from typing import *
import warnings

warnings.filterwarnings("ignore")

import exp.loader
import scene.util as scene_util
import system.base_scenes
import bullet2unity.states
from states_env import StatesEnv
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

    # Run all sets in experiment.
    opt = SYSTEM_OPTIONS[args.mode]
    set_name2opt = exp.loader.ExpLoader(exp_name=args.exp).set_name2opt

    if args.mode == "table1":
        assert not os.path.exists(opt.table1_path)

    if opt.render_bullet:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    set2success = {}

    for set_name, set_opt in set_name2opt.items():
        set2success[set_name] = {"n_success": 0, "n_or_success": 0, "n_trials": 0}

        if opt.save_states and not set_opt["save_states"]:
            continue
        set_loader = exp.loader.SetLoader(exp_name=args.exp, set_name=set_name)
        id2scene = set_loader.load_id2scene()
        task = set_opt["task"]

        for scene_id, scene in id2scene.items():
            if int(scene_id) != 99:
                continue
            # states_path = os.path.join(set_states_dir, f"{scene_idx:04}.p")
            bullet_cam_targets = {}

            # Modify the scene for placing, and determine placing destination.
            place_dst_xy, place_dest_object = None, None
            if task == "place":
                (
                    scene,
                    place_dst_xy,
                    place_dest_object,
                ) = scene_util.convert_scene_for_placing(opt, scene)

            if args.mode == "unity_dataset":
                env = StatesEnv(
                    opt=opt,
                    experiment=args.exp,
                    set_name=set_name,
                    scene_id=scene_id,
                    scene=scene,
                    task=task,
                    place_dst_xy=place_dst_xy,
                )
            else:
                env = DemoEnvironment(
                    opt=opt,
                    bullet_opt=BULLET_OPTIONS,
                    policy_opt=POLICY_OPTIONS,
                    name2policy_models=NAME2POLICY_MODELS,
                    vision_opt=VISION_OPTIONS,
                    exp_name=args.exp,
                    set_name=set_name,
                    scene_id=scene_id,
                    scene=scene,
                    task=task,
                    place_dst_xy=place_dst_xy,
                )

            # Initalize the directory if we are saving states.
            if opt.save_states:
                scene_loader = exp.loader.SceneLoader(
                    exp_name=args.exp, set_name=set_name, scene_id=scene_id
                )
                os.makedirs(scene_loader.states_dir)

            n_frames = 0
            frames_start = time.time()
            while 1:
                stage = env.stage
                # Save the current state, if the current stage is a stage that should
                # be saved for the current set. The state is the state of the world
                # BEFORE apply the action at the current timestep.
                if opt.save_states and env.stage == set_opt["stage"]:
                    scene_loader.save_state(
                        scene_id=scene_id, timestep=env.timestep, state=env.get_state()
                    )

                is_render_step = False
                if opt.render_unity:
                    # Modify the rendering frequency if we are using vision + policy,
                    # and it's during the planning or placing stage.
                    render_frequency = opt.render_frequency
                    if args.mode != "unity_dataset":
                        if opt.obs_mode == "vision":
                            if env.stage in "plan":
                                render_frequency = 1
                            elif env.stage == "place":
                                render_frequency = POLICY_OPTIONS.vision_delay

                    # Render unity and step.
                    if env.timestep % render_frequency == 0:
                        is_render_step = True
                        state_id = f"{scene_id}_{env.timestep:06}"
                        u_opt = get_unity_options(args.mode, opt, env)
                        for (rend_obs, rend_place, send_image, should_step) in u_opt:
                            render_state = compute_render_state(
                                env, place_dest_object, rend_obs, rend_place
                            )

                            # Compute camera targets.
                            new_targets = compute_bullet_camera_targets(env, send_image)
                            if new_targets is not None:
                                bullet_cam_targets = copy.deepcopy(new_targets)

                            # Compute the animation target.
                            b_ani_tar = compute_b_ani_tar(opt, env)

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
                                is_done, openrave_success = env.step()
                if not is_render_step:
                    is_done, openrave_success = env.step()

                # Break out if we're done with the sequence, or it failed.
                if is_done or not openrave_success:
                    set2success[set_name]["n_trials"] += 1
                    if env.check_success():
                        set2success[set_name]["n_success"] += 1
                    if openrave_success:
                        set2success[set_name]["n_or_success"] += 1
                    env.cleanup()
                    n_trials = set2success[set_name]["n_trials"]
                    n_success = set2success[set_name]["n_success"]
                    n_or_success = set2success[set_name]["n_or_success"]
                    success_rate = 0.0 if n_trials == 0 else n_success / n_trials
                    success_rate_wo_or = (
                        0.0 if n_or_success == 0 else n_success / n_or_success
                    )
                    avg_time = (
                        0 if n_trials == 0 else (time.time() - start_time) / n_trials
                    )
                    print(
                        f"Exp: {args.exp}\tSet: {set_name}\tTask: {task}\tScene ID: {scene_id}\tStage: {stage}\tTimestep: {env.timestep}\n"
                        f"Frame rate: {n_frames / (time.time() - frames_start):.2f}\tAvg trial time: {avg_time:.2f}\n"
                        f"Success rate: {success_rate:.2f} ({n_trials})\tSuccess w/o OR failures: {success_rate_wo_or:.2f} ({n_or_success})\t# Successes: {n_success}"
                    )
                    if args.mode == "table1":
                        util.save_json(path=opt.table1_path, data=set2success)
                    break
                n_frames += 1

    sys.exit(0)


def get_unity_options(mode, opt, env):
    if mode == "unity_dataset":
        unity_options = [(False, False, True, True)]
    else:
        render_place = env.task == "place"
        if opt.obs_mode == "vision":
            # render_place = opt.render_obs and env.task == "place"
            if env.stage in ["plan", "place"]:
                unity_options = [(False, False, True, True)]
                if opt.render_obs:
                    unity_options += [(True, render_place, False, False)]
            else:
                unity_options = [(opt.render_obs, render_place, False, True)]
        elif opt.obs_mode == "gt":
            unity_options = [(False, render_place, False, True)]
        else:
            raise ValueError(f"Invalid obs mode: {opt.obs_mode}")
    return unity_options


def compute_bullet_camera_targets(env, send_image):
    if not send_image:
        return None

    task = env.task
    if env.stage == "plan":
        cam_target = PLAN_TARGET_POSITION
    elif env.stage == "place" and env.stage_ts == 0:
        if task == "place":
            cam_target = env.place_dst_xy + [env.initial_obs[env.src_idx]["height"]]
        elif task == "stack":
            cam_target = bullet2unity.states.get_object_camera_target(
                bullet_odicts=env.initial_obs, oidx=env.dst_idx
            )
    else:
        return None

    # Set the camera target.
    bullet_camera_targets = bullet2unity.states.create_bullet_camera_targets(
        camera_control="position",
        should_save=False,
        should_send=send_image,
        position=cam_target,
    )
    return bullet_camera_targets


def compute_b_ani_tar(opt, env):
    if not opt.animate_head:
        return None
    task = env.task
    if env.stage in ["plan", "retract"]:
        b_ani_tar = None
    else:
        if env.stage in ["reach", "grasp"]:
            b_ani_tar = env.initial_obs[env.src_idx]["position"]
        elif env.stage in ["transport", "place", "release"]:
            if task == "place":
                b_ani_tar = env.place_dst_xy + [env.initial_obs[env.src_idx]["height"]]
            elif task == "stack":
                b_ani_tar = env.initial_obs[env.dst_idx]["position"]
            else:
                raise ValueError(f"Unsupported task: {task}")
        else:
            raise ValueError(f"Unsupported stage: {env.stage}.")
        b_ani_tar = copy.deepcopy(b_ani_tar)
        if env.stage in STAGE2ANIMATION_Z_OFFSET:
            z_offset = STAGE2ANIMATION_Z_OFFSET[env.stage]
        elif task in TASK2ANIMATION_Z_OFFSET:
            z_offset = TASK2ANIMATION_Z_OFFSET[task]
        b_ani_tar[2] += z_offset
    return b_ani_tar


def compute_render_state(env, place_dest_object, render_obs, render_place):
    state = env.get_state()

    # If we are rendering observations, add them to the
    # render state.
    render_state = copy.deepcopy(state)
    if render_obs:
        render_state = add_hallucinations_to_state(
            state=render_state, h_odicts=env.obs_to_render, color=None,
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
    parser.add_argument("mode", type=str, help="The mode of the system to run.")
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
