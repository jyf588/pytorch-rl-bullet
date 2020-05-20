"""Runs the bullet-unity system."""
import os
import sys
import time
import copy
import random
import pprint
import imageio
import argparse
import warnings
import numpy as np
import pybullet as p
from typing import *

warnings.filterwarnings("ignore")

import exp.loader
import system.policy
import system.options
import system.openrave
import bullet2unity.states
import scene.util as scene_util
from states_env import StatesEnv
import scene.generate as scene_gen
from system.env import DemoEnvironment
import my_pybullet_envs.utils as utils
import ns_vqa_dart.bullet.util as util
import bullet2unity.interface as interface
from exp.options import EXPERIMENT_OPTIONS
from system.vision_module import VisionModule

try:
    from ns_vqa_dart.scene_parse.detectron2.dash import DASHSegModule
except ImportError as e:
    print(e)
from system.options import (
    SYSTEM_OPTIONS,
    BULLET_OPTIONS,
    VISION_OPTIONS,
)

global args

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

    opt = system.options.set_unity_container_cfgs(opt=opt, port=args.port)
    system.openrave.check_clean_container(container_dir=opt.container_dir)

    success_metrics = {"overall": {}, "scenes": {}}
    examples = []
    for set_name, set_opt in set_name2opt.items():
        task = set_opt["task"]

        # Initialize data structures to track success.
        success_metrics["overall"][set_name] = {
            "n_success": 0,
            "n_or_success": 0,
            "n_trials": 0,
        }
        success_metrics["scenes"][set_name] = {}

        # Load scenes.
        scenes_dir = os.path.join(opt.scenes_root_dir, args.exp)
        scenes = scene_gen.load_scenes(load_dir=scenes_dir, task=task)

        if opt.task_subset is not None and task not in opt.task_subset:
            continue

        for scene_id, scene in enumerate(scenes):
            if opt.start_sid is not None and int(scene_id) < opt.start_sid:
                continue
            elif opt.end_sid is not None and int(scene_id) >= opt.end_sid:
                continue
            examples.append((set_name, set_opt, task, scene_id, scene))
    random.seed(opt.seed)
    random.shuffle(examples)

    # Define paths.
    if opt.unity_captures_dir is not None:
        util.delete_and_create_dir(opt.unity_captures_dir)
    run_time_str = util.get_time_dirname()
    run_dir = os.path.join(opt.root_outputs_dir, args.exp, opt.policy_id, run_time_str)
    outputs_dir = os.path.join(run_dir, "pickle")
    states_dir = os.path.join(run_dir, "states")
    successes_path = os.path.join(run_dir, "successes.json")
    os.makedirs(outputs_dir)
    os.makedirs(states_dir)

    # Preparing models.
    policy_opt, shape2policy_paths = system.options.get_policy_options_and_paths(
        policy_id=opt.policy_id
    )
    shape2policy_dict = system.policy.get_shape2policy_dict(
        opt=opt, policy_opt=policy_opt, shape2policy_paths=shape2policy_paths
    )
    vision_models_dict = load_vision_models() if opt.obs_mode == "vision" else {}

    # Manage experiment options.
    system.options.print_and_save_options(
        run_dir=run_dir,
        system_opt=opt,
        bullet_opt=BULLET_OPTIONS,
        policy_opt=policy_opt,
        vision_opt=VISION_OPTIONS,
    )

    # Connect to bullet.
    if opt.render_bullet:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    for idx in range(args.start_idx, len(examples)):
        set_name, set_opt, task, scene_id, scene = examples[idx]
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
                policy_opt=policy_opt,
                shape2policy_dict=shape2policy_dict,
                vision_opt=VISION_OPTIONS,
                exp_name=args.exp,
                set_name=set_name,
                scene_id=scene_id,
                scene=scene,
                task=task,
                outputs_dir=outputs_dir,
                place_dst_xy=place_dst_xy,
                vision_models_dict=vision_models_dict,
            )

        n_frames = 0
        frames_start = time.time()
        update_camera_target = False
        while 1:
            frame_id = f"{idx:04}_{task}_{scene_id}_{env.timestep:06}"
            stage = env.stage
            # The state is the state of the world BEFORE applying the action at the
            # current timestep. We currently save states primarily to check for
            # reproducibility of runs.
            if opt.save_states:
                state_path = os.path.join(states_dir, f"{frame_id}.p")
                util.save_pickle(state_path, data=env.get_state(), check_override=False)

            if (
                opt.obs_mode == "vision"
                and stage in ["plan", "place", "stack"]
                and env.stage_ts == 0
            ):
                update_camera_target = True

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
                            render_frequency = policy_opt.vision_delay

                # Render unity and step.
                if env.timestep % render_frequency == 0:
                    is_render_step = True
                    u_opt = get_unity_options(args.mode, opt, env)
                    for (rend_obs, rend_place, send_image, should_step) in u_opt:
                        # Compute camera targets.
                        if update_camera_target:
                            bullet_cam_targets = bullet2unity.states.compute_bullet_camera_targets_for_system(
                                opt,
                                env,
                                send_image,
                                save_image=opt.save_first_pov_image,
                            )
                            update_camera_target = False

                        render_state = bullet2unity.states.compute_render_state(
                            env,
                            place_dest_object,
                            bullet_cam_targets,
                            rend_obs,
                            rend_place,
                        )

                        # Compute the animation target.
                        b_ani_tar, head_speed = bullet2unity.states.compute_b_ani_tar(
                            opt, env
                        )

                        # Encode, send, receive, and decode.
                        message = interface.encode(
                            state_id=frame_id,
                            bullet_state=render_state,
                            bullet_animation_target=b_ani_tar,
                            head_speed=head_speed,
                            save_third_pov_image=opt.save_third_pov_image,
                            bullet_cam_targets=bullet_cam_targets,
                        )
                        await websocket.send(message)
                        reply = await websocket.recv()
                        data = interface.decode(
                            frame_id, reply, bullet_cam_targets=bullet_cam_targets,
                        )

                        # Hand the data to the env for processing.
                        if send_image:
                            env.set_unity_data(data)
                        if should_step:
                            is_done, openrave_success = env.step()
            if not is_render_step:
                is_done, openrave_success = env.step()

            # Skip retract if placing failed.
            if stage == "retract" and not env.check_success():
                is_done = True

            # Break out if we're done with the sequence, or it failed.
            if is_done or not openrave_success:
                success_metrics["overall"][set_name]["n_trials"] += 1
                trial_succeeded = False
                if openrave_success:
                    success_metrics["overall"][set_name]["n_or_success"] += 1
                    # Only check for task success if OR didn't fail.
                    if env.check_success():
                        trial_succeeded = True
                        success_metrics["overall"][set_name]["n_success"] += 1
                success_metrics["scenes"][set_name][scene_id] = {
                    "success": trial_succeeded,
                    "or_success": openrave_success,
                    "stage": env.stage,
                    "timestep": env.timestep,
                }
                env.cleanup()
                n_trials = success_metrics["overall"][set_name]["n_trials"]
                n_success = success_metrics["overall"][set_name]["n_success"]
                n_or_success = success_metrics["overall"][set_name]["n_or_success"]
                success_rate = 0.0 if n_trials == 0 else n_success / n_trials
                success_rate_wo_or = (
                    0.0 if n_or_success == 0 else n_success / n_or_success
                )
                avg_time = 0 if n_trials == 0 else (time.time() - start_time) / n_trials
                util.save_json(
                    path=successes_path, data=success_metrics, check_override=False
                )
                print(
                    f"Idx: {idx}\tExp: {args.exp}\tSet: {set_name}\tTask: {task}\tScene ID: {scene_id}\tStage: {stage}\tStage ts: {env.stage_ts}\tTimestep: {env.timestep}\n"
                    f"Frame rate: {n_frames / (time.time() - frames_start):.2f}\tAvg trial time: {avg_time:.2f}\n"
                    f"Success rate: {success_rate:.2f} ({n_trials})\tSuccess w/o OR failures: {success_rate_wo_or:.2f} ({n_or_success})\t# Successes: {n_success}\n"
                    f"Saved stats to: {successes_path}"
                )
                break
            n_frames += 1
    total_duration = time.time() - start_time
    print(f"Finished experiment.")
    print(f"Duration: {total_duration}")
    sys.exit(0)


def load_vision_models():
    vision_models_dict = {}
    seed = VISION_OPTIONS.seed
    if VISION_OPTIONS.use_segmentation_module:
        vision_models_dict["seg"] = DASHSegModule(
            seed=seed,
            mode="eval_single",
            dataset_name="eval_single",
            checkpoint_path=VISION_OPTIONS.seg_checkpoint_path,
        )
    if VISION_OPTIONS.separate_vision_modules:
        vision_models_dict["plan"] = VisionModule(
            seed=seed, load_checkpoint_path=VISION_OPTIONS.planning_checkpoint_path,
        )
        vision_models_dict["place"] = VisionModule(
            seed=seed, load_checkpoint_path=VISION_OPTIONS.placing_checkpoint_path,
        )
        vision_models_dict["stack"] = VisionModule(
            seed=seed, load_checkpoint_path=VISION_OPTIONS.stacking_checkpoint_path,
        )
    else:
        vision_models_dict["combined"] = VisionModule(
            seed=seed, load_checkpoint_path=VISION_OPTIONS.attr_checkpoint_path,
        )
    return vision_models_dict


def get_unity_options(mode, opt, env):
    if mode == "unity_dataset":
        unity_options = [(False, False, True, True)]
    else:
        render_place = env.task == "place"
        if opt.obs_mode == "vision":
            if env.stage in ["plan", "place"]:
                render_cur_step = True
                # if env.stage == "plan":
                #     render_cur_step = True
                # elif env.stage == "place" and env.stage_progress() < 0.2:
                #     render_cur_step = True

                unity_options = [(False, False, True, True)]
                if render_cur_step:
                    unity_options += [(True, render_place, False, False)]
            else:
                unity_options = [(opt.render_obs, render_place, False, True)]
        elif opt.obs_mode == "gt":
            unity_options = [(False, render_place, False, True)]
        else:
            raise ValueError(f"Invalid obs mode: {opt.obs_mode}")
    return unity_options


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str, help="The name of the experiment to run.")
    parser.add_argument("mode", type=str, help="The mode of the system to run.")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The example index to start running from.",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        # default="172.27.76.64",
        default="127.0.0.1",
        help="The hostname of the server.",
    )
    parser.add_argument(
        "--port", required=True, type=int, help="The port of the server."
    )
    args = parser.parse_args()

    # Start the python server.
    interface.run_server(hostname=args.hostname, port=args.port, handler=send_to_client)
