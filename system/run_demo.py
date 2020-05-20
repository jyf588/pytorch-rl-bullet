"""Runs the bullet-unity system."""
import os
import sys
import time
import copy
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

DEMO_COMMANDS = {
    0: [
        "Put the green cylinder on top of the blue box.",
        "Then, put the red box on top of the blue cylinder.",
    ],
    1: [
        "Pick up the red sphere that is to the left of the green box, and place it in front of the blue cylinder.",
        "Then, stack the yellow box on top of the blue cylinder.",
    ],
    2: ["Put the blue box to the left of the green cylinder",],
    3: ["Put the blue box to the left of the green cylinder",],
}

START_ARM_Q = {
    0: np.array([-0.238, 0.509, -0.255, -2.115, -0.743, 0.132, -0.209]),
    1: np.array([0.0] * 7),
    2: np.array([0.0] * 7),
    3: np.array([0.0] * 7),
}


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

    # Define paths.
    if opt.unity_captures_dir is not None:
        util.delete_and_create_dir(opt.unity_captures_dir)
    run_time_str = util.get_time_dirname()
    run_dir = os.path.join(opt.root_outputs_dir, args.exp, opt.policy_id, run_time_str)
    outputs_dir = os.path.join(run_dir, "pickle")
    states_dir = os.path.join(run_dir, "states")
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

    for set_name, set_opt in set_name2opt.items():

        # Load scenes.
        scenes_dir = os.path.join(opt.scenes_root_dir, args.exp)
        scenes = scene_gen.load_scenes(load_dir=scenes_dir, task=None)

        for scene_id, scene in enumerate(scenes):

            print("demo No.", scene_id)

            if opt.start_sid is not None and int(scene_id) < opt.start_sid:
                continue
            elif opt.end_sid is not None and int(scene_id) >= opt.end_sid:
                continue
            bullet_cam_targets = {}
            # Modify the scene for placing, and determine placing destination.
            place_dst_xy, place_dest_object = None, None  # deprecated

            assert args.mode != "unity_dataset"

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
                task=None,  # set from command in env.
                outputs_dir=outputs_dir,
                place_dst_xy=None,  # set from command in env.
                vision_models_dict=vision_models_dict,
                command=DEMO_COMMANDS[scene_id],
                start_q=START_ARM_Q[scene_id],
            )

            n_frames = 0
            frames_start = time.time()
            update_camera_target = False
            while 1:
                frame_id = f"{scene_id}_{env.timestep:06}"
                stage = env.stage

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
                            (
                                b_ani_tar,
                                head_speed,
                            ) = bullet2unity.states.compute_b_ani_tar(opt, env)

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

                # Break out if we're done with the sequence, or it failed.
                if is_done or not openrave_success:
                    env.cleanup()
                    print(
                        f"Exp: {args.exp} \tScene ID: {scene_id} \tStage: {stage} \tTimestep: {env.timestep}\tOR success: {openrave_success}\n"
                        f"Frame rate: {n_frames / (time.time() - frames_start):.2f}\t\n"
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
    # if mode == "unity_dataset":
    #     unity_options = [(False, False, True, True)]
    assert mode != "unity_dataset"

    render_place = False  # deprecated
    if opt.obs_mode == "vision":
        if env.stage in ["plan", "place"]:
            render_cur_step = opt.render_obs
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
