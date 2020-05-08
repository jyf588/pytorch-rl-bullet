"""Runs a bullet + unity demo."""

import argparse
import copy
import pprint
import sys
from typing import *

import bullet2unity.interface as interface
import bullet2unity.states
import system.base_scenes
from system.env import DemoEnvironment
from system.options import OPTIONS
from system.scene import SceneGenerator
import my_pybullet_envs.utils as utils
from ns_vqa_dart.bullet.random_objects import RandomObjectsGenerator

global args


ADD_SURROUNDING_OBJECTS = True
OBJECT_DIST_THRESH = 0.25
PLAN_TARGET_POSITION = [-0.06, 0.3, 0.0]
STAGE2ANIMATION_Z_OFFSET = {
    "plan": 0.3,
    "reach": 0.1,
    "grasp": 0.1,
    "retract": 0.3,
}
TASK2ANIMATION_Z_OFFSET = {"place": 0.1, "stack": 0.2}

DEMO_SCENE_IDS = [5, 6, 7, 9, 10, 11, 12, 13, 14]


async def send_to_client(websocket, path):
    """Sends and receives data to and from Unity.
    
    Args:
        websocket: The websocket protocol instance.
        path: The URI path.
    """
    scenes = generate_scenes(
        seed=args.seed,
        n_scenes=args.n_scenes,
        disable_orientation=args.disable_orientation,
    )

    use_control_skip = args.fast_mode

    # Used to initialize the pose of each trial with the pose of the last trial.
    init_fin_q, init_arm_q = None, None
    FAIL_SCENES = []  # [7, 8]
    for scene_idx in range(0, len(scenes)):
        # Skip failed scenes.
        if scene_idx in FAIL_SCENES:
            continue

        if args.demo_scenes and scene_idx not in DEMO_SCENE_IDS:
            continue
        scene = scenes[scene_idx]

        # Hard code top object to be green, and bottom object to be blue.
        scene[0]["color"] = "blue"
        scene[1]["color"] = "green"
        dst_shape = scene[0]["shape"]
        src_shape = scene[1]["shape"]
        command = f"Put the green {src_shape} on top of the blue {dst_shape}"

        print(f"scene {scene_idx}:")
        pprint.pprint(scene)

        # task = "stack" if scene_idx % 2 == 0 else "place"
        task = "stack"
        for obs_mode in ["vision"]:
            # Modify the scene for placing. We keep only the first object for
            # now, and set the placing destination xy location to be the
            # location of the original blue object (deleted).
            if task == "place":
                task_scene = copy.deepcopy(scene[1:])
                dest_object = copy.deepcopy(scene[1])
                place_dst_xy = scene[0]["position"][:2]
                dest_object["position"][0] = place_dst_xy[0]
                dest_object["position"][1] = place_dst_xy[1]
            elif task == "stack":
                task_scene = copy.deepcopy(scene)
                place_dst_xy = None
            env = DemoEnvironment(
                opt=OPTIONS,
                trial=scene_idx,
                scene=task_scene,
                task=task,
                command=command,
                observation_mode=obs_mode,
                renderer="unity",
                visualize_bullet=args.render_bullet,
                visualize_unity=False,
                place_dst_xy=place_dst_xy,
                init_fin_q=init_fin_q,
                init_arm_q=init_arm_q,
                use_control_skip=use_control_skip,
            )
            cam_target = None

            # Send states one by one.
            i = 0
            while 1:
                stage, stage_ts = env.get_current_stage()

                # Add options for early stopping during placing (since it's
                # slow for vision).
                # if (
                #     obs_mode == "vision"
                #     and stage == "place"
                #     and stage_ts % 25 == 0
                # ):
                #     response = input("Continue? [Y/N] ")
                #     if response == "N":
                #         break

                state = env.get_state()

                # Temporarily remove robot state.
                # state = {"objects": state["objects"]}
                render_frequency = args.render_frequency
                # Only have lucas look at / send images back when planning or placing.
                if obs_mode == "vision" and stage in ["plan", "place"]:
                    render_frequency = OPTIONS.vision_delay
                    unity_options = [(False, False, True, True)]
                    if args.render_obs:
                        unity_options += [(True, True, False, False)]

                    if stage == "plan":
                        cam_target = PLAN_TARGET_POSITION
                    elif stage == "place" and stage_ts == 0:
                        if task == "place":
                            cam_target = place_dst_xy + [
                                env.initial_obs[env.src_idx]["height"]
                            ]
                        elif task == "stack":
                            pprint.pprint(env.initial_obs)
                            cam_target = bullet2unity.states.get_object_camera_target(
                                bullet_odicts=env.initial_obs, oidx=env.dst_idx
                            )

                    # Set the camera target.
                    last_bullet_camera_targets = bullet2unity.states.create_bullet_camera_targets(
                        camera_control="position",
                        bullet_odicts=env.initial_obs,
                        use_oids=False,
                        should_save=False,
                        should_send=True,
                        position=cam_target,
                    )
                else:
                    if args.disable_unity:
                        render_frequency = None
                    elif args.fast_mode:
                        render_frequency = 70
                    if obs_mode == "vision":
                        unity_options = [(args.render_obs, True, False, True)]
                    elif obs_mode == "gt":
                        unity_options = [(False, True, False, True)]

                # Compute the animation target.
                if stage in ["plan", "retract"]:
                    b_ani_tar = None
                else:
                    if stage in ["reach", "grasp"]:
                        b_ani_tar = env.initial_obs[env.src_idx]["position"]
                    elif stage in ["transport", "place", "release"]:
                        if task == "place":
                            b_ani_tar = place_dst_xy + [
                                env.initial_obs[env.src_idx]["height"]
                            ]
                        elif task == "stack":
                            b_ani_tar = env.initial_obs[env.dst_idx][
                                "position"
                            ]
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
                # b_ani_tar = None
                # Rendering block.
                if render_frequency is not None and i % render_frequency == 0:
                    for (
                        render_obs,
                        render_hallucinations,
                        send_image,
                        should_step,
                    ) in unity_options:
                        # If we are rendering observations, add them to the
                        # render state.
                        render_state = copy.deepcopy(state)
                        if render_obs:
                            # h_odicts = compute_obs_w_gt_orn(
                            #     obs=env.obs, gt_odicts=scene
                            # )
                            h_odicts = env.obs
                            render_state = add_hallucinations_to_state(
                                state=render_state,
                                h_odicts=h_odicts,
                                color=None,
                            )
                        if render_hallucinations:
                            if task == "place":
                                place_target = copy.deepcopy(dest_object)
                                place_target["position"][2] -= (
                                    place_target["height"] / 2
                                )
                                place_target["height"] = 0.005
                                render_state = add_hallucinations_to_state(
                                    state=render_state,
                                    h_odicts=[place_target],
                                    color="clear",
                                )
                        if send_image:
                            bullet_camera_targets = last_bullet_camera_targets
                        else:
                            bullet_camera_targets = {}

                        state_id = f"{scene_idx:06}_{env.timestep:06}"

                        # Encode, send, receive, and decode.
                        message = interface.encode(
                            state_id=state_id,
                            bullet_state=render_state,
                            bullet_animation_target=b_ani_tar,
                            bullet_camera_targets=bullet_camera_targets,
                        )
                        await websocket.send(message)
                        reply = await websocket.recv()
                        data = interface.decode(
                            state_id,
                            reply,
                            bullet_camera_targets=bullet_camera_targets,
                        )

                        # Hand the data to the env for processing.
                        if send_image:
                            env.set_unity_data(data)
                        if should_step:
                            is_done, success = env.step()
                else:
                    is_done, success = env.step()

                # Break out if we're done with the sequence, or it failed.
                if is_done or not success:
                    # Only use qs for next trial if we finished the entire
                    # sequence and it was successful.
                    if is_done and success:
                        init_arm_q, init_fin_q = env.w.get_robot_q()
                    else:
                        init_arm_q, init_fin_q = None, None
                    env.cleanup()
                    break

                if stage != "plan":
                    i += 1
            # print(f"end of scene:")
            # pprint.pprint(scene)
            # input("Press enter to continue")
            # del env
    sys.exit(0)


def generate_scenes(seed: int, n_scenes: int, disable_orientation: bool):
    # Top object, with different min radius.
    generator_top = RandomObjectsGenerator(
        seed=seed,
        n_objs_bounds=(1, 1),
        obj_dist_thresh=OBJECT_DIST_THRESH,
        max_retries=50,
        shapes=["box", "cylinder"],
        colors=["blue", "green", "red", "yellow"],
        radius_bounds=(utils.HALF_W_MIN_BTM, utils.HALF_W_MAX),
        height_bounds=(utils.H_MIN, utils.H_MAX),
        x_bounds=(utils.TX_MIN, utils.TX_MAX),
        y_bounds=(utils.TY_MIN, utils.TY_MAX),
        z_bounds=(0.0, 0.0),
        mass_bounds=(utils.MASS_MIN, utils.MASS_MAX),
        mu_bounds=(OPTIONS.obj_mu, OPTIONS.obj_mu),
        position_mode="com",
        disable_orientation=disable_orientation,
    )
    generator_bottom = RandomObjectsGenerator(
        seed=seed,
        n_objs_bounds=(1, 1),
        obj_dist_thresh=OBJECT_DIST_THRESH,
        max_retries=50,
        shapes=["box", "cylinder"],
        colors=["blue", "green", "red", "yellow"],
        radius_bounds=(utils.HALF_W_MIN, utils.HALF_W_MAX),
        height_bounds=(utils.H_MIN, utils.H_MAX),
        x_bounds=(utils.TX_MIN, utils.TX_MAX),
        y_bounds=(utils.TY_MIN, utils.TY_MAX),
        z_bounds=(0.0, 0.0),
        mass_bounds=(utils.MASS_MIN, utils.MASS_MAX),
        mu_bounds=(OPTIONS.obj_mu, OPTIONS.obj_mu),
        position_mode="com",
        disable_orientation=disable_orientation,
    )
    # Remaining objects.
    generator_all = RandomObjectsGenerator(
        seed=seed,
        n_objs_bounds=(2, 4),  # Maximum number of objects allowed by OR is 6.
        obj_dist_thresh=OBJECT_DIST_THRESH,
        max_retries=50,
        shapes=["box", "cylinder", "sphere"],
        colors=["red", "yellow"],
        radius_bounds=(utils.HALF_W_MIN, utils.HALF_W_MAX),
        height_bounds=(utils.H_MIN, utils.H_MAX),
        x_bounds=(utils.TX_MIN, utils.TX_MAX),
        y_bounds=(utils.TY_MIN, utils.TY_MAX),
        z_bounds=(0.0, 0.0),
        mass_bounds=(utils.MASS_MIN, utils.MASS_MAX),
        mu_bounds=(OPTIONS.obj_mu, OPTIONS.obj_mu),
        position_mode="com",
        disable_orientation=disable_orientation,
    )
    scenes = []
    for _ in range(n_scenes):
        top_scene = generator_top.generate_tabletop_objects()
        bottom_scene = generator_bottom.generate_tabletop_objects(
            existing_odicts=top_scene
        )
        all_scene = []
        all_scene = generator_all.generate_tabletop_objects(
            existing_odicts=top_scene + bottom_scene
        )
        scene = top_scene + bottom_scene

        # We still generate surrounding objects even if this flag is turned off
        # because we want the top and bottom objects to be the same w/ and w/o
        # this flag.
        if ADD_SURROUNDING_OBJECTS:
            scene += all_scene
        scenes.append(scene)
    return scenes


def add_hallucinations_to_state(state: Dict, h_odicts: Dict, color: str):
    state = copy.deepcopy(state)
    h_odicts = copy.deepcopy(h_odicts)
    # print(f"h_odicts:")
    # pprint.pprint(h_odicts)
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
    # input("x")
    # print("state:")
    # pprint.pprint(state)
    return state


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        type=str,
        default="172.27.76.64",
        # default="localhost",
        help="The hostname of the server.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port of the server."
    )
    parser.add_argument(
        "--disable_unity",
        action="store_true",
        help="Whether to disable unity.",
    )
    parser.add_argument(
        "--render_bullet",
        action="store_true",
        help="Whether to render PyBullet using OpenGL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Seed to use for scene generation.",
    )
    parser.add_argument(
        "--n_scenes",
        type=int,
        default=100,
        help="Number of scenes to generate.",
    )
    parser.add_argument(
        "--demo_scenes",
        action="store_true",
        help="Whether to use the demo scenes.",
    )
    parser.add_argument(
        "--disable_orientation",
        action="store_true",
        help="Whether to disable randomizing orientation.",
    )
    parser.add_argument(
        "--render_frequency",
        type=int,
        default=2,
        help="The rendering frequency to use.",
    )
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        help="Whether to use fast mode. Useful for evaluation, not demo.",
    )
    parser.add_argument(
        "--render_obs",
        action="store_true",
        help="Whether to render the observations.",
    )
    args = parser.parse_args()

    if args.disable_unity:
        args.hostname = "localhost"

    # Start the python server.
    interface.run_server(
        hostname=args.hostname, port=args.port, handler=send_to_client
    )
