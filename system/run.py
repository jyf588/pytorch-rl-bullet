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


STAGE2CAMERA_CONTROL = {
    "plan": "center",
    "place": "position",
}
ADD_SURROUNDING_OBJECTS = False


async def send_to_client(websocket, path):
    """Sends and receives data to and from Unity.
    
    Args:
        websocket: The websocket protocol instance.
        path: The URI path.
    """
    scenes = generate_scenes()

    for scene_idx in range(1, len(scenes)):
        scene = scenes[scene_idx]

        # Hard code top object to be green, and bottom object to be blue.
        scene[0]["color"] = "blue"
        scene[1]["color"] = "green"
        dst_shape = scene[0]["shape"]
        src_shape = scene[1]["shape"]
        command = f"Put the green {src_shape} on top of the blue {dst_shape}"

        for task in ["stack", "place"]:
            # for task in ["place"]:
            for obs_mode in ["gt", "vision"]:
                # for obs_mode in ["vision"]:
                # for obs_mode in ["gt"]:

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
                    visualize_bullet=False,
                    visualize_unity=False,
                    place_dst_xy=place_dst_xy,
                )
                cam_target = None

                # Send states one by one.
                i = 0
                while 1:
                    stage, stage_ts = env.get_current_stage()
                    state = env.get_state()

                    # Temporarily remove robot state.
                    # state = {"objects": state["objects"]}

                    # Only have lucas look at / send images back when planning or placing.
                    if obs_mode == "vision" and stage in ["plan", "place"]:
                        render_frequency = 2
                        # unity_options = [(False, True)]
                        unity_options = [
                            (False, False, True, True),
                            (True, True, False, False),
                        ]
                        # unity_options = [(False, True, True)]

                        if stage == "place" and stage_ts == 0:
                            if task == "place":
                                cam_target = place_dst_xy + [
                                    env.initial_obs[0]["height"]
                                ]
                            elif task == "stack":
                                cam_target = bullet2unity.states.get_first_object_camera_target(
                                    bullet_odicts=env.initial_obs
                                )

                        # Turn on moving the camera each frame again.
                        last_bullet_camera_targets = bullet2unity.states.create_bullet_camera_targets(
                            camera_control=STAGE2CAMERA_CONTROL[stage],
                            bullet_odicts=env.initial_obs,
                            use_oids=False,
                            should_save=False,
                            should_send=True,
                            position=cam_target,
                        )
                    else:
                        render_frequency = 30
                        if obs_mode == "vision":
                            unity_options = [(True, True, False, True)]
                        elif obs_mode == "gt":
                            unity_options = [(False, True, False, True)]

                    # Rendering block.
                    if i % render_frequency == 0:
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
                                render_state = add_hallucinations_to_state(
                                    state=render_state,
                                    h_odicts=env.obs,
                                    color=None,
                                )
                                # input("x")

                            if render_hallucinations:
                                if task == "place":
                                    render_state = add_hallucinations_to_state(
                                        state=render_state,
                                        h_odicts=[dest_object],
                                        color="clear",
                                    )
                                    # input("x")
                            if send_image:
                                bullet_camera_targets = (
                                    last_bullet_camera_targets
                                )
                            else:
                                bullet_camera_targets = {}

                            state_id = f"{scene_idx:06}_{env.timestep:06}"

                            # Encode, send, receive, and decode.
                            message = interface.encode(
                                state_id=state_id,
                                bullet_state=render_state,
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
                                is_done = env.step()
                    else:
                        is_done = env.step()

                    # Break out if we're done.
                    if is_done:
                        break

                    if stage != "plan":
                        i += 1

                del env
    sys.exit(0)


def generate_scenes():
    # Top object, with different min radius.
    generator_top = RandomObjectsGenerator(
        seed=OPTIONS.seed,
        n_objs_bounds=(1, 1),
        obj_dist_thresh=0.2,
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
    )
    generator_bottom = RandomObjectsGenerator(
        seed=OPTIONS.seed,
        n_objs_bounds=(1, 1),
        obj_dist_thresh=0.2,
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
    )
    # Remaining objects.
    generator_all = RandomObjectsGenerator(
        seed=OPTIONS.seed,
        n_objs_bounds=(2, 5),
        obj_dist_thresh=0.2,
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
    )
    scenes = []
    for _ in range(100):
        top_scene = generator_top.generate_tabletop_objects()
        bottom_scene = generator_bottom.generate_tabletop_objects(
            existing_odicts=top_scene
        )
        all_scene = []
        if ADD_SURROUNDING_OBJECTS:
            all_scene = generator_all.generate_tabletop_objects(
                existing_odicts=top_scene + bottom_scene
            )

        scene = top_scene + bottom_scene + all_scene
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
        help="The hostname of the server.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port of the server."
    )
    args = parser.parse_args()

    # Start the python server.
    interface.run_server(
        hostname=args.hostname, port=args.port, handler=send_to_client
    )
