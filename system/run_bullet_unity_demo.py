"""Runs a bullet + unity demo."""

import argparse
import copy
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
    "place": "stack",
}


async def send_to_client(websocket, path):
    """Sends and receives data to and from Unity.
    
    Args:
        websocket: The websocket protocol instance.
        path: The URI path.
    """
    scenes = generate_scenes()

    for scene_idx in range(0, len(scenes)):
        scene = scenes[scene_idx]

        # Hard code top object to be green, and bottom object to be blue.
        scene[0]["shape"] = "cylinder"
        scene[0]["color"] = "blue"
        scene[0]["radius"] = utils.HALF_W_MIN_BTM  # Bottom object fatter
        scene[1]["shape"] = "cylinder"
        scene[1]["color"] = "green"
        dst_shape = scene[0]["shape"]
        src_shape = scene[1]["shape"]
        command = f"Put the green {src_shape} on top of the blue {dst_shape}"

        for obs_mode in ["gt", "vision"]:
            # for obs_mode in ["gt"]:
            # for obs_mode in ["vision"]:
            print(f"scene_idx: {scene_idx}")
            print(f"obs mode: {obs_mode}")

            env = DemoEnvironment(
                opt=OPTIONS,
                trial=scene_idx,
                scene=copy.deepcopy(scene),
                command=command,
                observation_mode=obs_mode,
                renderer="unity",
                visualize_bullet=False,
                visualize_unity=False,
            )

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
                    unity_options = [(False, True, True), (True, False, False)]
                    # unity_options = [(False, True, True)]

                    # Turn on moving the camera each frame again.
                    last_bullet_camera_targets = bullet2unity.states.create_bullet_camera_targets(
                        camera_control=STAGE2CAMERA_CONTROL[stage],
                        bullet_odicts=env.initial_obs,
                        use_oids=False,
                        should_save=False,
                        should_send=True,
                    )
                else:
                    render_frequency = 25
                    if obs_mode == "vision":
                        unity_options = [(True, False, True)]
                    elif obs_mode == "gt":
                        unity_options = [(False, False, True)]

                """
                Possible cases:
                
                    Render:
                        Get images.
                        Step (predict).
                        Render with predictions.
                    No render:
                        Step.
                """

                # Rendering block.
                if i % render_frequency == 0:
                    for render_obs, send_image, should_step in unity_options:
                        # If we are rendering observations, add them to the
                        # render state.
                        render_state = copy.deepcopy(state)
                        if render_obs:
                            render_state = add_obs_to_state(
                                state=render_state, obs=env.obs
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

                # Temporarily finish after planning.
                # if stage == "plan":
                #     break
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
        colors=["blue", "green"],
        radius_bounds=(utils.HALF_W_MIN_BTM, utils.HALF_W_MAX),
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
        n_objs_bounds=(1, 1),
        obj_dist_thresh=0.2,
        max_retries=50,
        shapes=["box", "cylinder"],
        colors=["blue", "green"],
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
        all_scene = generator_all.generate_tabletop_objects()
        scene = top_scene + all_scene
        scenes.append(scene)
    return scenes


def add_obs_to_state(state: Dict, obs: Dict):
    state = copy.deepcopy(state)
    if obs is not None:
        for oi, odict in enumerate(obs):
            odict["color"] = "red"
            state["objects"][f"o{oi}"] = odict
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
