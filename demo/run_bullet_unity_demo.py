"""A server interface between Python and Unity. This server loads states, sends
them to Unity, and processes received data from Unity.
"""

import asyncio
import argparse
import base64
import copy
import cv2
import functools
import imageio
from io import BytesIO
import json
import math
import numpy as np
import os
from PIL import Image
import pprint
import pybullet as p
import websockets
import random
import sys
import time
from typing import *

import bullet2unity.states
import demo.base_scenes
from demo.env import DemoEnvironment
from demo.options import OPTIONS
from demo.scene import SceneGenerator
import my_pybullet_envs.utils as utils
from ns_vqa_dart.bullet.random_objects import RandomObjectsGenerator

global args


def run_server(hostname: str, port: int, handler: Callable):
    """Runs the python server.

    Args:
        hostname: The hostname to run the server on.
        port: The port to run the server on.
        handler: The handler coroutine to run for each websocket connection.
    """
    print(f"Running server on {hostname}:{port}")

    start_server = websockets.serve(handler, hostname, port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


async def send_to_client(websocket, path):
    """Sends and receives data to and from Unity.
    
    Args:
        websocket: The websocket protocol instance.
        path: The URI path.
    """
    generator = RandomObjectsGenerator(
        seed=OPTIONS.seed,
        n_objs_bounds=(2, 2),
        obj_dist_thresh=0.2,
        max_retries=50,
        shapes=["box", "cylinder"],
        radius_bounds=(utils.HALF_W_MIN, utils.HALF_W_MAX),
        height_bounds=(utils.H_MIN, utils.H_MAX),
        x_bounds=(utils.TX_MIN, utils.TX_MAX),
        y_bounds=(utils.TY_MIN, utils.TY_MAX),
        z_bounds=(0.0, 0.0),
        mass_bounds=(utils.MASS_MIN, utils.MASS_MAX),
        mu_bounds=(OPTIONS.obj_mu, OPTIONS.obj_mu),
        position_mode="com",
    )
    scenes = [generator.generate_tabletop_objects() for _ in range(100)]

    for scene_idx in range(5, 100):
        scene = scenes[scene_idx]
        scene[0]["color"] = "green"
        scene[1]["color"] = "blue"
        src_shape = scene[0]["shape"]
        dst_shape = scene[1]["shape"]

        # for obs_mode in ["gt", "vision"]:
        for obs_mode in ["vision"]:
            print(f"scene_idx: {scene_idx}")
            print(f"obs mode: {obs_mode}")

            env = DemoEnvironment(
                opt=OPTIONS,
                scene=copy.deepcopy(scene),
                command=f"Put the green {src_shape} on top of the blue {dst_shape}",
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
                state = {"objects": state["objects"]}

                # Only have lucas look at / send images back when planning or placing.
                if obs_mode == "vision" and stage in ["plan", "place"]:
                    render_frequency = 2
                    # unity_options = [(False, True)]
                    unity_options = [(False, True), (True, False)]

                    if stage == "place" and stage_ts > 0:
                        pass
                    else:
                        last_bullet_camera_targets = {}
                        for tid, odict in enumerate(state["objects"].values()):
                            last_bullet_camera_targets[tid] = odict["position"]
                else:
                    render_frequency = 20
                    unity_options = [(True, False)]

                if i % render_frequency == 0:
                    # First, render only states and get images. Then, render
                    # both states and observations, but don't get images.
                    for render_obs, get_images in unity_options:
                        # If we are rendering observations, add them to the
                        # render state.
                        render_state = copy.deepcopy(state)
                        if render_obs:
                            render_state = add_obs_to_state(
                                state=render_state, obs=env.obs
                            )

                        # If we are getting images, get object indexes from the
                        # state.
                        if get_images:
                            bullet_camera_targets = last_bullet_camera_targets
                        else:
                            bullet_camera_targets = {}

                        state_id = f"{env.timestep:06}"
                        message = encode(
                            state_id=state_id,
                            bullet_state=render_state,
                            bullet_camera_targets=bullet_camera_targets,
                        )

                        # Send and get reply.
                        await websocket.send(message)
                        reply = await websocket.recv()

                        received_state_id, data = decode(
                            reply,
                            target_ids=list(bullet_camera_targets.keys()),
                        )

                        # Verify that the sent ID and received ID are equivalent.
                        assert received_state_id == state_id

                        # Hand the data to the env for processing.
                        if get_images:
                            env.set_unity_data(data)

                # If we've reached the end of the sequence, we are done.
                is_done = env.step()
                if is_done:
                    break

                if stage != "plan":
                    i += 1
            del env
    sys.exit(0)


def add_obs_to_state(state: Dict, obs: Dict):
    state = copy.deepcopy(state)
    if obs is not None:
        for oi, odict in enumerate(obs):
            odict["color"] = "red"
            state["objects"][f"o{oi}"] = odict
    return state


def encode(
    state_id: str, bullet_state: List[Any], bullet_camera_targets
) -> str:
    """Converts the provided bullet state into a Unity state, and encodes the
    message for sending to Unity.

    Args:
        state_id: The ID of the state.
        bullet_state: The Bullet state dictionary, with the format {
            "objects": {
                "<oid>": {
                    "shape": shape,
                    "color": color,
                    "radius": radius,
                    "height": height,
                    "orientation": [x, y, z, w],
                    "position": [x, y, z]
                },
                ...
            },
            "robot": {
                "<joint_name>": <joint_angle>,
                ...
            }
        }. Note that if "robot" key is not provided, the default robot pose 
        will be used.
        look_at_idxs: Object idxs to look at.

    Returns:
        message: The message to send to unity.
    """
    unity_state = bullet2unity.states.bullet2unity_state(
        bullet_state=bullet_state, bullet_camera_targets=bullet_camera_targets
    )

    # Combine the id and state, and stringify into a msg.
    message = [state_id] + unity_state
    message = str(message)
    print(f"Sending to unity: state {state_id}\tTime: {time.time()}")
    return message


def decode(reply: str, target_ids: List[int]):
    """Decodes messages received from Unity.

    Args:
        reply: The string message from the client, which is expected to be a
            comma separated string containing the following components:
            [
                <state_id>,
                <object_tag>,
                <camera_position>,
                <camera_orientation>,
                <image>,
                ...
            ]
        object_tags: A list of object tags.
    
    Returns:
        state_id: The state ID received in the reply.
        data: A dictionary containing the data in the message, in the format
            {
                <otag>:{
                    "camera_position": <camera_position>,
                    "camera_orientation": <camera_orientation>,
                    "image": <image>,
                },
                ...
            }
    """
    print(
        f"Received from client: {len(reply)} characters\tTime: {time.time()}"
    )
    # Split components by comma.
    reply = reply.split(",")
    print(f"Number of reply components: {len(reply)}")
    print(f"Attempting to parse unity data for {len(target_ids)} objects...")
    # Extract the state ID.
    state_id = reply[0]

    idx = 1
    data = {}
    for target_id in range(len(target_ids)):
        unity_target_id = int(reply[idx])
        print(f"target_id: {target_id}")
        print(f"unity_target_id: {unity_target_id}")
        assert unity_target_id == target_id
        idx += 1

        camera_position = [float(x) for x in reply[idx : idx + 3]]
        idx += 3
        camera_orientation = [float(x) for x in reply[idx : idx + 4]]
        idx += 4

        # Convert from base 64 to an image tensor.
        rgb = base64_to_numpy_image(b64=reply[idx])
        idx += 1
        seg_img = base64_to_numpy_image(b64=reply[idx])
        idx += 1

        # Store the data.
        data[int(unity_target_id)] = {
            "camera_position": camera_position,
            "camera_orientation": camera_orientation,
            "rgb": rgb,
            "seg_img": seg_img,
        }

    return state_id, data


def base64_to_numpy_image(b64: str) -> np.ndarray:
    """Decodes a base 64 string representation of an image into a numpy array.
    Args:
        b64: The base 64 text to decode.
    
    Returns:
        image: The numpy version of the image.
    """
    image = np.array(Image.open(BytesIO(base64.b64decode(b64))))
    return image


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
    run_server(hostname=args.hostname, port=args.port, handler=send_to_client)
