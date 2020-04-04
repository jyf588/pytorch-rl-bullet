"""A server interface between Python and Unity. This server loads states, sends
them to Unity, and processes received data from Unity.
"""

import asyncio
import argparse
import base64
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
    scene = SceneGenerator(
        base_scene=demo.base_scenes.SCENE, seed=OPTIONS.seed, mu=OPTIONS.obj_mu
    ).generate()

    env = DemoEnvironment(
        opt=OPTIONS,
        scene=scene,
        command="Put the green box on top of the blue cylinder",
        observation_mode="vision",
        renderer="unity",
        visualize_bullet=False,
        visualize_unity=False,
    )

    # Send states one by one.
    is_done = False
    while 1:
        stage, _ = env.get_current_stage()

        # Only have lucas look at / send images back when planning or placing.
        if stage in ["plan", "place"]:
            look_at_oids = env.world.oids
        else:
            look_at_oids = []

        state_id = f"{env.timestep:06}"
        message = encode(state_id, env.get_state(), look_at_oids)

        # Send and get reply.
        await websocket.send(message)
        reply = await websocket.recv()

        received_state_id, data = decode(reply, look_at_oids)

        # Verify that the sent ID and received ID are equivalent.
        assert received_state_id == state_id

        # Hand the data to the env for processing.
        env.set_unity_data(data)

        # If we've reached the end of the sequence, we are done.
        is_done = env.step()
        if is_done:
            sys.exit(0)


def encode(state_id: str, bullet_state: List[Any], look_at_oids) -> str:
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
    
    Returns:
        message: The message to send to unity.
    """
    unity_state = bullet2unity.states.bullet2unity_state(
        bullet_state=bullet_state, look_at_oids=look_at_oids
    )

    # Combine the id and state, and stringify into a msg.
    message = [state_id] + unity_state
    message = str(message)
    print(f"Sending to unity: state {state_id}\tTime: {time.time()}")
    return message


def decode(reply: str, look_at_oids: List[int]):
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
                "image": <image>,
                <otag>:{
                    "camera_position": <camera_position>,
                    "camera_orientation": <camera_orientation>,
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

    # Extract the state ID.
    state_id = reply[0]

    idx = 1
    data = {}
    for _ in range(len(look_at_oids)):
        unity_otag = reply[idx]
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
        data[int(unity_otag)] = {
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
